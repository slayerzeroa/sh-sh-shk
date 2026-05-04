from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from .env import load_local_env
from .models import CostBreakdown, GeneratedImage, GenerationRequest, ImageUsage, ModelEndpointConfig
from .pricing import calculate_image_cost


class ImageProvider(Protocol):
    def generate(self, request: GenerationRequest) -> GeneratedImage:
        ...


def build_provider_payload(request: GenerationRequest) -> dict[str, object]:
    return {
        "system_prompt": request.system_prompt,
        "prompt": request.positive_prompt,
        "negative_prompt": request.negative_prompt,
        "references": [
            {
                "asset_id": reference.asset_id,
                "character_id": reference.character_id,
                "source": reference.source,
            }
            for reference in request.references
        ],
        "seed": request.seed,
        "metadata": request.metadata,
    }


def build_image_model_payload(
    request: GenerationRequest,
    config: ModelEndpointConfig,
) -> dict[str, object]:
    payload = {
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "api_key_configured": bool(resolve_api_key(config, allow_missing=True)),
        "request": build_provider_payload(request),
    }
    if config.stage == "refine":
        if config.provider == "gemini" or config.model.startswith("gemini-"):
            payload["gemini_generate_content_request"] = build_gemini_image_request(request, config)
        else:
            payload["openai_image_request"] = build_openai_image_request(request, config)
    return payload


def build_bubble_fill_payload(
    request,
    config: ModelEndpointConfig,
) -> dict[str, object]:
    return {
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "api_key_configured": bool(resolve_api_key(config, allow_missing=True)),
        "system_prompt": request.system_prompt,
        "user_prompt": request.user_prompt,
        "lines": [
            {
                "speaker": line.speaker,
                "text": line.text,
                "slot_hint": line.slot_hint,
                "purpose": line.purpose,
            }
            for line in request.lines
        ],
        "max_output_tokens": request.max_output_tokens,
        "metadata": request.metadata,
    }


def resolve_api_key(config: ModelEndpointConfig, *, allow_missing: bool = False) -> str:
    load_local_env()
    if config.api_key:
        return config.api_key
    if config.api_key_env:
        value = os.getenv(config.api_key_env, "")
        if value:
            return value
    if allow_missing:
        return ""
    raise ValueError(
        f"Missing API key for model {config.model}. "
        f"Set {config.api_key_env or 'the configured API key env var'} or config.api_key."
    )


def build_openai_image_prompt(request: GenerationRequest) -> str:
    sections = [
        "SYSTEM CONTINUITY GUARDRAIL",
        request.system_prompt,
        "TARGET IMAGE REQUEST",
        request.positive_prompt,
    ]
    if request.negative_prompt:
        sections.extend(
            [
                "AVOID",
                request.negative_prompt,
            ]
        )
    return "\n\n".join(section for section in sections if section)


def build_gemini_image_prompt(request: GenerationRequest) -> str:
    sections = [
        "SYSTEM CONTINUITY GUARDRAIL",
        request.system_prompt,
        "TARGET IMAGE REQUEST",
        request.positive_prompt,
    ]
    if request.negative_prompt:
        sections.extend(["AVOID", request.negative_prompt])
    return "\n\n".join(section for section in sections if section)


def build_openai_image_request(
    request: GenerationRequest,
    config: ModelEndpointConfig,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": config.model,
        "prompt": build_openai_image_prompt(request),
        "size": config.size,
        "quality": config.quality,
        "moderation": config.moderation,
        "n": config.n,
        "output_format": config.output_format,
    }
    if config.background:
        payload["background"] = config.background
    if config.output_compression is not None:
        payload["output_compression"] = config.output_compression
    return payload


def size_to_aspect_ratio(size: str) -> str:
    mapping = {
        "1024x1024": "1:1",
        "1024x1536": "2:3",
        "1536x1024": "3:2",
        "1024x1792": "4:7",
        "1792x1024": "7:4",
    }
    return mapping.get(size, "1:1")


def build_gemini_image_request(
    request: GenerationRequest,
    config: ModelEndpointConfig,
) -> dict[str, object]:
    generation_config: dict[str, object] = {
        "responseModalities": ["TEXT", "IMAGE"],
        "imageConfig": {
            "aspectRatio": size_to_aspect_ratio(config.size),
        },
    }
    payload: dict[str, object] = {
        "contents": [
            {
                "parts": [
                    {
                        "text": build_gemini_image_prompt(request),
                    }
                ]
            }
        ],
        "generationConfig": generation_config,
    }
    return payload


def parse_openai_usage(payload: dict[str, object] | None) -> ImageUsage | None:
    if not payload:
        return None
    input_details = payload.get("input_tokens_details", {}) if isinstance(payload, dict) else {}
    output_details = payload.get("output_tokens_details", {}) if isinstance(payload, dict) else {}
    input_tokens = int(payload.get("input_tokens", 0) or 0)
    output_tokens = int(payload.get("output_tokens", 0) or 0)
    input_image_tokens = int(getattr(input_details, "get", lambda *_: 0)("image_tokens", 0) or 0)
    input_text_tokens = int(getattr(input_details, "get", lambda *_: 0)("text_tokens", 0) or 0)
    if input_text_tokens == 0 and input_tokens:
        input_text_tokens = input_tokens
    output_image_tokens = int(getattr(output_details, "get", lambda *_: 0)("image_tokens", 0) or 0)
    if output_image_tokens == 0 and output_tokens:
        output_image_tokens = output_tokens
    partial_image_tokens = int(
        getattr(output_details, "get", lambda *_: 0)("partial_image_tokens", 0) or 0
    )
    return ImageUsage(
        input_text_tokens=input_text_tokens,
        input_image_tokens=input_image_tokens,
        output_image_tokens=output_image_tokens,
        partial_image_tokens=partial_image_tokens,
        source="api",
    )


def _extract_modality_count(
    details: object,
    *,
    modality_names: tuple[str, ...],
) -> int:
    if not isinstance(details, list):
        return 0
    total = 0
    for item in details:
        if not isinstance(item, dict):
            continue
        modality = str(item.get("modality", "") or item.get("mimeType", "")).upper()
        if modality in modality_names:
            total += int(item.get("tokenCount", item.get("token_count", 0)) or 0)
    return total


def parse_gemini_usage(payload: dict[str, object] | None) -> ImageUsage | None:
    if not isinstance(payload, dict):
        return None
    prompt_text_tokens = _extract_modality_count(
        payload.get("promptTokensDetails"),
        modality_names=("TEXT",),
    )
    prompt_image_tokens = _extract_modality_count(
        payload.get("promptTokensDetails"),
        modality_names=("IMAGE",),
    )
    candidate_image_tokens = _extract_modality_count(
        payload.get("candidatesTokensDetails"),
        modality_names=("IMAGE",),
    )
    if prompt_text_tokens == 0:
        prompt_text_tokens = int(payload.get("promptTokenCount", 0) or 0)
    return ImageUsage(
        input_text_tokens=prompt_text_tokens,
        input_image_tokens=prompt_image_tokens,
        output_image_tokens=candidate_image_tokens,
        source="api",
    )


def estimate_generation_cost(
    request: GenerationRequest,
    config: ModelEndpointConfig,
    *,
    usage: ImageUsage | None = None,
) -> CostBreakdown:
    return calculate_image_cost(
        config.model,
        config.size,
        config.quality,
        usage=usage,
        prompt=build_gemini_image_prompt(request)
        if config.provider == "gemini" or config.model.startswith("gemini-")
        else build_openai_image_prompt(request),
        num_images=config.n,
    )


class MockImageProvider:
    provider_name = "mock"

    def __init__(self, config: ModelEndpointConfig | None = None):
        self.config = config

    def generate(self, request: GenerationRequest) -> GeneratedImage:
        payload = build_provider_payload(request)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        cost = estimate_generation_cost(request, self.config) if self.config is not None else None
        return GeneratedImage(
            asset_id=f"mock-{digest}",
            panel_id=request.panel_id,
            provider_name=self.provider_name,
            seed=request.seed,
            prompt_fingerprint=digest,
            model=self.config.model if self.config is not None else "",
            output_format=self.config.output_format if self.config is not None else "png",
            cost=cost,
        )


class OpenAIImageProvider:
    provider_name = "openai-image-api"

    def __init__(self, config: ModelEndpointConfig):
        self.config = config

    def generate(self, request: GenerationRequest) -> GeneratedImage:
        api_key = resolve_api_key(self.config)
        payload = build_openai_image_request(request, self.config)
        response_payload = self._post_json(payload, api_key)
        usage = parse_openai_usage(response_payload.get("usage"))
        image_item = self._get_first_image_item(response_payload)
        image_bytes = self._extract_image_bytes(image_item)
        saved_path = self._save_image(request.panel_id, image_bytes)
        revised_prompt = str(image_item.get("revised_prompt", ""))
        prompt_fingerprint = hashlib.sha256(payload["prompt"].encode("utf-8")).hexdigest()[:12]
        cost = estimate_generation_cost(request, self.config, usage=usage)
        return GeneratedImage(
            asset_id=self._resolve_asset_id(response_payload, request.panel_id, prompt_fingerprint),
            panel_id=request.panel_id,
            provider_name=self.provider_name,
            seed=request.seed,
            prompt_fingerprint=prompt_fingerprint,
            model=self.config.model,
            output_format=self.config.output_format,
            local_path=str(saved_path),
            revised_prompt=revised_prompt,
            usage=usage,
            cost=cost,
        )

    def _post_json(self, payload: dict[str, object], api_key: str) -> dict[str, object]:
        urls = [self._primary_url()]
        fallback_url = self._fallback_url()
        if fallback_url not in urls:
            urls.append(fallback_url)

        last_error: Exception | None = None
        for url in urls:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._build_headers(api_key),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 404 and url != urls[-1]:
                    last_error = RuntimeError(f"Image endpoint not found at {url}: {body}")
                    continue
                raise RuntimeError(f"OpenAI image request failed ({exc.code}): {body}") from exc
            except urllib.error.URLError as exc:
                last_error = exc
                break
        raise RuntimeError(f"OpenAI image request failed: {last_error}") from last_error

    def _build_headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        organization = os.getenv("OPENAI_ORGANIZATION", "")
        project = os.getenv("OPENAI_PROJECT", "") or os.getenv("OPENAI_PROJECT_ID", "")
        if organization:
            headers["OpenAI-Organization"] = organization
        if project:
            headers["OpenAI-Project"] = project
        return headers

    def _primary_url(self) -> str:
        base_url = (self.config.base_url or "https://api.openai.com").rstrip("/")
        return f"{base_url}/v1/images"

    def _fallback_url(self) -> str:
        base_url = (self.config.base_url or "https://api.openai.com").rstrip("/")
        return f"{base_url}/v1/images/generations"

    def _get_first_image_item(self, response_payload: dict[str, object]) -> dict[str, object]:
        data = response_payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("OpenAI image response did not contain any images.")
        item = data[0]
        if not isinstance(item, dict):
            raise RuntimeError("OpenAI image response had an unexpected data payload.")
        return item

    def _extract_image_bytes(self, image_item: dict[str, object]) -> bytes:
        b64_json = image_item.get("b64_json")
        if isinstance(b64_json, str) and b64_json:
            return base64.b64decode(b64_json)
        image_url = image_item.get("url")
        if isinstance(image_url, str) and image_url:
            with urllib.request.urlopen(image_url, timeout=self.config.timeout_seconds) as response:
                return response.read()
        raise RuntimeError("OpenAI image response did not contain b64_json or url output.")

    def _save_image(self, panel_id: str, image_bytes: bytes) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{panel_id}.{self.config.output_format}"
        destination.write_bytes(image_bytes)
        return destination.resolve()

    def _resolve_asset_id(
        self,
        response_payload: dict[str, object],
        panel_id: str,
        prompt_fingerprint: str,
    ) -> str:
        created = response_payload.get("created")
        if created is not None:
            return f"{self.config.model}-{panel_id}-{created}"
        return f"{self.config.model}-{prompt_fingerprint}"


class GeminiImageProvider:
    provider_name = "gemini-generate-content"

    def __init__(self, config: ModelEndpointConfig):
        self.config = config

    def generate(self, request: GenerationRequest) -> GeneratedImage:
        if self.config.n != 1:
            raise ValueError("Gemini image generation currently supports n=1 in this project.")
        api_key = resolve_api_key(self.config)
        payload = build_gemini_image_request(request, self.config)
        response_payload = self._post_json(payload, api_key)
        usage = parse_gemini_usage(response_payload.get("usageMetadata"))
        image_bytes, mime_type, text_response = self._extract_output_parts(response_payload)
        output_format = mime_type.split("/", 1)[-1] if "/" in mime_type else self.config.output_format
        saved_path = self._save_image(request.panel_id, image_bytes, output_format)
        prompt_fingerprint = hashlib.sha256(
            build_gemini_image_prompt(request).encode("utf-8")
        ).hexdigest()[:12]
        cost = estimate_generation_cost(request, self.config, usage=usage)
        return GeneratedImage(
            asset_id=self._resolve_asset_id(response_payload, request.panel_id, prompt_fingerprint),
            panel_id=request.panel_id,
            provider_name=self.provider_name,
            seed=request.seed,
            prompt_fingerprint=prompt_fingerprint,
            model=self.config.model,
            output_format=output_format,
            local_path=str(saved_path),
            revised_prompt=text_response,
            usage=usage,
            cost=cost,
        )

    def _post_json(self, payload: dict[str, object], api_key: str) -> dict[str, object]:
        url = self._url()
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini image request failed ({exc.code}): {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini image request failed: {exc}") from exc

    def _url(self) -> str:
        base_url = (self.config.base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        return f"{base_url}/v1beta/models/{self.config.model}:generateContent"

    def _extract_output_parts(self, response_payload: dict[str, object]) -> tuple[bytes, str, str]:
        candidates = response_payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Gemini image response did not contain any candidates.")
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise RuntimeError("Gemini image response had an unexpected candidate payload.")
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            raise RuntimeError("Gemini image response did not contain content.")
        parts = content.get("parts")
        if not isinstance(parts, list):
            raise RuntimeError("Gemini image response did not contain content parts.")

        collected_text: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if "text" in part and isinstance(part["text"], str):
                collected_text.append(part["text"])
            inline_data = part.get("inline_data") or part.get("inlineData")
            if isinstance(inline_data, dict):
                mime_type = str(
                    inline_data.get("mime_type")
                    or inline_data.get("mimeType")
                    or "image/png"
                )
                data = inline_data.get("data")
                if isinstance(data, str) and data:
                    return base64.b64decode(data), mime_type, "\n".join(collected_text).strip()
        raise RuntimeError("Gemini image response did not contain any inline image data.")

    def _save_image(self, panel_id: str, image_bytes: bytes, output_format: str) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{panel_id}.{output_format}"
        destination.write_bytes(image_bytes)
        return destination.resolve()

    def _resolve_asset_id(
        self,
        response_payload: dict[str, object],
        panel_id: str,
        prompt_fingerprint: str,
    ) -> str:
        response_id = response_payload.get("responseId")
        if response_id is not None:
            return f"{self.config.model}-{panel_id}-{response_id}"
        return f"{self.config.model}-{prompt_fingerprint}"
