from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import BatchJobInfo, GeneratedImage, GenerationRequest, ImageUsage, ModelEndpointConfig
from .providers import (
    build_gemini_image_request,
    estimate_generation_cost,
    parse_gemini_usage,
    resolve_api_key,
)


_COMPLETED_BATCH_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "BATCH_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
}


def read_manifest(source: Path) -> dict[str, object]:
    return json.loads(source.read_text(encoding="utf-8"))


def update_manifest(destination: Path, updates: dict[str, object]) -> Path:
    payload = read_manifest(destination) if destination.exists() else {}
    payload.update(updates)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


def write_json_artifact(destination: Path, payload: dict[str, object]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


@dataclass(frozen=True)
class GeminiBatchClient:
    config: ModelEndpointConfig

    def build_inline_batch_body(
        self,
        requests: tuple[GenerationRequest, ...],
        *,
        display_name: str,
        priority: int = 0,
    ) -> dict[str, object]:
        return {
            "batch": {
                "displayName": display_name,
                "inputConfig": {
                    "requests": {
                        "requests": [
                            {
                                "request": build_gemini_image_request(request, self.config),
                                "metadata": {
                                    "panelId": request.panel_id,
                                    "requestIndex": index,
                                },
                            }
                            for index, request in enumerate(requests)
                        ]
                    }
                },
                "priority": str(priority),
            }
        }

    def write_inline_manifest(
        self,
        requests: tuple[GenerationRequest, ...],
        *,
        display_name: str,
        destination: Path,
        extra: dict[str, object] | None = None,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "display_name": display_name,
            "provider": "gemini-batch",
            "model": self.config.model,
            "request_count": len(requests),
            "requests": [
                {
                    "panel_id": request.panel_id,
                    "seed": request.seed,
                    "metadata": request.metadata,
                    "batch_request": build_gemini_image_request(request, self.config),
                }
                for request in requests
            ],
        }
        if extra:
            payload.update(extra)
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return destination

    def submit_inline_batch(
        self,
        requests: tuple[GenerationRequest, ...],
        *,
        display_name: str,
        priority: int = 0,
        manifest_path: str = "",
    ) -> BatchJobInfo:
        api_key = resolve_api_key(self.config)
        body = self.build_inline_batch_body(requests, display_name=display_name, priority=priority)
        response = self._request_json(
            self._batch_endpoint(),
            method="POST",
            api_key=api_key,
            payload=body,
        )
        name = self._extract_batch_name(response)
        state = self._extract_batch_state(response)
        return BatchJobInfo(
            name=name,
            display_name=display_name,
            provider="gemini-batch",
            model=self.config.model,
            state=state,
            request_count=len(requests),
            manifest_path=manifest_path,
            raw_response=response,
        )

    def get_batch(self, name: str) -> BatchJobInfo:
        api_key = resolve_api_key(self.config)
        response = self._request_json(
            self._batch_get_url(name),
            method="GET",
            api_key=api_key,
        )
        return BatchJobInfo(
            name=name,
            display_name=str(response.get("displayName", name)),
            provider="gemini-batch",
            model=self.config.model,
            state=self._extract_batch_state(response),
            request_count=int(response.get("batchStats", {}).get("requestCount", 0) or 0),
            raw_response=response,
        )

    def wait_for_batch(
        self,
        name: str,
        *,
        poll_seconds: int = 30,
        timeout_seconds: int = 3600,
    ) -> BatchJobInfo:
        deadline = time.time() + timeout_seconds
        batch = self.get_batch(name)
        while batch.state not in _COMPLETED_BATCH_STATES:
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for batch job {name}")
            time.sleep(poll_seconds)
            batch = self.get_batch(name)
        return batch

    def materialize_inline_batch_results(
        self,
        batch_job: BatchJobInfo,
        requests: tuple[GenerationRequest, ...],
        *,
        output_dir: Path,
    ) -> tuple[GeneratedImage, ...]:
        inline_responses = self._extract_inline_responses(batch_job.raw_response)
        images: list[GeneratedImage] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(inline_responses):
            if not isinstance(item, dict):
                continue
            request = requests[min(index, len(requests) - 1)]
            response_payload = item.get("response")
            if not isinstance(response_payload, dict):
                continue
            try:
                image, output_path = self._parse_generated_image(response_payload, request, output_dir)
            except RuntimeError:
                continue
            images.append(image)
        return tuple(images)

    def _extract_inline_responses(self, raw_response: dict[str, object]) -> list[dict[str, object]]:
        if not isinstance(raw_response, dict):
            return []

        candidates: list[list[dict[str, object]]] = []

        direct_response = raw_response.get("response")
        if isinstance(direct_response, dict):
            inlined = direct_response.get("inlinedResponses")
            if isinstance(inlined, dict):
                nested = inlined.get("inlinedResponses")
                if isinstance(nested, list):
                    candidates.append([item for item in nested if isinstance(item, dict)])

        direct_output = raw_response.get("output")
        if isinstance(direct_output, dict):
            inlined = direct_output.get("inlinedResponses")
            if isinstance(inlined, dict):
                nested = inlined.get("inlinedResponses")
                if isinstance(nested, list):
                    candidates.append([item for item in nested if isinstance(item, dict)])

        metadata = raw_response.get("metadata")
        if isinstance(metadata, dict):
            output = metadata.get("output")
            if isinstance(output, dict):
                inlined = output.get("inlinedResponses")
                if isinstance(inlined, dict):
                    nested = inlined.get("inlinedResponses")
                    if isinstance(nested, list):
                        candidates.append([item for item in nested if isinstance(item, dict)])

        for items in candidates:
            if items:
                return items
        return []

    def _parse_generated_image(
        self,
        response_payload: dict[str, object],
        request: GenerationRequest,
        output_dir: Path,
    ) -> tuple[GeneratedImage, Path]:
        candidates = response_payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Batch image response did not contain any candidates.")
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise RuntimeError("Batch image response candidate payload was malformed.")
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            raise RuntimeError("Batch image response did not contain content.")
        parts = content.get("parts")
        if not isinstance(parts, list):
            raise RuntimeError("Batch image response did not contain content parts.")

        revised_text = []
        image_bytes: bytes | None = None
        extension = self.config.output_format
        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                revised_text.append(part["text"])
            inline_data = part.get("inline_data") or part.get("inlineData")
            if isinstance(inline_data, dict):
                mime_type = str(inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png")
                data = inline_data.get("data")
                if isinstance(data, str) and data:
                    image_bytes = base64.b64decode(data)
                    if "/" in mime_type:
                        extension = mime_type.split("/", 1)[1]
                    break
        if image_bytes is None:
            raise RuntimeError("Batch image response did not contain inline image data.")
        output_path = output_dir / f"{request.panel_id}.{extension}"
        output_path.write_bytes(image_bytes)
        usage = parse_gemini_usage(response_payload.get("usageMetadata"))
        cost = estimate_generation_cost(request, self.config, usage=usage)
        image = GeneratedImage(
            asset_id=f"{self.config.model}-{request.panel_id}",
            panel_id=request.panel_id,
            provider_name="gemini-batch",
            seed=request.seed,
            prompt_fingerprint="batch",
            model=self.config.model,
            output_format=extension,
            local_path=str(output_path.resolve()),
            revised_prompt="\n".join(revised_text).strip(),
            usage=usage,
            cost=cost,
        )
        return image, output_path

    def _request_json(
        self,
        url: str,
        *,
        method: str,
        api_key: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini batch request failed ({exc.code}): {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini batch request failed: {exc}") from exc

    def _batch_endpoint(self) -> str:
        base_url = (self.config.base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        return f"{base_url}/v1beta/models/{self.config.model}:batchGenerateContent"

    def _batch_get_url(self, name: str) -> str:
        base_url = (self.config.base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        normalized_name = name.lstrip("/")
        return f"{base_url}/v1beta/{normalized_name}"

    def _extract_batch_name(self, response: dict[str, object]) -> str:
        if isinstance(response.get("name"), str) and response["name"]:
            return str(response["name"])
        batch = response.get("batch")
        if isinstance(batch, dict) and isinstance(batch.get("name"), str):
            return str(batch["name"])
        metadata = response.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("name"), str):
            return str(metadata["name"])
        raise RuntimeError("Gemini batch response did not include a batch name.")

    def _extract_batch_state(self, response: dict[str, object]) -> str:
        state = response.get("state")
        if isinstance(state, dict):
            return str(state.get("name", "BATCH_STATE_UNSPECIFIED"))
        if isinstance(state, str):
            return state
        batch = response.get("batch")
        if isinstance(batch, dict):
            inner_state = batch.get("state")
            if isinstance(inner_state, dict):
                return str(inner_state.get("name", "BATCH_STATE_UNSPECIFIED"))
            if isinstance(inner_state, str):
                return inner_state
        metadata = response.get("metadata")
        if isinstance(metadata, dict):
            inner_state = metadata.get("state")
            if isinstance(inner_state, dict):
                return str(inner_state.get("name", "BATCH_STATE_UNSPECIFIED"))
            if isinstance(inner_state, str):
                return inner_state
        return "BATCH_STATE_UNSPECIFIED"
