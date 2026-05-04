from __future__ import annotations

from math import ceil

from .models import CostBreakdown, ImageUsage

# Sources:
# - https://developers.openai.com/api/docs/guides/image-generation
# - https://developers.openai.com/api/docs/pricing
# - https://ai.google.dev/gemini-api/docs/image-generation
# - https://ai.google.dev/gemini-api/docs/pricing
# Verified against official docs on 2026-04-17.

_TOKEN_RATES_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-image-1.5": {
        "text_input": 5.00,
        "image_input": 8.00,
        "image_output": 32.00,
    },
    "chatgpt-image-latest": {
        "text_input": 5.00,
        "image_input": 8.00,
        "image_output": 32.00,
    },
    "gpt-image-1": {
        "text_input": 5.00,
        "image_input": 10.00,
        "image_output": 40.00,
    },
    "gpt-image-1-mini": {
        "text_input": 2.00,
        "image_input": 2.50,
        "image_output": 8.00,
    },
    "gemini-2.5-flash-image": {
        "text_input": 0.30,
        "image_input": 0.30,
        "image_output": 30.00,
    },
}

_PER_IMAGE_OUTPUT_PRICES: dict[str, dict[str, dict[str, float]]] = {
    "gpt-image-1.5": {
        "low": {"1024x1024": 0.009, "1024x1536": 0.013, "1536x1024": 0.013},
        "medium": {"1024x1024": 0.034, "1024x1536": 0.050, "1536x1024": 0.050},
        "high": {"1024x1024": 0.133, "1024x1536": 0.200, "1536x1024": 0.200},
    },
    "chatgpt-image-latest": {
        "low": {"1024x1024": 0.009, "1024x1536": 0.013, "1536x1024": 0.013},
        "medium": {"1024x1024": 0.034, "1024x1536": 0.050, "1536x1024": 0.050},
        "high": {"1024x1024": 0.133, "1024x1536": 0.200, "1536x1024": 0.200},
    },
    "gpt-image-1": {
        "low": {"1024x1024": 0.011, "1024x1536": 0.016, "1536x1024": 0.016},
        "medium": {"1024x1024": 0.042, "1024x1536": 0.063, "1536x1024": 0.063},
        "high": {"1024x1024": 0.167, "1024x1536": 0.250, "1536x1024": 0.250},
    },
    "gpt-image-1-mini": {
        "low": {"1024x1024": 0.005, "1024x1536": 0.006, "1536x1024": 0.006},
        "medium": {"1024x1024": 0.011, "1024x1536": 0.015, "1536x1024": 0.015},
        "high": {"1024x1024": 0.036, "1024x1536": 0.052, "1536x1024": 0.052},
    },
    "gemini-2.5-flash-image": {
        "standard": {"1024x1024": 0.039, "1024x1536": 0.039, "1536x1024": 0.039},
    },
}


def approximate_text_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return ceil(len(normalized) / 4)


def normalize_image_model(model: str) -> str:
    normalized = model.strip().lower()
    if normalized not in _TOKEN_RATES_PER_MILLION:
        raise ValueError(f"Unsupported image pricing model: {model}")
    return normalized


def calculate_image_cost(
    model: str,
    size: str,
    quality: str,
    *,
    usage: ImageUsage | None = None,
    prompt: str = "",
    num_images: int = 1,
) -> CostBreakdown:
    normalized_model = normalize_image_model(model)
    rates = _TOKEN_RATES_PER_MILLION[normalized_model]
    notes: list[str] = []

    if normalized_model == "gemini-2.5-flash-image":
        return _calculate_gemini_image_cost(
            normalized_model,
            size,
            usage=usage,
            prompt=prompt,
            num_images=num_images,
        )

    if usage is not None:
        input_text_cost = usage.input_text_tokens * rates["text_input"] / 1_000_000
        input_image_cost = usage.input_image_tokens * rates["image_input"] / 1_000_000
        output_tokens = usage.output_image_tokens + usage.partial_image_tokens
        output_image_cost = output_tokens * rates["image_output"] / 1_000_000
        total_cost = input_text_cost + input_image_cost + output_image_cost
        if usage.partial_image_tokens:
            notes.append("Partial image token charges included from API usage.")
        return CostBreakdown(
            model=normalized_model,
            input_text_cost_usd=round(input_text_cost, 6),
            input_image_cost_usd=round(input_image_cost, 6),
            output_image_cost_usd=round(output_image_cost, 6),
            total_cost_usd=round(total_cost, 6),
            estimated=usage.source != "api",
            notes=tuple(notes or ["Calculated from token usage returned by the API."]),
        )

    try:
        output_price = _PER_IMAGE_OUTPUT_PRICES[normalized_model][quality][size] * max(1, num_images)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported size/quality combination for {normalized_model}: {size} / {quality}"
        ) from exc

    estimated_text_tokens = approximate_text_tokens(prompt)
    input_text_cost = estimated_text_tokens * rates["text_input"] / 1_000_000
    total_cost = output_price + input_text_cost
    notes.extend(
        [
            "Output image cost uses the official per-image pricing table.",
            "Input text cost uses a rough token estimate based on prompt length.",
            "Input image token cost is excluded because no edit-image usage was provided.",
        ]
    )
    return CostBreakdown(
        model=normalized_model,
        input_text_cost_usd=round(input_text_cost, 6),
        output_image_cost_usd=round(output_price, 6),
        total_cost_usd=round(total_cost, 6),
        estimated=True,
        notes=tuple(notes),
    )


def _calculate_gemini_image_cost(
    normalized_model: str,
    size: str,
    *,
    usage: ImageUsage | None,
    prompt: str,
    num_images: int,
) -> CostBreakdown:
    rates = _TOKEN_RATES_PER_MILLION[normalized_model]
    notes: list[str] = []

    if usage is not None:
        input_text_cost = usage.input_text_tokens * rates["text_input"] / 1_000_000
        input_image_cost = usage.input_image_tokens * rates["image_input"] / 1_000_000
        if usage.output_image_tokens:
            output_image_cost = usage.output_image_tokens * rates["image_output"] / 1_000_000
            notes.append("Calculated from Gemini usageMetadata output image tokens.")
        else:
            output_image_cost = _gemini_per_image_price(size) * max(1, num_images)
            notes.append("Gemini response did not expose image output tokens; used official per-image price.")
        total_cost = input_text_cost + input_image_cost + output_image_cost
        return CostBreakdown(
            model=normalized_model,
            input_text_cost_usd=round(input_text_cost, 6),
            input_image_cost_usd=round(input_image_cost, 6),
            output_image_cost_usd=round(output_image_cost, 6),
            total_cost_usd=round(total_cost, 6),
            estimated=usage.source != "api" or not usage.output_image_tokens,
            notes=tuple(notes),
        )

    estimated_text_tokens = approximate_text_tokens(prompt)
    input_text_cost = estimated_text_tokens * rates["text_input"] / 1_000_000
    output_price = _gemini_per_image_price(size) * max(1, num_images)
    total_cost = input_text_cost + output_price
    return CostBreakdown(
        model=normalized_model,
        input_text_cost_usd=round(input_text_cost, 6),
        output_image_cost_usd=round(output_price, 6),
        total_cost_usd=round(total_cost, 6),
        estimated=True,
        notes=(
            "Gemini 2.5 Flash Image standard pricing uses $0.039 per generated image.",
            "Prompt token cost uses the official $0.30 / 1M input token rate with rough token estimation.",
        ),
    )


def _gemini_per_image_price(size: str) -> float:
    return _PER_IMAGE_OUTPUT_PRICES["gemini-2.5-flash-image"]["standard"].get(size, 0.039)
