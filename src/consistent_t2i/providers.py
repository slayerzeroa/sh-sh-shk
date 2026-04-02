from __future__ import annotations

import hashlib
from typing import Protocol

from .models import GeneratedImage, GenerationRequest


class ImageProvider(Protocol):
    def generate(self, request: GenerationRequest) -> GeneratedImage:
        ...


class MockImageProvider:
    provider_name = "mock"

    def generate(self, request: GenerationRequest) -> GeneratedImage:
        digest = hashlib.sha256(
            f"{request.panel_id}|{request.seed}|{request.positive_prompt}".encode("utf-8")
        ).hexdigest()[:12]
        return GeneratedImage(
            asset_id=f"mock-{digest}",
            panel_id=request.panel_id,
            provider_name=self.provider_name,
            seed=request.seed,
            prompt_fingerprint=digest,
        )
