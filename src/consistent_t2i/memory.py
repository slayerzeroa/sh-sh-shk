from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .models import ReferenceImage


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


@dataclass
class ContinuityState:
    project_id: str
    style_id: str
    max_history_per_character: int = 6
    generated_assets: dict[str, list[ReferenceImage]] = field(default_factory=dict)
    approved_panels: dict[str, list[str]] = field(default_factory=dict)

    def register_reference(self, reference: ReferenceImage) -> None:
        bucket = self.generated_assets.setdefault(reference.character_id or "__global__", [])
        bucket.insert(0, reference)
        del bucket[self.max_history_per_character :]

    def register_approved_panel(self, panel_id: str, asset_ids: list[str]) -> None:
        self.approved_panels[panel_id] = list(asset_ids)

    def get_recent_references(self, character_id: str, limit: int = 3) -> tuple[ReferenceImage, ...]:
        return tuple(self.generated_assets.get(character_id, [])[:limit])

    def save(self, destination: Path) -> None:
        serializable = {
            "project_id": self.project_id,
            "style_id": self.style_id,
            "max_history_per_character": self.max_history_per_character,
            "generated_assets": {
                key: [asdict(reference) for reference in references]
                for key, references in self.generated_assets.items()
            },
            "approved_panels": self.approved_panels,
        }
        destination.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, source: Path) -> "ContinuityState":
        payload = json.loads(source.read_text(encoding="utf-8"))
        state = cls(
            project_id=payload["project_id"],
            style_id=payload["style_id"],
            max_history_per_character=payload.get("max_history_per_character", 6),
        )
        state.generated_assets = {
            key: [ReferenceImage(**reference) for reference in references]
            for key, references in payload.get("generated_assets", {}).items()
        }
        state.approved_panels = {
            panel_id: list(asset_ids)
            for panel_id, asset_ids in payload.get("approved_panels", {}).items()
        }
        return state
