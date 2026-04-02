from __future__ import annotations

from dataclasses import dataclass

from .memory import ContinuityState, stable_seed
from .models import CharacterBible, GenerationRequest, PanelSpec, ReferenceImage, StyleBible


@dataclass(frozen=True)
class GenerationPlanner:
    project_id: str
    style_bible: StyleBible
    characters: dict[str, CharacterBible]
    max_references_per_character: int = 3

    def build_request(self, panel: PanelSpec, continuity: ContinuityState) -> GenerationRequest:
        characters = [self.characters[character_id] for character_id in panel.characters]
        references = self._select_references(characters, continuity)
        prompt = self._build_positive_prompt(panel, characters)
        negative = self._build_negative_prompt(characters)
        seed_bundle = self._build_seed_bundle(panel)
        return GenerationRequest(
            panel_id=panel.panel_id,
            positive_prompt=prompt,
            negative_prompt=negative,
            references=references,
            seed=seed_bundle["composition_seed"],
            metadata={
                "path": "hot",
                "style_id": self.style_bible.style_id,
                "character_ids": panel.characters,
                "seed_bundle": seed_bundle,
                "lock_rules": {
                    "style_seed_locked": self.style_bible.lock_seed,
                    "immutable_trait_count": sum(len(character.immutable_traits) for character in characters),
                },
            },
        )

    def _select_references(
        self,
        characters: list[CharacterBible],
        continuity: ContinuityState,
    ) -> tuple[ReferenceImage, ...]:
        references: list[ReferenceImage] = []
        seen: set[str] = set()
        for character in characters:
            for reference in continuity.get_recent_references(
                character.character_id,
                limit=self.max_references_per_character,
            ):
                if reference.asset_id not in seen:
                    references.append(reference)
                    seen.add(reference.asset_id)
            for asset_id in character.reference_asset_ids:
                if asset_id not in seen:
                    references.append(
                        ReferenceImage(
                            asset_id=asset_id,
                            character_id=character.character_id,
                            note=f"Canonical reference for {character.display_name}",
                            source="canon",
                        )
                    )
                    seen.add(asset_id)
        return tuple(references)

    def _build_positive_prompt(self, panel: PanelSpec, characters: list[CharacterBible]) -> str:
        sections = [
            "STYLE LOCK",
            ", ".join(self.style_bible.positive_traits),
        ]
        if self.style_bible.render_rules:
            sections.extend(["STYLE RULES", ", ".join(self.style_bible.render_rules)])
        for character in characters:
            immutable = ", ".join(
                f"{trait.name}: {trait.expected_value}" for trait in character.immutable_traits
            )
            mutable = ", ".join(
                f"{trait.name}: {trait.expected_value}" for trait in character.mutable_traits
            )
            sections.extend(
                [
                    f"CHARACTER LOCK - {character.display_name}",
                    character.core_prompt,
                    immutable,
                ]
            )
            if mutable:
                sections.append(f"MUTABLE RANGE - {mutable}")
        sections.extend(
            [
                "PANEL GOAL",
                panel.narrative,
                (
                    f"location: {panel.location}; shot: {panel.shot_type}; "
                    f"emotion: {panel.emotion}; action: {panel.action or 'none'}"
                ),
            ]
        )
        if panel.props:
            sections.append(f"props: {', '.join(panel.props)}")
        if panel.dialogue:
            sections.append(f"dialogue: {' | '.join(panel.dialogue)}")
        return "\n".join(section for section in sections if section)

    def _build_negative_prompt(self, characters: list[CharacterBible]) -> str:
        negatives = list(self.style_bible.negative_traits)
        negatives.extend(
            [
                "style drift",
                "different face structure",
                "different hair color",
                "different eye color",
                "different costume silhouette",
            ]
        )
        for character in characters:
            negatives.extend(character.negative_traits)
        ordered = []
        seen: set[str] = set()
        for item in negatives:
            if item and item not in seen:
                ordered.append(item)
                seen.add(item)
        return ", ".join(ordered)

    def _build_seed_bundle(self, panel: PanelSpec) -> dict[str, int]:
        style_seed = stable_seed(self.project_id, self.style_bible.style_id, "style")
        composition_seed = stable_seed(
            self.project_id,
            self.style_bible.style_id,
            panel.panel_id,
            ",".join(panel.characters),
            panel.location,
            panel.shot_type,
        )
        character_seeds = {
            character_id: stable_seed(self.project_id, self.style_bible.style_id, character_id)
            for character_id in panel.characters
        }
        return {
            "style_seed": style_seed,
            "composition_seed": composition_seed,
            "character_seeds": character_seeds,
        }
