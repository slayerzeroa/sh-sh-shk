from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ConstraintLevel = Literal["hard", "soft"]


@dataclass(frozen=True)
class TraitConstraint:
    name: str
    expected_value: str
    level: ConstraintLevel = "hard"
    rationale: str = ""


@dataclass(frozen=True)
class StyleBible:
    style_id: str
    title: str
    positive_traits: tuple[str, ...]
    negative_traits: tuple[str, ...] = ()
    render_rules: tuple[str, ...] = ()
    lock_seed: bool = True


@dataclass(frozen=True)
class CharacterBible:
    character_id: str
    display_name: str
    core_prompt: str
    immutable_traits: tuple[TraitConstraint, ...]
    mutable_traits: tuple[TraitConstraint, ...] = ()
    negative_traits: tuple[str, ...] = ()
    reference_asset_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PanelSpec:
    panel_id: str
    narrative: str
    characters: tuple[str, ...]
    location: str
    shot_type: str
    emotion: str
    action: str = ""
    props: tuple[str, ...] = ()
    dialogue: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceImage:
    asset_id: str
    character_id: str | None
    note: str
    score: float = 1.0
    source: Literal["canon", "generated"] = "generated"


@dataclass(frozen=True)
class GenerationRequest:
    panel_id: str
    positive_prompt: str
    negative_prompt: str
    references: tuple[ReferenceImage, ...]
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedImage:
    asset_id: str
    panel_id: str
    provider_name: str
    seed: int
    prompt_fingerprint: str


@dataclass(frozen=True)
class ObservedTraits:
    panel_id: str
    character_id: str
    observed: dict[str, str]
    source: str = "manual"


@dataclass(frozen=True)
class DriftIssue:
    character_id: str
    trait_name: str
    expected_value: str
    observed_value: str
    severity: Literal["critical", "warning"]
    message: str


@dataclass(frozen=True)
class DriftReport:
    panel_id: str
    score: float
    issues: tuple[DriftIssue, ...]
    should_regenerate: bool


@dataclass(frozen=True)
class ColdPathDecision:
    panel_id: str
    should_repair: bool
    block_approval: bool
    reasons: tuple[str, ...]
