from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ConstraintLevel = Literal["hard", "soft"]
PipelineStage = Literal["storyboard", "sketch", "refine", "bubble"]
BubbleRenderMode = Literal["overlay", "in_image", "none"]
DraftLayout = Literal["stacked-2up", "single"]


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
class ModelEndpointConfig:
    stage: PipelineStage
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    api_key_env: str = ""
    size: str = "1024x1024"
    quality: str = "medium"
    background: str = ""
    output_format: str = "png"
    output_compression: int | None = None
    moderation: str = "auto"
    n: int = 1
    output_dir: str = "outputs"
    timeout_seconds: float = 120.0
    max_output_tokens: int | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CharacterBible:
    character_id: str
    display_name: str
    core_prompt: str
    immutable_traits: tuple[TraitConstraint, ...]
    mutable_traits: tuple[TraitConstraint, ...] = ()
    negative_traits: tuple[str, ...] = ()
    reference_asset_ids: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


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
class WebNovelInput:
    episode_id: str
    title: str
    source_text: str
    genre: str = ""
    tone: str = ""
    default_location: str = ""
    primary_characters: tuple[str, ...] = ()
    target_panel_count: int = 4
    adaptation_rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoryboardPlan:
    episode_id: str
    title: str
    premise: str
    panels: tuple[PanelSpec, ...]
    adaptation_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SketchPanel:
    panel_id: str
    summary: str
    camera_guide: str
    figure_guides: tuple[str, ...]
    prop_guides: tuple[str, ...] = ()
    bubble_slots: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SketchStoryboard:
    episode_id: str
    panels: tuple[SketchPanel, ...]
    drawing_rules: tuple[str, ...] = ()


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
    system_prompt: str
    positive_prompt: str
    negative_prompt: str
    references: tuple[ReferenceImage, ...]
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageUsage:
    input_text_tokens: int = 0
    input_image_tokens: int = 0
    output_image_tokens: int = 0
    partial_image_tokens: int = 0
    source: Literal["api", "estimate"] = "estimate"


@dataclass(frozen=True)
class CostBreakdown:
    model: str
    currency: str = "USD"
    input_text_cost_usd: float = 0.0
    input_image_cost_usd: float = 0.0
    output_image_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    estimated: bool = True
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeneratedImage:
    asset_id: str
    panel_id: str
    provider_name: str
    seed: int
    prompt_fingerprint: str
    model: str = ""
    output_format: str = "png"
    local_path: str = ""
    revised_prompt: str = ""
    usage: ImageUsage | None = None
    cost: CostBreakdown | None = None


@dataclass(frozen=True)
class BubbleLine:
    speaker: str
    text: str
    slot_hint: str
    purpose: Literal["dialogue", "caption", "sfx"] = "dialogue"
    speaker_character_id: str | None = None
    speaker_confidence: float = 0.0
    speaker_rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class BubbleFillRequest:
    panel_id: str
    system_prompt: str
    user_prompt: str
    lines: tuple[BubbleLine, ...]
    model: str
    max_output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BubbleRenderConfig:
    mode: BubbleRenderMode = "overlay"
    include_captions: bool = False
    force_exact_text: bool = True


@dataclass(frozen=True)
class BubblePlacement:
    panel_id: str
    slot_hint: str
    speaker: str
    speaker_character_id: str | None
    text: str
    purpose: Literal["dialogue", "caption", "sfx"]
    x: int
    y: int
    width: int
    height: int
    font_size: int
    line_height: int
    wrapped_lines: tuple[str, ...]
    shape: Literal["ellipse", "caption-box"] = "ellipse"
    tail_direction: Literal["left", "right", "down", "none"] = "none"


@dataclass(frozen=True)
class BubblePlacementPlan:
    panel_id: str
    canvas_width: int
    canvas_height: int
    placements: tuple[BubblePlacement, ...]


@dataclass(frozen=True)
class BubbleRenderArtifact:
    panel_id: str
    base_image_path: str
    overlay_svg_path: str
    composite_svg_path: str
    final_png_path: str = ""


@dataclass(frozen=True)
class DraftModeConfig:
    enabled: bool = False
    panels_per_image: int = 2
    layout: DraftLayout = "stacked-2up"


@dataclass(frozen=True)
class DraftPanelFrame:
    panel_id: str
    x: int
    y: int
    width: int
    height: int
    order: int


@dataclass(frozen=True)
class DraftSpread:
    spread_id: str
    panel_ids: tuple[str, ...]
    panels: tuple[PanelSpec, ...]
    frames: tuple[DraftPanelFrame, ...]
    layout: DraftLayout = "stacked-2up"
    rationale: tuple[str, ...] = ()


@dataclass(frozen=True)
class DraftSpreadImage:
    spread_id: str
    panel_ids: tuple[str, ...]
    image: GeneratedImage


@dataclass(frozen=True)
class BatchJobInfo:
    name: str
    display_name: str
    provider: str
    model: str
    state: str
    request_count: int
    manifest_path: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftRunResult:
    source: WebNovelInput
    spreads: tuple[DraftSpread, ...]
    requests: tuple[GenerationRequest, ...]
    images: tuple[DraftSpreadImage, ...] = ()
    bubble_plans: tuple[BubblePlacementPlan, ...] = ()
    bubble_artifacts: tuple[BubbleRenderArtifact, ...] = ()
    batch_job: BatchJobInfo | None = None


@dataclass(frozen=True)
class PipelineDraft:
    source: WebNovelInput
    storyboard: StoryboardPlan
    sketches: SketchStoryboard
    refined_requests: tuple[GenerationRequest, ...]
    bubble_requests: tuple[BubbleFillRequest, ...]


@dataclass(frozen=True)
class PipelineRunResult:
    draft: PipelineDraft
    generated_images: tuple[GeneratedImage, ...]
    bubble_plans: tuple[BubblePlacementPlan, ...] = ()
    bubble_artifacts: tuple[BubbleRenderArtifact, ...] = ()


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
