from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PIL import Image, ImageDraw

from .bubble_renderer import BubbleOverlayRenderer, parse_canvas_size
from .memory import ContinuityState, stable_seed
from .models import (
    BubbleFillRequest,
    BubblePlacement,
    BubblePlacementPlan,
    BubbleRenderArtifact,
    BubbleRenderConfig,
    DraftModeConfig,
    DraftPanelFrame,
    DraftSpread,
    DraftSpreadImage,
    GenerationRequest,
    GeneratedImage,
    ModelEndpointConfig,
    PanelSpec,
    SketchPanel,
    SketchStoryboard,
    StoryboardPlan,
)
from .planner import GenerationPlanner
from .providers import ImageProvider


@dataclass(frozen=True)
class DraftSpreadPlanner:
    generation_planner: GenerationPlanner
    draft_mode_config: DraftModeConfig

    def build_spreads(self, storyboard: StoryboardPlan) -> tuple[DraftSpread, ...]:
        if not self.draft_mode_config.enabled:
            return ()
        panels = list(storyboard.panels)
        spreads: list[DraftSpread] = []
        spread_index = 1
        while panels:
            first = panels.pop(0)
            grouped = [first]
            reasons: list[str] = ["Base panel selected for draft spread."]
            if (
                self.draft_mode_config.panels_per_image >= 2
                and panels
                and self._compatibility_score(first, panels[0]) >= 0.35
            ):
                second = panels.pop(0)
                grouped.append(second)
                reasons.append("Paired with adjacent panel based on draft compatibility score.")
            frames = self._build_frames(tuple(grouped))
            spreads.append(
                DraftSpread(
                    spread_id=f"{storyboard.episode_id}-draft-{spread_index:02d}",
                    panel_ids=tuple(panel.panel_id for panel in grouped),
                    panels=tuple(grouped),
                    frames=frames,
                    layout="stacked-2up" if len(grouped) > 1 else "single",
                    rationale=tuple(reasons),
                )
            )
            spread_index += 1
        return tuple(spreads)

    def build_requests(
        self,
        spreads: tuple[DraftSpread, ...],
        continuity: ContinuityState,
    ) -> tuple[GenerationRequest, ...]:
        return tuple(self._build_request(spread, continuity) for spread in spreads)

    def generate(
        self,
        spreads: tuple[DraftSpread, ...],
        continuity: ContinuityState,
        provider: ImageProvider,
    ) -> tuple[DraftSpreadImage, ...]:
        requests = self.build_requests(spreads, continuity)
        images: list[DraftSpreadImage] = []
        for spread, request in zip(spreads, requests, strict=True):
            image = provider.generate(request)
            image = self.prepare_spread_image(spread, image)
            images.append(
                DraftSpreadImage(
                    spread_id=spread.spread_id,
                    panel_ids=spread.panel_ids,
                    image=image,
                )
            )
        return tuple(images)

    def build_bubble_plans(
        self,
        spreads: tuple[DraftSpread, ...],
        sketches: SketchStoryboard,
        bubble_requests: tuple[BubbleFillRequest, ...],
        image_config: ModelEndpointConfig,
        render_config: BubbleRenderConfig,
    ) -> tuple[BubblePlacementPlan, ...]:
        if render_config.mode != "overlay":
            return ()
        sketch_map = {panel.panel_id: panel for panel in sketches.panels}
        bubble_map = {request.panel_id: request for request in bubble_requests}
        plans: list[BubblePlacementPlan] = []
        for spread in spreads:
            spread_width, spread_height = self._spread_canvas_size(image_config, spread)
            placements: list[BubblePlacement] = []
            for frame in spread.frames:
                sketch = sketch_map.get(frame.panel_id)
                bubble_request = bubble_map.get(frame.panel_id)
                if sketch is None or bubble_request is None:
                    continue
                frame_config = replace(image_config, size=f"{frame.width}x{frame.height}")
                renderer = BubbleOverlayRenderer(
                    image_config=frame_config,
                    render_config=render_config,
                )
                local_plan = renderer.build_plan(sketch, bubble_request)
                for placement in local_plan.placements:
                    placements.append(
                        replace(
                            placement,
                            x=placement.x + frame.x,
                            y=placement.y + frame.y,
                        )
                    )
            if placements:
                plans.append(
                    BubblePlacementPlan(
                        panel_id=spread.spread_id,
                        canvas_width=spread_width,
                        canvas_height=spread_height,
                        placements=tuple(placements),
                    )
                )
        return tuple(plans)

    def render_overlays(
        self,
        spread_images: tuple[DraftSpreadImage, ...],
        plans: tuple[BubblePlacementPlan, ...],
        image_config: ModelEndpointConfig,
        render_config: BubbleRenderConfig,
    ) -> tuple[BubbleRenderArtifact, ...]:
        if render_config.mode != "overlay":
            return ()
        image_map = {spread_image.spread_id: spread_image.image for spread_image in spread_images}
        renderer = BubbleOverlayRenderer(image_config=image_config, render_config=render_config)
        artifacts: list[BubbleRenderArtifact] = []
        for plan in plans:
            image = image_map.get(plan.panel_id)
            if image is None:
                continue
            artifact = renderer.render(image, plan)
            if artifact is not None:
                artifacts.append(artifact)
        return tuple(artifacts)

    def prepare_spread_image(
        self,
        spread: DraftSpread,
        image: GeneratedImage,
    ) -> GeneratedImage:
        if spread.layout != "stacked-2up" or len(spread.frames) < 2 or not image.local_path:
            return image
        base_path = Path(image.local_path)
        if not base_path.exists() or base_path.stem.endswith(".framed"):
            return image

        normalized_path = base_path.with_name(f"{base_path.stem}.framed{base_path.suffix}")
        with Image.open(base_path) as source_image:
            normalized = self._normalize_stacked_spread(source_image.convert("RGB"), spread)
            normalized.save(normalized_path)
        return replace(image, local_path=str(normalized_path.resolve()))

    def _build_request(self, spread: DraftSpread, continuity: ContinuityState) -> GenerationRequest:
        panel_requests = [
            self.generation_planner.build_request(panel, continuity)
            for panel in spread.panels
        ]
        unique_character_ids = tuple(
            dict.fromkeys(
                character_id
                for panel in spread.panels
                for character_id in panel.characters
            )
        )
        characters = [self.generation_planner.characters[character_id] for character_id in unique_character_ids]
        references = self._merge_references(panel_requests)
        positive_prompt = self._build_positive_prompt(spread, characters)
        negative_prompt = self._build_negative_prompt(characters)
        system_prompt = self._build_system_prompt(spread, characters)
        seed = stable_seed(
            self.generation_planner.project_id,
            spread.spread_id,
            spread.layout,
            "|".join(spread.panel_ids),
        )
        return GenerationRequest(
            panel_id=spread.spread_id,
            system_prompt=system_prompt,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            references=references,
            seed=seed,
            metadata={
                "pipeline_stage": "stage3_draft_spread",
                "style_id": self.generation_planner.style_bible.style_id,
                "draft_mode": {
                    "enabled": True,
                    "spread_id": spread.spread_id,
                    "layout": spread.layout,
                    "panel_ids": list(spread.panel_ids),
                    "frames": [asdict(frame) for frame in spread.frames],
                    "rationale": list(spread.rationale),
                },
            },
        )

    def _build_positive_prompt(self, spread: DraftSpread, characters) -> str:
        style = self.generation_planner.style_bible
        sections = [
            "DRAFT SPREAD MODE",
            "Create a rough webtoon draft spread with clear stacked panels and a clean gutter.",
            "If this spread has two panels, render EXACTLY TWO distinct comic cuts in one image.",
            "No speech bubbles, no word balloons, no caption boxes, no lettering, and no text in the image.",
            "Prioritize panel readability, acting, and continuity over detailed rendering polish.",
        ]
        sections.extend(self._build_layout_contract(spread))
        sections.extend(["STYLE LOCK", ", ".join(style.positive_traits)])
        if style.render_rules:
            sections.extend(["STYLE RULES", ", ".join(style.render_rules)])
        for character in characters:
            immutable = ", ".join(
                f"{trait.name}: {trait.expected_value}" for trait in character.immutable_traits
            )
            sections.extend(
                [
                    f"CHARACTER LOCK - {character.display_name}",
                    character.core_prompt,
                    immutable,
                ]
            )
        for order, panel in enumerate(spread.panels, start=1):
            frame_label = "TOP PANEL" if order == 1 else "BOTTOM PANEL"
            sections.extend(
                [
                    frame_label,
                    panel.narrative,
                    (
                        f"location: {panel.location}; shot: {panel.shot_type}; emotion: {panel.emotion}; "
                        f"action: {panel.action or 'hold pose'}"
                    ),
                ]
            )
            if panel.props:
                sections.append(f"{frame_label} props: {', '.join(panel.props)}")
        return "\n".join(sections)

    def _build_negative_prompt(self, characters) -> str:
        negatives = list(self.generation_planner.style_bible.negative_traits)
        negatives.extend(
            [
                "speech bubble",
                "word balloon",
                "caption box",
                "text overlay",
                "lettering",
                "subtitles",
                "logo",
                "panel misalignment",
                "overlapping panels",
                "single full-page illustration",
                "continuous scene spanning both panels",
                "art bleeding across the gutter",
                "tiny inset panels on white paper",
                "excess outer white margin around panels",
                "ambiguous diptych layout",
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

    def _build_system_prompt(self, spread: DraftSpread, characters) -> str:
        style = self.generation_planner.style_bible
        sections = [
            "You are the hidden continuity guardrail for a webtoon draft spread image generator.",
            "Generate a multi-panel draft spread with clean visual separation between panels.",
            "This is a strict layout task: each requested panel must read as its own cut, never as one split illustration.",
            "Do not render any speech bubble outlines, tails, text glyphs, captions, or sound effects.",
            f"Style bible: {style.title} ({style.style_id})",
            f"Locked style traits: {', '.join(style.positive_traits)}",
        ]
        if spread.layout == "stacked-2up" and len(spread.frames) == 2:
            sections.extend(
                [
                    "For this spread, render exactly two full-width rectangular panels stacked vertically.",
                    "Keep the gutter fully empty and white, add a strong black border around each panel, and do not let any scene elements cross the gutter.",
                    "Fill each panel confidently and avoid miniaturized panels floating inside a larger white page.",
                ]
            )
        for character in characters:
            immutable = ", ".join(
                f"{trait.name}={trait.expected_value}" for trait in character.immutable_traits
            )
            sections.append(f"Character canon - {character.display_name}: {immutable}")
        sections.append(
            f"Spread {spread.spread_id} contains {len(spread.panels)} ordered draft panels."
        )
        return "\n".join(sections)

    def _merge_references(
        self,
        panel_requests: list[GenerationRequest],
    ) -> tuple:
        ordered = []
        seen: set[str] = set()
        for request in panel_requests:
            for reference in request.references:
                if reference.asset_id not in seen:
                    ordered.append(reference)
                    seen.add(reference.asset_id)
        return tuple(ordered)

    def _build_frames(self, panels: tuple[PanelSpec, ...]) -> tuple[DraftPanelFrame, ...]:
        if len(panels) == 1:
            return (
                DraftPanelFrame(
                    panel_id=panels[0].panel_id,
                    x=0,
                    y=0,
                    width=1024,
                    height=1024,
                    order=1,
                ),
            )
        gutter = 56
        frame_height = (1024 - gutter) // 2
        return (
            DraftPanelFrame(panel_id=panels[0].panel_id, x=0, y=0, width=1024, height=frame_height, order=1),
            DraftPanelFrame(panel_id=panels[1].panel_id, x=0, y=frame_height + gutter, width=1024, height=frame_height, order=2),
        )

    def _compatibility_score(self, first: PanelSpec, second: PanelSpec) -> float:
        score = 0.0
        if self._normalized_tokens(first.location) == self._normalized_tokens(second.location):
            score += 0.42
        elif self._normalized_tokens(first.location) & self._normalized_tokens(second.location):
            score += 0.18

        overlap = len(set(first.characters) & set(second.characters))
        union = len(set(first.characters) | set(second.characters)) or 1
        score += overlap / union * 0.28

        if bool(first.dialogue) or bool(second.dialogue):
            score += 0.08
        if first.shot_type.split()[0] == second.shot_type.split()[0]:
            score += 0.08
        if len(first.props) + len(second.props) > 6:
            score -= 0.10
        if self._is_action_heavy(first) or self._is_action_heavy(second):
            score -= 0.15
        return score

    def _normalized_tokens(self, value: str) -> set[str]:
        return {token for token in re.split(r"[^a-zA-Z0-9가-힣]+", value.lower()) if token}

    def _is_action_heavy(self, panel: PanelSpec) -> bool:
        text = f"{panel.action} {panel.narrative}".lower()
        cues = ("run", "fight", "chase", "jump", "explode", "attack", "dash", "fall", "sprint", "combat", "액션", "달려", "뛰", "싸우", "추격", "넘어")
        return any(cue in text for cue in cues)

    def _spread_canvas_size(
        self,
        image_config: ModelEndpointConfig,
        spread: DraftSpread,
    ) -> tuple[int, int]:
        if spread.layout == "single":
            return parse_canvas_size(image_config.size)
        max_width = max(frame.x + frame.width for frame in spread.frames)
        max_height = max(frame.y + frame.height for frame in spread.frames)
        return max_width, max_height

    def _build_layout_contract(self, spread: DraftSpread) -> list[str]:
        if spread.layout != "stacked-2up" or len(spread.frames) != 2:
            return []
        top_frame, bottom_frame = spread.frames
        gutter = bottom_frame.y - (top_frame.y + top_frame.height)
        sections = [
            "LAYOUT CONTRACT",
            "Render EXACTLY TWO full-width rectangular comic panels stacked vertically.",
            f"Leave a fully empty white gutter between them of about {gutter}px on the 1024px canvas.",
            "Add a strong black rectangular border around each panel.",
            "Make each panel feel like its own cut from the sequence, not one continuous illustration split in half.",
            "Do not let characters, props, rain, speed lines, lighting, or perspective continue across the gutter.",
            "Keep outer margins minimal so the two panels fill the image clearly.",
        ]
        for frame in spread.frames:
            panel_label = "TOP PANEL" if frame.order == 1 else "BOTTOM PANEL"
            sections.append(
                f"{panel_label} frame: x={frame.x}, y={frame.y}, width={frame.width}, height={frame.height}"
            )
        return sections

    def _normalize_stacked_spread(self, image: Image.Image, spread: DraftSpread) -> Image.Image:
        spread_width = max(frame.x + frame.width for frame in spread.frames)
        spread_height = max(frame.y + frame.height for frame in spread.frames)
        scale_x = image.width / max(spread_width, 1)
        scale_y = image.height / max(spread_height, 1)
        scaled_frames = tuple(
            (
                int(round(frame.x * scale_x)),
                int(round(frame.y * scale_y)),
                max(1, int(round(frame.width * scale_x))),
                max(1, int(round(frame.height * scale_y))),
            )
            for frame in spread.frames
        )

        boundaries = [0]
        for index in range(len(scaled_frames) - 1):
            current = scaled_frames[index]
            following = scaled_frames[index + 1]
            midpoint = round(((current[1] + current[3]) + following[1]) / 2)
            boundaries.append(max(boundaries[-1], midpoint))
        boundaries.append(image.height)

        normalized = Image.new("RGB", image.size, (255, 255, 255))
        for index, frame in enumerate(scaled_frames):
            x, y, width, height = frame
            region_top = boundaries[index]
            region_bottom = boundaries[index + 1]
            region = image.crop((0, region_top, image.width, region_bottom))
            bbox = self._content_bbox(region)
            source = region if bbox is None else region.crop(bbox)
            fitted = source.resize((width, height), Image.Resampling.LANCZOS)
            normalized.paste(fitted, (x, y))

        border_width = max(3, round(min(image.size) * 0.004))
        draw = ImageDraw.Draw(normalized)
        for x, y, width, height in scaled_frames:
            draw.rectangle(
                (x, y, x + width - 1, y + height - 1),
                outline=(24, 24, 24),
                width=border_width,
            )
        return normalized

    def _content_bbox(self, image: Image.Image, threshold: int = 245) -> tuple[int, int, int, int] | None:
        mask = Image.new("L", image.size, 0)
        mask.putdata(
            [
                255 if (pixel[0] < threshold or pixel[1] < threshold or pixel[2] < threshold) else 0
                for pixel in image.getdata()
            ]
        )
        bbox = mask.getbbox()
        if bbox is None:
            return None
        left, top, right, bottom = bbox
        padding = 6
        return (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
