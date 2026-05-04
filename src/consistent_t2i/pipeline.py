from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from math import ceil

from .bubble_renderer import BubbleOverlayRenderer
from .memory import ContinuityState
from .models import (
    BubblePlacementPlan,
    BubbleRenderArtifact,
    BubbleFillRequest,
    BubbleLine,
    BubbleRenderConfig,
    CharacterBible,
    GeneratedImage,
    GenerationRequest,
    ModelEndpointConfig,
    PanelSpec,
    PipelineDraft,
    PipelineRunResult,
    SketchPanel,
    SketchStoryboard,
    StoryboardPlan,
    WebNovelInput,
)
from .planner import GenerationPlanner
from .providers import ImageProvider


@dataclass(frozen=True)
class NovelStoryboardPlanner:
    characters: dict[str, CharacterBible]
    default_shot_cycle: tuple[str, ...] = (
        "establishing shot",
        "medium shot",
        "close-up",
        "reaction shot",
    )
    default_location: str = "unspecified location"
    default_emotion: str = "dramatic tension"
    max_dialogue_per_panel: int = 2

    def build(self, source: WebNovelInput) -> StoryboardPlan:
        beats = self._segment_story(source)
        panels = tuple(
            self._build_panel(source, beat, index)
            for index, beat in enumerate(beats, start=1)
        )
        notes = [
            "Stage 1 converts prose into panel-sized story beats.",
            "Dialogue is trimmed early to protect later speech bubble token budgets.",
        ]
        if source.adaptation_rules:
            notes.append(f"Adaptation rules: {' | '.join(source.adaptation_rules)}")
        return StoryboardPlan(
            episode_id=source.episode_id,
            title=source.title,
            premise=self._summarize_premise(source.source_text),
            panels=panels,
            adaptation_notes=tuple(notes),
        )

    def _build_panel(self, source: WebNovelInput, beat: str, index: int) -> PanelSpec:
        dialogue = self._extract_dialogue(beat)
        cleaned_beat = self._clean_narrative(beat, dialogue)
        characters = self._detect_characters(cleaned_beat or beat, source)
        return PanelSpec(
            panel_id=f"{source.episode_id}-p{index:02d}",
            narrative=cleaned_beat or beat.strip(),
            characters=characters,
            location=self._infer_location(beat, source),
            shot_type=self.default_shot_cycle[(index - 1) % len(self.default_shot_cycle)],
            emotion=self._infer_emotion(beat, source),
            action=self._infer_action(cleaned_beat or beat),
            props=self._infer_props(beat),
            dialogue=dialogue,
        )

    def _segment_story(self, source: WebNovelInput) -> tuple[str, ...]:
        raw_chunks = [
            chunk.strip()
            for chunk in re.split(r"\n\s*\n", source.source_text)
            if chunk.strip()
        ]
        if len(raw_chunks) <= 1:
            raw_chunks = [
                chunk.strip()
                for chunk in re.split(r"(?<=[.!?])\s+|(?<=[다요죠])\s+", source.source_text)
                if chunk.strip()
            ]
        if not raw_chunks:
            raw_chunks = [source.source_text.strip()]
        target = max(1, source.target_panel_count)
        if len(raw_chunks) <= target:
            return tuple(raw_chunks)
        chunk_size = ceil(len(raw_chunks) / target)
        merged = [
            " ".join(raw_chunks[start : start + chunk_size]).strip()
            for start in range(0, len(raw_chunks), chunk_size)
        ]
        return tuple(merged[:target])

    def _detect_characters(self, beat: str, source: WebNovelInput) -> tuple[str, ...]:
        lowered = beat.lower()
        matched: list[str] = []
        for character_id, character in self.characters.items():
            aliases = {
                character_id.lower(),
                character.display_name.lower(),
                character.display_name.lower().split()[-1],
            }
            if any(alias in lowered for alias in aliases if alias):
                matched.append(character_id)
        if matched:
            return tuple(matched)
        if source.primary_characters:
            return source.primary_characters
        fallback = list(self.characters.keys())[: max(1, min(2, len(self.characters)))]
        return tuple(fallback)

    def _extract_dialogue(self, beat: str) -> tuple[str, ...]:
        extracted: list[str] = []
        extracted.extend(match.strip() for match in re.findall(r"[\"“](.*?)[\"”]", beat))
        for sentence in re.split(r"(?<=[.!?])\s+|\n", beat):
            sentence = sentence.strip()
            if ":" in sentence and sentence not in extracted:
                extracted.append(sentence)
        return tuple(extracted[: self.max_dialogue_per_panel])

    def _clean_narrative(self, beat: str, dialogue: tuple[str, ...]) -> str:
        cleaned = beat
        for line in dialogue:
            cleaned = cleaned.replace(line, "")
            cleaned = cleaned.replace(f'"{line}"', "")
            cleaned = cleaned.replace(f"“{line}”", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
        return cleaned

    def _infer_location(self, beat: str, source: WebNovelInput) -> str:
        mappings = {
            "school": "school interior",
            "corridor": "school corridor",
            "classroom": "classroom",
            "rain": "rainy exterior or window-side space",
            "street": "city street",
            "room": "private room",
            "복도": "school corridor",
            "교실": "classroom",
            "비": "rainy exterior or window-side space",
            "옥상": "school rooftop",
        }
        lowered = beat.lower()
        for keyword, location in mappings.items():
            if keyword in lowered:
                return location
        return source.default_location or self.default_location

    def _infer_emotion(self, beat: str, source: WebNovelInput) -> str:
        mappings = {
            "angry": "anger",
            "furious": "anger",
            "awkward": "awkward tension",
            "tense": "tension",
            "sad": "sadness",
            "cry": "sadness",
            "warm": "warmth",
            "romantic": "romantic tension",
            "nervous": "nervous anticipation",
            "surprised": "surprise",
            "긴장": "tension",
            "당황": "embarrassed surprise",
            "슬픔": "sadness",
            "설렘": "romantic tension",
        }
        lowered = beat.lower()
        for keyword, emotion in mappings.items():
            if keyword in lowered:
                return emotion
        return source.tone or self.default_emotion

    def _infer_action(self, beat: str) -> str:
        sentence = re.split(r"(?<=[.!?])\s+", beat.strip())[0]
        return sentence[:140] if sentence else "hold the dramatic moment"

    def _infer_props(self, beat: str) -> tuple[str, ...]:
        props: list[str] = []
        prop_keywords = {
            "umbrella": "umbrella",
            "locker": "lockers",
            "phone": "smartphone",
            "book": "book",
            "window": "window",
            "rain": "rain streaks",
            "우산": "umbrella",
            "사물함": "lockers",
            "핸드폰": "smartphone",
            "창문": "window",
        }
        lowered = beat.lower()
        for keyword, normalized in prop_keywords.items():
            if keyword in lowered and normalized not in props:
                props.append(normalized)
        return tuple(props)

    def _summarize_premise(self, source_text: str) -> str:
        cleaned = re.sub(r"\s+", " ", source_text).strip()
        return cleaned[:180] + ("..." if len(cleaned) > 180 else "")


@dataclass(frozen=True)
class StickFigureStoryboardPlanner:
    bubble_margin_note: str = "leave clean headroom for speech bubbles"

    def build(self, storyboard: StoryboardPlan) -> SketchStoryboard:
        panels = tuple(self._build_panel(panel) for panel in storyboard.panels)
        return SketchStoryboard(
            episode_id=storyboard.episode_id,
            panels=panels,
            drawing_rules=(
                "Use stick figures only, focusing on silhouette and eyeline.",
                "Lock camera angle and empty bubble lanes before detailed art.",
                self.bubble_margin_note,
            ),
        )

    def _build_panel(self, panel: PanelSpec) -> SketchPanel:
        figure_guides = tuple(
            self._build_figure_guide(character_id, index, panel)
            for index, character_id in enumerate(panel.characters)
        )
        bubble_slots = self._build_bubble_slots(panel)
        return SketchPanel(
            panel_id=panel.panel_id,
            summary=panel.narrative,
            camera_guide=f"{panel.shot_type}; keep silhouettes readable; {self.bubble_margin_note}",
            figure_guides=figure_guides,
            prop_guides=tuple(f"block prop: {prop}" for prop in panel.props),
            bubble_slots=bubble_slots,
            notes=(
                f"Emotion lock: {panel.emotion}",
                f"Action spine: {panel.action or 'hold pose'}",
            ),
        )

    def _build_figure_guide(self, character_id: str, index: int, panel: PanelSpec) -> str:
        anchors = ("left third", "center", "right third", "far background")
        anchor = anchors[min(index, len(anchors) - 1)]
        return (
            f"{character_id}: stick figure at {anchor}, "
            f"eyeline matched to {panel.emotion}, action '{panel.action or 'hold pose'}'"
        )

    def _build_bubble_slots(self, panel: PanelSpec) -> tuple[str, ...]:
        if not panel.dialogue:
            return ("caption lane reserved at top-left",)
        slots: list[str] = []
        for index, dialogue in enumerate(panel.dialogue, start=1):
            speaker = dialogue.split(":", 1)[0].strip() if ":" in dialogue else f"speaker-{index}"
            slots.append(f"bubble-{index}: near {speaker}, away from focal face")
        return tuple(slots)


@dataclass(frozen=True)
class DialogueSpeakerResolver:
    characters: dict[str, CharacterBible]

    def resolve(
        self,
        *,
        speaker_label: str,
        line_text: str,
        panel: PanelSpec,
        slot_hint: str,
        line_index: int,
        previous_line_speaker_id: str | None,
        previous_panel_speaker_id: str | None,
    ) -> tuple[str | None, float, tuple[str, ...]]:
        normalized_speaker = self._normalize(speaker_label)
        if normalized_speaker in {"caption", "narration", "내레이션", "sfx", "효과음"}:
            return None, 1.0, ("Non-dialogue speaker label detected.",)

        candidate_ids = [character_id for character_id in panel.characters if character_id in self.characters]
        if not candidate_ids:
            return None, 0.0, ("No known panel characters available for speaker inference.",)
        if len(candidate_ids) == 1:
            return candidate_ids[0], 0.96, ("Single visible character in panel.",)

        candidate_scores: dict[str, float] = {character_id: 0.0 for character_id in candidate_ids}
        candidate_reasons: dict[str, list[str]] = {character_id: [] for character_id in candidate_ids}

        for character_id in candidate_ids:
            aliases = self._aliases_for(character_id)
            context_text = " ".join((panel.narrative, panel.action, slot_hint, line_text))
            normalized_context = self._normalize(context_text)

            if normalized_speaker:
                if normalized_speaker in aliases:
                    candidate_scores[character_id] += 1.2
                    candidate_reasons[character_id].append("Exact speaker label alias match.")
                elif any(normalized_speaker in alias or alias in normalized_speaker for alias in aliases):
                    candidate_scores[character_id] += 0.72
                    candidate_reasons[character_id].append("Partial speaker label alias match.")

            alias_mentions = sum(1 for alias in aliases if alias and alias in normalized_context)
            if alias_mentions:
                score_gain = min(0.28, alias_mentions * 0.08)
                candidate_scores[character_id] += score_gain
                candidate_reasons[character_id].append("Character is mentioned in panel context.")

            if self._matches_slot_hint(slot_hint, aliases):
                candidate_scores[character_id] += 0.36
                candidate_reasons[character_id].append("Bubble slot hint points to this character.")

            if self._matches_context_actor(panel, character_id, aliases):
                candidate_scores[character_id] += 0.32
                candidate_reasons[character_id].append("Panel action/narrative suggests this character is the active speaker.")

            if previous_line_speaker_id and len(candidate_ids) == 2:
                if character_id != previous_line_speaker_id:
                    candidate_scores[character_id] += 0.22
                    candidate_reasons[character_id].append("Conversation alternation within the panel.")
                else:
                    candidate_scores[character_id] -= 0.04

            elif previous_panel_speaker_id and len(candidate_ids) == 2 and line_index == 0:
                if character_id != previous_panel_speaker_id:
                    candidate_scores[character_id] += 0.10
                    candidate_reasons[character_id].append("Likely turn switch from previous panel.")

        ranked = sorted(
            candidate_ids,
            key=lambda character_id: candidate_scores[character_id],
            reverse=True,
        )
        best_id = ranked[0]
        best_score = candidate_scores[best_id]
        second_score = candidate_scores[ranked[1]] if len(ranked) > 1 else 0.0
        if best_score < 0.12:
            return None, 0.0, ("No strong contextual speaker signal found.",)

        confidence = max(0.35, min(0.99, 0.52 + best_score * 0.28 + (best_score - second_score) * 0.46))
        reasons = tuple(candidate_reasons[best_id]) or ("Highest contextual score.",)
        return best_id, round(confidence, 3), reasons

    def _aliases_for(self, character_id: str) -> tuple[str, ...]:
        character = self.characters[character_id]
        base_aliases = {
            self._normalize(character.character_id),
            self._normalize(character.display_name),
        }
        base_aliases.update(
            self._normalize(alias)
            for alias in character.aliases
            if alias
        )
        name_parts = re.split(r"\s+", character.display_name)
        base_aliases.update(self._normalize(part) for part in name_parts if part)
        return tuple(alias for alias in base_aliases if alias)

    def _normalize(self, value: str) -> str:
        return re.sub(r"[\s\-_]+", "", value).lower()

    def _matches_slot_hint(self, slot_hint: str, aliases: tuple[str, ...]) -> bool:
        normalized_hint = self._normalize(slot_hint)
        return any(alias and alias in normalized_hint for alias in aliases)

    def _matches_context_actor(
        self,
        panel: PanelSpec,
        character_id: str,
        aliases: tuple[str, ...],
    ) -> bool:
        contexts = [panel.action, panel.narrative]
        action_cues = (
            "said",
            "says",
            "asked",
            "asks",
            "replied",
            "reply",
            "whispered",
            "shouted",
            "murmured",
            "offers",
            "offered",
            "extends",
            "extended",
            "hands",
            "handed",
            "speaks",
            "말했",
            "말한",
            "묻",
            "대답",
            "속삭",
            "외쳤",
            "중얼",
            "건넨",
            "내민",
        )
        for context in contexts:
            normalized = self._normalize(context)
            if not normalized:
                continue
            for alias in aliases:
                if not alias:
                    continue
                if alias in normalized:
                    if any(cue in normalized for cue in action_cues):
                        return True
        alias_mentions = sum(
            1
            for context in contexts
            for alias in aliases
            if alias and alias in self._normalize(context)
        )
        return alias_mentions >= 2


@dataclass
class SpeakerInferenceState:
    previous_panel_speaker_id: str | None = None


@dataclass(frozen=True)
class VisualRefinementPlanner:
    generation_planner: GenerationPlanner
    image_model: ModelEndpointConfig
    bubble_render_config: BubbleRenderConfig = field(default_factory=BubbleRenderConfig)

    def build_requests(
        self,
        storyboard: StoryboardPlan,
        sketches: SketchStoryboard,
        continuity: ContinuityState,
        bubble_requests: tuple[BubbleFillRequest, ...] | None = None,
    ) -> tuple[GenerationRequest, ...]:
        sketch_map = {panel.panel_id: panel for panel in sketches.panels}
        bubble_map = {request.panel_id: request for request in bubble_requests or ()}
        return tuple(
            self._merge_storyboard_and_sketch(
                panel,
                sketch_map[panel.panel_id],
                continuity,
                bubble_map.get(panel.panel_id),
            )
            for panel in storyboard.panels
        )

    def generate(
        self,
        storyboard: StoryboardPlan,
        sketches: SketchStoryboard,
        continuity: ContinuityState,
        provider: ImageProvider,
        bubble_requests: tuple[BubbleFillRequest, ...] | None = None,
    ) -> tuple[GeneratedImage, ...]:
        requests = self.build_requests(
            storyboard,
            sketches,
            continuity,
            bubble_requests=bubble_requests,
        )
        return tuple(provider.generate(request) for request in requests)

    def _merge_storyboard_and_sketch(
        self,
        panel: PanelSpec,
        sketch: SketchPanel,
        continuity: ContinuityState,
        bubble_request: BubbleFillRequest | None,
    ) -> GenerationRequest:
        base_request = self.generation_planner.build_request(panel, continuity)
        sketch_block = [
            "SKETCH STORYBOARD LOCK",
            sketch.summary,
            f"camera guide: {sketch.camera_guide}",
        ]
        if sketch.figure_guides:
            sketch_block.append(f"stick figure layout: {' | '.join(sketch.figure_guides)}")
        if sketch.prop_guides:
            sketch_block.append(f"prop blocks: {' | '.join(sketch.prop_guides)}")
        if sketch.bubble_slots:
            sketch_block.append(f"reserved bubble lanes: {' | '.join(sketch.bubble_slots)}")
        rendered_lines = self._select_lines_for_render(bubble_request)
        if bubble_request is not None and self.bubble_render_config.mode == "overlay":
            sketch_block.extend(
                [
                    "POST-PRODUCTION LETTERING MODE",
                    (
                        "Do not render any readable letters, words, or dialogue in the image. "
                        "Leave clean empty speech bubbles or clean bubble lanes for post-production lettering."
                    ),
                    "Bubble silhouettes, tails, and empty space are allowed, but text glyphs must remain empty.",
                ]
            )
            if rendered_lines:
                sketch_block.extend(
                    [
                        "POST-PRODUCTION LETTERING PLAN",
                        "These lines will be inserted later by a deterministic renderer, not by the image model.",
                    ]
                )
                sketch_block.extend(
                    [
                        f"{line.slot_hint} -> {line.speaker}: {line.text}"
                        for line in rendered_lines
                    ]
                )
        elif rendered_lines:
            sketch_block.extend(
                [
                    "IN-IMAGE BUBBLE LETTERING",
                    (
                        "Render the following speech bubble text inside the reserved bubble slots. "
                        "Keep lettering readable and anchored to the matching slot."
                    ),
                ]
            )
            if self.bubble_render_config.force_exact_text:
                sketch_block.append(
                    "Use the exact text provided below. Do not paraphrase, translate, or invent new dialogue."
                )
            sketch_block.extend(
                [
                    f"{line.slot_hint} -> {line.speaker}: {line.text}"
                    for line in rendered_lines
                ]
            )
        metadata = dict(base_request.metadata)
        metadata.update(
            {
                "pipeline_stage": "stage3_refine",
                "image_model": asdict(self.image_model),
                "sketch": {
                    "camera_guide": sketch.camera_guide,
                    "figure_guides": list(sketch.figure_guides),
                    "prop_guides": list(sketch.prop_guides),
                    "bubble_slots": list(sketch.bubble_slots),
                },
                "bubble_render": {
                    "mode": self.bubble_render_config.mode,
                    "render_text_in_image": self.bubble_render_config.mode == "in_image",
                    "render_text_overlay": self.bubble_render_config.mode == "overlay",
                    "include_captions": self.bubble_render_config.include_captions,
                    "force_exact_text": self.bubble_render_config.force_exact_text,
                    "visible_lines": [
                        {
                            "speaker": line.speaker,
                            "speaker_character_id": line.speaker_character_id,
                            "speaker_confidence": line.speaker_confidence,
                            "text": line.text,
                            "slot_hint": line.slot_hint,
                            "purpose": line.purpose,
                        }
                        for line in rendered_lines
                    ],
                },
            }
        )
        return replace(
            base_request,
            positive_prompt=f"{base_request.positive_prompt}\n" + "\n".join(sketch_block),
            metadata=metadata,
        )

    def _select_lines_for_render(
        self,
        bubble_request: BubbleFillRequest | None,
    ) -> tuple[BubbleLine, ...]:
        if self.bubble_render_config.mode == "none" or bubble_request is None:
            return ()
        allowed_purposes = {"dialogue"}
        if self.bubble_render_config.include_captions:
            allowed_purposes.add("caption")
        return tuple(
            line for line in bubble_request.lines if line.purpose in allowed_purposes and line.text
        )


@dataclass(frozen=True)
class SpeechBubblePlanner:
    model_config: ModelEndpointConfig
    speaker_resolver: DialogueSpeakerResolver | None = None
    fallback_caption_limit: int = 1

    def build_requests(
        self,
        storyboard: StoryboardPlan,
        sketches: SketchStoryboard,
    ) -> tuple[BubbleFillRequest, ...]:
        sketch_map = {panel.panel_id: panel for panel in sketches.panels}
        state = SpeakerInferenceState()
        return tuple(
            self._build_request(panel, sketch_map[panel.panel_id], state)
            for panel in storyboard.panels
        )

    def _build_request(
        self,
        panel: PanelSpec,
        sketch: SketchPanel,
        state: SpeakerInferenceState,
    ) -> BubbleFillRequest:
        lines = self._build_lines(panel, sketch, state)
        system_prompt = (
            "You fill webtoon speech bubbles with minimal token usage. "
            "Keep each line short, emotionally clear, and easy to letter."
        )
        user_sections = [
            f"panel_id: {panel.panel_id}",
            f"narrative: {panel.narrative}",
            f"emotion: {panel.emotion}",
            f"bubble slots: {' | '.join(sketch.bubble_slots)}",
            "Return compact JSON array items with slot, speaker, and text only.",
        ]
        metadata = {
            "pipeline_stage": "stage4_bubble",
            "model_config": asdict(self.model_config),
            "low_token_strategy": "trimmed-dialogue-first",
            "line_count": len(lines),
            "speaker_assignments": [
                {
                    "speaker": line.speaker,
                    "speaker_character_id": line.speaker_character_id,
                    "speaker_confidence": line.speaker_confidence,
                    "speaker_rationale": list(line.speaker_rationale),
                    "slot_hint": line.slot_hint,
                    "purpose": line.purpose,
                }
                for line in lines
            ],
        }
        return BubbleFillRequest(
            panel_id=panel.panel_id,
            system_prompt=system_prompt,
            user_prompt="\n".join(user_sections),
            lines=lines,
            model=self.model_config.model,
            max_output_tokens=self.model_config.max_output_tokens or 128,
            metadata=metadata,
        )

    def _build_lines(
        self,
        panel: PanelSpec,
        sketch: SketchPanel,
        state: SpeakerInferenceState,
    ) -> tuple[BubbleLine, ...]:
        previous_line_speaker_id: str | None = None
        lines: list[BubbleLine] = []
        if panel.dialogue:
            for index, dialogue in enumerate(panel.dialogue):
                line = self._dialogue_to_line(
                    dialogue,
                    panel,
                    sketch,
                    index,
                    previous_line_speaker_id=previous_line_speaker_id,
                    previous_panel_speaker_id=state.previous_panel_speaker_id,
                )
                lines.append(line)
                if line.speaker_character_id:
                    previous_line_speaker_id = line.speaker_character_id
                    state.previous_panel_speaker_id = line.speaker_character_id
            return tuple(lines)
        caption = self._compress_text(panel.narrative)
        return tuple(
            BubbleLine(
                speaker="caption",
                text=caption,
                slot_hint=sketch.bubble_slots[0] if sketch.bubble_slots else "caption lane",
                purpose="caption",
            )
            for _ in range(self.fallback_caption_limit)
        )

    def _dialogue_to_line(
        self,
        dialogue: str,
        panel: PanelSpec,
        sketch: SketchPanel,
        index: int,
        previous_line_speaker_id: str | None,
        previous_panel_speaker_id: str | None,
    ) -> BubbleLine:
        speaker = ""
        text = dialogue.strip()
        if ":" in dialogue:
            speaker, text = (part.strip() for part in dialogue.split(":", 1))
        purpose = "dialogue"
        normalized_speaker = speaker.lower()
        if normalized_speaker in {"caption", "narration", "내레이션"}:
            purpose = "caption"
        elif normalized_speaker in {"sfx", "효과음"}:
            purpose = "sfx"
        slot_hint = (
            sketch.bubble_slots[index]
            if index < len(sketch.bubble_slots)
            else f"bubble-{index + 1}"
        )
        speaker_character_id = None
        speaker_confidence = 0.0
        speaker_rationale: tuple[str, ...] = ()
        if self.speaker_resolver is not None:
            speaker_character_id, speaker_confidence, speaker_rationale = self.speaker_resolver.resolve(
                speaker_label=speaker,
                line_text=text,
                panel=panel,
                slot_hint=slot_hint,
                line_index=index,
                previous_line_speaker_id=previous_line_speaker_id,
                previous_panel_speaker_id=previous_panel_speaker_id,
            )
        return BubbleLine(
            speaker=speaker or "dialogue",
            text=self._compress_text(text),
            slot_hint=slot_hint,
            purpose=purpose,
            speaker_character_id=speaker_character_id,
            speaker_confidence=speaker_confidence,
            speaker_rationale=speaker_rationale,
        )

    def _compress_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= 60:
            return normalized
        return normalized[:57].rstrip() + "..."


@dataclass
class NovelToWebtoonPipeline:
    continuity: ContinuityState
    storyboard_planner: NovelStoryboardPlanner
    sketch_planner: StickFigureStoryboardPlanner
    refinement_planner: VisualRefinementPlanner
    bubble_planner: SpeechBubblePlanner

    def stage1_storyboard(self, source: WebNovelInput) -> StoryboardPlan:
        return self.storyboard_planner.build(source)

    def stage2_sketch_storyboard(self, storyboard: StoryboardPlan) -> SketchStoryboard:
        return self.sketch_planner.build(storyboard)

    def stage3_refine_requests(
        self,
        storyboard: StoryboardPlan,
        sketches: SketchStoryboard,
        bubble_requests: tuple[BubbleFillRequest, ...] | None = None,
    ) -> tuple[GenerationRequest, ...]:
        return self.refinement_planner.build_requests(
            storyboard,
            sketches,
            self.continuity,
            bubble_requests=bubble_requests,
        )

    def stage3_generate(
        self,
        storyboard: StoryboardPlan,
        sketches: SketchStoryboard,
        provider: ImageProvider,
        bubble_requests: tuple[BubbleFillRequest, ...] | None = None,
    ) -> tuple[GeneratedImage, ...]:
        return self.refinement_planner.generate(
            storyboard,
            sketches,
            self.continuity,
            provider,
            bubble_requests=bubble_requests,
        )

    def stage4_bubble_requests(
        self,
        storyboard: StoryboardPlan,
        sketches: SketchStoryboard,
    ) -> tuple[BubbleFillRequest, ...]:
        return self.bubble_planner.build_requests(storyboard, sketches)

    def build_draft(self, source: WebNovelInput) -> PipelineDraft:
        storyboard = self.stage1_storyboard(source)
        sketches = self.stage2_sketch_storyboard(storyboard)
        bubble_requests = self.stage4_bubble_requests(storyboard, sketches)
        refined_requests = self.stage3_refine_requests(
            storyboard,
            sketches,
            bubble_requests=bubble_requests,
        )
        return PipelineDraft(
            source=source,
            storyboard=storyboard,
            sketches=sketches,
            refined_requests=refined_requests,
            bubble_requests=bubble_requests,
        )

    def stage5_plan_bubbles(
        self,
        sketches: SketchStoryboard,
        bubble_requests: tuple[BubbleFillRequest, ...],
    ) -> tuple[BubblePlacementPlan, ...]:
        if self.refinement_planner.bubble_render_config.mode != "overlay":
            return ()
        sketch_map = {panel.panel_id: panel for panel in sketches.panels}
        renderer = BubbleOverlayRenderer(
            image_config=self.refinement_planner.image_model,
            render_config=self.refinement_planner.bubble_render_config,
        )
        plans = []
        for bubble_request in bubble_requests:
            sketch = sketch_map[bubble_request.panel_id]
            plan = renderer.build_plan(sketch, bubble_request)
            if plan.placements:
                plans.append(plan)
        return tuple(plans)

    def stage5_render_overlays(
        self,
        images: tuple[GeneratedImage, ...],
        bubble_plans: tuple[BubblePlacementPlan, ...],
    ) -> tuple[BubbleRenderArtifact, ...]:
        if self.refinement_planner.bubble_render_config.mode != "overlay":
            return ()
        image_map = {image.panel_id: image for image in images}
        renderer = BubbleOverlayRenderer(
            image_config=self.refinement_planner.image_model,
            render_config=self.refinement_planner.bubble_render_config,
        )
        artifacts: list[BubbleRenderArtifact] = []
        for plan in bubble_plans:
            image = image_map.get(plan.panel_id)
            if image is None:
                continue
            artifact = renderer.render(image, plan)
            if artifact is not None:
                artifacts.append(artifact)
        return tuple(artifacts)

    def run(self, source: WebNovelInput, provider: ImageProvider) -> PipelineRunResult:
        draft = self.build_draft(source)
        images = self.stage3_generate(
            draft.storyboard,
            draft.sketches,
            provider,
            bubble_requests=draft.bubble_requests,
        )
        bubble_plans = self.stage5_plan_bubbles(draft.sketches, draft.bubble_requests)
        bubble_artifacts = self.stage5_render_overlays(images, bubble_plans)
        return PipelineRunResult(
            draft=draft,
            generated_images=images,
            bubble_plans=bubble_plans,
            bubble_artifacts=bubble_artifacts,
        )
