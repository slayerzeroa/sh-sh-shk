from __future__ import annotations

import base64
import html
import math
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .models import (
    BubbleFillRequest,
    BubblePlacement,
    BubblePlacementPlan,
    BubbleRenderArtifact,
    BubbleRenderConfig,
    BubbleLine,
    GeneratedImage,
    ModelEndpointConfig,
    SketchPanel,
)


def parse_canvas_size(size: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", size.strip())
    if not match:
        return 1024, 1024
    return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True)
class _BubbleTextLayout:
    wrapped_lines: tuple[str, ...]
    font_size: int
    line_height: int
    bubble_width: int
    bubble_height: int


RenderableFont = ImageFont.FreeTypeFont | ImageFont.ImageFont


@dataclass(frozen=True)
class BubbleOverlayRenderer:
    image_config: ModelEndpointConfig
    render_config: BubbleRenderConfig
    margin_ratio: float = 0.06

    def build_plan(
        self,
        sketch: SketchPanel,
        bubble_request: BubbleFillRequest,
    ) -> BubblePlacementPlan:
        canvas_width, canvas_height = parse_canvas_size(self.image_config.size)
        placements: list[BubblePlacement] = []
        face_zones = self._build_face_zones(sketch, canvas_width, canvas_height)
        visible_lines = self._select_visible_lines(bubble_request.lines)
        for index, line in enumerate(visible_lines, start=1):
            placement = self._build_placement(
                sketch=sketch,
                line=line,
                index=index,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                existing_placements=tuple(placements),
                face_zones=face_zones,
            )
            placements.append(placement)
        return BubblePlacementPlan(
            panel_id=sketch.panel_id,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            placements=tuple(placements),
        )

    def render(
        self,
        image: GeneratedImage,
        plan: BubblePlacementPlan,
    ) -> BubbleRenderArtifact | None:
        if not plan.placements:
            return None

        base_image_path = image.local_path
        if not base_image_path:
            return None

        base_path = Path(base_image_path)
        output_root = base_path.parent
        output_root.mkdir(parents=True, exist_ok=True)
        overlay_path = output_root / f"{base_path.stem}.overlay.svg"
        composite_path = output_root / f"{base_path.stem}.final.svg"
        final_png_path = output_root / f"{base_path.stem}.final.png"

        overlay_path.write_text(
            self._build_overlay_svg(plan),
            encoding="utf-8",
        )
        composite_path.write_text(
            self._build_composite_svg(base_path, plan),
            encoding="utf-8",
        )
        self._build_composite_png(base_path, final_png_path, plan)
        return BubbleRenderArtifact(
            panel_id=plan.panel_id,
            base_image_path=str(base_path.resolve()),
            overlay_svg_path=str(overlay_path.resolve()),
            composite_svg_path=str(composite_path.resolve()),
            final_png_path=str(final_png_path.resolve()),
        )

    def _select_visible_lines(self, lines: tuple[BubbleLine, ...]) -> tuple[BubbleLine, ...]:
        if self.render_config.mode != "overlay":
            return ()
        allowed_purposes = {"dialogue"}
        if self.render_config.include_captions:
            allowed_purposes.add("caption")
        return tuple(
            line for line in lines if line.purpose in allowed_purposes and line.text.strip()
        )

    def _build_placement(
        self,
        *,
        sketch: SketchPanel,
        line: BubbleLine,
        index: int,
        canvas_width: int,
        canvas_height: int,
        existing_placements: tuple[BubblePlacement, ...],
        face_zones: tuple[tuple[int, int, int, int], ...],
    ) -> BubblePlacement:
        text_layout = self._layout_bubble_text(
            text=line.text,
            purpose=line.purpose,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )

        x, y = self._resolve_anchor(
            sketch=sketch,
            slot_hint=line.slot_hint,
            speaker=line.speaker,
            speaker_character_id=line.speaker_character_id,
            index=index,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            bubble_width=text_layout.bubble_width,
            bubble_height=text_layout.bubble_height,
            existing_placements=existing_placements,
            face_zones=face_zones,
        )

        shape: Literal["ellipse", "caption-box"] = (
            "caption-box" if line.purpose == "caption" else "ellipse"
        )
        tail_direction = self._infer_tail_direction(
            sketch,
            line.speaker,
            line.speaker_character_id,
            x,
            text_layout.bubble_width,
            canvas_width,
        )

        return BubblePlacement(
            panel_id=sketch.panel_id,
            slot_hint=line.slot_hint,
            speaker=line.speaker,
            speaker_character_id=line.speaker_character_id,
            text=line.text,
            purpose=line.purpose,
            x=x,
            y=y,
            width=text_layout.bubble_width,
            height=text_layout.bubble_height,
            font_size=text_layout.font_size,
            line_height=text_layout.line_height,
            wrapped_lines=text_layout.wrapped_lines,
            shape=shape,
            tail_direction=tail_direction,
        )

    def _resolve_anchor(
        self,
        *,
        sketch: SketchPanel,
        slot_hint: str,
        speaker: str,
        speaker_character_id: str | None,
        index: int,
        canvas_width: int,
        canvas_height: int,
        bubble_width: int,
        bubble_height: int,
        existing_placements: tuple[BubblePlacement, ...],
        face_zones: tuple[tuple[int, int, int, int], ...],
    ) -> tuple[int, int]:
        margin_x = int(canvas_width * self.margin_ratio)
        margin_y = int(canvas_height * self.margin_ratio)
        lowered_slot = slot_hint.lower()

        if "top-left" in lowered_slot:
            return margin_x, margin_y

        anchor_x = self._speaker_anchor_x(sketch, speaker, speaker_character_id, canvas_width)
        if anchor_x is None:
            fractions = (0.18, 0.52, 0.74)
            anchor_x = int(canvas_width * fractions[min(index - 1, len(fractions) - 1)])
        candidate_positions = self._candidate_positions(
            anchor_x=anchor_x,
            index=index,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            bubble_width=bubble_width,
            bubble_height=bubble_height,
            margin_x=margin_x,
            margin_y=margin_y,
            prefers_away_from_face="away from focal face" in lowered_slot,
        )
        best_position = candidate_positions[0]
        best_score = float("-inf")
        for position in candidate_positions:
            score = self._score_position(
                position=position,
                bubble_width=bubble_width,
                bubble_height=bubble_height,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                anchor_x=anchor_x,
                existing_placements=existing_placements,
                face_zones=face_zones,
            )
            if score > best_score:
                best_score = score
                best_position = position
        return best_position

    def _speaker_anchor_x(
        self,
        sketch: SketchPanel,
        speaker: str,
        speaker_character_id: str | None,
        canvas_width: int,
    ) -> int | None:
        if speaker_character_id:
            for guide in sketch.figure_guides:
                lowered_guide = guide.lower()
                if speaker_character_id.lower() in lowered_guide:
                    return self._extract_anchor_from_guide(lowered_guide, canvas_width)
        lowered_speaker = speaker.lower()
        for guide in sketch.figure_guides:
            lowered_guide = guide.lower()
            if lowered_speaker and lowered_speaker in lowered_guide:
                return self._extract_anchor_from_guide(lowered_guide, canvas_width)
        if lowered_speaker in {"caption", "narration"}:
            return int(canvas_width * 0.24)
        return None

    def _extract_anchor_from_guide(self, guide: str, canvas_width: int) -> int:
        if "left third" in guide:
            return int(canvas_width * 0.24)
        if "right third" in guide:
            return int(canvas_width * 0.74)
        if "center" in guide:
            return int(canvas_width * 0.50)
        return int(canvas_width * 0.50)

    def _infer_tail_direction(
        self,
        sketch: SketchPanel,
        speaker: str,
        speaker_character_id: str | None,
        x: int,
        bubble_width: int,
        canvas_width: int,
    ) -> Literal["left", "right", "down", "none"]:
        anchor_x = self._speaker_anchor_x(sketch, speaker, speaker_character_id, canvas_width)
        if anchor_x is None:
            return "none"
        bubble_center = x + bubble_width // 2
        if anchor_x < bubble_center - 20:
            return "left"
        if anchor_x > bubble_center + 20:
            return "right"
        return "down"

    def _layout_bubble_text(
        self,
        *,
        text: str,
        purpose: str,
        canvas_width: int,
        canvas_height: int,
    ) -> _BubbleTextLayout:
        is_dialogue = purpose == "dialogue"
        max_width = int(canvas_width * (0.41 if is_dialogue else 0.50))
        min_width = int(canvas_width * (0.22 if is_dialogue else 0.28))
        min_height = int(canvas_height * (0.13 if is_dialogue else 0.11))
        max_height = int(canvas_height * (0.28 if is_dialogue else 0.22))
        base_font_size = max(24, int(canvas_width * (0.030 if is_dialogue else 0.027)))
        min_font_size = max(20, int(base_font_size * 0.76))

        best_layout: _BubbleTextLayout | None = None
        best_score = float("inf")
        for font_size in range(base_font_size, min_font_size - 1, -2):
            font = self._load_font(font_size)
            line_height = self._line_height(font_size, purpose)
            padding_x = max(30, int(font_size * (1.34 if is_dialogue else 1.18)))
            padding_top = max(22, int(font_size * 0.92))
            padding_bottom = max(26, int(font_size * 1.04))
            max_text_width = max(120, max_width - padding_x * 2)
            wrapped_lines = self._wrap_text(text, max_width=max_text_width, font=font)
            max_line_width = max(self._measure_text_width(line, font=font) for line in wrapped_lines)
            text_block_height = self._measure_text_block_height(
                wrapped_lines,
                font=font,
                line_height=line_height,
            )
            bubble_width = max(min_width, min(max_width, int(math.ceil(max_line_width + padding_x * 2))))
            bubble_height = max(
                min_height,
                int(math.ceil(text_block_height + padding_top + padding_bottom)),
            )
            layout = _BubbleTextLayout(
                wrapped_lines=wrapped_lines,
                font_size=font_size,
                line_height=line_height,
                bubble_width=bubble_width,
                bubble_height=bubble_height,
            )
            score = len(wrapped_lines) * 10 + abs((bubble_width / max(bubble_height, 1)) - 1.18) * 4
            if bubble_height > max_height:
                score += 100 + (bubble_height - max_height)
            if score < best_score:
                best_layout = layout
                best_score = score
            if bubble_height <= max_height and len(wrapped_lines) <= 4:
                return layout

        if best_layout is None:
            fallback_font_size = min_font_size
            fallback_font = self._load_font(fallback_font_size)
            fallback_line_height = self._line_height(fallback_font_size, purpose)
            fallback_lines = self._wrap_text(text, max_width=max_width - 64, font=fallback_font)
            return _BubbleTextLayout(
                wrapped_lines=fallback_lines,
                font_size=fallback_font_size,
                line_height=fallback_line_height,
                bubble_width=max_width,
                bubble_height=min_height,
            )
        return best_layout

    def _wrap_text(self, text: str, *, max_width: int, font: RenderableFont) -> tuple[str, ...]:
        cleaned = re.sub(r"\s+", " ", text.strip())
        if not cleaned:
            return ("",)
        words = cleaned.split(" ")
        lines: list[str] = []
        current = ""
        for word in words:
            if not word:
                continue
            candidate = word if not current else f"{current} {word}"
            if self._measure_text_width(candidate, font=font) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            split_word = self._split_long_token(word, max_width=max_width, font=font)
            if len(split_word) > 1:
                lines.extend(split_word[:-1])
            current = split_word[-1]
        if current:
            lines.append(current)
        return tuple(lines or [cleaned])

    def _split_long_token(
        self,
        token: str,
        *,
        max_width: int,
        font: RenderableFont,
    ) -> tuple[str, ...]:
        segments: list[str] = []
        current = ""
        for char in token:
            candidate = current + char
            if current and self._measure_text_width(candidate, font=font) > max_width:
                segments.append(current)
                current = char
                continue
            current = candidate
        if current:
            segments.append(current)
        return tuple(segments or [token])

    def _line_height(self, font_size: int, purpose: str) -> int:
        ratio = 1.18 if purpose == "dialogue" else 1.24
        return max(font_size + 6, int(font_size * ratio))

    def _measure_text_width(
        self,
        text: str,
        *,
        font: RenderableFont,
        stroke_width: int = 1,
    ) -> int:
        bbox = self._measure_text_bbox(text, font=font, stroke_width=stroke_width)
        return bbox[2] - bbox[0]

    def _measure_text_height(
        self,
        text: str,
        *,
        font: RenderableFont,
        stroke_width: int = 1,
    ) -> int:
        bbox = self._measure_text_bbox(text, font=font, stroke_width=stroke_width)
        return bbox[3] - bbox[1]

    def _measure_text_bbox(
        self,
        text: str,
        *,
        font: RenderableFont,
        stroke_width: int = 1,
    ) -> tuple[int, int, int, int]:
        probe = Image.new("L", (1, 1), 0)
        return ImageDraw.Draw(probe).textbbox((0, 0), text, font=font, stroke_width=stroke_width)

    def _measure_text_block_height(
        self,
        lines: tuple[str, ...],
        *,
        font: RenderableFont,
        line_height: int,
    ) -> int:
        if not lines:
            return 0
        max_line_height = max(self._measure_text_height(line, font=font) for line in lines)
        return max_line_height + max(0, len(lines) - 1) * line_height

    def _build_overlay_svg(self, plan: BubblePlacementPlan) -> str:
        elements = self._build_bubble_elements(plan)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{plan.canvas_width}" '
            f'height="{plan.canvas_height}" viewBox="0 0 {plan.canvas_width} {plan.canvas_height}">\n'
            f"{self._build_defs()}\n"
            f'  <rect width="100%" height="100%" fill="transparent"/>\n'
            f"{elements}\n"
            f"</svg>\n"
        )

    def _build_composite_svg(self, base_path: Path, plan: BubblePlacementPlan) -> str:
        mime_type = mimetypes.guess_type(base_path.name)[0] or "image/png"
        data_uri = f"data:{mime_type};base64,{base64.b64encode(base_path.read_bytes()).decode('ascii')}"
        elements = self._build_bubble_elements(plan)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{plan.canvas_width}" '
            f'height="{plan.canvas_height}" viewBox="0 0 {plan.canvas_width} {plan.canvas_height}">\n'
            f"{self._build_defs()}\n"
            f'  <image href="{data_uri}" x="0" y="0" width="{plan.canvas_width}" height="{plan.canvas_height}"/>\n'
            f"{elements}\n"
            f"</svg>\n"
        )

    def _build_composite_png(
        self,
        base_path: Path,
        output_path: Path,
        plan: BubblePlacementPlan,
    ) -> None:
        base_image = Image.open(base_path).convert("RGBA")
        overlay = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
        shadow = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        overlay_draw = ImageDraw.Draw(overlay)
        text_draw = ImageDraw.Draw(overlay)

        for placement in plan.placements:
            if placement.shape == "caption-box":
                self._draw_caption_box(shadow_draw, overlay_draw, placement)
            else:
                self._draw_dialogue_bubble(shadow_draw, overlay_draw, placement)
            self._draw_text(text_draw, placement)

        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
        composed = Image.alpha_composite(base_image, shadow)
        composed = Image.alpha_composite(composed, overlay)
        composed.save(output_path)

    def _build_defs(self) -> str:
        return (
            "  <defs>\n"
            "    <filter id=\"bubbleShadow\" x=\"-20%\" y=\"-20%\" width=\"140%\" height=\"160%\">\n"
            "      <feDropShadow dx=\"0\" dy=\"5\" stdDeviation=\"5\" flood-color=\"#111111\" flood-opacity=\"0.18\"/>\n"
            "    </filter>\n"
            "    <filter id=\"captionShadow\" x=\"-20%\" y=\"-20%\" width=\"140%\" height=\"160%\">\n"
            "      <feDropShadow dx=\"0\" dy=\"3\" stdDeviation=\"3\" flood-color=\"#111111\" flood-opacity=\"0.14\"/>\n"
            "    </filter>\n"
            "  </defs>"
        )

    def _build_bubble_elements(self, plan: BubblePlacementPlan) -> str:
        lines: list[str] = []
        for placement in plan.placements:
            if placement.shape == "caption-box":
                lines.append(self._render_caption_box(placement))
            else:
                lines.append(self._render_ellipse_bubble(placement))
            lines.append(self._render_text(placement))
        return "\n".join(lines)

    def _render_ellipse_bubble(self, placement: BubblePlacement) -> str:
        bubble_path = self._build_webtoon_bubble_path(placement)
        tail = self._render_tail(placement)
        bubble = (
            f'  <path d="{bubble_path}" fill="white" fill-opacity="0.98" stroke="#111111" '
            f'stroke-width="5" stroke-linejoin="round" stroke-linecap="round" filter="url(#bubbleShadow)"/>\n'
            f'  <path d="{bubble_path}" fill="none" stroke="#ffffff" stroke-width="1.6" opacity="0.75" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        return (tail + "\n" if tail else "") + bubble

    def _render_caption_box(self, placement: BubblePlacement) -> str:
        return (
            f'  <rect x="{placement.x}" y="{placement.y}" width="{placement.width}" '
            f'height="{placement.height}" rx="28" ry="28" fill="white" fill-opacity="0.98" '
            f'stroke="#111111" stroke-width="4" filter="url(#captionShadow)"/>\n'
            f'  <rect x="{placement.x + 4}" y="{placement.y + 4}" width="{placement.width - 8}" '
            f'height="{placement.height - 8}" rx="24" ry="24" fill="none" stroke="#ffffff" '
            f'stroke-width="1.2" opacity="0.72"/>'
        )

    def _render_tail(self, placement: BubblePlacement) -> str:
        tail_points = self._tail_outline(placement)
        if not tail_points:
            return ""
        path = self._polygon_path(tail_points)
        return (
            f'  <path d="{path}" fill="white" fill-opacity="0.98" stroke="#111111" stroke-width="4.2" '
            f'stroke-linejoin="round" stroke-linecap="round" filter="url(#bubbleShadow)"/>\n'
            f'  <path d="{path}" fill="none" stroke="#ffffff" stroke-width="1.5" opacity="0.7" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def _render_text(self, placement: BubblePlacement) -> str:
        center_x = placement.x + placement.width / 2
        font = self._load_font(placement.font_size)
        top_y = self._text_top_y(placement, font)
        safe_lines = [html.escape(line) for line in placement.wrapped_lines]
        tspan_lines = []
        for index, line in enumerate(safe_lines):
            if index == 0:
                tspan_lines.append(
                    f'    <tspan x="{center_x:.1f}" y="{top_y:.1f}">{line}</tspan>'
                )
                continue
            tspan_lines.append(f'    <tspan x="{center_x:.1f}" dy="{placement.line_height}">{line}</tspan>')
        return (
            f'  <text text-anchor="middle" dominant-baseline="hanging" '
            f'font-family="Pretendard, Noto Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif" '
            f'font-size="{placement.font_size}" font-weight="800" fill="#111111" letter-spacing="-0.35" '
            f'style="paint-order: stroke; stroke: rgba(255,255,255,0.74); stroke-width: 0.85px;">'
            + "\n"
            + "\n".join(tspan_lines)
            + "\n  </text>"
        )

    def _build_webtoon_bubble_path(self, placement: BubblePlacement) -> str:
        return self._polygon_path(self._bubble_outline_points(placement, samples=36))

    def _draw_dialogue_bubble(
        self,
        shadow_draw: ImageDraw.ImageDraw,
        overlay_draw: ImageDraw.ImageDraw,
        placement: BubblePlacement,
    ) -> None:
        tail_points = self._tail_outline(placement)
        if tail_points:
            shadow_draw.polygon([(x, y + 4) for x, y in tail_points], fill=(17, 17, 17, 40))
            overlay_draw.polygon(tail_points, fill=(255, 255, 255, 248), outline=(17, 17, 17, 255), width=4)
        points = self._bubble_outline_points(placement, samples=36)
        shadow_points = [(x, y + 4) for x, y in points]
        shadow_draw.polygon(shadow_points, fill=(17, 17, 17, 45))
        overlay_draw.polygon(points, fill=(255, 255, 255, 250), outline=(17, 17, 17, 255), width=5)
        highlight_points = self._inset_points(points, inset=7)
        overlay_draw.line(highlight_points + [highlight_points[0]], fill=(255, 255, 255, 160), width=1)

    def _draw_caption_box(
        self,
        shadow_draw: ImageDraw.ImageDraw,
        overlay_draw: ImageDraw.ImageDraw,
        placement: BubblePlacement,
    ) -> None:
        rect = [placement.x, placement.y, placement.x + placement.width, placement.y + placement.height]
        shadow_rect = [rect[0], rect[1] + 3, rect[2], rect[3] + 3]
        shadow_draw.rounded_rectangle(shadow_rect, radius=28, fill=(17, 17, 17, 38))
        overlay_draw.rounded_rectangle(rect, radius=28, fill=(255, 255, 255, 248), outline=(17, 17, 17, 255), width=4)
        inner = [rect[0] + 4, rect[1] + 4, rect[2] - 4, rect[3] - 4]
        overlay_draw.rounded_rectangle(inner, radius=24, outline=(255, 255, 255, 180), width=1)

    def _draw_text(self, draw: ImageDraw.ImageDraw, placement: BubblePlacement) -> None:
        font = self._load_font(placement.font_size)
        center_x = placement.x + placement.width / 2
        top_y = self._text_top_y(placement, font)
        for index, line in enumerate(placement.wrapped_lines):
            y = top_y + index * placement.line_height
            text_width = self._measure_text_width(line, font=font)
            x = center_x - text_width / 2
            draw.text(
                (x, y),
                line,
                font=font,
                fill=(17, 17, 17, 255),
                stroke_width=1,
                stroke_fill=(255, 255, 255, 180),
            )

    def _bubble_outline_points(
        self,
        placement: BubblePlacement,
        *,
        samples: int,
    ) -> list[tuple[float, float]]:
        cx = placement.x + placement.width / 2
        cy = placement.y + placement.height / 2
        rx = placement.width / 2
        ry = placement.height / 2
        exponent = 2.8 if placement.width >= placement.height else 2.55
        outline: list[tuple[float, float]] = []
        for index in range(samples):
            theta = (2 * math.pi * index / samples) - (math.pi / 2)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            superellipse = (abs(cos_t) ** exponent + abs(sin_t) ** exponent) ** (-1.0 / exponent)
            wobble = 1.0 + 0.035 * math.sin(theta * 3 + 0.35) + 0.022 * math.sin(theta * 5 - 0.8)
            x = cx + cos_t * rx * superellipse * wobble * (1.0 + 0.015 * math.sin(theta * 2 - 0.3))
            y = cy + sin_t * ry * superellipse * wobble * (1.0 + 0.02 * math.cos(theta * 2 + 0.4))
            if sin_t < -0.62:
                y -= min(ry * 0.05, 5.0)
            if sin_t > 0.52:
                y += min(ry * 0.06, 6.0)
            if cos_t > 0.55:
                x += min(rx * 0.025, 5.0)
            outline.append((x, y))
        return outline

    def _quadratic_point(
        self,
        start: tuple[float, float],
        control: tuple[float, float],
        end: tuple[float, float],
        t: float,
    ) -> tuple[float, float]:
        x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0]
        y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1]
        return x, y

    def _tail_outline(self, placement: BubblePlacement) -> list[tuple[float, float]]:
        if placement.tail_direction == "none":
            return []
        if placement.tail_direction == "left":
            base_outer = (placement.x + placement.width * 0.24, placement.y + placement.height * 0.78)
            base_inner = (placement.x + placement.width * 0.38, placement.y + placement.height * 0.66)
            tip = (placement.x - placement.width * 0.08, placement.y + placement.height + placement.height * 0.20)
            control_a = (placement.x + placement.width * 0.10, placement.y + placement.height * 0.98)
            control_b = (placement.x + placement.width * 0.18, placement.y + placement.height * 0.72)
        elif placement.tail_direction == "right":
            base_outer = (placement.x + placement.width * 0.76, placement.y + placement.height * 0.78)
            base_inner = (placement.x + placement.width * 0.62, placement.y + placement.height * 0.66)
            tip = (placement.x + placement.width * 1.08, placement.y + placement.height + placement.height * 0.20)
            control_a = (placement.x + placement.width * 0.90, placement.y + placement.height * 0.98)
            control_b = (placement.x + placement.width * 0.82, placement.y + placement.height * 0.72)
        else:
            base_outer = (placement.x + placement.width * 0.43, placement.y + placement.height * 0.95)
            base_inner = (placement.x + placement.width * 0.57, placement.y + placement.height * 0.95)
            tip = (placement.x + placement.width * 0.50, placement.y + placement.height + placement.height * 0.24)
            control_a = (placement.x + placement.width * 0.45, placement.y + placement.height + placement.height * 0.08)
            control_b = (placement.x + placement.width * 0.55, placement.y + placement.height + placement.height * 0.08)
        outward_curve = [self._quadratic_point(base_outer, control_a, tip, t / 8) for t in range(9)]
        inward_curve = [self._quadratic_point(tip, control_b, base_inner, t / 8) for t in range(1, 9)]
        return outward_curve + inward_curve

    def _polygon_path(self, points: list[tuple[float, float]]) -> str:
        if not points:
            return ""
        commands = [f"M {points[0][0]:.1f},{points[0][1]:.1f}"]
        commands.extend(f"L {x:.1f},{y:.1f}" for x, y in points[1:])
        commands.append("Z")
        return " ".join(commands)

    def _inset_points(
        self,
        points: list[tuple[float, float]],
        *,
        inset: float,
    ) -> list[tuple[float, float]]:
        if not points:
            return []
        center_x = sum(x for x, _ in points) / len(points)
        center_y = sum(y for _, y in points) / len(points)
        inset_points: list[tuple[float, float]] = []
        for x, y in points:
            dx = x - center_x
            dy = y - center_y
            distance = math.hypot(dx, dy)
            if distance <= inset:
                inset_points.append((x, y))
                continue
            scale = (distance - inset) / distance
            inset_points.append((center_x + dx * scale, center_y + dy * scale))
        return inset_points

    def _text_top_y(
        self,
        placement: BubblePlacement,
        font: RenderableFont,
    ) -> float:
        block_height = self._measure_text_block_height(
            placement.wrapped_lines,
            font=font,
            line_height=placement.line_height,
        )
        return placement.y + (placement.height - block_height) / 2

    def _load_font(self, size: int) -> RenderableFont:
        font_candidates = [
            r"C:\Windows\Fonts\malgunbd.ttf",
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]
        for candidate in font_candidates:
            path = Path(candidate)
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size=size)
                except OSError:
                    continue
        return ImageFont.load_default()

    def _build_face_zones(
        self,
        sketch: SketchPanel,
        canvas_width: int,
        canvas_height: int,
    ) -> tuple[tuple[int, int, int, int], ...]:
        zones: list[tuple[int, int, int, int]] = []
        for guide in sketch.figure_guides:
            anchor_x = self._extract_anchor_from_guide(guide.lower(), canvas_width)
            zone_width = int(canvas_width * 0.22)
            zone_height = int(canvas_height * 0.28)
            x = max(0, anchor_x - zone_width // 2)
            y = int(canvas_height * 0.12)
            zones.append((x, y, zone_width, zone_height))
        return tuple(zones)

    def _candidate_positions(
        self,
        *,
        anchor_x: int,
        index: int,
        canvas_width: int,
        canvas_height: int,
        bubble_width: int,
        bubble_height: int,
        margin_x: int,
        margin_y: int,
        prefers_away_from_face: bool,
    ) -> tuple[tuple[int, int], ...]:
        top_y = margin_y + int(canvas_height * 0.02)
        mid_y = margin_y + int(canvas_height * 0.16) + (index - 1) * int(canvas_height * 0.08)
        low_y = margin_y + int(canvas_height * 0.30) + (index - 1) * int(canvas_height * 0.05)
        center_x = max(margin_x, min(canvas_width - margin_x - bubble_width, anchor_x - bubble_width // 2))
        left_x = max(margin_x, min(canvas_width - margin_x - bubble_width, anchor_x - bubble_width - int(canvas_width * 0.04)))
        right_x = max(margin_x, min(canvas_width - margin_x - bubble_width, anchor_x + int(canvas_width * 0.04)))
        positions = [
            (center_x, top_y),
            (left_x, mid_y),
            (right_x, mid_y),
            (center_x, low_y),
            (margin_x, top_y),
            (canvas_width - margin_x - bubble_width, top_y),
        ]
        if prefers_away_from_face:
            positions = [positions[1], positions[2], positions[0], positions[3], positions[4], positions[5]]
        return tuple(dict.fromkeys(positions))

    def _score_position(
        self,
        *,
        position: tuple[int, int],
        bubble_width: int,
        bubble_height: int,
        canvas_width: int,
        canvas_height: int,
        anchor_x: int,
        existing_placements: tuple[BubblePlacement, ...],
        face_zones: tuple[tuple[int, int, int, int], ...],
    ) -> float:
        x, y = position
        rect = (x, y, bubble_width, bubble_height)
        score = 0.0
        bubble_center_x = x + bubble_width / 2
        score -= abs(bubble_center_x - anchor_x) / canvas_width
        score -= y / canvas_height * 0.18

        for placement in existing_placements:
            overlap = self._intersection_area(
                rect,
                (placement.x, placement.y, placement.width, placement.height),
            )
            score -= overlap / max(bubble_width * bubble_height, 1) * 5.0
            expanded_overlap = self._intersection_area(
                self._expand_rect(rect, padding=24),
                self._expand_rect(
                    (placement.x, placement.y, placement.width, placement.height),
                    padding=24,
                ),
            )
            score -= expanded_overlap / max(bubble_width * bubble_height, 1) * 1.4

        for zone in face_zones:
            overlap = self._intersection_area(rect, zone)
            score -= overlap / max(bubble_width * bubble_height, 1) * 4.2

        if x < 0 or y < 0 or x + bubble_width > canvas_width or y + bubble_height > canvas_height:
            score -= 10
        return score

    def _intersection_area(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> int:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)
        if right <= left or bottom <= top:
            return 0
        return (right - left) * (bottom - top)

    def _expand_rect(
        self,
        rect: tuple[int, int, int, int],
        *,
        padding: int,
    ) -> tuple[int, int, int, int]:
        x, y, width, height = rect
        return (x - padding, y - padding, width + padding * 2, height + padding * 2)
