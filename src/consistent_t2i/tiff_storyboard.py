from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .demo_data import build_demo_characters
from .models import WebNovelInput
from .pipeline import NovelStoryboardPlanner


@dataclass(frozen=True)
class ContiPanel:
    panel_id: str
    narrative: str
    dialogue: tuple[str, ...]
    location: str
    emotion: str


@dataclass(frozen=True)
class RenderConfig:
    page_width: int = 1240
    page_height: int = 1754
    margin: int = 56
    gutter: int = 24
    header_height: int = 110
    panel_padding: int = 20
    panels_per_page: int = 4
    font_path: str | None = None
    bundle_tiff: bool = False


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    path: str
    panel_ids: tuple[str, ...]
    pixel_mode: str
    compression: str
    file_size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "page_number": self.page_number,
            "path": self.path,
            "panel_ids": list(self.panel_ids),
            "pixel_mode": self.pixel_mode,
            "compression": self.compression,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass(frozen=True)
class RenderedStoryboard:
    title: str
    episode_id: str
    panel_count: int
    pages: tuple[RenderedPage, ...]
    manifest_path: str
    source_copy_path: str
    bundle_path: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "episode_id": self.episode_id,
            "panel_count": self.panel_count,
            "manifest_path": self.manifest_path,
            "source_copy_path": self.source_copy_path,
            "bundle_path": self.bundle_path,
            "pages": [page.to_dict() for page in self.pages],
        }


class StoryboardPlanner:
    def __init__(self, planner: NovelStoryboardPlanner | None = None) -> None:
        self.planner = planner or NovelStoryboardPlanner(characters=build_demo_characters())

    def build(
        self,
        *,
        source_text: str,
        title: str = "웹소설 콘티",
        episode_id: str,
        target_panel_count: int,
    ) -> tuple[ContiPanel, ...]:
        storyboard = self.planner.build(
            WebNovelInput(
                episode_id=episode_id,
                title=title,
                source_text=source_text,
                target_panel_count=max(1, target_panel_count),
            )
        )
        return tuple(
            ContiPanel(
                panel_id=panel.panel_id,
                narrative=panel.narrative,
                dialogue=panel.dialogue,
                location=panel.location,
                emotion=panel.emotion,
            )
            for panel in storyboard.panels
        )


def render_novel_to_tiff(
    *,
    source_text: str,
    title: str,
    episode_id: str,
    target_panel_count: int,
    output_dir: str | Path,
    render_config: RenderConfig | None = None,
    storyboard_planner: StoryboardPlanner | None = None,
) -> RenderedStoryboard:
    config = render_config or RenderConfig()
    planner = storyboard_planner or StoryboardPlanner()
    panels = planner.build(
        source_text=source_text,
        title=title,
        episode_id=episode_id,
        target_panel_count=target_panel_count,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    source_copy_path = output_root / "source.txt"
    source_copy_path.write_text(source_text.strip() + "\n", encoding="utf-8")

    page_paths: list[Path] = []
    pages: list[RenderedPage] = []
    for page_number, page_panels in enumerate(_chunked(panels, max(1, config.panels_per_page)), start=1):
        page_image = _render_page(
            title=title,
            episode_id=episode_id,
            page_number=page_number,
            page_panels=page_panels,
            config=config,
        )
        page_path = output_root / f"page-{page_number:03d}.tiff"
        page_image.save(page_path, format="TIFF", compression="group4")
        page_paths.append(page_path)
        pages.append(
            RenderedPage(
                page_number=page_number,
                path=str(page_path.resolve()),
                panel_ids=tuple(panel.panel_id for panel in page_panels),
                pixel_mode="1",
                compression="group4",
                file_size_bytes=page_path.stat().st_size,
            )
        )

    bundle_path = ""
    if config.bundle_tiff and page_paths:
        resolved_bundle = output_root / "storyboard.tiff"
        _write_bundle(page_paths, resolved_bundle)
        bundle_path = str(resolved_bundle.resolve())

    result = RenderedStoryboard(
        title=title,
        episode_id=episode_id,
        panel_count=len(panels),
        pages=tuple(pages),
        manifest_path=str((output_root / "manifest.json").resolve()),
        source_copy_path=str(source_copy_path.resolve()),
        bundle_path=bundle_path,
    )
    manifest_path = Path(result.manifest_path)
    manifest_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def format_tiff_storyboard_summary(result: RenderedStoryboard) -> str:
    lines = [
        f"Title: {result.title}",
        f"Episode: {result.episode_id}",
        f"Panels: {result.panel_count}",
        f"Pages: {len(result.pages)}",
        f"Manifest: {result.manifest_path}",
        "",
        "[TIFF Pages]",
    ]
    for page in result.pages:
        lines.extend(
            [
                f"- page {page.page_number}",
                f"  path: {page.path}",
                f"  panels: {', '.join(page.panel_ids)}",
                f"  mode/compression: {page.pixel_mode} / {page.compression}",
                f"  bytes: {page.file_size_bytes}",
            ]
        )
    if result.bundle_path:
        lines.extend(["", f"Combined TIFF: {result.bundle_path}"])
    return "\n".join(lines)


def _render_page(
    *,
    title: str,
    episode_id: str,
    page_number: int,
    page_panels: tuple[ContiPanel, ...],
    config: RenderConfig,
) -> Image.Image:
    canvas = Image.new("L", (config.page_width, config.page_height), 255)
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(config.font_path, 34)
    body_font = _load_font(config.font_path, 22)
    label_font = _load_font(config.font_path, 18)

    draw.text(
        (config.margin, config.margin),
        title,
        fill=0,
        font=title_font,
    )
    draw.text(
        (config.margin, config.margin + 42),
        f"{episode_id}  page {page_number:03d}",
        fill=0,
        font=label_font,
    )

    content_top = config.margin + config.header_height
    available_height = config.page_height - content_top - config.margin
    panel_count = max(1, len(page_panels))
    panel_height = (available_height - (config.gutter * (panel_count - 1))) // panel_count
    panel_width = config.page_width - (config.margin * 2)

    for index, panel in enumerate(page_panels):
        top = content_top + index * (panel_height + config.gutter)
        left = config.margin
        right = left + panel_width
        bottom = top + panel_height
        draw.rectangle((left, top, right, bottom), outline=0, width=4)

        body_left = left + config.panel_padding
        body_top = top + config.panel_padding
        body_right = right - config.panel_padding
        draw.text(
            (body_left, body_top),
            f"{panel.panel_id} | {panel.location} | {panel.emotion}",
            fill=0,
            font=label_font,
        )
        cursor_y = body_top + _line_height(label_font) + 10
        cursor_y = _draw_wrapped_block(
            draw,
            text=panel.narrative,
            font=body_font,
            left=body_left,
            top=cursor_y,
            right=body_right,
            bottom=bottom - config.panel_padding,
            line_gap=6,
        )
        if panel.dialogue and cursor_y < bottom - config.panel_padding - (_line_height(body_font) * 2):
            cursor_y += 6
            draw.line(
                (body_left, cursor_y, body_right, cursor_y),
                fill=0,
                width=2,
            )
            cursor_y += 10
            for line in panel.dialogue:
                cursor_y = _draw_wrapped_block(
                    draw,
                    text=f"- {line}",
                    font=body_font,
                    left=body_left,
                    top=cursor_y,
                    right=body_right,
                    bottom=bottom - config.panel_padding,
                    line_gap=4,
                )
                if cursor_y >= bottom - config.panel_padding - _line_height(body_font):
                    break

    return canvas.point(lambda value: 0 if value < 200 else 255, mode="1")


def _draw_wrapped_block(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    left: int,
    top: int,
    right: int,
    bottom: int,
    line_gap: int,
) -> int:
    cursor_y = top
    for line in _wrap_text(draw, text, font, right - left):
        next_y = cursor_y + _line_height(font)
        if next_y > bottom:
            break
        draw.text((left, cursor_y), line, fill=0, font=font)
        cursor_y = next_y + line_gap
    return cursor_y


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
) -> tuple[str, ...]:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return ()
    tokens = normalized.split(" ")
    if len(tokens) == 1:
        tokens = list(normalized)
        separator = ""
    else:
        separator = " "
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = token if not current else f"{current}{separator}{token}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = token
            continue
        lines.append(_truncate_token(draw, token, font, max_width))
        current = ""
    if current:
        lines.append(current)
    return tuple(lines)


def _truncate_token(
    draw: ImageDraw.ImageDraw,
    token: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    current = ""
    for character in token:
        candidate = current + character
        if draw.textlength(candidate, font=font) > max_width:
            if current:
                return current
            return character
        current = candidate
    return current


def _line_height(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> int:
    bbox = font.getbbox("Ag가")
    return max(20, bbox[3] - bbox[1] + 2)


def _load_font(
    font_path: str | None,
    size: int,
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidates = [candidate for candidate in (font_path, _default_korean_font_path()) if candidate]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _default_korean_font_path() -> str | None:
    windir = Path(os.environ.get("WINDIR", "C:\\Windows"))
    for candidate in ("malgun.ttf", "malgunbd.ttf", "gulim.ttc"):
        font_path = windir / "Fonts" / candidate
        if font_path.exists():
            return str(font_path)
    return None


def _chunked(items: tuple[ContiPanel, ...], size: int) -> tuple[tuple[ContiPanel, ...], ...]:
    return tuple(
        items[index : index + size]
        for index in range(0, len(items), size)
    )


def _write_bundle(page_paths: list[Path], destination: Path) -> None:
    images = [Image.open(page_path) for page_path in page_paths]
    try:
        first, *rest = images
        first.save(destination, format="TIFF", save_all=True, append_images=rest, compression="group4")
    finally:
        for image in images:
            image.close()
