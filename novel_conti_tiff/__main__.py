from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import RenderConfig, render_novel_to_tiff


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m novel_conti_tiff",
        description="Render a web novel into 1-bit TIFF storyboard sheets with very small memory usage.",
    )
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--text", help="Inline novel text.")
    input_group.add_argument("--text-file", help="Read novel text from a UTF-8 text file.")
    input_group.add_argument("--stdin", action="store_true", help="Read novel text from stdin.")
    parser.add_argument("--title", default="웹소설 콘티", help="Storyboard title.")
    parser.add_argument("--episode-id", default="storyboard", help="Stable id for panel labels.")
    parser.add_argument("--panel-count", type=int, default=8, help="Target number of panels.")
    parser.add_argument("--panels-per-page", type=int, default=4, help="How many panels to place on each TIFF page.")
    parser.add_argument("--output-dir", default="outputs/novel_conti_tiff", help="Directory for TIFF output.")
    parser.add_argument("--font-path", default="", help="Optional font path for Korean text rendering.")
    parser.add_argument("--bundle", action="store_true", help="Also write a combined multi-page TIFF after page rendering.")
    parser.add_argument(
        "--format",
        choices=("summary", "json"),
        default="summary",
        help="CLI output format.",
    )
    return parser


def resolve_source_text(args: argparse.Namespace, stdin_stream: object = sys.stdin) -> str:
    if args.text:
        return args.text.strip()
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    if args.stdin:
        return stdin_stream.read().strip()
    return ""


def format_summary(result) -> str:
    lines = [
        f"Title: {result.title}",
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


def main(argv: list[str] | None = None, stdin_stream: object = sys.stdin) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    source_text = resolve_source_text(args, stdin_stream=stdin_stream)
    if not source_text:
        raise SystemExit("Provide source text with --text, --text-file, or --stdin.")

    render_config = RenderConfig(
        panels_per_page=max(1, args.panels_per_page),
        font_path=args.font_path or None,
        bundle_tiff=args.bundle,
    )
    result = render_novel_to_tiff(
        source_text=source_text,
        title=args.title,
        episode_id=args.episode_id,
        target_panel_count=max(1, args.panel_count),
        output_dir=args.output_dir,
        render_config=render_config,
    )

    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return
    print(format_summary(result))


if __name__ == "__main__":
    main()
