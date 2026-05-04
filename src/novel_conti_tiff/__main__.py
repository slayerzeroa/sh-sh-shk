from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from consistent_t2i.tiff_storyboard import RenderConfig, format_tiff_storyboard_summary, render_novel_to_tiff


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


def resolve_source_text(args: argparse.Namespace, stdin_stream: TextIO = sys.stdin) -> str:
    if args.text:
        return args.text.strip()
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    if args.stdin:
        return stdin_stream.read().strip()
    return ""


def main(argv: list[str] | None = None, stdin_stream: TextIO = sys.stdin) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    source_text = resolve_source_text(args, stdin_stream=stdin_stream)
    if not source_text:
        raise SystemExit("Provide source text with --text, --text-file, or --stdin.")

    result = render_novel_to_tiff(
        source_text=source_text,
        title=args.title,
        episode_id=args.episode_id,
        target_panel_count=max(1, args.panel_count),
        output_dir=args.output_dir,
        render_config=RenderConfig(
            panels_per_page=max(1, args.panels_per_page),
            font_path=args.font_path or None,
            bundle_tiff=args.bundle,
        ),
    )

    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return
    print(format_tiff_storyboard_summary(result))


if __name__ == "__main__":
    main()
