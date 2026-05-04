from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from .batch_api import GeminiBatchClient, read_manifest, update_manifest, write_json_artifact
from .control_models import (
    CONTROL_MODEL_CONSISTENCY_KEEPER,
    CONTROL_MODEL_TIFF_STORYBOARD,
    normalize_control_model,
)
from .demo_data import (
    build_demo_bubble_model_config,
    build_demo_characters,
    build_demo_gemini_image_model_config,
    build_demo_image_model_config,
    build_demo_novel,
    build_demo_panel,
    build_demo_state,
    build_demo_style,
)
from .env import load_local_env
from .drift import DriftAnalyzer
from .draft_mode import DraftSpreadPlanner
from .engine import ConsistentT2IEngine
from .models import (
    BatchJobInfo,
    BubbleLine,
    BubbleRenderConfig,
    BubbleFillRequest,
    DraftModeConfig,
    DraftPanelFrame,
    DraftRunResult,
    DraftSpread,
    DraftSpreadImage,
    GenerationRequest,
    ModelEndpointConfig,
    ObservedTraits,
    PanelSpec,
    PipelineRunResult,
    ReferenceImage,
    SketchPanel,
    SketchStoryboard,
    WebNovelInput,
)
from .pipeline import (
    DialogueSpeakerResolver,
    NovelStoryboardPlanner,
    NovelToWebtoonPipeline,
    SpeechBubblePlanner,
    StickFigureStoryboardPlanner,
    VisualRefinementPlanner,
)
from .planner import GenerationPlanner
from .providers import GeminiImageProvider, MockImageProvider, OpenAIImageProvider
from .tiff_storyboard import RenderConfig as TiffRenderConfig, format_tiff_storyboard_summary, render_novel_to_tiff
from .visual_bible import build_visual_bible_workspace, format_visual_bible_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m consistent_t2i",
        description="Test the web novel -> webtoon draft pipeline with demo or pasted text.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--demo", action="store_true", help="Run the bundled demo sample.")
    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="Paste multi-line text directly into the terminal, then finish with a single END line.",
    )
    input_group.add_argument("--stdin", action="store_true", help="Read the source text from stdin.")
    input_group.add_argument("--text", help="Provide the source text inline.")
    input_group.add_argument("--text-file", help="Read the source text from a UTF-8 text file.")
    parser.add_argument("--title", default="Pasted Novel Draft", help="Episode title for custom input.")
    parser.add_argument("--episode-id", default="user-episode", help="Episode id for custom input.")
    parser.add_argument("--genre", default="", help="Optional genre hint for stage 1.")
    parser.add_argument("--tone", default="", help="Optional tone hint for stage 1.")
    parser.add_argument(
        "--default-location",
        default="",
        help="Optional fallback location if the text does not imply one.",
    )
    parser.add_argument(
        "--panel-count",
        type=int,
        default=4,
        help="Target panel count for stage 1 storyboard splitting.",
    )
    parser.add_argument(
        "--control-model",
        default="consistent_t2i",
        help=(
            "Choose which root feature is controlled from src. "
            "Supported: consistent_t2i, novel_conti_tiff, consistency_keeper."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "openai", "gemini"),
        default="mock",
        help="`mock` only builds requests. `openai` and `gemini` perform real image generation and save files.",
    )
    parser.add_argument(
        "--image-model",
        default="",
        help="Override the stage 3 image model, e.g. gpt-image-1.5 or gemini-2.5-flash-image.",
    )
    parser.add_argument(
        "--image-size",
        default="",
        help="Override the image size, e.g. 1024x1024, 1024x1536, 1536x1024.",
    )
    parser.add_argument(
        "--image-quality",
        choices=("low", "medium", "high", "standard", ""),
        default="",
        help="Override the image quality. OpenAI GPT Image uses low/medium/high; Gemini uses standard.",
    )
    parser.add_argument(
        "--image-background",
        default="",
        help="Optional background override, e.g. transparent.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpeg", "webp", ""),
        default="",
        help="Override the saved output format.",
    )
    parser.add_argument(
        "--image-output-dir",
        default="",
        help="Directory where real generated images will be saved.",
    )
    parser.add_argument(
        "--cheap-image",
        action="store_true",
        help="Force the lowest-cost image settings available for the selected provider.",
    )
    parser.add_argument(
        "--draft-mode",
        action="store_true",
        help="Generate draft spreads by combining adjacent panels into a single image.",
    )
    parser.add_argument(
        "--draft-panels-per-image",
        type=int,
        default=2,
        help="How many panels to combine into one draft spread image.",
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Use the Gemini Batch API instead of immediate synchronous generation.",
    )
    parser.add_argument(
        "--batch-wait",
        action="store_true",
        help="Wait for the Gemini Batch API job to finish and download inline results.",
    )
    parser.add_argument(
        "--batch-poll-seconds",
        type=int,
        default=30,
        help="Polling interval in seconds when waiting for a batch job.",
    )
    parser.add_argument(
        "--batch-resume-name",
        default="",
        help="Resume and materialize a previously submitted Gemini batch job by job name.",
    )
    parser.add_argument(
        "--batch-resume-manifest",
        default="",
        help="Resume and materialize a previously submitted Gemini batch job from a saved manifest file.",
    )
    parser.add_argument(
        "--render-bubble-text",
        action="store_true",
        help="Render stage 4 dialogue text with stable post-processing bubble overlays.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "summary"),
        default="summary",
        help="Output format. `summary` is easier for quick manual testing.",
    )
    parser.add_argument(
        "--build-visual-bible",
        action="store_true",
        help="Compile a full-novel continuity bible into a root workspace for image-model consistency.",
    )
    parser.add_argument(
        "--visual-bible-dir",
        default="visual_bible",
        help="Root workspace directory where the visual-bible source copy and generated guides will be written.",
    )
    parser.add_argument(
        "--visual-bible-max-characters",
        type=int,
        default=10,
        help="Maximum number of character cards to generate in visual-bible mode.",
    )
    parser.add_argument(
        "--visual-bible-evidence-limit",
        type=int,
        default=4,
        help="Maximum number of evidence sentences kept per visual-bible section.",
    )
    parser.add_argument(
        "--tiff-panels-per-page",
        type=int,
        default=4,
        help="How many storyboard panels to place on each TIFF page in novel_conti_tiff mode.",
    )
    parser.add_argument(
        "--tiff-output-dir",
        default="outputs/novel_conti_tiff",
        help="Directory where TIFF storyboard pages will be written in novel_conti_tiff mode.",
    )
    parser.add_argument(
        "--tiff-font-path",
        default="",
        help="Optional font path for TIFF storyboard rendering.",
    )
    parser.add_argument(
        "--tiff-bundle",
        action="store_true",
        help="Also write a combined multi-page TIFF in novel_conti_tiff mode.",
    )
    return parser


def build_demo_engine(
    image_config: ModelEndpointConfig | None = None,
    bubble_render_config: BubbleRenderConfig | None = None,
) -> tuple[ConsistentT2IEngine, NovelToWebtoonPipeline, ModelEndpointConfig]:
    style = build_demo_style()
    characters = build_demo_characters()
    state = build_demo_state()
    runtime_image_config = image_config or build_demo_image_model_config()
    engine = ConsistentT2IEngine(
        continuity=state,
        planner=GenerationPlanner(
            project_id=state.project_id,
            style_bible=style,
            characters=characters,
        ),
        drift_analyzer=DriftAnalyzer(characters=characters),
    )
    pipeline = NovelToWebtoonPipeline(
        continuity=state,
        storyboard_planner=NovelStoryboardPlanner(characters=characters),
        sketch_planner=StickFigureStoryboardPlanner(),
        refinement_planner=VisualRefinementPlanner(
            generation_planner=engine.planner,
            image_model=runtime_image_config,
            bubble_render_config=bubble_render_config or BubbleRenderConfig(),
        ),
        bubble_planner=SpeechBubblePlanner(
            model_config=build_demo_bubble_model_config(),
            speaker_resolver=DialogueSpeakerResolver(characters=characters),
        ),
    )
    return engine, pipeline, runtime_image_config


def default_image_config_for_provider(provider_name: str) -> ModelEndpointConfig:
    if provider_name == "gemini":
        return build_demo_gemini_image_model_config()
    return build_demo_image_model_config()


def read_interactive_text(stream: object = sys.stdin) -> str:
    print("Paste your web novel text below.", file=sys.stderr)
    print("Finish with a single line containing only END", file=sys.stderr)
    lines: list[str] = []
    while True:
        line = stream.readline()
        if line == "":
            break
        if line.rstrip("\r\n") == "END":
            break
        lines.append(line)
    return "".join(lines).strip()


def resolve_source_text(args: argparse.Namespace, stdin_stream: object = sys.stdin) -> str:
    if args.text:
        return args.text.strip()
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    if args.stdin:
        return stdin_stream.read().strip()
    if args.interactive:
        return read_interactive_text(stdin_stream)
    return ""


def build_input_from_args(args: argparse.Namespace, stdin_stream: object = sys.stdin) -> WebNovelInput:
    source_text = resolve_source_text(args, stdin_stream=stdin_stream)
    if not source_text:
        return build_demo_novel()
    return WebNovelInput(
        episode_id=args.episode_id,
        title=args.title,
        source_text=source_text,
        genre=args.genre,
        tone=args.tone,
        default_location=args.default_location,
        target_panel_count=max(1, args.panel_count),
    )


def apply_image_overrides(args: argparse.Namespace, config: ModelEndpointConfig) -> ModelEndpointConfig:
    updated = config
    if args.image_model:
        updated = replace(updated, model=args.image_model)
        lowered_model = args.image_model.lower()
        if lowered_model.startswith("gemini-"):
            updated = replace(updated, provider="gemini", api_key_env="GEMINI_API_KEY", quality="standard")
        elif lowered_model.startswith("gpt-image") or lowered_model.startswith("chatgpt-image"):
            updated = replace(updated, provider="openai", api_key_env="OPENAI_API_KEY")
    if args.image_size:
        updated = replace(updated, size=args.image_size)
    if args.image_quality:
        updated = replace(updated, quality=args.image_quality)
    if args.image_background:
        updated = replace(updated, background=args.image_background)
    if args.image_format:
        updated = replace(updated, output_format=args.image_format)
    if args.image_output_dir:
        updated = replace(updated, output_dir=args.image_output_dir)
    return updated


def apply_cheap_image_profile(args: argparse.Namespace, config: ModelEndpointConfig) -> ModelEndpointConfig:
    if not args.cheap_image:
        return config

    updated = config
    provider = updated.provider
    model = updated.model.lower()

    if not args.image_size:
        updated = replace(updated, size="1024x1024")

    if provider == "openai" or model.startswith("gpt-image") or model.startswith("chatgpt-image"):
        if not args.image_quality:
            updated = replace(updated, quality="low")
    elif provider == "gemini" or model.startswith("gemini-"):
        if not args.image_quality:
            updated = replace(updated, quality="standard")

    return updated


def build_bubble_render_config(args: argparse.Namespace) -> BubbleRenderConfig:
    mode = "overlay"
    if not args.render_bubble_text:
        mode = "overlay"
    return BubbleRenderConfig(mode=mode)


def build_draft_mode_config(args: argparse.Namespace) -> DraftModeConfig:
    return DraftModeConfig(
        enabled=args.draft_mode,
        panels_per_image=min(2, max(1, args.draft_panels_per_image)),
    )


def has_explicit_source(args: argparse.Namespace) -> bool:
    return any((args.demo, args.text, args.text_file, args.stdin, args.interactive))


def resolve_control_mode(args: argparse.Namespace) -> str:
    try:
        return normalize_control_model(
            args.control_model,
            build_visual_bible=args.build_visual_bible,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def build_tiff_render_config(args: argparse.Namespace) -> TiffRenderConfig:
    return TiffRenderConfig(
        panels_per_page=max(1, args.tiff_panels_per_page),
        font_path=args.tiff_font_path or None,
        bundle_tiff=args.tiff_bundle,
    )


def build_image_provider(provider_name: str, config: ModelEndpointConfig):
    if provider_name == "openai":
        return OpenAIImageProvider(config)
    if provider_name == "gemini":
        return GeminiImageProvider(config)
    return MockImageProvider(config)


def build_demo_payload(
    engine: ConsistentT2IEngine,
    pipeline: NovelToWebtoonPipeline,
    image_provider,
) -> dict[str, object]:
    panel = build_demo_panel()
    request = engine.build_generation_request(panel)
    image = engine.generate_panel(panel, image_provider)
    pipeline_run = pipeline.run(build_demo_novel(), image_provider)
    observations = [
        ObservedTraits(
            panel_id=panel.panel_id,
            character_id="haeun",
            observed={
                "hair_color": "black",
                "hair_style": "long straight hair with blunt bangs",
                "eye_color": "dark brown",
                "signature_item": "silver hairpin shaped like a crescent moon",
            },
        )
    ]
    report = engine.evaluate_observations(panel.panel_id, observations)
    review = engine.review_and_approve_generation(panel, image, observations)
    return {
        "request": asdict(request),
        "generated_image": asdict(image),
        "drift_report": asdict(report),
        "cold_path_review": asdict(review),
        "pipeline_run": asdict(pipeline_run),
    }


def _default_batch_display_name(source: WebNovelInput, draft_mode: DraftModeConfig) -> str:
    prefix = "draft" if draft_mode.enabled else "panel"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_title = re.sub(r"[^a-zA-Z0-9가-힣_-]+", "-", source.title).strip("-") or source.episode_id
    return f"{prefix}-{safe_title}-{timestamp}"


def _serialize_request(request) -> dict[str, object]:
    return {
        "panel_id": request.panel_id,
        "system_prompt": request.system_prompt,
        "positive_prompt": request.positive_prompt,
        "negative_prompt": request.negative_prompt,
        "references": [asdict(reference) for reference in request.references],
        "seed": request.seed,
        "metadata": request.metadata,
    }


def _deserialize_request(payload: dict[str, object]):
    references = tuple(
        ReferenceImage(**reference)
        for reference in payload.get("references", [])
        if isinstance(reference, dict)
    )
    return GenerationRequest(
        panel_id=str(payload.get("panel_id", "")),
        system_prompt=str(payload.get("system_prompt", "")),
        positive_prompt=str(payload.get("positive_prompt", "")),
        negative_prompt=str(payload.get("negative_prompt", "")),
        references=references,
        seed=int(payload.get("seed", 0) or 0),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _serialize_sketches(sketches: SketchStoryboard) -> dict[str, object]:
    return {
        "episode_id": sketches.episode_id,
        "panels": [asdict(panel) for panel in sketches.panels],
        "drawing_rules": list(sketches.drawing_rules),
    }


def _deserialize_sketches(payload: dict[str, object]) -> SketchStoryboard:
    return SketchStoryboard(
        episode_id=str(payload.get("episode_id", "")),
        panels=tuple(
            SketchPanel(**panel)
            for panel in payload.get("panels", [])
            if isinstance(panel, dict)
        ),
        drawing_rules=tuple(payload.get("drawing_rules", []) or []),
    )


def _serialize_bubble_requests(bubble_requests: tuple[BubbleFillRequest, ...]) -> list[dict[str, object]]:
    return [asdict(request) for request in bubble_requests]


def _deserialize_bubble_requests(items: list[dict[str, object]]) -> tuple[BubbleFillRequest, ...]:
    restored = []
    for item in items:
        if not isinstance(item, dict):
            continue
        lines = tuple(
            BubbleLine(**line)
            for line in item.get("lines", [])
            if isinstance(line, dict)
        )
        restored.append(
            BubbleFillRequest(
                panel_id=str(item.get("panel_id", "")),
                system_prompt=str(item.get("system_prompt", "")),
                user_prompt=str(item.get("user_prompt", "")),
                lines=lines,
                model=str(item.get("model", "")),
                max_output_tokens=int(item.get("max_output_tokens", 0) or 0),
                metadata=dict(item.get("metadata", {}) or {}),
            )
        )
    return tuple(restored)


def _deserialize_spreads(items: list[dict[str, object]]) -> tuple[DraftSpread, ...]:
    restored = []
    for item in items:
        if not isinstance(item, dict):
            continue
        frames = tuple(
            DraftPanelFrame(**frame)
            for frame in item.get("frames", [])
            if isinstance(frame, dict)
        )
        panels = tuple(
            PanelSpec(**panel)
            for panel in item.get("panels", [])
            if isinstance(panel, dict)
        )
        restored.append(
            DraftSpread(
                spread_id=str(item.get("spread_id", "")),
                panel_ids=tuple(item.get("panel_ids", []) or []),
                panels=panels,
                frames=frames,
                layout=str(item.get("layout", "stacked-2up")),
                rationale=tuple(item.get("rationale", []) or []),
            )
        )
    return tuple(restored)


def _build_draft_manifest_payload(
    *,
    source: WebNovelInput,
    spreads: tuple[DraftSpread, ...],
    requests,
    sketches: SketchStoryboard,
    bubble_requests: tuple[BubbleFillRequest, ...],
    image_config: ModelEndpointConfig,
    display_name: str,
    output_root: Path,
) -> dict[str, object]:
    return {
        "display_name": display_name,
        "provider": "gemini-batch",
        "model": image_config.model,
        "source": asdict(source),
        "output_dir": str(output_root.resolve()),
        "image_config": asdict(image_config),
        "draft_spreads": [asdict(spread) for spread in spreads],
        "requests_serialized": [_serialize_request(request) for request in requests],
        "sketches": _serialize_sketches(sketches),
        "bubble_requests": _serialize_bubble_requests(bubble_requests),
    }


def _find_manifest_for_batch_name(output_root: Path, batch_name: str) -> Path | None:
    manifests = sorted(output_root.glob("*.manifest.json"))
    for manifest in manifests:
        try:
            payload = read_manifest(manifest)
        except Exception:
            continue
        batch_job = payload.get("batch_job", {}) if isinstance(payload, dict) else {}
        if (not batch_job) and isinstance(payload.get("extra"), dict):
            batch_job = payload["extra"].get("batch_job", {})
        if isinstance(batch_job, dict) and batch_job.get("name") == batch_name:
            return manifest
    if len(manifests) == 1:
        return manifests[0]
    return None


def run_draft_mode(
    *,
    source: WebNovelInput,
    draft_planner: DraftSpreadPlanner,
    pipeline: NovelToWebtoonPipeline,
    provider_name: str,
    image_provider,
    image_config: ModelEndpointConfig,
    bubble_render_config: BubbleRenderConfig,
    batch_mode: bool,
    batch_wait: bool,
    batch_poll_seconds: int,
) -> DraftRunResult:
    storyboard = pipeline.stage1_storyboard(source)
    sketches = pipeline.stage2_sketch_storyboard(storyboard)
    bubble_requests = pipeline.stage4_bubble_requests(storyboard, sketches)
    spreads = draft_planner.build_spreads(storyboard)
    requests = draft_planner.build_requests(spreads, pipeline.continuity)
    output_root = Path(image_config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if batch_mode:
        if provider_name != "gemini":
            raise ValueError("Batch mode is currently implemented for the Gemini provider only.")
        batch_client = GeminiBatchClient(image_config)
        display_name = _default_batch_display_name(source, draft_planner.draft_mode_config)
        manifest_path = output_root / f"{display_name}.manifest.json"
        batch_client.write_inline_manifest(
            requests,
            display_name=display_name,
            destination=manifest_path,
            extra=_build_draft_manifest_payload(
                source=source,
                spreads=spreads,
                requests=requests,
                sketches=sketches,
                bubble_requests=bubble_requests,
                image_config=image_config,
                display_name=display_name,
                output_root=output_root,
            ),
        )
        batch_job = batch_client.submit_inline_batch(
            requests,
            display_name=display_name,
            manifest_path=str(manifest_path.resolve()),
        )
        submit_response_path = write_json_artifact(
            output_root / f"{display_name}.submit-response.json",
            batch_job.raw_response,
        )
        update_manifest(
            manifest_path,
            {
                "batch_job": {
                    "name": batch_job.name,
                    "display_name": batch_job.display_name,
                    "provider": batch_job.provider,
                    "model": batch_job.model,
                    "state": batch_job.state,
                    "request_count": batch_job.request_count,
                    "manifest_path": batch_job.manifest_path,
                },
                "submit_response_path": str(submit_response_path.resolve()),
            },
        )
        images: tuple = ()
        bubble_plans: tuple = ()
        bubble_artifacts: tuple = ()
        if batch_wait:
            batch_job = batch_client.wait_for_batch(
                batch_job.name,
                poll_seconds=batch_poll_seconds,
            )
            if batch_job.state in {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}:
                generated_images = batch_client.materialize_inline_batch_results(
                    batch_job,
                    requests,
                    output_dir=output_root,
                )
                latest_response_path = write_json_artifact(
                    output_root / f"{display_name}.latest-response.json",
                    batch_job.raw_response,
                )
                spread_map = {spread.spread_id: spread for spread in spreads}
                reconstructed: list[DraftSpreadImage] = []
                for image in generated_images:
                    spread = spread_map.get(image.panel_id)
                    if spread is None:
                        continue
                    prepared_image = draft_planner.prepare_spread_image(spread, image)
                    reconstructed.append(
                        DraftSpreadImage(
                            spread_id=spread.spread_id,
                            panel_ids=spread.panel_ids,
                            image=prepared_image,
                        )
                    )
                images = tuple(reconstructed)
                bubble_plans = draft_planner.build_bubble_plans(
                    spreads,
                    sketches,
                    bubble_requests,
                    image_config,
                    bubble_render_config,
                )
                bubble_artifacts = draft_planner.render_overlays(
                    images,
                    bubble_plans,
                    image_config,
                    bubble_render_config,
                )
                update_manifest(
                    manifest_path,
                    {
                        "batch_job": {
                            "name": batch_job.name,
                            "display_name": batch_job.display_name,
                            "provider": batch_job.provider,
                            "model": batch_job.model,
                            "state": batch_job.state,
                            "request_count": batch_job.request_count,
                            "manifest_path": batch_job.manifest_path,
                        },
                        "latest_response_path": str(latest_response_path.resolve()),
                        "materialized_image_count": len(images),
                        "materialized_images": [asdict(item) for item in images],
                        "bubble_artifacts": [asdict(artifact) for artifact in bubble_artifacts],
                    },
                )
        return DraftRunResult(
            source=source,
            spreads=spreads,
            requests=requests,
            images=images,
            bubble_plans=bubble_plans,
            bubble_artifacts=bubble_artifacts,
            batch_job=batch_job,
        )

    spread_images = draft_planner.generate(spreads, pipeline.continuity, image_provider)
    bubble_plans = draft_planner.build_bubble_plans(
        spreads,
        sketches,
        bubble_requests,
        image_config,
        bubble_render_config,
    )
    bubble_artifacts = draft_planner.render_overlays(
        spread_images,
        bubble_plans,
        image_config,
        bubble_render_config,
    )
    return DraftRunResult(
        source=source,
        spreads=spreads,
        requests=requests,
        images=spread_images,
        bubble_plans=bubble_plans,
        bubble_artifacts=bubble_artifacts,
    )


def resume_draft_batch(
    *,
    batch_name: str,
    manifest_path: str,
    image_config: ModelEndpointConfig,
    bubble_render_config: BubbleRenderConfig,
    batch_wait: bool,
    batch_poll_seconds: int,
) -> DraftRunResult:
    output_root = Path(image_config.output_dir)
    resolved_manifest: Path | None = Path(manifest_path) if manifest_path else None
    if resolved_manifest is None and batch_name:
        resolved_manifest = _find_manifest_for_batch_name(output_root, batch_name)

    manifest_payload: dict[str, object] = {}
    if resolved_manifest is not None and resolved_manifest.exists():
        manifest_payload = read_manifest(resolved_manifest)
    elif manifest_path:
        raise ValueError(f"Batch manifest not found: {manifest_path}")

    extra_payload = manifest_payload.get("extra", {}) if isinstance(manifest_payload.get("extra"), dict) else {}

    stored_job = manifest_payload.get("batch_job", {}) if isinstance(manifest_payload, dict) else {}
    if not stored_job and extra_payload:
        stored_job = extra_payload.get("batch_job", {})
    effective_batch_name = batch_name or (stored_job.get("name") if isinstance(stored_job, dict) else "")
    if not effective_batch_name:
        raise ValueError("Could not determine batch job name. Provide --batch-resume-name or a manifest with batch_job.name.")

    batch_client = GeminiBatchClient(image_config)
    batch_job = batch_client.get_batch(str(effective_batch_name))
    if batch_wait and batch_job.state not in {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}:
        batch_job = batch_client.wait_for_batch(
            batch_job.name,
            poll_seconds=batch_poll_seconds,
        )

    latest_response_path = write_json_artifact(
        output_root / f"{re.sub(r'[^a-zA-Z0-9_-]+', '-', batch_job.display_name)}.latest-response.json",
        batch_job.raw_response,
    )

    requests = tuple(
        _deserialize_request(item)
        for item in (
            manifest_payload.get("requests_serialized", [])
            if isinstance(manifest_payload.get("requests_serialized"), list)
            else extra_payload.get("requests_serialized", [])
            if isinstance(extra_payload.get("requests_serialized"), list)
            else []
        )
        if isinstance(item, dict)
    )
    if not requests:
        requests = tuple(
            GenerationRequest(
                panel_id=str(item.get("panel_id", "")),
                system_prompt="",
                positive_prompt="",
                negative_prompt="",
                references=(),
                seed=int(item.get("seed", 0) or 0),
                metadata=dict(item.get("metadata", {}) or {}),
            )
            for item in (
                manifest_payload.get("requests", [])
                if isinstance(manifest_payload.get("requests"), list)
                else extra_payload.get("requests", [])
                if isinstance(extra_payload.get("requests"), list)
                else []
            )
            if isinstance(item, dict)
        )

    spreads = _deserialize_spreads(
        manifest_payload.get("draft_spreads", [])
        if isinstance(manifest_payload.get("draft_spreads"), list)
        else extra_payload.get("draft_spreads", [])
        if extra_payload
        else []
    )
    sketches_payload = (
        manifest_payload.get("sketches", {})
        if isinstance(manifest_payload.get("sketches"), dict)
        else extra_payload.get("sketches", {})
        if isinstance(extra_payload.get("sketches"), dict)
        else {}
    )
    sketches = _deserialize_sketches(sketches_payload) if isinstance(sketches_payload, dict) and sketches_payload else SketchStoryboard(episode_id="", panels=())
    bubble_requests = _deserialize_bubble_requests(
        manifest_payload.get("bubble_requests", [])
        if isinstance(manifest_payload.get("bubble_requests"), list)
        else extra_payload.get("bubble_requests", [])
        if isinstance(extra_payload.get("bubble_requests"), list)
        else []
    )
    source_payload = (
        manifest_payload.get("source", {})
        if isinstance(manifest_payload.get("source"), dict)
        else extra_payload.get("source", {})
        if isinstance(extra_payload.get("source"), dict)
        else {}
    )
    source = WebNovelInput(**source_payload) if isinstance(source_payload, dict) and source_payload else WebNovelInput(
        episode_id="batch-resume",
        title=str(manifest_payload.get("display_name", "Resumed Batch")),
        source_text="",
    )

    images: tuple[DraftSpreadImage, ...] = ()
    bubble_plans: tuple = ()
    bubble_artifacts: tuple = ()
    if batch_job.state in {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"} and requests:
        draft_planner: DraftSpreadPlanner | None = None
        if spreads:
            draft_planner = DraftSpreadPlanner(
                generation_planner=GenerationPlanner(
                    project_id="resume",
                    style_bible=build_demo_style(),
                    characters=build_demo_characters(),
                ),
                draft_mode_config=DraftModeConfig(enabled=True),
            )
        generated_images = batch_client.materialize_inline_batch_results(
            batch_job,
            requests,
            output_dir=output_root,
        )
        spread_map = {spread.spread_id: spread for spread in spreads}
        images = tuple(
            DraftSpreadImage(
                spread_id=spread_map[image.panel_id].spread_id,
                panel_ids=spread_map[image.panel_id].panel_ids,
                image=(
                    draft_planner.prepare_spread_image(spread_map[image.panel_id], image)
                    if draft_planner is not None
                    else image
                ),
            )
            for image in generated_images
            if image.panel_id in spread_map
        )
        if draft_planner is not None and spreads and sketches.panels and bubble_requests:
            bubble_plans = draft_planner.build_bubble_plans(
                spreads,
                sketches,
                bubble_requests,
                image_config,
                bubble_render_config,
            )
            bubble_artifacts = draft_planner.render_overlays(
                images,
                bubble_plans,
                image_config,
                bubble_render_config,
            )

    if resolved_manifest is not None:
        update_manifest(
            resolved_manifest,
            {
                "batch_job": {
                    "name": batch_job.name,
                    "display_name": batch_job.display_name,
                    "provider": batch_job.provider,
                    "model": batch_job.model,
                    "state": batch_job.state,
                    "request_count": batch_job.request_count,
                    "manifest_path": str(resolved_manifest.resolve()),
                },
                "latest_response_path": str(latest_response_path.resolve()),
                "materialized_image_count": len(images),
                "materialized_images": [asdict(item) for item in images],
                "bubble_artifacts": [asdict(artifact) for artifact in bubble_artifacts],
            },
        )
        batch_job = BatchJobInfo(
            name=batch_job.name,
            display_name=batch_job.display_name,
            provider=batch_job.provider,
            model=batch_job.model,
            state=batch_job.state,
            request_count=batch_job.request_count,
            manifest_path=str(resolved_manifest.resolve()),
            raw_response=batch_job.raw_response,
        )

    return DraftRunResult(
        source=source,
        spreads=spreads,
        requests=requests,
        images=images,
        bubble_plans=bubble_plans,
        bubble_artifacts=bubble_artifacts,
        batch_job=batch_job,
    )


def _sum_known_costs(result: PipelineRunResult) -> float:
    return round(
        sum(
            image.cost.total_cost_usd
            for image in result.generated_images
            if image.cost is not None
        ),
        6,
    )


def _sum_known_draft_costs(result: DraftRunResult) -> float:
    return round(
        sum(
            spread_image.image.cost.total_cost_usd
            for spread_image in result.images
            if spread_image.image.cost is not None
        ),
        6,
    )


def format_draft_summary(result: DraftRunResult) -> str:
    total_cost = _sum_known_draft_costs(result)
    artifact_map = {artifact.panel_id: artifact for artifact in result.bubble_artifacts}
    lines = [
        f"Episode: {result.source.title} ({result.source.episode_id})",
        f"Draft spreads: {len(result.spreads)}",
    ]
    if total_cost:
        lines.append(f"Draft total image cost: ${total_cost:.6f}")
    if result.batch_job is not None:
        lines.append(
            f"Batch job: {result.batch_job.name} [{result.batch_job.state}] manifest={result.batch_job.manifest_path or '(none)'}"
        )
        lines.append(f"Batch materialized images: {len(result.images)}/{len(result.requests)}")
    lines.extend(["", "[Draft Spreads]"])
    image_map = {spread_image.spread_id: spread_image.image for spread_image in result.images}
    for spread in result.spreads:
        image = image_map.get(spread.spread_id)
        cost_label = "(not generated yet)"
        saved_label = "(not generated yet)"
        if image is not None:
            if image.cost is not None:
                qualifier = "estimated" if image.cost.estimated else "actual"
                cost_label = f"${image.cost.total_cost_usd:.6f} {qualifier}"
            saved_label = image.local_path or "(provider did not save a file)"
        artifact = artifact_map.get(spread.spread_id)
        lines.extend(
            [
                f"- {spread.spread_id}",
                f"  panels: {', '.join(spread.panel_ids)}",
                f"  layout: {spread.layout}",
                f"  rationale: {' | '.join(spread.rationale)}",
                f"  cost: {cost_label}",
                f"  saved: {saved_label}",
                f"  stage5: {artifact.composite_svg_path if artifact else '(no overlay artifact)'}",
            ]
        )
    return "\n".join(lines)


def format_pipeline_summary(result: PipelineRunResult) -> str:
    total_cost = _sum_known_costs(result)
    artifact_map = {artifact.panel_id: artifact for artifact in result.bubble_artifacts}
    lines = [
        f"Episode: {result.draft.source.title} ({result.draft.source.episode_id})",
        f"Panels: {len(result.draft.storyboard.panels)}",
    ]
    if total_cost:
        lines.append(f"Stage 3 total image cost: ${total_cost:.6f}")
    lines.extend(
        [
            "",
            "[Stage 1] Text Storyboard",
        ]
    )
    for panel in result.draft.storyboard.panels:
        lines.extend(
            [
                f"- {panel.panel_id}",
                f"  narrative: {panel.narrative}",
                f"  shot/emotion: {panel.shot_type} / {panel.emotion}",
                f"  dialogue: {' | '.join(panel.dialogue) if panel.dialogue else '(none)'}",
            ]
        )
    lines.append("")
    lines.append("[Stage 2] Stick Figure Storyboard")
    for sketch in result.draft.sketches.panels:
        lines.extend(
            [
                f"- {sketch.panel_id}",
                f"  camera: {sketch.camera_guide}",
                f"  figures: {' | '.join(sketch.figure_guides)}",
                f"  bubbles: {' | '.join(sketch.bubble_slots)}",
            ]
        )
    lines.append("")
    lines.append("[Stage 3] Generated Images")
    for request, image in zip(result.draft.refined_requests, result.generated_images, strict=True):
        cost_label = "(unknown)"
        if image.cost is not None:
            qualifier = "estimated" if image.cost.estimated else "actual"
            cost_label = f"${image.cost.total_cost_usd:.6f} {qualifier}"
        bubble_render = request.metadata.get("bubble_render", {})
        bubble_lines = bubble_render.get("visible_lines", []) if isinstance(bubble_render, dict) else []
        bubble_render_mode = (
            str(bubble_render.get("mode"))
            if isinstance(bubble_render, dict)
            else "none"
        )
        artifact = artifact_map.get(request.panel_id)
        if bubble_render_mode == "overlay" and bubble_lines:
            bubble_text_label = "overlay - " + " | ".join(
                (
                    f"{line['speaker_character_id']}/{line['speaker']}({line.get('speaker_confidence', 0):.2f}): {line['text']}"
                    if line.get("speaker_character_id")
                    else f"{line['speaker']}: {line['text']}"
                )
                for line in bubble_lines
            )
        elif bubble_render_mode == "overlay":
            bubble_text_label = "overlay - no dialogue lines"
        elif bubble_render_mode == "in_image" and bubble_lines:
            bubble_text_label = "in-image - " + " | ".join(
                (
                    f"{line['speaker_character_id']}/{line['speaker']}({line.get('speaker_confidence', 0):.2f}): {line['text']}"
                    if line.get("speaker_character_id")
                    else f"{line['speaker']}: {line['text']}"
                )
                for line in bubble_lines
            )
        else:
            bubble_text_label = "disabled"
        lines.extend(
            [
                f"- {request.panel_id}",
                f"  model: {image.model or request.metadata['image_model']['model']}",
                f"  seed: {request.seed}",
                f"  cost: {cost_label}",
                f"  saved: {image.local_path or '(mock provider: no file saved)'}",
                f"  bubble text: {bubble_text_label}",
                f"  stage5: {artifact.composite_svg_path if artifact else '(no overlay artifact)'}",
                f"  prompt excerpt: {request.positive_prompt[:160].replace(chr(10), ' ')}...",
            ]
        )
    lines.append("")
    lines.append("[Stage 4] Bubble Requests")
    for bubble_request in result.draft.bubble_requests:
        bubble_preview = " | ".join(line.text for line in bubble_request.lines) or "(none)"
        lines.extend(
            [
                f"- {bubble_request.panel_id}",
                f"  model/tokens: {bubble_request.model} / {bubble_request.max_output_tokens}",
                f"  lines: {bubble_preview}",
            ]
        )
    lines.append("")
    lines.append("[Stage 5] Bubble Overlays")
    if result.bubble_artifacts:
        for artifact in result.bubble_artifacts:
            lines.extend(
                [
                    f"- {artifact.panel_id}",
                    f"  overlay: {artifact.overlay_svg_path}",
                    f"  final: {artifact.composite_svg_path}",
                    f"  final png: {artifact.final_png_path}",
                ]
            )
    else:
        lines.append("- (no overlay artifacts generated)")
    return "\n".join(lines)


def main(argv: list[str] | None = None, stdin_stream: object = sys.stdin) -> None:
    load_local_env()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    control_mode = resolve_control_mode(args)

    if control_mode == CONTROL_MODEL_CONSISTENCY_KEEPER:
        if not has_explicit_source(args):
            raise SystemExit(
                "Visual bible mode requires source text via --text, --text-file, --stdin, --interactive, or --demo."
            )
        source = build_input_from_args(args, stdin_stream=stdin_stream)
        document, visual_bible_result = build_visual_bible_workspace(
            source,
            workspace_dir=args.visual_bible_dir,
            max_characters=max(1, args.visual_bible_max_characters),
            evidence_limit=max(1, args.visual_bible_evidence_limit),
        )
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "document": asdict(document),
                        "build": asdict(visual_bible_result),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return
        print(format_visual_bible_summary(visual_bible_result))
        return

    if control_mode == CONTROL_MODEL_TIFF_STORYBOARD:
        if not has_explicit_source(args):
            raise SystemExit(
                "novel_conti_tiff mode requires source text via --text, --text-file, --stdin, --interactive, or --demo."
            )
        source = build_input_from_args(args, stdin_stream=stdin_stream)
        result = render_novel_to_tiff(
            source_text=source.source_text,
            title=source.title,
            episode_id=source.episode_id,
            target_panel_count=source.target_panel_count,
            output_dir=args.tiff_output_dir,
            render_config=build_tiff_render_config(args),
        )
        if args.format == "json":
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
            return
        print(format_tiff_storyboard_summary(result))
        return

    image_config = apply_image_overrides(args, default_image_config_for_provider(args.provider))
    image_config = apply_cheap_image_profile(args, image_config)
    bubble_render_config = build_bubble_render_config(args)
    draft_mode_config = build_draft_mode_config(args)
    engine, pipeline, runtime_image_config = build_demo_engine(
        image_config=image_config,
        bubble_render_config=bubble_render_config,
    )
    image_provider = build_image_provider(args.provider, runtime_image_config)
    draft_planner = DraftSpreadPlanner(
        generation_planner=engine.planner,
        draft_mode_config=draft_mode_config,
    )

    try:
        if args.demo and not any((args.text, args.text_file, args.stdin, args.interactive)):
            demo_result = pipeline.run(build_demo_novel(), image_provider)
            if args.format == "json":
                if args.provider == "mock":
                    payload = build_demo_payload(engine, pipeline, image_provider)
                    print(json.dumps(payload, indent=2, ensure_ascii=False))
                    return
                print(json.dumps({"pipeline_run": asdict(demo_result)}, indent=2, ensure_ascii=False))
                return
            print(format_pipeline_summary(demo_result))
            return

        if args.batch_resume_name or args.batch_resume_manifest:
            if args.provider != "gemini":
                raise ValueError("Batch resume is currently implemented for the Gemini provider only.")
            resumed = resume_draft_batch(
                batch_name=args.batch_resume_name,
                manifest_path=args.batch_resume_manifest,
                image_config=runtime_image_config,
                bubble_render_config=bubble_render_config,
                batch_wait=args.batch_wait,
                batch_poll_seconds=args.batch_poll_seconds,
            )
            if args.format == "json":
                print(json.dumps(asdict(resumed), indent=2, ensure_ascii=False))
                return
            print(format_draft_summary(resumed))
            return

        source = build_input_from_args(args, stdin_stream=stdin_stream)
        if args.batch_mode and not args.draft_mode:
            raise ValueError("Batch mode is currently implemented together with --draft-mode.")
        if draft_mode_config.enabled:
            draft_result = run_draft_mode(
                source=source,
                draft_planner=draft_planner,
                pipeline=pipeline,
                provider_name=args.provider,
                image_provider=image_provider,
                image_config=runtime_image_config,
                bubble_render_config=bubble_render_config,
                batch_mode=args.batch_mode,
                batch_wait=args.batch_wait,
                batch_poll_seconds=args.batch_poll_seconds,
            )
            if args.format == "json":
                print(json.dumps(asdict(draft_result), indent=2, ensure_ascii=False))
                return
            print(format_draft_summary(draft_result))
            return
        pipeline_result = pipeline.run(source, image_provider)
        if args.format == "json":
            print(json.dumps(asdict(pipeline_result), indent=2, ensure_ascii=False))
            return
        print(format_pipeline_summary(pipeline_result))
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
