from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from PIL import Image, ImageDraw

from consistent_t2i import __main__ as cli
from consistent_t2i.batch_api import GeminiBatchClient
from consistent_t2i.demo_data import (
    build_demo_bubble_model_config,
    build_demo_characters,
    build_demo_gemini_image_model_config,
    build_demo_image_model_config,
    build_demo_novel,
    build_demo_state,
    build_demo_style,
)
from consistent_t2i.draft_mode import DraftSpreadPlanner
from consistent_t2i.models import BubbleRenderConfig, DraftModeConfig, GeneratedImage
from consistent_t2i.pipeline import (
    DialogueSpeakerResolver,
    NovelStoryboardPlanner,
    NovelToWebtoonPipeline,
    SpeechBubblePlanner,
    StickFigureStoryboardPlanner,
    VisualRefinementPlanner,
)
from consistent_t2i.planner import GenerationPlanner
from consistent_t2i.providers import (
    MockImageProvider,
    build_bubble_fill_payload,
    build_gemini_image_request,
    build_image_model_payload,
    build_openai_image_request,
    resolve_api_key,
)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.characters = build_demo_characters()
        self.state = build_demo_state()
        self.style = build_demo_style()
        self.novel = build_demo_novel()
        self.generation_planner = GenerationPlanner(
            project_id=self.state.project_id,
            style_bible=self.style,
            characters=self.characters,
        )
        self.image_model = build_demo_image_model_config()
        self.bubble_model = build_demo_bubble_model_config()
        self.speaker_resolver = DialogueSpeakerResolver(characters=self.characters)
        self.pipeline = NovelToWebtoonPipeline(
            continuity=self.state,
            storyboard_planner=NovelStoryboardPlanner(characters=self.characters),
            sketch_planner=StickFigureStoryboardPlanner(),
            refinement_planner=VisualRefinementPlanner(
                generation_planner=self.generation_planner,
                image_model=self.image_model,
            ),
            bubble_planner=SpeechBubblePlanner(
                model_config=self.bubble_model,
                speaker_resolver=self.speaker_resolver,
            ),
        )

    def test_stage1_builds_panelized_storyboard_from_novel(self) -> None:
        storyboard = self.pipeline.stage1_storyboard(self.novel)
        self.assertEqual(storyboard.episode_id, self.novel.episode_id)
        self.assertGreaterEqual(len(storyboard.panels), 2)
        self.assertLessEqual(len(storyboard.panels), self.novel.target_panel_count)
        self.assertIn("Stage 1 converts prose", storyboard.adaptation_notes[0])
        self.assertIn("haeun", storyboard.panels[0].characters)

    def test_stage2_converts_storyboard_into_stick_figure_guides(self) -> None:
        storyboard = self.pipeline.stage1_storyboard(self.novel)
        sketches = self.pipeline.stage2_sketch_storyboard(storyboard)
        self.assertEqual(len(sketches.panels), len(storyboard.panels))
        self.assertIn("stick figure", sketches.drawing_rules[0].lower())
        self.assertTrue(sketches.panels[0].figure_guides)
        self.assertTrue(sketches.panels[0].bubble_slots)

    def test_stage3_refine_request_carries_sketch_and_model_metadata(self) -> None:
        storyboard = self.pipeline.stage1_storyboard(self.novel)
        sketches = self.pipeline.stage2_sketch_storyboard(storyboard)
        bubble_requests = self.pipeline.stage4_bubble_requests(storyboard, sketches)
        request = self.pipeline.stage3_refine_requests(
            storyboard,
            sketches,
            bubble_requests=bubble_requests,
        )[0]
        payload = build_image_model_payload(request, self.image_model)
        openai_request = build_openai_image_request(request, self.image_model)
        self.assertIn("SKETCH STORYBOARD LOCK", request.positive_prompt)
        self.assertIn("POST-PRODUCTION LETTERING MODE", request.positive_prompt)
        self.assertNotIn("IN-IMAGE BUBBLE LETTERING", request.positive_prompt)
        self.assertEqual(request.metadata["pipeline_stage"], "stage3_refine")
        self.assertEqual(payload["model"], "gpt-image-1.5")
        self.assertEqual(openai_request["model"], "gpt-image-1.5")
        self.assertEqual(openai_request["size"], "1024x1024")
        self.assertEqual(openai_request["quality"], "low")
        self.assertFalse(payload["api_key_configured"])
        self.assertEqual(request.metadata["bubble_render"]["mode"], "overlay")
        self.assertTrue(request.metadata["bubble_render"]["render_text_overlay"])

    def test_stage4_builds_low_token_bubble_fill_requests(self) -> None:
        storyboard = self.pipeline.stage1_storyboard(self.novel)
        sketches = self.pipeline.stage2_sketch_storyboard(storyboard)
        request = self.pipeline.stage4_bubble_requests(storyboard, sketches)[0]
        payload = build_bubble_fill_payload(request, self.bubble_model)
        self.assertEqual(request.model, "gemma4")
        self.assertEqual(request.max_output_tokens, 120)
        self.assertEqual(request.metadata["pipeline_stage"], "stage4_bubble")
        self.assertEqual(payload["max_output_tokens"], 120)
        self.assertTrue(payload["lines"])

    def test_gemini_refine_payload_can_be_built_without_network_call(self) -> None:
        gemini_model = build_demo_gemini_image_model_config()
        gemini_pipeline = NovelToWebtoonPipeline(
            continuity=self.state,
            storyboard_planner=NovelStoryboardPlanner(characters=self.characters),
            sketch_planner=StickFigureStoryboardPlanner(),
            refinement_planner=VisualRefinementPlanner(
                generation_planner=self.generation_planner,
                image_model=gemini_model,
            ),
            bubble_planner=SpeechBubblePlanner(
                model_config=self.bubble_model,
                speaker_resolver=self.speaker_resolver,
            ),
        )
        storyboard = gemini_pipeline.stage1_storyboard(self.novel)
        sketches = gemini_pipeline.stage2_sketch_storyboard(storyboard)
        request = gemini_pipeline.stage3_refine_requests(storyboard, sketches)[0]
        payload = build_image_model_payload(request, gemini_model)
        gemini_request = build_gemini_image_request(request, gemini_model)
        self.assertEqual(payload["model"], "gemini-2.5-flash-image")
        self.assertIn("gemini_generate_content_request", payload)
        self.assertEqual(
            gemini_request["generationConfig"]["imageConfig"]["aspectRatio"],
            "1:1",
        )

    def test_gemini_api_key_can_be_loaded_from_dotenv(self) -> None:
        config = build_demo_gemini_image_model_config()
        self.assertTrue(resolve_api_key(config))

    def test_stage5_bubble_plan_uses_dialogue_lines_for_overlay(self) -> None:
        render_pipeline = NovelToWebtoonPipeline(
            continuity=self.state,
            storyboard_planner=NovelStoryboardPlanner(characters=self.characters),
            sketch_planner=StickFigureStoryboardPlanner(),
            refinement_planner=VisualRefinementPlanner(
                generation_planner=self.generation_planner,
                image_model=self.image_model,
                bubble_render_config=BubbleRenderConfig(mode="overlay"),
            ),
            bubble_planner=SpeechBubblePlanner(
                model_config=self.bubble_model,
                speaker_resolver=self.speaker_resolver,
            ),
        )
        storyboard = render_pipeline.stage1_storyboard(self.novel)
        sketches = render_pipeline.stage2_sketch_storyboard(storyboard)
        bubble_requests = render_pipeline.stage4_bubble_requests(storyboard, sketches)
        request = render_pipeline.stage3_refine_requests(storyboard, sketches, bubble_requests=bubble_requests)[1]
        bubble_plans = render_pipeline.stage5_plan_bubbles(sketches, bubble_requests)
        self.assertIn("POST-PRODUCTION LETTERING PLAN", request.positive_prompt)
        self.assertIn("These lines will be inserted later by a deterministic renderer", request.positive_prompt)
        self.assertEqual(request.metadata["bubble_render"]["mode"], "overlay")
        self.assertEqual(len(bubble_plans), 1)
        self.assertEqual(bubble_plans[0].panel_id, "ep01-p02")
        self.assertEqual(bubble_plans[0].placements[0].text, "You dropped this.")
        self.assertEqual(bubble_plans[0].placements[0].speaker_character_id, "dohyun")

    def test_stage5_render_overlays_writes_svg_artifacts_for_real_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "panel.png"
            Image.new("RGBA", (1024, 1024), (255, 255, 255, 255)).save(image_path)

            render_pipeline = NovelToWebtoonPipeline(
                continuity=self.state,
                storyboard_planner=NovelStoryboardPlanner(characters=self.characters),
                sketch_planner=StickFigureStoryboardPlanner(),
                refinement_planner=VisualRefinementPlanner(
                    generation_planner=self.generation_planner,
                    image_model=self.image_model,
                    bubble_render_config=BubbleRenderConfig(mode="overlay"),
                ),
                bubble_planner=SpeechBubblePlanner(
                    model_config=self.bubble_model,
                    speaker_resolver=self.speaker_resolver,
                ),
            )
            storyboard = render_pipeline.stage1_storyboard(self.novel)
            sketches = render_pipeline.stage2_sketch_storyboard(storyboard)
            bubble_requests = render_pipeline.stage4_bubble_requests(storyboard, sketches)
            bubble_plans = render_pipeline.stage5_plan_bubbles(sketches, bubble_requests)
            image = MockImageProvider(self.image_model).generate(
                render_pipeline.stage3_refine_requests(
                    storyboard,
                    sketches,
                    bubble_requests=bubble_requests,
                )[1]
            )
            image = image.__class__(
                **{**image.__dict__, "panel_id": "ep01-p02", "local_path": str(image_path.resolve())}
            )
            artifacts = render_pipeline.stage5_render_overlays((image,), bubble_plans)
            self.assertEqual(len(artifacts), 1)
            self.assertTrue(Path(artifacts[0].overlay_svg_path).exists())
            self.assertTrue(Path(artifacts[0].composite_svg_path).exists())
            self.assertTrue(Path(artifacts[0].final_png_path).exists())
            overlay_text = Path(artifacts[0].overlay_svg_path).read_text(encoding="utf-8")
            self.assertIn("feDropShadow", overlay_text)
            self.assertIn("<path d=\"M", overlay_text)

    def test_run_generates_images_without_mutating_continuity(self) -> None:
        result = self.pipeline.run(self.novel, MockImageProvider(self.image_model))
        self.assertEqual(len(result.generated_images), len(result.draft.storyboard.panels))
        self.assertEqual(self.state.generated_assets, {})
        self.assertEqual(self.state.approved_panels, {})
        self.assertIsNotNone(result.generated_images[0].cost)

    def test_korean_speaker_label_maps_to_correct_character_id(self) -> None:
        source = self.novel.__class__(
            episode_id="ep-kr",
            title="Korean Speaker Mapping",
            source_text="비 오는 복도에서 두 사람이 마주친다. 도현: 이거 떨어뜨렸어.",
            primary_characters=("haeun", "dohyun"),
            target_panel_count=2,
        )
        storyboard = self.pipeline.stage1_storyboard(source)
        sketches = self.pipeline.stage2_sketch_storyboard(storyboard)
        bubble_requests = self.pipeline.stage4_bubble_requests(storyboard, sketches)
        dialogue_line = bubble_requests[1].lines[0]
        self.assertEqual(dialogue_line.speaker, "도현")
        self.assertEqual(dialogue_line.speaker_character_id, "dohyun")

    def test_unlabeled_second_turn_is_inferred_from_conversation_flow(self) -> None:
        from consistent_t2i.models import PanelSpec, SketchPanel, SketchStoryboard, StoryboardPlan

        panel = PanelSpec(
            panel_id="turn-test-p01",
            narrative="Haeun and Dohyun face each other in the hallway.",
            characters=("haeun", "dohyun"),
            location="hallway",
            shot_type="medium shot",
            emotion="tense",
            action="Dohyun waits for an answer while Haeun looks up at him.",
            dialogue=("도현: 괜찮아?", "응, 나 괜찮아."),
        )
        sketch = SketchPanel(
            panel_id="turn-test-p01",
            summary=panel.narrative,
            camera_guide="medium shot",
            figure_guides=(
                "haeun: stick figure at left third, eyeline matched to tense",
                "dohyun: stick figure at right third, eyeline matched to tense",
            ),
            bubble_slots=(
                "bubble-1: near 도현, away from focal face",
                "bubble-2: near haeun, away from focal face",
            ),
        )
        storyboard = StoryboardPlan(
            episode_id="turn-test",
            title="Turn Test",
            premise=panel.narrative,
            panels=(panel,),
        )
        sketches = SketchStoryboard(
            episode_id="turn-test",
            panels=(sketch,),
        )
        bubble_requests = self.pipeline.stage4_bubble_requests(storyboard, sketches)
        self.assertEqual(bubble_requests[0].lines[0].speaker_character_id, "dohyun")
        self.assertEqual(bubble_requests[0].lines[1].speaker_character_id, "haeun")

    def test_draft_mode_groups_adjacent_compatible_panels(self) -> None:
        draft_planner = DraftSpreadPlanner(
            generation_planner=self.generation_planner,
            draft_mode_config=DraftModeConfig(enabled=True, panels_per_image=2),
        )
        storyboard = self.pipeline.stage1_storyboard(self.novel)
        spreads = draft_planner.build_spreads(storyboard)
        self.assertEqual(len(spreads), 2)
        self.assertEqual(spreads[0].panel_ids, ("ep01-p01", "ep01-p02"))
        self.assertEqual(spreads[1].panel_ids, ("ep01-p03",))

    def test_draft_mode_request_forbids_bubbles_and_text(self) -> None:
        draft_planner = DraftSpreadPlanner(
            generation_planner=self.generation_planner,
            draft_mode_config=DraftModeConfig(enabled=True, panels_per_image=2),
        )
        storyboard = self.pipeline.stage1_storyboard(self.novel)
        requests = draft_planner.build_requests(draft_planner.build_spreads(storyboard), self.state)
        self.assertIn("No speech bubbles, no word balloons", requests[0].positive_prompt)
        self.assertIn("Render EXACTLY TWO full-width rectangular comic panels stacked vertically.", requests[0].positive_prompt)
        self.assertIn("Add a strong black rectangular border around each panel.", requests[0].positive_prompt)
        self.assertIn("continuous scene spanning both panels", requests[0].negative_prompt)
        self.assertIn("Do not render any speech bubble outlines", requests[0].system_prompt)
        self.assertIn("strict layout task", requests[0].system_prompt)

    def test_draft_mode_normalizes_stacked_spread_into_two_clear_panels(self) -> None:
        draft_planner = DraftSpreadPlanner(
            generation_planner=self.generation_planner,
            draft_mode_config=DraftModeConfig(enabled=True, panels_per_image=2),
        )
        storyboard = self.pipeline.stage1_storyboard(self.novel)
        spread = draft_planner.build_spreads(storyboard)[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "draft.png"
            image = Image.new("RGB", (1024, 1024), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((80, 56, 944, 458), fill=(198, 214, 236), outline=(24, 24, 24), width=4)
            draw.rectangle((80, 566, 944, 968), fill=(236, 214, 198), outline=(24, 24, 24), width=4)
            image.save(raw_path)

            generated = GeneratedImage(
                asset_id="draft-test",
                panel_id=spread.spread_id,
                provider_name="gemini-generate-content",
                seed=123,
                prompt_fingerprint="draft",
                local_path=str(raw_path),
            )
            normalized = draft_planner.prepare_spread_image(spread, generated)

            self.assertTrue(normalized.local_path.endswith(".framed.png"))
            with Image.open(normalized.local_path) as framed:
                framed = framed.convert("RGB")
                self.assertEqual(framed.size, (1024, 1024))
                self.assertLess(sum(framed.getpixel((2, 2))), 120)
                self.assertEqual(framed.getpixel((512, 512)), (255, 255, 255))
                self.assertNotEqual(framed.getpixel((512, 240)), (255, 255, 255))
                self.assertNotEqual(framed.getpixel((512, 780)), (255, 255, 255))


class CliTests(unittest.TestCase):
    def test_build_input_from_inline_text(self) -> None:
        parser = cli.build_arg_parser()
        args = parser.parse_args(
            [
                "--text",
                "첫 문장입니다. 두 번째 문장입니다.",
                "--title",
                "사용자 테스트",
                "--episode-id",
                "ep-user",
                "--panel-count",
                "2",
            ]
        )
        source = cli.build_input_from_args(args)
        self.assertEqual(source.title, "사용자 테스트")
        self.assertEqual(source.episode_id, "ep-user")
        self.assertEqual(source.target_panel_count, 2)
        self.assertIn("첫 문장", source.source_text)

    def test_interactive_reader_stops_at_end_line(self) -> None:
        fake_input = io.StringIO("첫 줄\n둘째 줄\nEND\n무시됨\n")
        fake_stderr = io.StringIO()
        with redirect_stderr(fake_stderr):
            text = cli.read_interactive_text(fake_input)
        self.assertEqual(text, "첫 줄\n둘째 줄")
        self.assertIn("Finish with a single line containing only END", fake_stderr.getvalue())

    def test_main_supports_summary_output_from_pasted_text(self) -> None:
        fake_stdout = io.StringIO()
        fake_stderr = io.StringIO()
        with redirect_stdout(fake_stdout), redirect_stderr(fake_stderr):
            cli.main(
                [
                    "--text",
                    "비 오는 복도에서 두 사람이 마주친다. 도현: 이거 떨어뜨렸어.",
                    "--title",
                    "붙여넣기 테스트",
                    "--episode-id",
                    "paste-01",
                    "--panel-count",
                    "2",
                    "--format",
                    "summary",
                ]
            )
        output = fake_stdout.getvalue()
        self.assertIn("Episode: 붙여넣기 테스트 (paste-01)", output)
        self.assertIn("Stage 3 total image cost:", output)
        self.assertIn("[Stage 1] Text Storyboard", output)
        self.assertIn("[Stage 4] Bubble Requests", output)
        self.assertIn("[Stage 5] Bubble Overlays", output)

    def test_main_summary_mentions_enabled_bubble_text_when_flag_is_set(self) -> None:
        fake_stdout = io.StringIO()
        fake_stderr = io.StringIO()
        with redirect_stdout(fake_stdout), redirect_stderr(fake_stderr):
            cli.main(
                [
                    "--text",
                    "비 오는 복도에서 두 사람이 마주친다. 도현: 이거 떨어뜨렸어.",
                    "--title",
                    "붙여넣기 테스트",
                    "--episode-id",
                    "paste-02",
                    "--panel-count",
                    "2",
                    "--render-bubble-text",
                    "--format",
                    "summary",
                ]
            )
        output = fake_stdout.getvalue()
        self.assertIn("bubble text: overlay - dohyun/도현(", output)

    def test_main_supports_draft_mode_summary_with_mock_provider(self) -> None:
        fake_stdout = io.StringIO()
        fake_stderr = io.StringIO()
        with redirect_stdout(fake_stdout), redirect_stderr(fake_stderr):
            cli.main(
                [
                    "--provider",
                    "mock",
                    "--image-model",
                    "gemini-2.5-flash-image",
                    "--draft-mode",
                    "--draft-panels-per-image",
                    "2",
                    "--text",
                    "비 오는날 복도에서 두 사람이 마주친다. 도현: 이거 떨어뜨렸어.",
                    "--panel-count",
                    "2",
                    "--format",
                    "summary",
                ]
            )
        output = fake_stdout.getvalue()
        self.assertIn("Draft spreads: 1", output)
        self.assertIn("[Draft Spreads]", output)
        self.assertIn("layout: stacked-2up", output)

    def test_format_draft_summary_shows_batch_materialized_count(self) -> None:
        state = build_demo_state()
        generation_planner = GenerationPlanner(
            project_id=state.project_id,
            style_bible=build_demo_style(),
            characters=build_demo_characters(),
        )
        planner = DraftSpreadPlanner(
            generation_planner=generation_planner,
            draft_mode_config=DraftModeConfig(enabled=True, panels_per_image=2),
        )
        novel = build_demo_novel()
        storyboard = NovelStoryboardPlanner(characters=build_demo_characters()).build(novel)
        spreads = planner.build_spreads(storyboard)
        requests = planner.build_requests(spreads, state)
        result = cli.DraftRunResult(
            source=novel,
            spreads=spreads,
            requests=requests,
            images=(),
            batch_job=cli.BatchJobInfo(
                name="batches/test",
                display_name="test",
                provider="gemini-batch",
                model="gemini-2.5-flash-image",
                state="JOB_STATE_SUCCEEDED",
                request_count=len(requests),
            ),
        )
        summary = cli.format_draft_summary(result)
        self.assertIn("Batch materialized images: 0/", summary)

    def test_apply_image_overrides_updates_runtime_config(self) -> None:
        parser = cli.build_arg_parser()
        args = parser.parse_args(
            [
                "--image-model",
                "gpt-image-1",
                "--image-size",
                "1024x1024",
                "--image-quality",
                "high",
                "--image-output-dir",
                "outputs/custom",
            ]
        )
        config = cli.apply_image_overrides(args, build_demo_image_model_config())
        self.assertEqual(config.model, "gpt-image-1")
        self.assertEqual(config.size, "1024x1024")
        self.assertEqual(config.quality, "high")
        self.assertEqual(config.output_dir, "outputs/custom")

    def test_apply_cheap_image_profile_for_openai_uses_low_cost_defaults(self) -> None:
        parser = cli.build_arg_parser()
        args = parser.parse_args(["--cheap-image"])
        config = cli.apply_cheap_image_profile(args, build_demo_image_model_config())
        self.assertEqual(config.size, "1024x1024")
        self.assertEqual(config.quality, "low")

    def test_apply_cheap_image_profile_for_gemini_keeps_standard_cheap_mode(self) -> None:
        parser = cli.build_arg_parser()
        args = parser.parse_args(["--cheap-image", "--provider", "gemini"])
        config = cli.apply_cheap_image_profile(args, build_demo_gemini_image_model_config())
        self.assertEqual(config.size, "1024x1024")
        self.assertEqual(config.quality, "standard")

    def test_build_bubble_render_config_tracks_cli_flag(self) -> None:
        parser = cli.build_arg_parser()
        args = parser.parse_args(["--render-bubble-text"])
        config = cli.build_bubble_render_config(args)
        self.assertEqual(config.mode, "overlay")

    def test_default_image_config_for_gemini_provider_uses_gemini_model(self) -> None:
        config = cli.default_image_config_for_provider("gemini")
        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.model, "gemini-2.5-flash-image")

    def test_apply_image_overrides_switches_to_gemini_provider_for_gemini_model(self) -> None:
        parser = cli.build_arg_parser()
        args = parser.parse_args(["--image-model", "gemini-2.5-flash-image"])
        config = cli.apply_image_overrides(args, build_demo_image_model_config())
        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.api_key_env, "GEMINI_API_KEY")
        self.assertEqual(config.quality, "standard")

    def test_main_returns_clean_error_when_openai_key_is_missing(self) -> None:
        with self.assertRaises(SystemExit) as context:
            cli.main(
                [
                    "--text",
                    "실제 호출 테스트",
                    "--provider",
                    "openai",
                    "--format",
                    "summary",
                ]
            )
        self.assertIn("Missing API key for model gpt-image-1.5", str(context.exception))


class BatchApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.characters = build_demo_characters()
        self.state = build_demo_state()
        self.generation_planner = GenerationPlanner(
            project_id=self.state.project_id,
            style_bible=build_demo_style(),
            characters=self.characters,
        )
        self.draft_planner = DraftSpreadPlanner(
            generation_planner=self.generation_planner,
            draft_mode_config=DraftModeConfig(enabled=True, panels_per_image=2),
        )
        self.storyboard = NovelStoryboardPlanner(characters=self.characters).build(build_demo_novel())
        self.requests = self.draft_planner.build_requests(
            self.draft_planner.build_spreads(self.storyboard),
            self.state,
        )
        self.client = GeminiBatchClient(build_demo_gemini_image_model_config())

    def test_build_inline_batch_body_contains_requests_and_metadata(self) -> None:
        body = self.client.build_inline_batch_body(self.requests, display_name="draft-test")
        self.assertEqual(body["batch"]["displayName"], "draft-test")
        self.assertEqual(len(body["batch"]["inputConfig"]["requests"]["requests"]), len(self.requests))
        first_metadata = body["batch"]["inputConfig"]["requests"]["requests"][0]["metadata"]
        self.assertEqual(first_metadata["panelId"], self.requests[0].panel_id)

    def test_write_inline_manifest_writes_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            written = self.client.write_inline_manifest(
                self.requests,
                display_name="draft-test",
                destination=manifest_path,
                extra={"draft": True},
            )
            self.assertTrue(written.exists())
            payload = written.read_text(encoding="utf-8")
            self.assertIn("draft-test", payload)
            self.assertIn("\"draft\": true", payload)

    def test_materialize_inline_batch_results_reads_top_level_response_path(self) -> None:
        png_bytes = io.BytesIO()
        Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(png_bytes, format="PNG")
        batch_job = cli.BatchJobInfo(
            name="batches/test",
            display_name="test",
            provider="gemini-batch",
            model="gemini-2.5-flash-image",
            state="BATCH_STATE_SUCCEEDED",
            request_count=1,
            raw_response={
                "response": {
                    "inlinedResponses": {
                        "inlinedResponses": [
                            {
                                "response": {
                                    "candidates": [
                                        {
                                            "content": {
                                                "parts": [
                                                    {"text": "draft"},
                                                    {
                                                        "inlineData": {
                                                            "mimeType": "image/png",
                                                            "data": __import__("base64").b64encode(png_bytes.getvalue()).decode("ascii"),
                                                        }
                                                    },
                                                ]
                                            }
                                        }
                                    ],
                                    "usageMetadata": {"promptTokenCount": 10},
                                }
                            }
                        ]
                    }
                }
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            images = self.client.materialize_inline_batch_results(
                batch_job,
                self.requests,
                output_dir=Path(tmpdir),
            )
            self.assertEqual(len(images), 1)
            self.assertTrue(Path(images[0].local_path).exists())


if __name__ == "__main__":
    unittest.main()
