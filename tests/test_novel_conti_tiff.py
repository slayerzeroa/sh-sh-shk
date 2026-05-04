from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from novel_conti_tiff import RenderConfig, StoryboardPlanner, render_novel_to_tiff
from novel_conti_tiff import __main__ as cli


class StoryboardPlannerTests(unittest.TestCase):
    def test_build_splits_korean_text_into_story_panels(self) -> None:
        planner = StoryboardPlanner()
        panels = planner.build(
            source_text=(
                "비 오는 복도에서 두 사람이 마주친다. "
                "도현: 이거 떨어뜨렸어. "
                "하은은 놀라서 고개를 든다. "
                "둘 사이에 어색한 정적이 흐른다."
            ),
            episode_id="hallway",
            target_panel_count=4,
        )

        self.assertGreaterEqual(len(panels), 3)
        self.assertEqual(panels[0].panel_id, "hallway-p01")
        self.assertIn("school corridor", panels[0].location)
        self.assertEqual(panels[1].dialogue[0], "도현: 이거 떨어뜨렸어.")


class TiffRenderTests(unittest.TestCase):
    def test_render_writes_1bit_group4_tiff_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = render_novel_to_tiff(
                source_text=(
                    "비 오는 복도에서 두 사람이 마주친다. "
                    "도현: 이거 떨어뜨렸어. "
                    "하은은 놀라서 손을 뻗는다. "
                    "창문 너머로 빗줄기가 보인다."
                ),
                title="비 오는 복도",
                episode_id="hallway",
                target_panel_count=4,
                output_dir=tmpdir,
                render_config=RenderConfig(panels_per_page=2),
            )

            self.assertEqual(result.panel_count, 4)
            self.assertEqual(len(result.pages), 2)
            self.assertTrue(Path(result.manifest_path).exists())

            first_page = Path(result.pages[0].path)
            self.assertTrue(first_page.exists())
            self.assertLess(first_page.stat().st_size, 200_000)

            with Image.open(first_page) as image:
                self.assertEqual(image.mode, "1")
                self.assertEqual(image.tag_v2.get(259), 4)
                self.assertEqual(image.size, (1240, 1754))

    def test_bundle_option_writes_multi_page_tiff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = render_novel_to_tiff(
                source_text=(
                    "첫 장면이다. 둘째 장면이다. 셋째 장면이다. 넷째 장면이다. "
                    "다섯째 장면이다. 여섯째 장면이다."
                ),
                title="묶음 테스트",
                episode_id="bundle",
                target_panel_count=6,
                output_dir=tmpdir,
                render_config=RenderConfig(panels_per_page=2, bundle_tiff=True),
            )

            self.assertTrue(result.bundle_path)
            bundle_path = Path(result.bundle_path)
            self.assertTrue(bundle_path.exists())
            with Image.open(bundle_path) as image:
                self.assertEqual(image.mode, "1")
                self.assertEqual(image.tag_v2.get(259), 4)
                self.assertEqual(image.n_frames, 3)


class CliTests(unittest.TestCase):
    def test_cli_json_output_mentions_manifest_and_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_stdout = io.StringIO()
            with redirect_stdout(fake_stdout):
                cli.main(
                    [
                        "--text",
                        "비 오는 복도에서 두 사람이 마주친다. 도현: 이거 떨어뜨렸어. 하은은 놀라서 고개를 든다.",
                        "--title",
                        "CLI 테스트",
                        "--episode-id",
                        "cli-01",
                        "--panel-count",
                        "3",
                        "--output-dir",
                        tmpdir,
                        "--format",
                        "json",
                    ]
                )

            payload = json.loads(fake_stdout.getvalue())
            self.assertEqual(payload["title"], "CLI 테스트")
            self.assertEqual(payload["panel_count"], 3)
            self.assertTrue(Path(payload["manifest_path"]).exists())
            self.assertEqual(len(payload["pages"]), 1)


if __name__ == "__main__":
    unittest.main()
