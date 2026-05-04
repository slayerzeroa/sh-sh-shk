from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from consistent_t2i import __main__ as consistent_cli
from consistent_t2i.control_models import (
    CONTROL_MODEL_CONSISTENCY_KEEPER,
    CONTROL_MODEL_CONSISTENT_T2I,
    CONTROL_MODEL_TIFF_STORYBOARD,
    normalize_control_model,
)
from consistent_t2i.tiff_storyboard import RenderConfig, render_novel_to_tiff
from novel_conti_tiff import __main__ as tiff_cli


SAMPLE_TEXT = """
비 오는 복도에서 두 사람이 마주친다.
도현: 이거 떨어뜨렸어.

해은은 젖은 머리핀을 고쳐 꽂는다.
""".strip()


class ControlModelTests(unittest.TestCase):
    def test_normalize_control_model_accepts_typo_alias(self) -> None:
        self.assertEqual(normalize_control_model("conisstent_t2i"), CONTROL_MODEL_CONSISTENT_T2I)
        self.assertEqual(normalize_control_model("tiff"), CONTROL_MODEL_TIFF_STORYBOARD)
        self.assertEqual(
            normalize_control_model("anything", build_visual_bible=True),
            CONTROL_MODEL_CONSISTENCY_KEEPER,
        )


class TiffStoryboardTests(unittest.TestCase):
    def test_render_novel_to_tiff_writes_manifest_pages_and_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = render_novel_to_tiff(
                source_text=SAMPLE_TEXT,
                title="비 복도",
                episode_id="rain-hall",
                target_panel_count=3,
                output_dir=tmpdir,
                render_config=RenderConfig(panels_per_page=2, bundle_tiff=True),
            )

            self.assertEqual(result.title, "비 복도")
            self.assertEqual(result.episode_id, "rain-hall")
            self.assertEqual(result.panel_count, 2)
            self.assertEqual(len(result.pages), 1)
            self.assertTrue(Path(result.source_copy_path).exists())
            self.assertTrue(Path(result.manifest_path).exists())
            self.assertTrue(Path(result.bundle_path).exists())

            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["episode_id"], "rain-hall")
            self.assertEqual(manifest["pages"][0]["pixel_mode"], "1")
            self.assertEqual(manifest["pages"][0]["compression"], "group4")

    def test_consistent_cli_routes_to_tiff_control_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_stdout = io.StringIO()
            with redirect_stdout(fake_stdout):
                consistent_cli.main(
                    [
                        "--control-model",
                        "tiff",
                        "--text",
                        SAMPLE_TEXT,
                        "--title",
                        "비 복도",
                        "--episode-id",
                        "rain-hall",
                        "--tiff-output-dir",
                        tmpdir,
                        "--format",
                        "summary",
                    ]
                )

            output = fake_stdout.getvalue()
            self.assertIn("Title: 비 복도", output)
            self.assertIn("Episode: rain-hall", output)
            self.assertIn("[TIFF Pages]", output)
            self.assertTrue(any(Path(tmpdir).glob("page-*.tiff")))

    def test_packaged_novel_conti_tiff_cli_stays_usable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_stdout = io.StringIO()
            with redirect_stdout(fake_stdout):
                tiff_cli.main(
                    [
                        "--text",
                        SAMPLE_TEXT,
                        "--title",
                        "패키지 테스트",
                        "--episode-id",
                        "pkg-test",
                        "--output-dir",
                        tmpdir,
                        "--format",
                        "summary",
                    ]
                )

            output = fake_stdout.getvalue()
            self.assertIn("Title: 패키지 테스트", output)
            self.assertIn("[TIFF Pages]", output)
            self.assertTrue((Path(tmpdir) / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
