from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from consistent_t2i import __main__ as cli
from consistent_t2i.models import WebNovelInput
from consistent_t2i.visual_bible import build_visual_bible_workspace


SAMPLE_NOVEL_TEXT = """
비가 내리는 늦은 오후, 서린고 복도는 젖은 운동화 냄새로 가득했다.
해은은 검은 긴 생머리를 어깨 아래로 늘어뜨린 채 창가에 섰다.
그녀의 은색 초승달 머리핀이 젖은 앞머리 사이에서 반짝였다.
도현은 큰 키에 짙은 남색 교복 재킷을 걸치고 회색 눈으로 해은을 바라봤다.
선배는 젖은 소매를 걷어 올리고 우산 손잡이를 쥐었다.
학생회가 장악한 학교에서는 밤마다 옥상 출입이 금지되어 있었다.
도현: 우산 가져왔어.
해은: 고마워.

다음 날 옥상에서 해은은 마른 교복 셔츠 위에 회색 가디건을 걸쳤다.
그녀는 젖은 우산 대신 검은 책을 품에 안고 있었다.
""".strip()

CONFLICT_NOVEL_TEXT = """
해은은 검은 긴 생머리를 어깨 아래로 늘어뜨리고 있었다.
잠시 후 해은은 금발처럼 빛나는 머리를 털어내며 창가를 돌아봤다.
""".strip()


class VisualBibleTests(unittest.TestCase):
    def build_source(self, text: str = SAMPLE_NOVEL_TEXT) -> WebNovelInput:
        return WebNovelInput(
            episode_id="novel-full",
            title="서린고의 비",
            source_text=text,
        )

    def test_workspace_builder_writes_layered_visual_bible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            document, result = build_visual_bible_workspace(
                self.build_source(),
                workspace_dir=tmpdir,
                max_characters=5,
                evidence_limit=3,
            )

            self.assertEqual(result.character_count, 2)
            self.assertGreaterEqual(result.scene_count, 2)
            self.assertTrue(Path(result.source_copy_path).exists())
            self.assertTrue(Path(result.json_path).exists())
            self.assertTrue(Path(result.world_path).exists())
            self.assertTrue(Path(result.characters_path).exists())
            self.assertTrue(Path(result.appearance_path).exists())
            self.assertTrue(Path(result.scene_state_path).exists())
            self.assertTrue(Path(result.prompt_pack_path).exists())
            self.assertTrue(Path(result.conflict_path).exists())

            haeun = next(character for character in document.characters if character.name == "해은")
            dohyun = next(character for character in document.characters if character.name == "도현")

            self.assertTrue(any("검은 긴 생머리" in evidence.text for evidence in haeun.fixed_appearance["hair"]))
            self.assertTrue(any("초승달 머리핀" in evidence.text for evidence in haeun.fixed_appearance["accessories"]))
            self.assertTrue(any("가디건" in evidence.text for evidence in haeun.variable_appearance["outfit"]))
            self.assertIn("선배", dohyun.aliases)
            self.assertTrue(any("선배는 젖은 소매" in sentence for sentence in dohyun.evidence))
            self.assertTrue(document.prompt_pack.character_cards)
            self.assertGreaterEqual(len(document.prompt_pack.scene_prompts), 2)

            world_markdown = Path(result.world_path).read_text(encoding="utf-8")
            self.assertIn("student council", world_markdown)
            self.assertIn("rain", world_markdown)

            scene_state_markdown = Path(result.scene_state_path).read_text(encoding="utf-8")
            self.assertIn("scene-01", scene_state_markdown)
            self.assertIn("scene-02", scene_state_markdown)

            prompt_pack = json.loads(Path(result.prompt_pack_path).read_text(encoding="utf-8"))
            self.assertIn("global_positive_prompt", prompt_pack)
            self.assertEqual(len(prompt_pack["character_cards"]), 2)

            payload = json.loads(Path(result.json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["document"]["title"], "서린고의 비")
            self.assertIn("scene_states", payload["document"])

    def test_conflict_report_captures_incompatible_fixed_traits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            document, result = build_visual_bible_workspace(
                self.build_source(CONFLICT_NOVEL_TEXT),
                workspace_dir=tmpdir,
                max_characters=3,
                evidence_limit=3,
            )

            self.assertEqual(result.character_count, 1)
            self.assertGreaterEqual(result.conflict_count, 1)
            self.assertTrue(document.conflicts)
            self.assertTrue(any("black hair" in conflict.variants for conflict in document.conflicts))
            self.assertTrue(any("blonde hair" in conflict.variants for conflict in document.conflicts))

            conflict_markdown = Path(result.conflict_path).read_text(encoding="utf-8")
            self.assertIn("black hair", conflict_markdown)
            self.assertIn("blonde hair", conflict_markdown)

    def test_cli_visual_bible_mode_prints_upgraded_summary_and_writes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_stdout = io.StringIO()
            with redirect_stdout(fake_stdout):
                cli.main(
                    [
                        "--build-visual-bible",
                        "--text",
                        SAMPLE_NOVEL_TEXT,
                        "--title",
                        "서린고의 비",
                        "--episode-id",
                        "novel-full",
                        "--visual-bible-dir",
                        tmpdir,
                        "--format",
                        "summary",
                    ]
                )

            output = fake_stdout.getvalue()
            self.assertIn("Visual bible: 서린고의 비 (novel-full)", output)
            self.assertIn("Detected characters: 2", output)
            self.assertIn("Detected scenes:", output)
            self.assertIn("Prompt pack:", output)
            self.assertIn("Conflict report:", output)

            output_dir = Path(tmpdir) / "output"
            self.assertTrue(any(output_dir.glob("*.visual-bible.json")))
            self.assertTrue(any(output_dir.glob("*.appearance-locks.md")))
            self.assertTrue(any(output_dir.glob("*.scene-state.md")))
            self.assertTrue(any(output_dir.glob("*.prompt-pack.json")))


if __name__ == "__main__":
    unittest.main()
