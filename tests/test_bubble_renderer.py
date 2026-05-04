from __future__ import annotations

import unittest

from consistent_t2i.bubble_renderer import BubbleOverlayRenderer
from consistent_t2i.demo_data import build_demo_image_model_config
from consistent_t2i.models import BubbleFillRequest, BubbleLine, BubbleRenderConfig, SketchPanel


class BubbleRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = BubbleOverlayRenderer(
            image_config=build_demo_image_model_config(),
            render_config=BubbleRenderConfig(mode="overlay"),
        )
        self.sketch = SketchPanel(
            panel_id="bubble-test-p01",
            summary="Two students face each other in a school hallway.",
            camera_guide="medium shot",
            figure_guides=(
                "haeun: stick figure at left third, eyeline matched",
                "dohyun: stick figure at right third, eyeline matched",
            ),
            bubble_slots=(
                "bubble-1: near dohyun, away from focal face",
                "bubble-2: near haeun, away from focal face",
            ),
        )

    def test_long_dialogue_keeps_padding_inside_bubble(self) -> None:
        request = BubbleFillRequest(
            panel_id=self.sketch.panel_id,
            system_prompt="",
            user_prompt="",
            model="test-bubble",
            max_output_tokens=120,
            lines=(
                BubbleLine(
                    speaker="도현",
                    speaker_character_id="dohyun",
                    text="이거 떨어뜨렸어, 네가 아까 급하게 지나가다가 놓친 것 같아서 먼저 챙겨뒀어.",
                    slot_hint=self.sketch.bubble_slots[0],
                ),
            ),
        )

        placement = self.renderer.build_plan(self.sketch, request).placements[0]
        font = self.renderer._load_font(placement.font_size)
        max_line_width = max(
            self.renderer._measure_text_width(line, font=font)
            for line in placement.wrapped_lines
        )
        text_block_height = self.renderer._measure_text_block_height(
            placement.wrapped_lines,
            font=font,
            line_height=placement.line_height,
        )

        self.assertGreaterEqual(placement.width - max_line_width, 56)
        self.assertGreaterEqual(placement.height - text_block_height, 42)
        self.assertLessEqual(len(placement.wrapped_lines), 4)

    def test_multiple_dialogues_keep_visual_gap(self) -> None:
        request = BubbleFillRequest(
            panel_id=self.sketch.panel_id,
            system_prompt="",
            user_prompt="",
            model="test-bubble",
            max_output_tokens=120,
            lines=(
                BubbleLine(
                    speaker="도현",
                    speaker_character_id="dohyun",
                    text="이거 떨어뜨렸어.",
                    slot_hint=self.sketch.bubble_slots[0],
                ),
                BubbleLine(
                    speaker="하은",
                    speaker_character_id="haeun",
                    text="아, 고마워. 진짜 못 본 줄 알았어.",
                    slot_hint=self.sketch.bubble_slots[1],
                ),
            ),
        )

        placements = self.renderer.build_plan(self.sketch, request).placements
        first = self.renderer._expand_rect(
            (placements[0].x, placements[0].y, placements[0].width, placements[0].height),
            padding=18,
        )
        second = self.renderer._expand_rect(
            (placements[1].x, placements[1].y, placements[1].width, placements[1].height),
            padding=18,
        )

        self.assertEqual(self.renderer._intersection_area(first, second), 0)

    def test_dialogue_tail_is_closed_filled_shape(self) -> None:
        request = BubbleFillRequest(
            panel_id=self.sketch.panel_id,
            system_prompt="",
            user_prompt="",
            model="test-bubble",
            max_output_tokens=120,
            lines=(
                BubbleLine(
                    speaker="도현",
                    speaker_character_id="dohyun",
                    text="이거 떨어뜨렸어.",
                    slot_hint=self.sketch.bubble_slots[0],
                ),
            ),
        )

        placement = self.renderer.build_plan(self.sketch, request).placements[0]
        tail_svg = self.renderer._render_tail(placement)

        self.assertIn('fill="white"', tail_svg)
        self.assertIn(" Z", tail_svg)


if __name__ == "__main__":
    unittest.main()
