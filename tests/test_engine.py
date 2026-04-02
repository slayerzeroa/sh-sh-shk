from __future__ import annotations

import unittest
from pathlib import Path

from consistent_t2i.demo_data import (
    build_demo_characters,
    build_demo_panel,
    build_demo_state,
    build_demo_style,
)
from consistent_t2i.drift import DriftAnalyzer
from consistent_t2i.engine import ConsistentT2IEngine
from consistent_t2i.memory import ContinuityState
from consistent_t2i.models import ObservedTraits, ReferenceImage
from consistent_t2i.planner import GenerationPlanner
from consistent_t2i.providers import MockImageProvider


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.style = build_demo_style()
        self.characters = build_demo_characters()
        self.panel = build_demo_panel()
        self.state = build_demo_state()
        self.engine = ConsistentT2IEngine(
            continuity=self.state,
            planner=GenerationPlanner(
                project_id=self.state.project_id,
                style_bible=self.style,
                characters=self.characters,
            ),
            drift_analyzer=DriftAnalyzer(characters=self.characters),
        )

    def test_approved_generation_is_saved_as_future_reference(self) -> None:
        image = self.engine.generate_panel(self.panel, MockImageProvider())
        self.engine.approve_generation(self.panel, image)
        recent = self.state.get_recent_references("haeun")
        self.assertEqual(recent[0].asset_id, image.asset_id)
        self.assertIn(self.panel.panel_id, self.state.approved_panels)

    def test_generate_panel_does_not_mutate_continuity_before_approval(self) -> None:
        image = self.engine.generate_panel(self.panel, MockImageProvider())
        self.assertTrue(image.asset_id.startswith("mock-"))
        self.assertEqual(self.state.get_recent_references("haeun"), ())
        self.assertNotIn(self.panel.panel_id, self.state.approved_panels)

    def test_continuity_state_round_trip(self) -> None:
        image = self.engine.generate_panel(self.panel, MockImageProvider())
        self.engine.approve_generation(self.panel, image)
        destination = Path(".test-continuity.json")
        try:
            self.state.save(destination)
            loaded = ContinuityState.load(destination)
        finally:
            destination.unlink(missing_ok=True)
        self.assertEqual(loaded.project_id, self.state.project_id)
        self.assertEqual(
            loaded.get_recent_references("dohyun")[0].asset_id,
            image.asset_id,
        )

    def test_drift_feedback_can_gate_reuse(self) -> None:
        report = self.engine.evaluate_observations(
            self.panel.panel_id,
            [
                ObservedTraits(
                    panel_id=self.panel.panel_id,
                    character_id="dohyun",
                    observed={
                        "hair_color": "bright red",
                        "hair_style": "slightly tousled short hair",
                        "eye_color": "gray brown",
                        "signature_item": "school blazer with gold crest",
                    },
                )
            ],
        )
        self.assertTrue(report.should_regenerate)

    def test_cold_path_review_blocks_approval_on_critical_drift(self) -> None:
        image = self.engine.generate_panel(self.panel, MockImageProvider())
        review = self.engine.review_generation(
            self.panel.panel_id,
            [
                ObservedTraits(
                    panel_id=self.panel.panel_id,
                    character_id="haeun",
                    observed={
                        "hair_color": "silver",
                        "hair_style": "long straight hair with blunt bangs",
                        "eye_color": "dark brown",
                        "signature_item": "silver hairpin shaped like a crescent moon",
                    },
                )
            ],
        )
        self.assertTrue(review.should_repair)
        self.assertTrue(review.block_approval)
        self.assertIn("hair_color", review.reasons[0])
        with self.assertRaises(ValueError):
            self.engine.approve_generation(self.panel, image, review=review)

    def test_clean_cold_path_review_allows_approval(self) -> None:
        image = self.engine.generate_panel(self.panel, MockImageProvider())
        review = self.engine.review_generation(
            self.panel.panel_id,
            [
                ObservedTraits(
                    panel_id=self.panel.panel_id,
                    character_id="haeun",
                    observed={
                        "hair_color": "black",
                        "hair_style": "long straight hair with blunt bangs",
                        "eye_color": "dark brown",
                        "signature_item": "silver hairpin shaped like a crescent moon",
                    },
                )
            ],
        )
        self.engine.approve_generation(self.panel, image, review=review)
        self.assertEqual(self.state.get_recent_references("haeun")[0].asset_id, image.asset_id)

    def test_review_and_approve_generation_uses_safe_default_flow(self) -> None:
        image = self.engine.generate_panel(self.panel, MockImageProvider())
        observations = [
            ObservedTraits(
                panel_id=self.panel.panel_id,
                character_id="haeun",
                observed={
                    "hair_color": "black",
                    "hair_style": "long straight hair with blunt bangs",
                    "eye_color": "dark brown",
                    "signature_item": "silver hairpin shaped like a crescent moon",
                },
            )
        ]
        review = self.engine.review_and_approve_generation(self.panel, image, observations)
        self.assertFalse(review.block_approval)
        self.assertEqual(self.state.get_recent_references("haeun")[0].asset_id, image.asset_id)

    def test_cold_path_review_does_not_mutate_continuity(self) -> None:
        review = self.engine.review_generation(
            self.panel.panel_id,
            [
                ObservedTraits(
                    panel_id=self.panel.panel_id,
                    character_id="haeun",
                    observed={
                        "hair_color": "silver",
                        "hair_style": "long straight hair with blunt bangs",
                        "eye_color": "dark brown",
                        "signature_item": "silver hairpin shaped like a crescent moon",
                    },
                )
            ],
        )
        self.assertTrue(review.block_approval)
        self.assertEqual(self.state.get_recent_references("haeun"), ())
        self.assertNotIn(self.panel.panel_id, self.state.approved_panels)

    def test_continuity_state_keeps_recent_history_bounded(self) -> None:
        self.state.max_history_per_character = 2
        for asset_id in ("ref-1", "ref-2", "ref-3"):
            self.state.register_reference(
                ReferenceImage(
                    asset_id=asset_id,
                    character_id="haeun",
                    note=f"reference {asset_id}",
                )
            )
        recent = self.state.get_recent_references("haeun")
        self.assertEqual(tuple(reference.asset_id for reference in recent), ("ref-3", "ref-2"))


if __name__ == "__main__":
    unittest.main()
