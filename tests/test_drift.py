from __future__ import annotations

import unittest

from consistent_t2i.demo_data import build_demo_characters, build_demo_panel
from consistent_t2i.drift import DriftAnalyzer
from consistent_t2i.models import ObservedTraits


class DriftAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.characters = build_demo_characters()
        self.panel = build_demo_panel()
        self.analyzer = DriftAnalyzer(characters=self.characters)

    def test_hard_trait_mismatch_requires_regeneration(self) -> None:
        report = self.analyzer.evaluate(
            self.panel.panel_id,
            [
                ObservedTraits(
                    panel_id=self.panel.panel_id,
                    character_id="haeun",
                    observed={
                        "hair_color": "blonde",
                        "hair_style": "long straight hair with blunt bangs",
                        "eye_color": "dark brown",
                        "signature_item": "silver hairpin shaped like a crescent moon",
                    },
                )
            ],
        )
        self.assertTrue(report.should_regenerate)
        self.assertEqual(report.issues[0].trait_name, "hair_color")
        self.assertLess(report.score, 1.0)

    def test_matching_traits_keep_full_score(self) -> None:
        report = self.analyzer.evaluate(
            self.panel.panel_id,
            [
                ObservedTraits(
                    panel_id=self.panel.panel_id,
                    character_id="dohyun",
                    observed={
                        "hair_color": "dark ash brown",
                        "hair_style": "slightly tousled short hair",
                        "eye_color": "gray brown",
                        "signature_item": "school blazer with gold crest",
                    },
                )
            ],
        )
        self.assertEqual(report.score, 1.0)
        self.assertFalse(report.should_regenerate)

    def test_missing_observed_traits_do_not_trigger_false_drift(self) -> None:
        report = self.analyzer.evaluate(
            self.panel.panel_id,
            [
                ObservedTraits(
                    panel_id=self.panel.panel_id,
                    character_id="dohyun",
                    observed={
                        "hair_color": "dark ash brown",
                    },
                )
            ],
        )
        self.assertEqual(report.score, 1.0)
        self.assertFalse(report.should_regenerate)


if __name__ == "__main__":
    unittest.main()
