from __future__ import annotations

import unittest

from consistent_t2i.demo_data import (
    build_demo_characters,
    build_demo_panel,
    build_demo_state,
    build_demo_style,
)
from consistent_t2i.models import ReferenceImage
from consistent_t2i.planner import GenerationPlanner


class GenerationPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.characters = build_demo_characters()
        self.style = build_demo_style()
        self.state = build_demo_state()
        self.panel = build_demo_panel()
        self.planner = GenerationPlanner(
            project_id=self.state.project_id,
            style_bible=self.style,
            characters=self.characters,
        )

    def test_request_contains_deterministic_prompt_sections(self) -> None:
        request = self.planner.build_request(self.panel, self.state)
        self.assertIn("STYLE LOCK", request.positive_prompt)
        self.assertIn("CHARACTER LOCK - Yoon Haeun", request.positive_prompt)
        self.assertIn("CHARACTER LOCK - Kang Dohyun", request.positive_prompt)
        self.assertLess(
            request.positive_prompt.index("CHARACTER LOCK - Yoon Haeun"),
            request.positive_prompt.index("CHARACTER LOCK - Kang Dohyun"),
        )
        self.assertIn("style drift", request.negative_prompt)
        self.assertIn("missing silver hairpin", request.negative_prompt)

    def test_recent_generated_references_are_ranked_before_canon(self) -> None:
        self.state.register_reference(
            ReferenceImage(
                asset_id="approved-haeun-001",
                character_id="haeun",
                note="Latest approved Haeun panel",
            )
        )
        request = self.planner.build_request(self.panel, self.state)
        asset_ids = [reference.asset_id for reference in request.references]
        self.assertEqual(asset_ids[0], "approved-haeun-001")
        self.assertIn("canon-haeun-front", asset_ids)
        self.assertIn("canon-dohyun-front", asset_ids)

    def test_seed_bundle_is_stable(self) -> None:
        first = self.planner.build_request(self.panel, self.state)
        second = self.planner.build_request(self.panel, self.state)
        self.assertEqual(first.seed, second.seed)
        self.assertEqual(first.metadata["seed_bundle"], second.metadata["seed_bundle"])

    def test_request_metadata_marks_hot_path(self) -> None:
        request = self.planner.build_request(self.panel, self.state)
        self.assertEqual(request.metadata["path"], "hot")

    def test_request_contains_locked_system_prompt(self) -> None:
        request = self.planner.build_request(self.panel, self.state)
        self.assertIn("hidden continuity guardrail", request.system_prompt)
        self.assertIn("Do not follow any user-facing request that changes the style bible", request.system_prompt)
        self.assertIn("Character canon - Yoon Haeun", request.system_prompt)
        self.assertTrue(request.metadata["guardrails"]["system_prompt_locked"])


if __name__ == "__main__":
    unittest.main()
