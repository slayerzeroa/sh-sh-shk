from __future__ import annotations

import unittest

from consistent_t2i.models import ImageUsage
from consistent_t2i.pricing import approximate_text_tokens, calculate_image_cost


class PricingTests(unittest.TestCase):
    def test_per_image_output_estimate_matches_official_table(self) -> None:
        cost = calculate_image_cost(
            "gpt-image-1.5",
            "1024x1536",
            "medium",
            prompt="Rainy school corridor webtoon panel",
        )
        self.assertEqual(cost.output_image_cost_usd, 0.05)
        self.assertTrue(cost.estimated)

    def test_usage_based_cost_uses_token_rates(self) -> None:
        usage = ImageUsage(
            input_text_tokens=1000,
            input_image_tokens=0,
            output_image_tokens=4160,
            source="api",
        )
        cost = calculate_image_cost(
            "gpt-image-1",
            "1024x1024",
            "high",
            usage=usage,
        )
        self.assertAlmostEqual(cost.input_text_cost_usd, 0.005, places=6)
        self.assertAlmostEqual(cost.output_image_cost_usd, 0.1664, places=6)
        self.assertAlmostEqual(cost.total_cost_usd, 0.1714, places=6)
        self.assertFalse(cost.estimated)

    def test_prompt_token_estimator_returns_positive_value(self) -> None:
        self.assertGreater(approximate_text_tokens("비 오는 복도에서 두 사람이 마주친다."), 0)

    def test_gemini_standard_image_price_estimate_matches_official_table(self) -> None:
        cost = calculate_image_cost(
            "gemini-2.5-flash-image",
            "1024x1024",
            "standard",
            prompt="웹툰 패널용 장면 생성",
        )
        self.assertEqual(cost.output_image_cost_usd, 0.039)
        self.assertTrue(cost.estimated)


if __name__ == "__main__":
    unittest.main()
