from __future__ import annotations

import json
from dataclasses import asdict

from .demo_data import build_demo_characters, build_demo_panel, build_demo_state, build_demo_style
from .drift import DriftAnalyzer
from .engine import ConsistentT2IEngine
from .models import ObservedTraits
from .planner import GenerationPlanner
from .providers import MockImageProvider


def main() -> None:
    style = build_demo_style()
    characters = build_demo_characters()
    state = build_demo_state()
    engine = ConsistentT2IEngine(
        continuity=state,
        planner=GenerationPlanner(
            project_id=state.project_id,
            style_bible=style,
            characters=characters,
        ),
        drift_analyzer=DriftAnalyzer(characters=characters),
    )

    panel = build_demo_panel()
    request = engine.build_generation_request(panel)
    image = engine.generate_panel(panel, MockImageProvider())
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

    payload = {
        "request": asdict(request),
        "generated_image": asdict(image),
        "drift_report": asdict(report),
        "cold_path_review": asdict(review),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
