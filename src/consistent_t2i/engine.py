from __future__ import annotations

from dataclasses import dataclass

from .drift import DriftAnalyzer
from .memory import ContinuityState
from .models import ColdPathDecision, GeneratedImage, ObservedTraits, PanelSpec
from .paths import ColdPathController, RealtimeGenerationPath
from .planner import GenerationPlanner
from .providers import ImageProvider


@dataclass
class ConsistentT2IEngine:
    continuity: ContinuityState
    planner: GenerationPlanner
    drift_analyzer: DriftAnalyzer

    @property
    def hot_path(self) -> RealtimeGenerationPath:
        return RealtimeGenerationPath(continuity=self.continuity, planner=self.planner)

    @property
    def cold_path(self) -> ColdPathController:
        return ColdPathController(drift_analyzer=self.drift_analyzer)

    def build_generation_request(self, panel: PanelSpec):
        return self.hot_path.build_generation_request(panel)

    def generate_panel(self, panel: PanelSpec, provider: ImageProvider) -> GeneratedImage:
        return self.hot_path.generate_panel(panel, provider)

    def approve_generation(
        self,
        panel: PanelSpec,
        image: GeneratedImage,
        note: str = "Approved panel result",
        review: ColdPathDecision | None = None,
    ) -> None:
        self.hot_path.approve_generation(panel, image, note=note, review=review)

    def evaluate_observations(self, panel_id: str, observations: list[ObservedTraits]):
        return self.cold_path.evaluate_observations(panel_id, observations)

    def review_generation(self, panel_id: str, observations: list[ObservedTraits]):
        return self.cold_path.review_generation(panel_id, observations)

    def review_and_approve_generation(
        self,
        panel: PanelSpec,
        image: GeneratedImage,
        observations: list[ObservedTraits],
        note: str = "Approved panel result",
    ) -> ColdPathDecision:
        review = self.review_generation(panel.panel_id, observations)
        self.approve_generation(panel, image, note=note, review=review)
        return review
