from __future__ import annotations

from dataclasses import dataclass

from .drift import DriftAnalyzer
from .memory import ContinuityState
from .models import ColdPathDecision, DriftReport, GeneratedImage, ObservedTraits, PanelSpec, ReferenceImage
from .planner import GenerationPlanner
from .providers import ImageProvider


@dataclass
class RealtimeGenerationPath:
    continuity: ContinuityState
    planner: GenerationPlanner

    def build_generation_request(self, panel: PanelSpec):
        return self.planner.build_request(panel, self.continuity)

    def generate_panel(self, panel: PanelSpec, provider: ImageProvider) -> GeneratedImage:
        request = self.build_generation_request(panel)
        return provider.generate(request)

    def approve_generation(
        self,
        panel: PanelSpec,
        image: GeneratedImage,
        note: str = "Approved panel result",
        review: ColdPathDecision | None = None,
    ) -> None:
        if review is not None and review.block_approval:
            raise ValueError(
                f"Cannot approve panel {panel.panel_id}: cold path review blocked approval."
            )
        for character_id in panel.characters:
            self.continuity.register_reference(
                ReferenceImage(
                    asset_id=image.asset_id,
                    character_id=character_id,
                    note=f"{note}: {panel.panel_id}",
                    source="generated",
                )
            )
        self.continuity.register_approved_panel(panel.panel_id, [image.asset_id])


@dataclass(frozen=True)
class ColdPathController:
    drift_analyzer: DriftAnalyzer

    def evaluate_observations(self, panel_id: str, observations: list[ObservedTraits]) -> DriftReport:
        return self.drift_analyzer.evaluate(panel_id, observations)

    def review_generation(self, panel_id: str, observations: list[ObservedTraits]) -> ColdPathDecision:
        report = self.evaluate_observations(panel_id, observations)
        reasons = tuple(issue.message for issue in report.issues) or ("No continuity issues detected.",)
        return ColdPathDecision(
            panel_id=panel_id,
            should_repair=report.should_regenerate,
            block_approval=report.should_regenerate,
            reasons=reasons,
        )
