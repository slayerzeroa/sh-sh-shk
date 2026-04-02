from __future__ import annotations

from dataclasses import dataclass

from .models import CharacterBible, DriftIssue, DriftReport, ObservedTraits


@dataclass(frozen=True)
class DriftAnalyzer:
    characters: dict[str, CharacterBible]

    def evaluate(self, panel_id: str, observations: list[ObservedTraits]) -> DriftReport:
        issues: list[DriftIssue] = []
        total_weight = 0
        mismatch_weight = 0
        for observation in observations:
            character = self.characters[observation.character_id]
            for trait in character.immutable_traits:
                total_weight += 2 if trait.level == "hard" else 1
                observed_value = observation.observed.get(trait.name)
                if observed_value is None:
                    continue
                if observed_value == trait.expected_value:
                    continue
                severity = "critical" if trait.level == "hard" else "warning"
                mismatch_weight += 2 if severity == "critical" else 1
                issues.append(
                    DriftIssue(
                        character_id=character.character_id,
                        trait_name=trait.name,
                        expected_value=trait.expected_value,
                        observed_value=observed_value,
                        severity=severity,
                        message=(
                            f"{character.display_name} trait drifted: "
                            f"{trait.name} expected '{trait.expected_value}' "
                            f"but observed '{observed_value}'."
                        ),
                    )
                )
        score = 1.0 if total_weight == 0 else max(0.0, 1.0 - (mismatch_weight / total_weight))
        should_regenerate = any(issue.severity == "critical" for issue in issues)
        return DriftReport(
            panel_id=panel_id,
            score=round(score, 3),
            issues=tuple(issues),
            should_regenerate=should_regenerate,
        )
