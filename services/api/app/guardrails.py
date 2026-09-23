from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailDecision:
    status: str
    selected_grade: int | None
    confidence: float
    reason: str | None = None


def decide(probabilities: list[float], *, minimum_confidence: float = 0.55) -> GuardrailDecision:
    if len(probabilities) != 5 or any(value < 0 or value > 1 for value in probabilities):
        return GuardrailDecision("abstained", None, 0.0, "invalid_probability_vector")
    total = sum(probabilities)
    if not 0.99 <= total <= 1.01:
        return GuardrailDecision("abstained", None, 0.0, "probabilities_not_normalized")
    confidence = max(probabilities)
    selected = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    if confidence < minimum_confidence:
        return GuardrailDecision("abstained", None, confidence, "confidence_below_threshold")
    return GuardrailDecision("completed", selected, confidence)
