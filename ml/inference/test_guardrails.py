from ml.inference.guardrails import decide


def test_guardrail_returns_grade_for_valid_confident_vector() -> None:
    decision = decide([0.02, 0.03, 0.80, 0.10, 0.05])
    assert decision.status == "completed"
    assert decision.selected_grade == 2


def test_guardrail_abstains_for_low_confidence() -> None:
    decision = decide([0.20, 0.20, 0.20, 0.20, 0.20])
    assert decision.status == "abstained"
    assert decision.reason == "confidence_below_threshold"


def test_guardrail_abstains_for_out_of_distribution_input() -> None:
    decision = decide([0.02, 0.03, 0.80, 0.10, 0.05], out_of_distribution=True)
    assert decision.status == "abstained"
    assert decision.selected_grade is None
