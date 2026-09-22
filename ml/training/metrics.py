from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score


def classification_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(targets, predictions, weights="quadratic")),
    }


def expected_calibration_error(targets: list[int], probabilities: list[list[float]], bins: int = 10) -> float:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(targets, dtype=int)
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    error = 0.0
    for low, high in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins + 1)[1:]):
        selected = (confidence > low) & (confidence <= high if high < 1 else confidence <= 1)
        if selected.any():
            error += selected.mean() * abs((predictions[selected] == y[selected]).mean() - confidence[selected].mean())
    return float(error)

