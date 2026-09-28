import numpy as np

from src.modeling.thresholds import evaluate_thresholds


def test_evaluate_thresholds_returns_expected_columns():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.9])

    result = evaluate_thresholds(
        y_true=y_true,
        y_proba=y_proba,
        thresholds=[0.5],
    )

    assert list(result.columns) == [
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "Coverage",
        "Customers Flagged",
    ]

    assert len(result) == 1