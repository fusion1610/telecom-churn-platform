import numpy as np
import pandas as pd

from src.modeling.calibration import (
    create_calibration_table,
    evaluate_calibration,
)


def test_evaluate_calibration_returns_brier_score():
    y_true = pd.Series([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6])

    result = evaluate_calibration(
        y_true,
        y_proba,
        n_bins=3,
    )

    assert "brier_score" in result
    assert 0.0 <= result["brier_score"] <= 1.0


def test_create_calibration_table():
    y_true = pd.Series([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6])

    table = create_calibration_table(
        y_true,
        y_proba,
        n_bins=3,
    )

    assert len(table) == 3

    assert list(table.columns) == [
        "Mean Predicted Probability",
        "Observed Churn Rate",
    ]