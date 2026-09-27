import numpy as np
import pandas as pd

from src.modeling.thresholds import (
    evaluate_threshold,
    evaluate_thresholds,
)


def test_evaluate_threshold():
    y_true = pd.Series([0, 0, 1, 1])

    y_proba = np.array([
        0.10,
        0.20,
        0.80,
        0.90,
    ])

    result = evaluate_threshold(
        y_true=y_true,
        y_proba=y_proba,
        threshold=0.50,
    )

    assert result["Customers Flagged"] == 2
    assert result["Coverage"] == 0.5
    assert result["Precision"] == 1.0
    assert result["Recall"] == 1.0
    assert result["F1"] == 1.0


def test_zero_positive_predictions():
    y_true = pd.Series([0, 0, 1, 1])

    y_proba = np.array([
        0.10,
        0.20,
        0.30,
        0.40,
    ])

    result = evaluate_threshold(
        y_true=y_true,
        y_proba=y_proba,
        threshold=0.50,
    )

    assert result["Customers Flagged"] == 0
    assert result["Coverage"] == 0.0
    assert result["Precision"] == 0.0
    assert result["Recall"] == 0.0
    assert result["F1"] == 0.0


def test_evaluate_multiple_thresholds():
    y_true = pd.Series([0, 0, 1, 1])

    y_proba = np.array([
        0.10,
        0.20,
        0.80,
        0.90,
    ])

    results = evaluate_thresholds(
        y_true=y_true,
        y_proba=y_proba,
        thresholds=[0.25, 0.50, 0.75],
    )

    assert len(results) == 3
    assert list(results.columns) == [
        "Threshold",
        "Customers Flagged",
        "Coverage",
        "Precision",
        "Recall",
        "F1",
    ]