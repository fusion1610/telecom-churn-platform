import numpy as np
import pandas as pd

from src.modeling.evaluation import evaluate_test_predictions


def test_evaluate_test_predictions():
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.10, 0.20, 0.80, 0.90])

    result = evaluate_test_predictions(
        y_true=y_true,
        y_proba=y_proba,
        threshold=0.50,
    )

    assert result["Threshold"] == 0.50
    assert result["PR-AUC"] == 1.0
    assert result["ROC-AUC"] == 1.0
    assert result["Precision"] == 1.0
    assert result["Recall"] == 1.0
    assert result["F1"] == 1.0
    assert result["Customers Flagged"] == 2
    assert result["Coverage"] == 0.5


def test_zero_predictions_are_handled():
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.10, 0.20, 0.30, 0.40])

    result = evaluate_test_predictions(
        y_true=y_true,
        y_proba=y_proba,
        threshold=0.50,
    )

    assert result["Customers Flagged"] == 0
    assert result["Precision"] == 0.0
    assert result["Recall"] == 0.0
    assert result["F1"] == 0.0
    assert result["Coverage"] == 0.0