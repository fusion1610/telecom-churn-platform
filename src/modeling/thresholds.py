from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_threshold(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict:
    """
    Evaluate classification performance at a single probability threshold.
    """

    y_pred = (y_proba >= threshold).astype(int)

    customers_flagged = int(y_pred.sum())

    total_customers = len(y_true)

    coverage = customers_flagged / total_customers

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    return {
        "Threshold": threshold,
        "Customers Flagged": customers_flagged,
        "Coverage": coverage,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

def evaluate_thresholds(y_true: pd.Series, y_proba: np.ndarray, thresholds: list[float] | np.ndarray) -> pd.DataFrame:
    """
    Evaluate model performance across multiple probability thresholds.
    """

    results = [
        evaluate_threshold(y_true=y_true, y_proba=y_proba, threshold=threshold)
        for threshold in thresholds
    ]

    return pd.DataFrame(results)
