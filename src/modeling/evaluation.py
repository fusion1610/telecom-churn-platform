from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_test_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> pd.Series:
    """Evaluate probability and threshold-based metrics on a held-out test set."""

    y_pred = (y_proba >= threshold).astype(int)

    return pd.Series(
        {
            "Threshold": threshold,
            "PR-AUC": average_precision_score(y_true, y_proba),
            "ROC-AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Customers Flagged": int(y_pred.sum()),
            "Coverage": float(y_pred.mean()),
        }
    )