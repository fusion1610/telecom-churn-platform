import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_thresholds(
    y_true,
    y_proba,
    thresholds=None,
):
    """Evaluate classification performance across probability thresholds."""

    if thresholds is None:
        thresholds = np.arange(0.01, 0.51, 0.01)

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        rows.append(
            {
                "Threshold": float(threshold),
                "Precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "Coverage": float(y_pred.mean()),
                "Customers Flagged": int(y_pred.sum()),
            }
        )

    return pd.DataFrame(rows)