from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


def evaluate_calibration(
    y_true: pd.Series,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    fraction_positive, mean_predicted_value = calibration_curve(
        y_true,
        y_proba,
        n_bins=n_bins,
        strategy="quantile",
    )

    return {
        "brier_score": brier_score_loss(y_true, y_proba),
        "fraction_positive": fraction_positive,
        "mean_predicted_value": mean_predicted_value,
    }


def create_calibration_table(
    y_true: pd.Series,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    result = evaluate_calibration(
        y_true=y_true,
        y_proba=y_proba,
        n_bins=n_bins,
    )

    return pd.DataFrame(
        {
            "Mean Predicted Probability": result["mean_predicted_value"],
            "Observed Churn Rate": result["fraction_positive"],
        }
    )


def build_calibrated_model(
    estimator,
    method: str,
    cv: int = 5,
) -> CalibratedClassifierCV:
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(
            "method must be either 'sigmoid' or 'isotonic'."
        )

    return CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=cv,
    )