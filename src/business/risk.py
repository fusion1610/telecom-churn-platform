import numpy as np
import pandas as pd


def calculate_revenue_at_risk(
    churn_probability,
    total_revenue,
):
    """Calculate expected revenue exposure from predicted churn."""
    probability = np.asarray(churn_probability, dtype=float)
    revenue = np.asarray(total_revenue, dtype=float)

    if probability.shape != revenue.shape:
        raise ValueError(
            "churn_probability and total_revenue must have matching shapes."
        )

    if np.any(~np.isfinite(probability)):
        raise ValueError(
            "churn_probability must contain only finite values."
        )

    if np.any((probability < 0) | (probability > 1)):
        raise ValueError(
            "churn_probability must be between 0 and 1."
        )

    if np.any(~np.isfinite(revenue)):
        raise ValueError(
            "total_revenue must contain only finite values."
        )

    if np.any(revenue < 0):
        raise ValueError(
            "total_revenue cannot be negative."
        )

    return probability * revenue


def build_risk_table(
    data,
    churn_probability,
    threshold,
):
    """Build an observation-level churn risk and revenue table."""
    probability = np.asarray(churn_probability, dtype=float)

    if len(data) != len(probability):
        raise ValueError(
            "data and churn_probability must have matching lengths."
        )

    if "TotalRevenue" not in data.columns:
        raise ValueError(
            "data must contain a TotalRevenue column."
        )

    result = data.copy()

    result["Churn Probability"] = probability

    result["Revenue at Risk"] = calculate_revenue_at_risk(
        churn_probability=probability,
        total_revenue=result["TotalRevenue"].to_numpy(),
    )

    result["Retention Flag"] = (
        result["Churn Probability"] >= threshold
    )

    return result

def assign_risk_band(churn_probability):
    """Assign descriptive risk bands to churn probabilities."""
    probability = pd.Series(churn_probability).copy()

    if probability.isna().any():
        raise ValueError(
            "churn_probability cannot contain missing values."
        )

    if ((probability < 0) | (probability > 1)).any():
        raise ValueError(
            "churn_probability must be between 0 and 1."
        )

    return pd.Series(
        np.select(
            [
                probability < 0.05,
                probability < 0.07,
            ],
            [
                "Low",
                "Moderate",
            ],
            default="High",
        ),
        index=probability.index,
        name="Risk Band",
    )