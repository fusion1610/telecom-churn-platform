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

def assign_priority_tier(
    revenue_at_risk,
    priority_quantiles=(0.75, 0.90),
):
    revenue = pd.Series(revenue_at_risk).copy()

    if revenue.isna().any():
        raise ValueError("revenue_at_risk cannot contain missing values.")

    if (revenue < 0).any():
        raise ValueError("revenue_at_risk cannot be negative.")

    if len(priority_quantiles) != 2:
        raise ValueError("priority_quantiles must contain exactly two values.")

    q75, q90 = revenue.quantile(priority_quantiles).to_numpy()

    return pd.Series(
        np.select(
            [
                revenue <= q75,
                revenue <= q90,
            ],
            [
                "Standard",
                "Priority",
            ],
            default="Critical",
        ),
        index=revenue.index,
        name="Priority Tier",
    )

def assign_retention_action(risk_band, priority_tier):
    risk = pd.Series(risk_band)
    priority = pd.Series(priority_tier)

    if len(risk) != len(priority):
        raise ValueError(
            "risk_band and priority_tier must have the same length."
        )

    action_map = {
        ("Low", "Standard"): "Normal lifecycle management",
        ("Low", "Priority"): "Monitor customer and review revenue exposure",
        ("Low", "Critical"): "Revenue-focused review despite low predicted churn",
        ("Moderate", "Standard"): "Add to monitoring workflow",
        ("Moderate", "Priority"): "Proactive retention review",
        ("Moderate", "Critical"): "Prioritized retention review",
        ("High", "Standard"): "Targeted retention review",
        ("High", "Priority"): "Proactive retention intervention",
        ("High", "Critical"): "Highest-priority retention review",
    }

    result = pd.Series(
        [
            action_map.get((r, p), "Review required")
            for r, p in zip(risk, priority)
        ],
        index=risk.index,
        name="Retention Action",
    )

    return result

def assign_retention_urgency(risk_band, priority_tier):
    risk = pd.Series(risk_band)
    priority = pd.Series(priority_tier)

    if len(risk) != len(priority):
        raise ValueError(
            "risk_band and priority_tier must have the same length."
        )

    urgency_map = {
        ("Low", "Standard"): "Low",
        ("Low", "Priority"): "Medium",
        ("Low", "Critical"): "Medium",
        ("Moderate", "Standard"): "Medium",
        ("Moderate", "Priority"): "High",
        ("Moderate", "Critical"): "High",
        ("High", "Standard"): "High",
        ("High", "Priority"): "Very High",
        ("High", "Critical"): "Very High",
    }

    return pd.Series(
        [
            urgency_map.get((r, p), "Review")
            for r, p in zip(risk, priority)
        ],
        index=risk.index,
        name="Retention Urgency",
    )