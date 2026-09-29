import pandas as pd
import pytest

from src.business.risk import (
    assign_retention_action,
    assign_retention_urgency,
)


def test_assign_retention_action():
    risk_band = pd.Series(
        ["Low", "Moderate", "High"],
        index=[10, 20, 30],
    )
    priority_tier = pd.Series(
        ["Standard", "Priority", "Critical"],
        index=[10, 20, 30],
    )

    result = assign_retention_action(
        risk_band,
        priority_tier,
    )

    assert result.index.tolist() == [10, 20, 30]
    assert result.name == "Retention Action"
    assert result.iloc[0] == "Normal lifecycle management"
    assert result.iloc[1] == "Proactive retention review"
    assert result.iloc[2] == "Highest-priority retention review"


def test_assign_retention_urgency():
    risk_band = pd.Series(["Low", "Moderate", "High"])
    priority_tier = pd.Series(["Standard", "Priority", "Critical"])

    result = assign_retention_urgency(
        risk_band,
        priority_tier,
    )

    assert result.tolist() == [
        "Low",
        "High",
        "Very High",
    ]


def test_retention_action_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        assign_retention_action(
            pd.Series(["Low"]),
            pd.Series(["Standard", "Critical"]),
        )


def test_retention_urgency_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        assign_retention_urgency(
            pd.Series(["Low"]),
            pd.Series(["Standard", "Critical"]),
        )