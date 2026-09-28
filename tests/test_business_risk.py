import numpy as np
import pandas as pd
import pytest

from src.business.risk import (
    calculate_revenue_at_risk,
    build_risk_table,
)


def test_calculate_revenue_at_risk():
    probabilities = np.array([0.10, 0.50, 1.00])
    revenue = np.array([100.0, 200.0, 500.0])

    result = calculate_revenue_at_risk(
        churn_probability=probabilities,
        total_revenue=revenue,
    )

    np.testing.assert_allclose(
        result,
        np.array([10.0, 100.0, 500.0]),
    )


def test_calculate_revenue_at_risk_rejects_invalid_probability():
    with pytest.raises(ValueError):
        calculate_revenue_at_risk(
            churn_probability=np.array([-0.1]),
            total_revenue=np.array([100.0]),
        )

    with pytest.raises(ValueError):
        calculate_revenue_at_risk(
            churn_probability=np.array([1.1]),
            total_revenue=np.array([100.0]),
        )


def test_calculate_revenue_at_risk_rejects_negative_revenue():
    with pytest.raises(ValueError):
        calculate_revenue_at_risk(
            churn_probability=np.array([0.5]),
            total_revenue=np.array([-100.0]),
        )


def test_build_risk_table():
    data = pd.DataFrame(
        {
            "PID": ["A", "B", "C"],
            "TotalRevenue": [100.0, 200.0, 300.0],
            "CHURN": [0, 1, 0],
        }
    )

    probabilities = np.array([0.02, 0.07, 0.10])

    result = build_risk_table(
        data=data,
        churn_probability=probabilities,
        threshold=0.07,
    )

    assert list(result["PID"]) == ["A", "B", "C"]

    np.testing.assert_allclose(
        result["Churn Probability"],
        [0.02, 0.07, 0.10],
    )

    np.testing.assert_allclose(
        result["Revenue at Risk"],
        [2.0, 14.0, 30.0],
    )

    assert result["Retention Flag"].tolist() == [False, True, True]


def test_build_risk_table_requires_matching_lengths():
    data = pd.DataFrame(
        {
            "PID": ["A", "B"],
            "TotalRevenue": [100.0, 200.0],
        }
    )

    with pytest.raises(ValueError):
        build_risk_table(
            data=data,
            churn_probability=np.array([0.5]),
            threshold=0.07,
        )