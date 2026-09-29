import pandas as pd
import pytest

from src.business.risk import assign_risk_band


def test_assign_risk_band():
    probabilities = pd.Series(
        [0.01, 0.049999, 0.05, 0.069999, 0.07, 0.10]
    )

    result = assign_risk_band(probabilities)

    assert result.tolist() == [
        "Low",
        "Low",
        "Moderate",
        "Moderate",
        "High",
        "High",
    ]


def test_assign_risk_band_rejects_invalid_probability():
    with pytest.raises(ValueError):
        assign_risk_band(pd.Series([-0.01]))

    with pytest.raises(ValueError):
        assign_risk_band(pd.Series([1.01]))


def test_assign_risk_band_preserves_index():
    probabilities = pd.Series(
        [0.01, 0.08],
        index=[10, 20],
    )

    result = assign_risk_band(probabilities)

    assert result.index.tolist() == [10, 20]