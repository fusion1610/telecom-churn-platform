import pandas as pd
import pytest

from src.business.risk import assign_priority_tier


def test_assign_priority_tier():
    revenue_at_risk = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0,
         6.0, 7.0, 8.0, 9.0, 10.0]
    )

    result = assign_priority_tier(revenue_at_risk)

    assert result.name == "Priority Tier"
    assert len(result) == len(revenue_at_risk)
    assert set(result).issubset(
        {"Standard", "Priority", "Critical"}
    )
    assert "Standard" in set(result)
    assert "Priority" in set(result)
    assert "Critical" in set(result)


def test_assign_priority_tier_rejects_negative_values():
    with pytest.raises(ValueError):
        assign_priority_tier(pd.Series([-1.0]))


def test_assign_priority_tier_rejects_missing_values():
    with pytest.raises(ValueError):
        assign_priority_tier(pd.Series([1.0, None]))


def test_assign_priority_tier_preserves_index():
    revenue_at_risk = pd.Series(
        [1.0, 2.0, 3.0],
        index=[10, 20, 30],
    )

    result = assign_priority_tier(revenue_at_risk)

    assert result.index.tolist() == [10, 20, 30]