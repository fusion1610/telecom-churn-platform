import pandas as pd
import pytest

from src.data.validation import (
    normalize_column_names,
    validate_columns,
    validate_target,
)

def test_column_names_are_normalized():
    
    df = pd.DataFrame(columns=[' PID ', 'AvgMobileRevenue '])

    normalized = normalize_column_names(df)

    assert 'PID' in normalized.columns
    assert 'AvgMobileRevenue' in normalized.columns

def test_invalid_target_is_rejected():
    df = pd.DataFrame({
            'CHURN': [0, 1, 2]
        })

    with pytest.raises(ValueError):
        validate_target(df)

def test_missing_column_is_rejected():
    
    df = pd.DataFrame({
            "PID": [1, 2, 3]
        })

    with pytest.raises(ValueError):
        validate_columns(df)