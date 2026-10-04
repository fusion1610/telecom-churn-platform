import pandas as pd
import pytest
from pathlib import Path

from src.data.validation import (
    normalize_column_names,
    validate_columns,
    validate_target,
    validate_dataset
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

def make_valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PID": [1, 2],
            "CRM_PID_Value_Segment": ["Bronze", "Silver"],
            "EffectiveSegment": ["SOHO", "SME"],
            "Billing_ZIP": ["12345", "54321"],
            "KA_name": ["KA1", "KA2"],
            "Active_subscribers": [5, 10],
            "Not_Active_subscribers": [1, 2],
            "Suspended_subscribers": [0, 0],
            "Total_SUBs": [6, 12],
            "AvgMobileRevenue": [50.0, 100.0],
            "AvgFIXRevenue": [10.0, 20.0],
            "TotalRevenue": [60.0, 120.0],
            "ARPU": [10.0, 10.0],
            "CHURN": ["No", "Yes"],
        }
    )

def test_missing_columns_are_rejected():
    df = make_valid_dataframe().drop(columns=["ARPU"])

    with pytest.raises(
        ValueError,
        match="Missing columns",
    ):
        validate_columns(df)

def test_unexpected_columns_are_rejected():
    df = make_valid_dataframe()
    df["UnexpectedColumn"] = 123

    with pytest.raises(
        ValueError,
        match="Unexpected columns",
    ):
        validate_columns(df)

def test_non_empty_dataset_passes_row_count_validation():
    from src.data.validation import validate_row_count

    df = make_valid_dataframe()

    validate_row_count(df)

def test_empty_dataset_is_rejected():
    from src.data.validation import validate_row_count

    df = make_valid_dataframe().iloc[0:0]

    with pytest.raises(
        ValueError,
        match="Dataset contains zero rows",
    ):
        validate_row_count(df)

def test_missing_target_is_rejected():
    df = make_valid_dataframe().drop(columns=["CHURN"])

    with pytest.raises(
        ValueError,
        match="CHURN column is missing",
    ):
        validate_target(df)

def test_valid_target_passes():
    df = make_valid_dataframe()

    validate_target(df)

def test_load_dataset_rejects_missing_file(tmp_path):
    from src.data.validation import load_dataset

    missing_file = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Dataset not found",
    ):
        load_dataset(missing_file)

def test_validate_dataset_runs_all_checks(tmp_path):
    df = make_valid_dataframe()
    file_path = tmp_path / "telecom_churn.csv"

    df.to_csv(file_path, index=False)

    validated = validate_dataset(file_path)

    assert list(validated.columns) == list(df.columns)
    assert len(validated) == len(df)
    assert set(validated["CHURN"]) == {"Yes", "No"}
    assert validated["PID"].tolist() == df["PID"].tolist()