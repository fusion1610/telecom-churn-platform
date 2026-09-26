from pathlib import Path
import pandas as pd

DATASET_PATH = Path('./data/raw/telecom_churn.csv')

EXPECTED_COLUMNS = {
    'PID',
    'CRM_PID_Value_Segment',
    'EffectiveSegment',
    'Billing_ZIP',
    'KA_name',
    'Active_subscribers',
    'Not_Active_subscribers',
    'Suspended_subscribers',
    'Total_SUBs',
    'AvgMobileRevenue',
    'AvgFIXRevenue',
    'TotalRevenue',
    'ARPU',
    'CHURN',
}

def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the raw telecom churn dataset."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    return pd.read_csv(file_path)

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize whitespace around column names."""
    
    df = df.copy()
    df.columns = df.columns.str.strip()

    return df

def validate_columns(df: pd.DataFrame) -> None:
    """Validate that the dataset contains the expected columns."""

    actual_columns = set(df.columns)

    missing_columns = EXPECTED_COLUMNS - actual_columns
    unexpected_columns = actual_columns - EXPECTED_COLUMNS

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns: {sorted(unexpected_columns)}"
        )

def validate_row_count(df: pd.DataFrame) -> None:
    """Validate that the dataset contains records."""

    if len(df) == 0:
        raise ValueError('Dataset contains zero rows')

def validate_target(df: pd.DataFrame) -> None:
    """Validate the churn target."""

    if 'CHURN' not in df.columns:
        raise ValueError('CHURN column is missing')

    unique_values = set(df["CHURN"].dropna().unique())

    expected_values = {"Yes", "No"}

    if not unique_values.issubset(expected_values):
        raise ValueError(
            f"CHURN contains unexpected values: {sorted(unique_values)}"
        )

def validate_dataset(file_path: Path) -> pd.DataFrame:
    """Run all schema-level validation checks."""

    df = load_dataset(file_path)

    df = normalize_column_names(df)

    validate_columns(df)
    validate_row_count(df)
    validate_target(df)

    return df

if __name__ == "__main__":
    dataframe = validate_dataset(DATASET_PATH)

    print("Dataset validation successful.")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print(f"Columns: {list(dataframe.columns)}")