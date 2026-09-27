from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data.validation import normalize_column_names


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telecom_churn.csv"


# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

TARGET_COLUMN = "CHURN"

CATEGORICAL_FEATURES = [
    "CRM_PID_Value_Segment",
    "EffectiveSegment",
]

NUMERIC_FEATURES = [
    "Active_subscribers",
    "Not_Active_subscribers",
    "Suspended_subscribers",
    "Total_SUBs",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "ARPU",
]

EXCLUDED_FEATURES = [
    "PID",
    "Billing_ZIP",
    "KA_name",
    "TotalRevenue",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the immutable raw telecom churn dataset.

    Parameters
    ----------
    path:
        Path to the raw CSV file.

    Returns
    -------
    pandas.DataFrame
        Raw dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    df = pd.read_csv(path)
    df = normalize_column_names(df)
    return df


# ---------------------------------------------------------------------------
# Raw-data preprocessing
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic feature preparation that does not learn parameters
    from the dataset.

    Operations:
    - validate required columns
    - convert structural subscriber missing values to zero
    - reconstruct the single missing ARPU value from TotalRevenue / Total_SUBs
    - select the baseline feature set

    The input DataFrame is copied and is never modified in place.
    """

    REQUIRED_PREPROCESSING_COLUMNS = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
        + ["TotalRevenue", TARGET_COLUMN]
    )

    missing = [
        col for col in REQUIRED_PREPROCESSING_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()

    # Structural missingness: these missing values were established during
    # Milestone 1 to represent zero rather than unknown quantities.
    data["Not_Active_subscribers"] = (
        data["Not_Active_subscribers"].fillna(0)
    )

    data["Suspended_subscribers"] = (
        data["Suspended_subscribers"].fillna(0)
    )

    # ARPU has one missing observation. Reconstruct it only when the
    # necessary source values are available.
    arpu_missing = data["ARPU"].isna()
    can_reconstruct_arpu = (
        arpu_missing
        & data["TotalRevenue"].notna()
        & data["Total_SUBs"].notna()
        & data["Total_SUBs"].ne(0)
    )

    reconstructed_arpu = (
        data.loc[can_reconstruct_arpu, "TotalRevenue"]
        / data.loc[can_reconstruct_arpu, "Total_SUBs"]
    )

    data.loc[can_reconstruct_arpu, "ARPU"] = (
        reconstructed_arpu.round(2)
    )

    # Return only the approved baseline feature set.
    return data[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()


# ---------------------------------------------------------------------------
# Target preparation
# ---------------------------------------------------------------------------

def prepare_target(df: pd.DataFrame) -> pd.Series:
    """
    Convert the raw CHURN target from Yes/No to 1/0.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    unexpected_values = set(df[TARGET_COLUMN].dropna().unique()) - {
        "Yes",
        "No",
    }

    if unexpected_values:
        raise ValueError(
            f"Unexpected target values: {unexpected_values}"
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "CHURN contains missing values."
        )

    return (
        df[TARGET_COLUMN]
        .map({"No": 0, "Yes": 1})
        .astype("int8")
        .rename(TARGET_COLUMN)
    )


# ---------------------------------------------------------------------------
# Scikit-learn preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor(
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build the reusable sklearn preprocessing transformer.

    Parameters
    ----------
    scale_numeric:
        If True, standardize numeric features. This is appropriate for
        Logistic Regression. Tree-based models can use an equivalent
        pipeline without scaling.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted preprocessing transformer.
    """

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Complete preparation helper
# ---------------------------------------------------------------------------

def prepare_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the raw dataset into X and y before train/test splitting.

    This function performs only deterministic preparation. Learned
    preprocessing such as imputation statistics, scaling parameters, and
    categorical encoding is handled by the sklearn transformer after
    the train/test split.
    """

    X = prepare_features(df)
    y = prepare_target(df)

    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    return X, y