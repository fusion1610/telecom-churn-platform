import pandas as pd
import pytest

from src.data.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    load_raw_data,
    prepare_dataset,
    prepare_features,
    prepare_target,
)


def test_raw_dataset_loads():
    df = load_raw_data()

    assert len(df) == 8453
    assert len(df.columns) == 14


def test_prepare_features_returns_baseline_features():
    df = load_raw_data()

    X = prepare_features(df)

    expected_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    assert list(X.columns) == expected_columns
    assert len(X) == 8453


def test_structural_missing_values_become_zero():
    df = load_raw_data()

    X = prepare_features(df)

    assert X["Not_Active_subscribers"].isna().sum() == 0
    assert X["Suspended_subscribers"].isna().sum() == 0


def test_arpu_missing_value_is_reconstructed():
    df = load_raw_data()

    X = prepare_features(df)

    assert X["ARPU"].isna().sum() == 0

    # The known missing ARPU observation reconstructs to 20.09.
    assert (X["ARPU"] == 20.09).sum() >= 1


def test_target_is_encoded_correctly():
    df = load_raw_data()

    y = prepare_target(df)

    assert y.name == TARGET_COLUMN
    assert y.dtype == "int8"
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 549


def test_prepare_dataset_preserves_row_count():
    df = load_raw_data()

    X, y = prepare_dataset(df)

    assert len(X) == len(y) == 8453


def test_preprocessor_can_fit_and_transform():
    df = load_raw_data()

    X, _ = prepare_dataset(df)

    preprocessor = build_preprocessor()

    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == 8453
    assert X_transformed.shape[1] > len(NUMERIC_FEATURES)


def test_unseen_categories_do_not_break_transform():
    df = load_raw_data()

    X, _ = prepare_dataset(df)

    train = X.iloc[:8000].copy()
    test = X.iloc[8000:].copy()

    # Introduce a category that cannot exist in the training data.
    test.loc[test.index[0], "CRM_PID_Value_Segment"] = (
        "Synthetic_Unseen_Category"
    )

    preprocessor = build_preprocessor()

    preprocessor.fit(train)

    transformed_test = preprocessor.transform(test)

    assert transformed_test.shape[0] == len(test)