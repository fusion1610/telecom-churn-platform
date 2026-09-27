import pandas as pd

from src.data.split import create_train_test_split


def make_test_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "feature": range(100),
        }
    )

    y = pd.Series(
        ["Yes"] * 20 + ["No"] * 80,
        name="CHURN",
    )

    return X, y


def test_split_preserves_total_row_count():
    X, y = make_test_data()

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_split_is_stratified():
    X, y = make_test_data()

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    train_rate = (y_train == "Yes").mean()
    test_rate = (y_test == "Yes").mean()

    assert abs(train_rate - test_rate) < 0.02


def test_split_is_reproducible():
    X, y = make_test_data()

    split_1 = create_train_test_split(X, y)
    split_2 = create_train_test_split(X, y)

    X_train_1, X_test_1, y_train_1, y_test_1 = split_1
    X_train_2, X_test_2, y_train_2, y_test_2 = split_2

    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)


def test_split_contains_both_target_classes():
    X, y = make_test_data()

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    assert set(y_train) == {"Yes", "No"}
    assert set(y_test) == {"Yes", "No"}


def test_split_preserves_feature_columns():
    X, y = make_test_data()

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    assert list(X_train.columns) == list(X.columns)
    assert list(X_test.columns) == list(X.columns)


def test_target_name_is_preserved():
    X, y = make_test_data()

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    assert y_train.name == "CHURN"
    assert y_test.name == "CHURN"