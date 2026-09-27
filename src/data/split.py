import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "CHURN"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a stratified train/test split.

    The target distribution is preserved across train and test sets.
    """

    if TARGET_COLUMN != y.name:
        raise ValueError(
            f"Expected target named '{TARGET_COLUMN}', "
            f"but received '{y.name}'."
        )

    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def report_split(
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Print target distribution for train and test sets."""

    print(f"Training rows: {len(y_train)}")
    print(f"Test rows: {len(y_test)}")
    print()

    print("Training target distribution:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True).mul(100))
    print()

    print("Test target distribution:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True).mul(100))

if __name__ == "__main__":
    from pathlib import Path

    from src.data.preprocessing import (
        load_raw_data,
        prepare_dataset,
    )

    raw_path = Path("data/raw/telecom_churn.csv")

    df = load_raw_data(raw_path)

    X, y = prepare_dataset(df)

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y,
    )

    print(f"Total rows: {len(X)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print()

    report_split(y_train, y_test)