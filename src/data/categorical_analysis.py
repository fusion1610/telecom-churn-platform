from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/raw/telecom_churn.csv")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load and normalize the dataset."""

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    return df


def analyze_categorical_features(df: pd.DataFrame) -> None:
    """Analyze cardinality and category sizes."""

    categorical_columns = [
        "CRM_PID_Value_Segment",
        "EffectiveSegment",
        "Billing_ZIP",
        "KA_name",
    ]

    for column in categorical_columns:
        print(f"\n{column}")
        print("=" * len(column))

        print(f"Unique values: {df[column].nunique(dropna=True)}")
        print(f"Missing values: {df[column].isna().sum()}")

        summary = (
            df.groupby(column, dropna=False)
            .agg(
                rows=("CHURN", "size"),
                churned=("CHURN", lambda x: (x == "Yes").sum()),
            )
        )

        summary["churn_rate_pct"] = (
            summary["churned"] / summary["rows"] * 100
        )

        summary = summary.sort_values(
            ["rows", "churn_rate_pct"],
            ascending=[True, False],
        )

        print(summary.to_string())


if __name__ == "__main__":
    dataframe = load_dataset(DATASET_PATH)

    analyze_categorical_features(dataframe)