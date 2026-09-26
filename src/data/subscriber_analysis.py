from pathlib import Path
import pandas as pd

DATASET_PATH = Path("data/raw/telecom_churn.csv")

def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load and normalize the raw dataset."""

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    return df

def analyze_subscriber_missingness(df: pd.DataFrame) -> None:
    """Investigate whether missing subscriber values behave like structural zeros."""

    subscriber_columns = [
        "Active_subscribers",
        "Not_Active_subscribers",
        "Suspended_subscribers",
    ]

    print("Subscriber missingness analysis")
    print("==============================")

    # 1. Missing counts
    print("\nMissing counts:")
    print(df[subscriber_columns].isna().sum())

    # 2. Number of missing subscriber fields per row
    missing_per_row = df[subscriber_columns].isna().sum(axis=1)

    print("\nRows by number of missing subscriber fields:")
    print(missing_per_row.value_counts().sort_index())

    # 3. Test each subscriber category independently
    print("\nRecoverability from Total_SUBs")
    print("------------------------------")

    for column in [
        "Not_Active_subscribers",
        "Suspended_subscribers",
    ]:
        other_columns = [
            c for c in subscriber_columns
            if c != column
        ]

        missing_mask = df[column].isna()

        recovered_value = (
            df.loc[missing_mask, "Total_SUBs"]
            - df.loc[missing_mask, other_columns].fillna(0).sum(axis=1)
        )

        print(f"\n{column}")
        print(f"Missing rows: {missing_mask.sum()}")

        if len(recovered_value) > 0:
            print(
                f"Recovered minimum: {recovered_value.min()}"
            )
            print(
                f"Recovered maximum: {recovered_value.max()}"
            )
            print(
                f"Recovered mean: {recovered_value.mean():.4f}"
            )
            print(
                f"Negative recovered values: "
                f"{(recovered_value < 0).sum()}"
            )

            print("Recovered value distribution:")
            print(recovered_value.value_counts().head(10))

    # 4. Missingness by churn status
    print("\nMissingness by churn status")
    print("--------------------------")

    churn_missingness = (
        df.groupby("CHURN")[subscriber_columns]
        .apply(lambda group: group.isna().sum())
    )

    print(churn_missingness)

    # 5. Rows where multiple subscriber fields are missing
    multiple_missing = df[missing_per_row >= 2]

    print("\nRows with 2+ missing subscriber fields:")
    print(len(multiple_missing))

    if not multiple_missing.empty:
        print(
            multiple_missing[
                subscriber_columns + ["Total_SUBs", "CHURN"]
            ].head(10)
        )


if __name__ == "__main__":
    df = load_dataset(DATASET_PATH)

    analyze_subscriber_missingness(df)