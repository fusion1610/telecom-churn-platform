from pathlib import Path
import pandas as pd

DATASET_PATH = Path('./data/raw/telecom_churn.csv')

def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load and normalize the raw dataset."""

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    return df

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages."""

    report = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_percentage': df.isna().mean() * 100
    })

    return report.sort_values(
        by="missing_count",
        ascending=False,
    )

def check_subscriber_relationship(df: pd.DataFrame) -> None:
    """Check whether Total_SUBs equals the sum of subscriber categories."""

    subscriber_columns = [
        "Active_subscribers",
        "Not_Active_subscribers",
        "Suspended_subscribers",
    ]

    subscriber_sum = df[subscriber_columns].fillna(0).sum(axis=1)

    mismatches = df["Total_SUBs"] != subscriber_sum

    mismatch_count = mismatches.sum()

    print("\nSubscriber relationship check")
    print("--------------------------------")
    print(f"Rows checked: {len(df)}")
    print(f"Mismatches: {mismatch_count}")

    if mismatch_count > 0:
        print("\nExample mismatches:")
        print(
            df.loc[
                mismatches,
                subscriber_columns + ["Total_SUBs"]
            ].head(10)
        )

if __name__ == "__main__":
    df = load_dataset(DATASET_PATH)

    print("Missing-value report")
    print("====================")
    print(missing_value_report(df).to_string())

    check_subscriber_relationship(df)