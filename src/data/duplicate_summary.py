from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/raw/telecom_churn.csv")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load and normalize the dataset."""

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    return df


def summarize_repeated_pids(df: pd.DataFrame) -> None:
    """Quantify and inspect repeated PID records."""

    pid_counts = df["PID"].value_counts()
    repeated_pids = pid_counts[pid_counts > 1]

    repeated_records = df[
        df["PID"].isin(repeated_pids.index)
    ].copy()

    print("Repeated PID summary")
    print("====================")

    print(f"Total rows: {len(df)}")
    print(f"Unique PID values: {df['PID'].nunique()}")
    print(f"Repeated PID values: {len(repeated_pids)}")
    print(f"Rows belonging to repeated PIDs: {len(repeated_records)}")

    print(
        f"Percentage of rows belonging to repeated PIDs: "
        f"{len(repeated_records) / len(df) * 100:.4f}%"
    )

    # ---------------------------------------------------------
    # Check whether repeated records are duplicates excluding PID
    # ---------------------------------------------------------

    non_pid_columns = [
        column
        for column in df.columns
        if column != "PID"
    ]

    repeated_without_pid_duplicates = (
        repeated_records
        .duplicated(
            subset=non_pid_columns,
            keep=False,
        )
    )

    print(
        "\nRepeated records that are identical "
        "excluding PID:"
    )
    print(repeated_without_pid_duplicates.sum())

    if repeated_without_pid_duplicates.any():
        print(
            repeated_records.loc[
                repeated_without_pid_duplicates
            ].to_string(index=False)
        )

    # ---------------------------------------------------------
    # Determine whether repeated PID groups contain
    # multiple churn labels
    # ---------------------------------------------------------

    churn_counts = (
        repeated_records
        .groupby("PID")["CHURN"]
        .nunique()
    )

    conflicting_churn = churn_counts[churn_counts > 1]

    print("\nRepeated PIDs with conflicting CHURN:")
    print(len(conflicting_churn))

    if not conflicting_churn.empty:
        print(conflicting_churn)

    # ---------------------------------------------------------
    # Final recommendation
    # ---------------------------------------------------------

    print("\nConclusion")
    print("----------")
    print(
        "PID will not be used as a predictive feature "
        "or treated as a unique customer key."
    )
    print(
        "Repeated PID observations will be retained "
        "as separate records."
    )


if __name__ == "__main__":
    dataframe = load_dataset(DATASET_PATH)

    summarize_repeated_pids(dataframe)