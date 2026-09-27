from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/raw/telecom_churn.csv")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load and normalize the dataset."""

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    return df


def analyze_target(df: pd.DataFrame) -> None:
    """Validate and summarize the CHURN target."""

    target = df['CHURN']

    print("CHURN target analysis")
    print("=====================")

    # ---------------------------------------------------------
    # 1. Missing target values
    # ---------------------------------------------------------

    missing_count = target.isna().sum()
    print("\nMissing CHURN values:")
    print(missing_count)

    # ---------------------------------------------------------
    # 2. Raw unique values
    # ---------------------------------------------------------

    print("\nUnique CHURN values:")
    print(target.unique())

    # ---------------------------------------------------------
    # 3. Whitespace / casing investigation
    # ---------------------------------------------------------

    print("\nRaw target value representation:")

    for value in target.dropna().unique():
        print(repr(value))

    # ---------------------------------------------------------
    # 4. Class distribution
    # ---------------------------------------------------------

    counts = target.value_counts(dropna=False)
    percentages = target.value_counts(
        normalize=True,
        dropna=False,
    ) * 100

    distribution = pd.DataFrame({
        "count": counts,
        "percentage": percentages,
    })

    print("\nTarget distribution:")
    print(distribution)

    # ---------------------------------------------------------
    # 5. Validate expected labels
    # ---------------------------------------------------------

    expected_values = {"Yes", "No"}

    actual_values = set(target.dropna().unique())

    unexpected_values = actual_values - expected_values

    print("\nUnexpected target values:")
    print(unexpected_values)

    # ---------------------------------------------------------
    # 6. Repeated PID churn conflicts
    # ---------------------------------------------------------

    pid_counts = df["PID"].value_counts()
    repeated_pids = pid_counts[pid_counts > 1]

    repeated_records = df[
        df["PID"].isin(repeated_pids.index)
    ]

    churn_per_pid = (
        repeated_records
        .groupby("PID")["CHURN"]
        .nunique()
    )

    conflicting_pids = churn_per_pid[
        churn_per_pid > 1
    ]

    print("\nRepeated PIDs with conflicting CHURN:")
    print(len(conflicting_pids))

    if not conflicting_pids.empty:
        print(conflicting_pids)

    # ---------------------------------------------------------
    # 7. Final assessment
    # ---------------------------------------------------------

    print("\nTarget validation summary")
    print("-------------------------")

    if missing_count == 0 and not unexpected_values:
        print("Target contains valid non-missing Yes/No labels.")
    else:
        print("Target requires additional investigation.")


if __name__ == "__main__":
    dataframe = load_dataset(DATASET_PATH)

    analyze_target(dataframe)