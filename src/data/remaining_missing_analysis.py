from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/raw/telecom_churn.csv")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load and normalize the raw dataset."""

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    return df

def analyze_remaining_missing_values(df: pd.DataFrame) -> None:
    """Inspect the small number of remaining missing values."""

    columns = [
        "CRM_PID_Value_Segment",
        "Billing_ZIP",
        "ARPU",
    ]

    for column in columns:
        print("\n" + "=" * 70)
        print(f"COLUMN: {column}")
        print("=" * 70)

        missing = df[df[column].isna()]

        print(f"Missing rows: {len(missing)}")

        if missing.empty:
            continue

        display_columns = [
            "PID",
            "CRM_PID_Value_Segment",
            "EffectiveSegment",
            "Billing_ZIP",
            "KA_name",
            "Active_subscribers",
            "Not_Active_subscribers",
            "Suspended_subscribers",
            "Total_SUBs",
            "AvgMobileRevenue",
            "AvgFIXRevenue",
            "TotalRevenue",
            "ARPU",
            "CHURN",
        ]

        print("\nAffected rows:")
        print(missing[display_columns].to_string(index=False))

        print("\nChurn distribution:")
        print(missing["CHURN"].value_counts(dropna=False))

        # Check whether the single missing ARPU can be reconstructed.
    print("\n" + "=" * 70)
    print("ARPU RECONSTRUCTION CHECK")
    print("=" * 70)

    arpu_missing = df[df["ARPU"].isna()].copy()

    if not arpu_missing.empty:
        arpu_missing["calculated_arpu"] = (
            arpu_missing["TotalRevenue"]
            / arpu_missing["Total_SUBs"]
        )

        print(
            arpu_missing[
                [
                    "PID",
                    "TotalRevenue",
                    "Total_SUBs",
                    "ARPU",
                    "calculated_arpu",
                    "CHURN",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    df = load_dataset(DATASET_PATH)

    analyze_remaining_missing_values(df)