from pathlib import Path

import numpy as np
import pandas as pd


DATASET_PATH = Path("data/raw/telecom_churn.csv")


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load and normalize the dataset."""

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    return df


def check_revenue_relationships(df: pd.DataFrame) -> None:
    """Check deterministic relationships between revenue features."""
    print("Revenue relationship analysis")
    print("=============================")

    # ---------------------------------------------------------
    # TotalRevenue relationship
    # ---------------------------------------------------------

    calculated_total = (
        df["AvgMobileRevenue"]
        + df["AvgFIXRevenue"]
    )

    revenue_difference = (
        df["TotalRevenue"] - calculated_total
    ).abs()

    print("\nTotalRevenue relationship")
    print("-------------------------")
    print(
        "Rows where TotalRevenue != "
        "AvgMobileRevenue + AvgFIXRevenue:"
    )
    print((revenue_difference > 1e-6).sum())

    print(
        f"Maximum absolute difference: "
        f"{revenue_difference.max()}"
    )

    # ---------------------------------------------------------
    # ARPU relationship
    # ---------------------------------------------------------

    valid_arpu = (
        df["Total_SUBs"] > 0
    ) & df["ARPU"].notna()

    calculated_arpu = (
        df.loc[valid_arpu, "TotalRevenue"]
        / df.loc[valid_arpu, "Total_SUBs"]
    )

    arpu_difference = (
        df.loc[valid_arpu, "ARPU"]
        - calculated_arpu
    ).abs()

    print("\nARPU relationship")
    print("-----------------")
    print(
        "Rows where ARPU differs from "
        "TotalRevenue / Total_SUBs by > 0.01:"
    )
    print((arpu_difference > 0.01).sum())

    print(
        f"Maximum absolute difference: "
        f"{arpu_difference.max():.6f}"
    )

    print(
        f"Mean absolute difference: "
        f"{arpu_difference.mean():.6f}"
    )

def check_numeric_correlations(df: pd.DataFrame) -> None:
    """Display correlations between numeric features."""

    print("\nNumeric feature correlations")
    print("============================")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    correlation = (
        df[numeric_columns]
        .corr()
        .round(3)
    )

    print(correlation.to_string())


def check_target_relationships(df: pd.DataFrame) -> None:
    """Compare numeric features across churn classes."""

    print("\nNumeric features by CHURN")
    print("=========================")

    numeric_columns = [
        "Active_subscribers",
        "Not_Active_subscribers",
        "Suspended_subscribers",
        "Total_SUBs",
        "AvgMobileRevenue",
        "AvgFIXRevenue",
        "TotalRevenue",
        "ARPU",
    ]

    summary = (
        df.groupby("CHURN")[numeric_columns]
        .agg(["mean", "median"])
        .round(2)
    )

    print(summary.to_string())


def check_categorical_target_relationships(
    df: pd.DataFrame,
) -> None:
    """Inspect churn rates across categorical features."""

    print("\nCategorical feature churn rates")
    print("===============================")

    categorical_columns = [
        "CRM_PID_Value_Segment",
        "EffectiveSegment",
        "KA_name",
    ]

    for column in categorical_columns:
        print(f"\n{column}")
        print("-" * len(column))

        summary = (
            df.groupby(column, dropna=False)["CHURN"]
            .apply(
                lambda x: (x == "Yes").mean() * 100
            )
            .sort_values(ascending=False)
        )

        print(summary.round(2).to_string())


if __name__ == "__main__":
    dataframe = load_dataset(DATASET_PATH)

    check_revenue_relationships(dataframe)
    check_numeric_correlations(dataframe)
    check_target_relationships(dataframe)
    check_categorical_target_relationships(dataframe)