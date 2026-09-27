from pathlib import Path

import pandas as pd


DATASET_PATH = Path('data/raw/telecom_churn.csv')


def load_dataset(file_path: Path) -> pd.DataFrame:
    '''Load and normalize the raw dataset.'''

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    return df

def analyze_duplicates(df: pd.DataFrame) -> None:
    '''Investigate exact duplicates and repeated PIDs.'''

    print('Duplicate analysis')
    print('==================')

    # ---------------------------------------------------------
    # 1. Exact duplicate rows
    # ---------------------------------------------------------

    exact_duplicates = df.duplicated().sum()

    print('\nExact duplicate rows:')
    print(exact_duplicates)

    # ---------------------------------------------------------
    # 2. Duplicate PID values
    # ---------------------------------------------------------

    pid_counts = df['PID'].value_counts()

    duplicate_pids = pid_counts[pid_counts > 1]

    print('\nRepeated PID values:')
    print(duplicate_pids.value_counts().sort_index())

    # ---------------------------------------------------------
    # 3. Inspect all repeated PID records
    # ---------------------------------------------------------

    repeated_pid_values = duplicate_pids.index

    repeated_records = df[df['PID'].isin(repeated_pid_values)].sort_values('PID')

    print('\nRepeated PID records:')
    print(repeated_records.to_string(index=False))

    # ---------------------------------------------------------
    # 4. Compare whether repeated PID records are identical
    # ---------------------------------------------------------

    columns_without_pid = [column for column in df.columns if column != 'PID']

    grouped = repeated_records.groupby('PID')

    print('\nRepeated PID comparison')
    print('-----------------------')

    for pid, group in grouped:
        unique_records = group[columns_without_pid].drop_duplicates()

        print(
            f'PID: {pid}'
            f'{len(group)} rows,'
            f'{len(unique_records)} unique non-PID records'
        )

    # ---------------------------------------------------------
    # 5. Churn consistency
    # ---------------------------------------------------------

    churn_consistency = grouped['CHURN'].nunique()

    inconsistent_churn = churn_consistency[churn_consistency > 1]

    print('\nRepeated PIDs with conflicting CHURN:')
    print(len(inconsistent_churn))

    if not inconsistent_churn.empty:
        print(inconsistent_churn)

    # ---------------------------------------------------------
    # 6. Segment consistency
    # ---------------------------------------------------------

    for column in ['CRM_PID_Value_Segment','EffectiveSegment','KA_name']:
        consistency = grouped[column].nunique(dropna=False)

        inconsistent = consistency[consistency > 1]

        print(
            f"\nRepeated PIDs with conflicting {column}:"
        )
        print(len(inconsistent))

        if not inconsistent.empty:
            print(inconsistent)


if __name__ == "__main__":
    dataframe = load_dataset(DATASET_PATH)

    analyze_duplicates(dataframe)
