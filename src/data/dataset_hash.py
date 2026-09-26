from pathlib import Path
import hashlib

DATASET_PATH = Path('./data/raw/telecom_churn.csv')

def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

if __name__ == '__main__':
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    file_hash = calculate_sha256(DATASET_PATH)

    print(f"File: {DATASET_PATH}")
    print(f"SHA-256: {file_hash}")