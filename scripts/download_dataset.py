from pathlib import Path
from urllib.request import urlopen
import hashlib
import os
import tempfile
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = DATA_DIR / "telecom_churn.csv"

EXPECTED_SHA256 = (
    "36509d631e29a70e4a6e01c0deef42f47d35c112ea29d94b21b67b0a2e26c523"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    dataset_url = os.environ.get("DATASET_URL")

    if not dataset_url:
        raise RuntimeError(
            "DATASET_URL environment variable is not set."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from controlled CI source:")
    print(dataset_url)

    with tempfile.NamedTemporaryFile(suffix=".zip") as temporary_file:
        with urlopen(dataset_url, timeout=120) as response:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                temporary_file.write(chunk)

        temporary_file.flush()

        with zipfile.ZipFile(temporary_file.name) as archive:
            csv_files = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]

            if len(csv_files) != 1:
                raise RuntimeError(
                    f"Expected exactly one CSV in archive; "
                    f"found: {csv_files}"
                )

            with archive.open(csv_files[0]) as source:
                with OUTPUT_FILE.open("wb") as destination:
                    destination.write(source.read())

    actual_sha256 = sha256_file(OUTPUT_FILE)

    print(f"Dataset SHA-256: {actual_sha256}")

    if actual_sha256 != EXPECTED_SHA256:
        OUTPUT_FILE.unlink(missing_ok=True)

        raise RuntimeError(
            "Dataset checksum mismatch.\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {actual_sha256}"
        )

    print("Dataset checksum verified successfully.")


if __name__ == "__main__":
    main()