
"""Download the original UCI diabetes readmission dataset."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
from tempfile import TemporaryDirectory

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "diabetic_data.csv"
MAPPING_FILE = ROOT / "IDS_mapping.csv"


def main():
    if DATA_FILE.exists() and MAPPING_FILE.exists():
        print("Dataset files already exist.")
        return

    with TemporaryDirectory() as temp:
        archive = Path(temp) / "diabetes.zip"

        print("Downloading dataset from UCI...")
        urlretrieve(DATA_URL, archive)

        with ZipFile(archive) as z:
            for filename in ("diabetic_data.csv", "IDS_mapping.csv"):
                with z.open(filename) as source:
                    destination = ROOT / filename
                    destination.write_bytes(source.read())
                    print(f"Extracted: {destination.name}")

    print("Dataset setup complete.")


if __name__ == "__main__":
    main()
