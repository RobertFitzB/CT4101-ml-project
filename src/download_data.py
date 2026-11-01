"""Download the UCI Bank Marketing dataset and extract bank-full.csv into data/raw/.

Run from the repository root:
    python src/download_data.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TARGET_FILE = "bank-full.csv"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / TARGET_FILE
    if out_path.exists():
        print(f"{out_path} already exists, skipping download.")
        return

    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))

    # The UCI archive contains bank.zip, which in turn contains bank-full.csv
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
    out_path.write_bytes(inner.read(TARGET_FILE))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
