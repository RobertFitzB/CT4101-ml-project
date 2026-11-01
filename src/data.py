"""Loading and feature-preparation helpers shared by the notebooks."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "bank-full.csv"

# duration is only known after the call has ended, so it would leak the outcome
LEAKY_COLUMNS = ["duration"]
# day of the month has no plausible link to a client subscribing. Combined with month
# it picks out specific calendar dates (with no year), which will not recur for future
# calls. See section 6 of notebooks/01_data_audit_and_baseline.ipynb for the evidence.
DROPPED_COLUMNS = ["day"]
TARGET = "y"


def load_raw(path=DATA_PATH):
    """Read bank-full.csv exactly as published (semicolon-separated)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/download_data.py` first."
        )
    return pd.read_csv(path, sep=";")


def prepare_features(df):
    """Row-wise cleaning that needs no fitting, so it is safe to apply before the split.

    Returns the feature table X and the binary target y (1 = subscribed).
    """
    df = df.drop(columns=LEAKY_COLUMNS + DROPPED_COLUMNS)

    # pdays = -1 means the client was never contacted in a previous campaign.
    # Make that explicit as its own flag so -1 is not read as a real number of days.
    df["previously_contacted"] = (df["pdays"] != -1).astype(int)
    df["pdays"] = df["pdays"].where(df["pdays"] != -1, 0)

    y = (df[TARGET] == "yes").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y
