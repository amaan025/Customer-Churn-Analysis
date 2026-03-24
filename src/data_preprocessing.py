"""Load raw Telco churn data and produce a cleaned dataframe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_CSV = PROCESSED_DIR / "cleaned_telco_churn.csv"


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Kaggle Telco Customer Churn CSV."""
    p = Path(path) if path is not None else RAW_DATA
    return pd.read_csv(p)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes, drop invalid rows and duplicates, remove customerID."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out = out.dropna()
    out = out.drop(columns=["customerID"], errors="ignore")
    out = out.drop_duplicates()
    return out.reset_index(drop=True)


def save_cleaned_data(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Write cleaned data to ``data/processed``."""
    p = Path(path) if path is not None else CLEANED_CSV
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def load_cleaned_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned CSV produced by :func:`save_cleaned_data`."""
    p = Path(path) if path is not None else CLEANED_CSV
    return pd.read_csv(p)
