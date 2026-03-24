"""Feature engineering for Telco churn (tenure groups, services, derived metrics)."""

from __future__ import annotations

import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin tenure into ``tenure_group`` categories."""
    out = df.copy()
    out["tenure_group"] = pd.cut(
        out["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48-72"],
    )
    return out


def encode_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode listed service columns as 1 if ``Yes``, else 0."""
    out = df.copy()
    for col in SERVICE_COLUMNS:
        if col in out.columns:
            out[col] = out[col].isin(["Yes"]).astype(int)
    return out


def add_derived_charges_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``TotalServices`` and ``AvgMonthlySpend`` (requires encoded services)."""
    out = df.copy()
    out["TotalServices"] = out[SERVICE_COLUMNS].sum(axis=1)
    out["AvgMonthlySpend"] = out["TotalCharges"] / out["tenure"].replace(0, 1)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full pipeline: tenure bins, binary service flags,
    total services, and average monthly spend.
    """
    return add_derived_charges_features(encode_service_columns(add_tenure_group(df)))


def prepare_modeling_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Map churn to 0/1 and return features ``X`` and target ``y``."""
    d = df.copy()
    y = d["Churn"].map({"No": 0, "Yes": 1})
    X = d.drop(columns=["Churn"])
    return X, y
