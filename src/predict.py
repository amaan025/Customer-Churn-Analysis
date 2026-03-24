"""Load a saved pipeline and score new customer rows."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def load_pipeline(model_path: str | Path | None = None) -> Pipeline:
    """
    Load a ``Pipeline`` saved by ``train_model.train_and_save_pipelines``.

    If ``model_path`` is None, loads the best model listed in ``metrics.json`` when present,
    otherwise ``random_forest_pipeline.joblib``.
    """
    if model_path is not None:
        return joblib.load(model_path)

    metrics_file = MODELS_DIR / "metrics.json"
    if metrics_file.exists():
        import json

        with open(metrics_file, encoding="utf-8") as f:
            meta = json.load(f)
        fname = meta.get("best_model_file", "random_forest_pipeline.joblib")
        path = MODELS_DIR / fname
        if path.exists():
            return joblib.load(path)

    fallback = MODELS_DIR / "random_forest_pipeline.joblib"
    if fallback.exists():
        return joblib.load(fallback)
    raise FileNotFoundError(
        f"No model found under {MODELS_DIR}. Train models first or pass model_path."
    )


def predict_proba_churn(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return probability of churn (class 1) for each row."""
    return pipeline.predict_proba(X)[:, 1]


def predict_labels(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return predicted class labels (0 = stay, 1 = churn)."""
    return pipeline.predict(X)
