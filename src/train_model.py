"""Train sklearn pipelines (preprocess + classifier) and persist them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def _infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    return categorical_cols, numerical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols, numerical_cols = _infer_column_types(X)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def logistic_regression_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Return an unfitted ``Pipeline`` (column transformer + :class:`~sklearn.linear_model.LogisticRegression`)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def train_test_prepare(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> list[str]:
    categorical_cols, numerical_cols = _infer_column_types(X)
    ohe = preprocessor.named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out(categorical_cols)
    return numerical_cols + list(cat_features)


def get_models() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "xgboost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
        ),
    }


def _display_name(key: str) -> str:
    return key.replace("_", " ").title()


def compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    models: dict[str, Any] | None = None,
) -> pd.DataFrame:
    models = models or get_models()
    results: list[dict[str, Any]] = []
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        results.append(
            {
                "Model": _display_name(name),
                "key": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob),
            }
        )
    return pd.DataFrame(results)


def train_and_save_pipelines(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    output_dir: Path | None = None,
    models: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Fit each model, save ``*.joblib`` under ``models/``, write ``metrics.json`` and ``model_comparison.csv``."""
    output_dir = output_dir or MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = build_preprocessor(X_train)
    models = models or get_models()
    fitted: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []

    for key, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        fitted[key] = pipeline
        joblib.dump(pipeline, output_dir / f"{key}_pipeline.joblib")

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model_key": key,
                "display_name": _display_name(key),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
            }
        )

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    df.to_csv(output_dir / "model_comparison.csv", index=False)

    best_key = df.iloc[0]["model_key"]
    meta = {
        "best_model_key": best_key,
        "best_model_file": f"{best_key}_pipeline.joblib",
        "metrics": rows,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    joblib.dump(
        {
            "preprocessor_template": build_preprocessor(X_train),
            "categorical_cols": _infer_column_types(X_train)[0],
            "numerical_cols": _infer_column_types(X_train)[1],
        },
        output_dir / "preprocessor_info.joblib",
    )

    return df, fitted


def run_full_training_from_cleaned_csv(
    cleaned_csv: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Load cleaned data, engineer features, split, train, and save all pipelines."""
    from .data_preprocessing import CLEANED_CSV, load_cleaned_data
    from .feature_engineering import engineer_features, prepare_modeling_xy

    path = cleaned_csv or CLEANED_CSV
    df = load_cleaned_data(path)
    df = engineer_features(df)
    X, y = prepare_modeling_xy(df)
    X_train, X_test, y_train, y_test = train_test_prepare(X, y)
    return train_and_save_pipelines(
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    out, _ = run_full_training_from_cleaned_csv()
    print(out.to_string(index=False))
    print("Saved to", MODELS_DIR)
