# Saved churn models

Artifacts produced by `python -m src.train_model` or by the notebook section **Persist models** (which calls `train_and_save_pipelines`).

| File | Purpose |
|------|---------|
| `*_pipeline.joblib` | Full sklearn `Pipeline` (preprocessing + estimator). Load with `joblib.load` or `src.predict.load_pipeline()`. |
| `metrics.json` | Test-set metrics for each model and `best_model_key` / `best_model_file`. |
| `model_comparison.csv` | Same metrics in tabular form (sorted by `roc_auc` when saved). |
| `preprocessor_info.joblib` | Column name lists and an unfitted `ColumnTransformer` template (optional; each pipeline already embeds a fitted preprocessor). |

**Recommended inference:** use `src.predict.load_pipeline()` (defaults to the best model in `metrics.json`) and pass a `DataFrame` with the same columns as training after cleaning + `engineer_features()`.
