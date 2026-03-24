# Customer churn prediction (Telco)

End-to-end **tabular ML** project: preprocessing, feature engineering, model comparison, serialized pipelines, and probability scoring for **retention prioritization**.

## Contents

| Path | Role |
|------|------|
| `notebooks/churn_analysis_refactored.ipynb` | Main narrative: business problem, metrics, model selection, ROC curve, and step-by-step reasoning. Regenerate with `python scripts/build_churn_notebook.py`. |
| `src/` | Library code: `data_preprocessing`, `feature_engineering`, `train_model`, `predict`. |
| `models/` | Saved `*_pipeline.joblib`, `metrics.json`, `model_comparison.csv` (see `models/README.md`). |
| `requirements.txt` | Python dependencies. |
| `reports/PROJECT_REPORT.md` | Full project report with embedded figures under `reports/figures/` (regenerate with `python scripts/generate_report_figures.py`). |

## Quick start

```bash
pip install -r requirements.txt
python -m src.train_model          # train + save all models to models/
```

Run the notebook from the `notebooks/` directory (or set project root on `sys.path` as in the first code cell).
