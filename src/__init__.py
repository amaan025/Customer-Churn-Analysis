"""Telco churn: preprocessing, feature engineering, training, and prediction."""

from .data_preprocessing import (
    CLEANED_CSV,
    RAW_DATA,
    clean_dataframe,
    load_cleaned_data,
    load_raw_data,
    save_cleaned_data,
)
from .feature_engineering import (
    SERVICE_COLUMNS,
    add_derived_charges_features,
    add_tenure_group,
    encode_service_columns,
    engineer_features,
    prepare_modeling_xy,
)
from .predict import load_pipeline, predict_labels, predict_proba_churn
from .train_model import (
    build_preprocessor,
    compare_models,
    get_feature_names_after_preprocessing,
    get_models,
    logistic_regression_pipeline,
    train_and_save_pipelines,
    train_test_prepare,
)

__all__ = [
    "CLEANED_CSV",
    "RAW_DATA",
    "SERVICE_COLUMNS",
    "add_derived_charges_features",
    "add_tenure_group",
    "encode_service_columns",
    "build_preprocessor",
    "clean_dataframe",
    "compare_models",
    "engineer_features",
    "get_feature_names_after_preprocessing",
    "get_models",
    "logistic_regression_pipeline",
    "load_cleaned_data",
    "load_pipeline",
    "load_raw_data",
    "prepare_modeling_xy",
    "predict_labels",
    "predict_proba_churn",
    "save_cleaned_data",
    "train_and_save_pipelines",
    "train_test_prepare",
]
