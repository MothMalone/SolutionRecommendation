"""Data helpers (schemas, splits, metafeatures)."""

from .schemas import Dataset, ensure_dataset
from .splits import split_train_val_test
from .metafeatures import extract_enhanced_metafeatures, build_metafeatures_matrix, load_metafeatures_csv
from .loaders import load_openml_dataset, load_kaggle_dataset, load_dummy_dataset, load_csv_dataset
from .openml_analysis import (
    compute_outlier_ratio,
    compute_scale_ratio,
    evaluate_dataset_issues,
    get_cc18_dataset_ids,
    load_all_datasets_metadata,
    select_imputation_candidates,
)

__all__ = [
    "Dataset",
    "ensure_dataset",
    "split_train_val_test",
    "extract_enhanced_metafeatures",
    "build_metafeatures_matrix",
    "load_metafeatures_csv",
    "load_openml_dataset",
    "load_kaggle_dataset",
    "load_dummy_dataset",
    "load_csv_dataset",
    "load_all_datasets_metadata",
    "select_imputation_candidates",
    "get_cc18_dataset_ids",
    "compute_scale_ratio",
    "compute_outlier_ratio",
    "evaluate_dataset_issues",
]
