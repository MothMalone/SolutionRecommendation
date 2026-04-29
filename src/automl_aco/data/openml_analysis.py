"""OpenML-direct metadata and issue analysis helpers."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .loaders import load_openml_dataset


def _import_openml_api():
    try:
        import openml  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "openml package is required for direct OpenML metadata/suite access. "
            "Install with: pip install openml"
        ) from exc
    return openml


def load_all_datasets_metadata() -> pd.DataFrame:
    """Fetch OpenML dataset metadata as a DataFrame."""
    openml = _import_openml_api()
    df = openml.datasets.list_datasets(output_format="dataframe")
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if "did" not in df.columns:
        raise ValueError("OpenML metadata table does not contain 'did' column.")
    return df.copy()


def select_imputation_candidates(
    all_datasets_meta: pd.DataFrame,
    max_instances: int = 10_000,
) -> pd.DataFrame:
    """Select datasets likely useful for imputation benchmarks."""
    df = all_datasets_meta.copy()
    for col in ("NumberOfInstances", "NumberOfMissingValues"):
        if col not in df.columns:
            raise ValueError(f"Required metadata column missing: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    filtered = df[
        (df["NumberOfInstances"] <= float(max_instances))
        & (df["NumberOfMissingValues"] > 0)
    ]
    return filtered.sort_values(by="NumberOfMissingValues", ascending=False)


def get_cc18_dataset_ids() -> List[int]:
    """Return OpenML-CC18 dataset IDs (suite 99)."""
    openml = _import_openml_api()
    suite_cc18 = openml.study.get_suite(99)
    return [int(did) for did in suite_cc18.data]


def compute_scale_ratio(df: pd.DataFrame) -> float:
    """Compute max/min feature range ratio for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return 1.0
    ranges = (numeric.max(axis=0) - numeric.min(axis=0)).replace([np.inf, -np.inf], np.nan).dropna()
    positive = ranges[ranges > 0]
    if positive.empty:
        return 1.0
    min_range = float(positive.min())
    max_range = float(positive.max())
    if min_range <= 0:
        return 1.0
    return float(max_range / min_range)


def compute_outlier_ratio(df: pd.DataFrame, iqr_k: float = 1.5) -> float:
    """Compute row-level outlier ratio using IQR rule over numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return 0.0

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr
    outlier_mask = (numeric.lt(lower) | numeric.gt(upper)).fillna(False)
    return float(outlier_mask.any(axis=1).mean())


def evaluate_dataset_issues(
    dataset_ids: Iterable[int],
    loader: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    max_samples_default: int = 10_000,
    verbose: bool = False,
) -> pd.DataFrame:
    """Evaluate scaling/outlier diagnostics on a list of dataset IDs."""
    dataset_loader = loader or load_openml_dataset
    rows: List[Dict[str, Any]] = []

    for dataset_id in dataset_ids:
        loaded = dataset_loader(dataset_id, verbose=verbose)
        if loaded is None or "X" not in loaded:
            rows.append(
                {
                    "ID": int(dataset_id),
                    "Name": f"D_{dataset_id}",
                    "N_Samples": np.nan,
                    "N_Features": np.nan,
                    "Scale_Ratio": np.nan,
                    "Outlier_Ratio": np.nan,
                    "Load_Status": "failed",
                }
            )
            continue

        X = loaded["X"].copy()
        if len(X) > max_samples_default:
            X = X.sample(n=max_samples_default, random_state=42)

        rows.append(
            {
                "ID": int(dataset_id),
                "Name": loaded.get("name", f"D_{dataset_id}"),
                "N_Samples": int(X.shape[0]),
                "N_Features": int(X.shape[1]),
                "Scale_Ratio": compute_scale_ratio(X),
                "Outlier_Ratio": compute_outlier_ratio(X),
                "Load_Status": "ok",
            }
        )

    return pd.DataFrame(rows)

