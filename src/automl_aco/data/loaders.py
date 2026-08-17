"""Dataset loading helpers (OpenML, Kaggle, dummy, CSV)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml

try:
    import openml as openml_api  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openml_api = None


def load_dummy_dataset(dataset_id: Any, verbose: bool = False) -> Dict[str, Any]:
    if verbose:
        print(f"Loaded dataset {dataset_id}")
    return {"id": dataset_id, "name": f"D_{dataset_id}"}


def _coerce_task_target(y: pd.Series) -> Tuple[pd.Series, str]:
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    if y.nunique() > 50 and y.dtype.kind in "iufc":
        task_type = "regression"
    else:
        task_type = "classification"
        y = y.astype(int)
    return y, task_type


def _prepare_dataset_from_xy(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_id: Any,
    test_dataset_ids: Optional[list],
    max_samples_if_test: int,
    max_samples_default: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    if isinstance(X, pd.DataFrame):
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype(str)

    X = X.dropna(axis=1, how="all")
    mask = ~pd.isna(y)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    y, task_type = _coerce_task_target(y)

    if task_type == "classification":
        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= 5].index
        mask = y.isin(valid_classes)
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

    max_samples = max_samples_if_test if (test_dataset_ids and dataset_id in test_dataset_ids) else max_samples_default
    if len(X) > max_samples:
        X, y = shuffle(X, y, n_samples=max_samples, random_state=42)
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

    if verbose:
        print(f"Loaded dataset {dataset_id}")
        print(f"  Shape: {X.shape}")
        print(f"  Task: {task_type}")
        print(f"  Target classes: {len(np.unique(y)) if task_type=='classification' else 'N/A'}")

    return {"id": dataset_id, "name": f"D_{dataset_id}", "X": X, "y": y, "task_type": task_type}


def _detect_target_column(df: pd.DataFrame) -> str:
    for candidate in ("target", "class", "label", "y"):
        if candidate in df.columns:
            return candidate
    # Last column fallback for generic exported OpenML CSVs.
    return str(df.columns[-1])


def _load_local_openml_csv(dataset_id: Any, local_data_folder: str) -> Optional[pd.DataFrame]:
    dataset_id_str = str(dataset_id)
    candidates = (
        os.path.join(local_data_folder, f"{dataset_id_str}.csv"),
        os.path.join(local_data_folder, f"{dataset_id_str}.csv.zip"),
        os.path.join(local_data_folder, f"{dataset_id_str}.zip"),
    )
    for file_path in candidates:
        if not os.path.exists(file_path):
            continue
        try:
            return pd.read_csv(file_path, compression="infer")
        except Exception:
            continue
    return None


def _load_openml_dataset_direct_api(dataset_id: Any) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Load OpenML dataset via openml-python API when available."""
    if openml_api is None:
        return None

    dataset = openml_api.datasets.get_dataset(dataset_id, download_data=True)
    target_attr = getattr(dataset, "default_target_attribute", None)
    X, y, _categorical, _attributes = dataset.get_data(
        target=target_attr,
        dataset_format="dataframe",
    )
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if y is None:
        target_column = _detect_target_column(X)
        y = X[target_column].copy()
        X = X.drop(columns=[target_column]).copy()
    elif isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError(f"OpenML dataset {dataset_id} returned empty target frame")
        y = y.iloc[:, 0].copy()
    elif not isinstance(y, pd.Series):
        y = pd.Series(y)
    else:
        y = y.copy()

    return X.copy(), y


def load_openml_dataset(
    dataset_id: Any,
    test_dataset_ids: Optional[list] = None,
    verbose: bool = False,
    local_data_folder: Optional[str] = None,
    use_direct_api: bool = True,
    prefer_local: bool = True,
    max_samples_if_test: int = 100000,
) -> Optional[Dict[str, Any]]:
    """Load OpenML dataset, preferring a local ``<id>.csv`` when one is supplied.

    ``prefer_local`` (default True) makes a present local CSV authoritative instead of a mere
    fallback-on-exception. That ordering matters for correctness, not convenience: several
    evaluation datasets share an OpenML id with a DIFFERENT table of the same name. DiffPrep's
    `pol` is 15,000 rows; OpenML 722 fetched through the API and row-capped is 5,000. With the old
    ordering an API that happened to be reachable silently won, so the run scored the wrong data
    (observed on Kaggle: "722 [native] 5000 rows x 48 features"), and ids the API could reach but
    not parse failed outright instead of using the file that was sitting right there.

    Set ``prefer_local=False`` to restore the old API-first behaviour.
    """
    direct_api_error: Optional[Exception] = None
    sklearn_error: Optional[Exception] = None

    if prefer_local and local_data_folder:
        local_df = _load_local_openml_csv(dataset_id=dataset_id, local_data_folder=local_data_folder)
        if local_df is not None:
            target_column = _detect_target_column(local_df)
            if verbose:
                print(
                    f"Using local CSV for dataset {dataset_id} from {local_data_folder} "
                    f"(target={target_column}); OpenML API not consulted"
                )
            return _prepare_dataset_from_xy(
                X=local_df.drop(columns=[target_column]).copy(),
                y=local_df[target_column].copy(),
                dataset_id=dataset_id,
                test_dataset_ids=test_dataset_ids,
                max_samples_if_test=max_samples_if_test,
                max_samples_default=5000,
                verbose=verbose,
            )

    try:
        if use_direct_api:
            try:
                direct_loaded = _load_openml_dataset_direct_api(dataset_id=dataset_id)
                if direct_loaded is not None:
                    X_direct, y_direct = direct_loaded
                    return _prepare_dataset_from_xy(
                        X=X_direct,
                        y=y_direct,
                        dataset_id=dataset_id,
                        test_dataset_ids=test_dataset_ids,
                        max_samples_if_test=max_samples_if_test,
                        max_samples_default=5000,
                        verbose=verbose,
                    )
            except Exception as exc:
                direct_api_error = exc
                if verbose:
                    print(f"Direct OpenML API load failed for dataset {dataset_id}: {exc}")

        try:
            dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
        except ValueError as e:
            if "Sparse ARFF" in str(e):
                if verbose:
                    print(f"Retrying dataset {dataset_id} with as_frame=False...")
                dataset = fetch_openml(data_id=dataset_id, as_frame=False, parser="auto")
            else:
                raise e

        return _prepare_dataset_from_xy(
            X=dataset.data.copy(),
            y=dataset.target,
            dataset_id=dataset_id,
            test_dataset_ids=test_dataset_ids,
            max_samples_if_test=max_samples_if_test,
            max_samples_default=5000,
            verbose=verbose,
        )
    except Exception as e:
        sklearn_error = e
        if local_data_folder:
            local_df = _load_local_openml_csv(dataset_id=dataset_id, local_data_folder=local_data_folder)
            if local_df is not None:
                target_column = _detect_target_column(local_df)
                if verbose:
                    print(
                        f"OpenML API failed for dataset {dataset_id}; "
                        f"falling back to local file in {local_data_folder} (target={target_column})"
                    )
                X = local_df.drop(columns=[target_column]).copy()
                y = local_df[target_column].copy()
                try:
                    return _prepare_dataset_from_xy(
                        X=X,
                        y=y,
                        dataset_id=dataset_id,
                        test_dataset_ids=test_dataset_ids,
                        max_samples_if_test=max_samples_if_test,
                        max_samples_default=5000,
                        verbose=verbose,
                    )
                except Exception as local_exc:
                    if verbose:
                        print(f"Local fallback processing failed for dataset {dataset_id}: {local_exc}")
        if verbose:
            if direct_api_error is not None:
                print(f"Failed to load dataset {dataset_id} via sklearn fetch_openml: {sklearn_error}")
                print(f"Earlier direct OpenML API error for dataset {dataset_id}: {direct_api_error}")
            else:
                print(f"Failed to load dataset {dataset_id}: {e}")
        return None


def load_kaggle_dataset(
    dataset_id: Any,
    data_folder: str = "/kaggle/input/openml",
    target_column: str = "target",
    test_dataset_ids: Optional[list] = None,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load dataset from Kaggle input folder with error handling and automatic problem type detection."""
    try:
        file_path = os.path.join(data_folder, f"{dataset_id}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        dataset = pd.read_csv(file_path)

        if target_column not in dataset.columns:
            raise ValueError(f"No '{target_column}' column found in dataset {dataset_id}")

        return _prepare_dataset_from_xy(
            X=dataset.drop(columns=[target_column]).copy(),
            y=dataset[target_column].copy(),
            dataset_id=dataset_id,
            test_dataset_ids=test_dataset_ids,
            max_samples_if_test=8000,
            max_samples_default=5000,
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"Failed to load dataset {dataset_id}: {e}")
        return None


def load_csv_dataset(
    csv_path: str,
    target_column: str,
    dataset_id: Any = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found in dataset")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    return {"id": dataset_id, "name": f"D_{dataset_id}" if dataset_id is not None else "dataset", "X": X, "y": y}
