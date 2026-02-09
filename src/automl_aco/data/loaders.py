"""Dataset loading helpers (OpenML, Kaggle, dummy, CSV)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml


def load_dummy_dataset(dataset_id: Any, verbose: bool = False) -> Dict[str, Any]:
    if verbose:
        print(f"Loaded dataset {dataset_id}")
    return {"id": dataset_id, "name": f"D_{dataset_id}"}


def load_openml_dataset(dataset_id: Any, test_dataset_ids: Optional[list] = None, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Load OpenML dataset with error handling and automatic problem type detection."""
    try:
        try:
            dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
        except ValueError as e:
            if "Sparse ARFF" in str(e):
                if verbose:
                    print(f"Retrying dataset {dataset_id} with as_frame=False...")
                dataset = fetch_openml(data_id=dataset_id, as_frame=False, parser="auto")
            else:
                raise e

        X = dataset.data.copy()
        y = dataset.target

        if isinstance(X, pd.DataFrame):
            for col in X.select_dtypes(include=["object", "category"]).columns:
                X[col] = X[col].astype(str)

        if y.dtype == "object" or y.dtype.name == "category":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)

        X = X.dropna(axis=1, how="all")
        mask = ~pd.isna(y)
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

        if y.nunique() > 50 and y.dtype.kind in "iufc":
            task_type = "regression"
        else:
            task_type = "classification"
            y = y.astype(int)

        if task_type == "classification":
            class_counts = y.value_counts()
            valid_classes = class_counts[class_counts >= 5].index
            mask = y.isin(valid_classes)
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)

        max_samples = 100000 if (test_dataset_ids and dataset_id in test_dataset_ids) else 5000
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
    except Exception as e:
        if verbose:
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

        X = dataset.drop(columns=[target_column]).copy()
        y = dataset[target_column].copy()

        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype(str)

        if y.dtype == "object" or y.dtype.name == "category":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)

        X = X.dropna(axis=1, how="all")
        mask = ~pd.isna(y)
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

        if y.nunique() > 20 and y.dtype.kind in "iufc":
            task_type = "regression"
        else:
            task_type = "classification"
            y = y.astype(int)

        if task_type == "classification":
            class_counts = y.value_counts()
            valid_classes = class_counts[class_counts >= 5].index
            mask = y.isin(valid_classes)
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)

        max_samples = 8000 if (test_dataset_ids and dataset_id in test_dataset_ids) else 5000
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
