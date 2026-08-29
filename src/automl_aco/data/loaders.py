"""Dataset loading helpers (OpenML, Kaggle, dummy, CSV)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import json
import os
import tempfile
import time
from pathlib import Path
import pandas as pd
import numpy as np
import requests
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
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

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
    force_task_type: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if isinstance(X, pd.DataFrame):
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype(str)

    X = X.dropna(axis=1, how="all")
    mask = ~pd.isna(y)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    if force_task_type == "regression":
        y = pd.to_numeric(y, errors="coerce")
        numeric_mask = y.notna()
        X = X.loc[numeric_mask].reset_index(drop=True)
        y = y.loc[numeric_mask].reset_index(drop=True)
        task_type = "regression"
    elif force_task_type == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)
        y = y.astype(int)
        task_type = "classification"
    else:
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


GITLAB_OPENML_ROOT = "https://gitlab.com/data/d/openml"

# Dataset 42932 (Avila) has no default target in its OpenML metadata.  Its
# Parquet table stores the class in column ``10`` and retains two indicators
# from its original provider split; those indicators must never be features.
GITLAB_TARGET_OVERRIDES = {42932: "10"}
GITLAB_IGNORE_COLUMN_OVERRIDES = {42932: {"train", "test"}}


def _metadata_attributes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _download_gitlab_openml_file(
    dataset_id: int,
    relative_path: str,
    destination: Path,
    *,
    retries: int = 3,
) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"{GITLAB_OPENML_ROOT}/{dataset_id}/-/raw/master/{relative_path}"
    errors = []
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(20, 300),
                allow_redirects=True,
                headers={"User-Agent": "ACORec-AutoDP36/1.0"},
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            if not temporary.exists() or temporary.stat().st_size == 0:
                raise IOError(f"empty response from {url}")
            os.replace(temporary, destination)
            return destination
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(" | ".join(errors))


def load_gitlab_openml_dataset(
    dataset_id: Any,
    *,
    test_dataset_ids: Optional[Iterable[int]] = None,
    regression_dataset_ids: Optional[Iterable[int]] = None,
    verbose: bool = False,
    cache_dir: Optional[str] = None,
    local_data_folder: Optional[str] = None,
    max_samples_if_test: int = 100000,
) -> Optional[Dict[str, Any]]:
    """Load an OpenML dataset from DataGit's GitLab Parquet mirror.

    A local ``<dataset_id>.csv`` in ``cache_dir`` is authoritative. This lets a
    mixed evaluation suite use exact frozen snapshots (including synthetic IDs
    such as DiffPrep's Google dataset 100000) while missing IDs still download
    from the mirror.

    Files are cached by dataset ID. The Parquet magic bytes are checked before
    parsing so Git LFS pointers, HTML error pages, and truncated downloads fail
    clearly instead of surfacing later as opaque ``ArrowInvalid`` errors.
    """
    did = int(dataset_id)
    cache_root = Path(cache_dir or tempfile.gettempdir()) / "openml_gitlab" / str(did)
    try:
        local_folder = local_data_folder or cache_dir
        if local_folder:
            local_df = _load_local_openml_csv(dataset_id=did, local_data_folder=local_folder)
            if local_df is not None:
                target = _detect_target_column(local_df)
                force_task_type = (
                    "regression"
                    if regression_dataset_ids
                    and did in {int(value) for value in regression_dataset_ids}
                    else None
                )
                prepared = _prepare_dataset_from_xy(
                    X=local_df.drop(columns=[target]).copy(),
                    y=local_df[target].copy(),
                    dataset_id=did,
                    test_dataset_ids=list(test_dataset_ids or []),
                    max_samples_if_test=max_samples_if_test,
                    max_samples_default=5000,
                    force_task_type=force_task_type,
                    verbose=verbose,
                )
                prepared["download_backend"] = "local-csv"
                return prepared

        metadata_path = _download_gitlab_openml_file(
            did, "dataset/metadata.json", cache_root / "metadata.json"
        )
        parquet_path = _download_gitlab_openml_file(
            did, "dataset/tables/data.pq", cache_root / "data.pq"
        )
        with parquet_path.open("rb") as source:
            header = source.read(4)
            source.seek(-4, os.SEEK_END)
            footer = source.read(4)
        if header != b"PAR1" or footer != b"PAR1":
            parquet_path.unlink(missing_ok=True)
            raise ValueError(
                f"GitLab dataset {did} is not a complete Parquet file "
                f"(header={header!r}, footer={footer!r})"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        description = metadata.get("data_set_description", metadata)
        targets = _metadata_attributes(
            GITLAB_TARGET_OVERRIDES.get(did)
            or description.get("default_target_attribute")
        )
        if len(targets) != 1:
            raise ValueError(f"expected one target for dataset {did}, got {targets}")

        frame = pd.read_parquet(parquet_path)
        frame.columns = frame.columns.astype(str)
        target = targets[0]
        if target not in frame.columns:
            raise KeyError(f"target {target!r} is absent from GitLab dataset {did}")
        excluded = {target}
        excluded.update(_metadata_attributes(description.get("ignore_attribute")))
        excluded.update(_metadata_attributes(description.get("row_id_attribute")))
        excluded.update(GITLAB_IGNORE_COLUMN_OVERRIDES.get(did, set()))
        features = [column for column in frame.columns if column not in excluded]
        force_task_type = (
            "regression"
            if regression_dataset_ids and did in {int(value) for value in regression_dataset_ids}
            else None
        )
        prepared = _prepare_dataset_from_xy(
            X=frame[features].copy(),
            y=frame[target].copy(),
            dataset_id=did,
            test_dataset_ids=list(test_dataset_ids or []),
            max_samples_if_test=max_samples_if_test,
            max_samples_default=5000,
            force_task_type=force_task_type,
            verbose=verbose,
        )
        prepared["download_backend"] = "gitlab-parquet"
        return prepared
    except Exception as exc:
        if verbose:
            print(f"Failed to load GitLab/OpenML dataset {did}: {type(exc).__name__}: {exc}")
        return None


def load_openml_dataset(
    dataset_id: Any,
    test_dataset_ids: Optional[list] = None,
    regression_dataset_ids: Optional[Iterable[int]] = None,
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
    force_task_type = (
        "regression"
        if regression_dataset_ids
        and int(dataset_id) in {int(value) for value in regression_dataset_ids}
        else None
    )

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
                force_task_type=force_task_type,
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
                        force_task_type=force_task_type,
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
            force_task_type=force_task_type,
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
                        force_task_type=force_task_type,
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
