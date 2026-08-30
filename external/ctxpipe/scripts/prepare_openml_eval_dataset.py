#!/usr/bin/env python3
"""Prepare OpenML datasets in CtxPipe folder format.

Output layout per dataset:
<output_dir>/openml_<id>/data.csv
<output_dir>/openml_<id>/info.json

The loading/filtering behavior mirrors SoluRec's loader logic.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def _parse_dataset_ids(raw: str) -> List[int]:
    tokens = [t for t in re.split(r"[\s,]+", raw.strip()) if t]
    result: List[int] = []
    seen = set()
    for token in tokens:
        did = int(token)
        if did not in seen:
            result.append(did)
            seen.add(did)
    return result


def _load_ids_from_file(path: Path) -> List[int]:
    return _parse_dataset_ids(path.read_text())


def _fetch_openml_compat(dataset_id: int, as_frame: bool):
    try:
        return fetch_openml(data_id=dataset_id, as_frame=as_frame, parser="auto")
    except TypeError:
        # Older scikit-learn versions do not support parser arg.
        return fetch_openml(data_id=dataset_id, as_frame=as_frame)


def load_openml_dataset(
    dataset_id: int,
    test_dataset_ids: Optional[List[int]] = None,
    max_samples: int = 5000,
    max_test_samples: int = 100000,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    try:
        try:
            dataset = _fetch_openml_compat(dataset_id=dataset_id, as_frame=True)
        except ValueError as e:
            if "Sparse ARFF" in str(e):
                if verbose:
                    print(f"[{dataset_id}] retry with as_frame=False due to Sparse ARFF")
                dataset = _fetch_openml_compat(dataset_id=dataset_id, as_frame=False)
            else:
                raise

        X = dataset.data.copy()
        y = dataset.target

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

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

        sample_cap = (
            max_test_samples
            if test_dataset_ids is not None and dataset_id in test_dataset_ids
            else max_samples
        )
        if len(X) > sample_cap:
            X, y = shuffle(X, y, n_samples=sample_cap, random_state=42)
            X = X.reset_index(drop=True)
            y = pd.Series(y).reset_index(drop=True)

        if verbose:
            print(
                f"[{dataset_id}] loaded shape={X.shape} task={task_type} "
                f"classes={y.nunique() if task_type == 'classification' else 'N/A'}"
            )

        return {"id": dataset_id, "name": f"openml_{dataset_id}", "X": X, "y": y, "task_type": task_type}
    except Exception as e:
        if verbose:
            print(f"[{dataset_id}] failed: {e}")
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _iter_dataset_ids(args: argparse.Namespace) -> List[int]:
    ids: List[int] = []
    if args.dataset_ids:
        ids.extend(_parse_dataset_ids(args.dataset_ids))
    if args.dataset_ids_file:
        ids.extend(_load_ids_from_file(Path(args.dataset_ids_file)))

    deduped: List[int] = []
    seen = set()
    for did in ids:
        if did not in seen:
            deduped.append(did)
            seen.add(did)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OpenML eval datasets for CtxPipe.")
    parser.add_argument(
        "--dataset-ids",
        type=str,
        default="",
        help="Comma/space/newline separated OpenML dataset IDs.",
    )
    parser.add_argument(
        "--dataset-ids-file",
        type=str,
        default="",
        help="Path to text file containing OpenML dataset IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/diffprep_dataset",
        help="CtxPipe dataset folder root.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="target",
        help="Target column name written into data.csv.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Sample cap for non-test dataset IDs.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=100000,
        help="Sample cap for IDs treated as test datasets.",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip regression datasets (CtxPipe currently focuses on classification).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset folders if present.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-dataset details.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="",
        help="Optional explicit path for summary JSON.",
    )

    args = parser.parse_args()
    dataset_ids = _iter_dataset_ids(args)
    if not dataset_ids:
        raise SystemExit("No dataset IDs provided. Use --dataset-ids or --dataset-ids-file.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "requested_ids": dataset_ids,
        "output_dir": str(output_dir),
        "prepared": [],
        "skipped": [],
        "failed": [],
    }

    for dataset_id in dataset_ids:
        ds = load_openml_dataset(
            dataset_id=dataset_id,
            test_dataset_ids=dataset_ids,
            max_samples=args.max_samples,
            max_test_samples=args.max_test_samples,
            verbose=args.verbose,
        )
        if ds is None:
            summary["failed"].append({"dataset_id": dataset_id, "reason": "load_failed"})
            continue

        if args.skip_regression and ds["task_type"] != "classification":
            summary["skipped"].append(
                {"dataset_id": dataset_id, "reason": "regression_dataset"}
            )
            continue

        ds_dir = output_dir / f"openml_{dataset_id}"
        if ds_dir.exists() and not args.force:
            summary["skipped"].append(
                {
                    "dataset_id": dataset_id,
                    "reason": "already_exists (use --force to overwrite)",
                    "path": str(ds_dir),
                }
            )
            continue

        ds_dir.mkdir(parents=True, exist_ok=True)
        df = ds["X"].copy()
        y_series = pd.Series(ds["y"]).reset_index(drop=True)
        if ds["task_type"] == "classification":
            y_series = y_series.astype(int)
        df[args.target_column] = y_series

        data_csv_path = ds_dir / "data.csv"
        info_json_path = ds_dir / "info.json"
        df.to_csv(data_csv_path, index=False)

        info_payload = {
            "label": args.target_column,
            "openml_dataset_id": int(dataset_id),
            "task_type": ds["task_type"],
            "rows": int(df.shape[0]),
            "feature_columns": int(df.shape[1] - 1),
            "target_unique": int(y_series.nunique()),
            "source": "openml",
        }
        _write_json(info_json_path, info_payload)

        summary["prepared"].append(
            {
                "dataset_id": dataset_id,
                "dataset_name": f"openml_{dataset_id}",
                "path": str(ds_dir),
                "rows": int(df.shape[0]),
                "feature_columns": int(df.shape[1] - 1),
                "task_type": ds["task_type"],
            }
        )

    summary_path = Path(args.summary_path) if args.summary_path else output_dir / "_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Prepared: {len(summary['prepared'])}")
    print(f"Skipped:  {len(summary['skipped'])}")
    print(f"Failed:   {len(summary['failed'])}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
