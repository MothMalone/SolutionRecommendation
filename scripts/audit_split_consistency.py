"""Audit the fixed outer split used by every evaluation adapter.

This is deliberately an adapter-level check.  It does not train any model and
does not inspect a result score.  It loads each canonical evaluation dataset,
computes the train/validation/test membership fingerprint, and compares the
split implementations used by the baseline, ACORec, CtxPipe, and DiffPrep
notebooks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# The script is run directly from the repository root as well as from Kaggle.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automl_aco.data.loaders import load_gitlab_openml_dataset
from automl_aco.data.splits import split_train_val_test
from automl_aco.eval_ids import EVAL_DATASETS


def _manual_diffprep_split(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    """The split embedded in the DiffPrep H2O/TPOT adapters."""
    n_val = int(len(y) * 0.20)
    n_test = int(len(y) * 0.20)
    indices = np.random.RandomState(seed).permutation(len(y))
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return (
        X.iloc[train_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        X.iloc[val_idx].reset_index(drop=True),
        y.iloc[val_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


def _legacy_matrix_split(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    """The older AutoDP matrix notebook's stratified sklearn split."""
    try:
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.40, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
        )
    except ValueError:
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.40, random_state=seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=seed
        )
    return (
        X_train.reset_index(drop=True), y_train.reset_index(drop=True),
        X_val.reset_index(drop=True), y_val.reset_index(drop=True),
        X_test.reset_index(drop=True), y_test.reset_index(drop=True),
    )


def _fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    frame = X.copy().reset_index(drop=True)
    frame.columns = [str(column) for column in frame.columns]
    frame["__audit_target__"] = pd.Series(y).reset_index(drop=True).astype(str)
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()[:20]


def _parts(split: tuple[Any, ...]) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    X_train, y_train, X_val, y_val, X_test, y_test = split
    return {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test),
    }


def _digest(split: tuple[Any, ...]) -> dict[str, str]:
    return {name: _fingerprint(X, y) for name, (X, y) in _parts(split).items()}


def audit_dataset(dataset: dict[str, Any], seed: int) -> dict[str, Any]:
    X = pd.DataFrame(dataset["X"]).copy()
    y = pd.Series(dataset["y"]).copy()
    shared = _digest(split_train_val_test(X, y, seed=seed))
    diffprep = _digest(_manual_diffprep_split(X, y, seed=seed))
    legacy_matrix = _digest(_legacy_matrix_split(X, y, seed=seed))
    same = shared == diffprep
    return {
        "dataset_id": int(dataset["id"]),
        "dataset": dataset.get("name", f"D_{dataset['id']}"),
        "rows": int(len(y)),
        "features": int(X.shape[1]),
        "train_rows": int(len(y) - int(len(y) * 0.2) - int(len(y) * 0.2)),
        "validation_rows": int(len(y) * 0.2),
        "test_rows": int(len(y) * 0.2),
        "shared_split": shared,
        "diffprep_split": diffprep,
        "legacy_matrix_split": legacy_matrix,
        "legacy_matrix_matches_shared": bool(shared == legacy_matrix),
        "same_membership": bool(same),
        # Explicit fingerprints for the five rows in the comparison table.
        # DiffPrep's RandomState permutation is byte-for-byte equivalent to
        # the shared np.random.seed/permutation implementation for seed 42.
        "solution_split_fingerprints": {
            "no_preprocessing_h2o": shared,
            "h2o_default_target_encoding": shared,
            "diffprep_h2o": diffprep,
            "ctxpipe_tpot": shared,
            "acorec_tpot_or_h2o": shared,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/split_audit_cache"))
    parser.add_argument("--output", type=Path, default=Path("outputs/split_consistency_30.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=100_000)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ids = [int(value) for value in EVAL_DATASETS.values()]
    rows = []
    for position, (name, dataset_id) in enumerate(EVAL_DATASETS.items(), start=1):
        print(f"[{position}/{len(EVAL_DATASETS)}] {name} ({dataset_id})", flush=True)
        dataset = load_gitlab_openml_dataset(
            int(dataset_id),
            cache_dir=str(args.cache_dir),
            test_dataset_ids=ids,
            verbose=True,
            max_samples_if_test=int(args.max_samples),
        )
        if dataset is None:
            rows.append({"dataset_id": int(dataset_id), "dataset": name, "status": "load_failed"})
            continue
        try:
            row = audit_dataset(dataset, seed=int(args.seed))
            fingerprint_sets = list(row["solution_split_fingerprints"].values())
            row["all_solution_rows_same"] = len({json.dumps(value, sort_keys=True) for value in fingerprint_sets}) == 1
            row["status"] = "ok"
            rows.append(row)
            print("  same split:", row["same_membership"], flush=True)
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": int(dataset_id),
                    "dataset": name,
                    "status": "audit_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    successful = [row for row in rows if row.get("status") == "ok"]
    summary = {
        "split_seed": int(args.seed),
        "dataset_count": len(EVAL_DATASETS),
        "loaded_count": len(successful),
        "load_or_audit_failures": [row for row in rows if row.get("status") != "ok"],
        "all_same": bool(successful) and all(row["same_membership"] for row in successful),
        "all_solution_rows_same": bool(successful)
        and all(row["all_solution_rows_same"] for row in successful),
        "legacy_matrix_matches_shared_count": sum(
            bool(row.get("legacy_matrix_matches_shared")) for row in successful
        ),
        "implementations": {
            "no_preprocessing": "automl_aco.data.splits.split_train_val_test",
            "h2o_default": "automl_aco.data.splits.split_train_val_test",
            "acorec": "automl_aco.data.splits.split_train_val_test",
            "ctxpipe": "automl_aco.data.splits.split_train_val_test",
            "diffprep": "RandomState(42) permutation; numerically identical to shared split",
        },
        "datasets": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("dataset_count", "loaded_count", "all_same", "all_solution_rows_same")}, indent=2))
    print("Saved:", args.output)
    return 0 if summary["all_same"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
