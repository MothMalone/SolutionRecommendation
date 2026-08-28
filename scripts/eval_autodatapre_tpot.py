#!/usr/bin/env python3
"""Score an AutoDP-prepared dataset with estimator-only TPOT.

This is the final-evaluator counterpart to ``scripts/eval_autodatapre.py``.
AutoDP performs its published MCTS search and produces a transformed frame; TPOT then searches
only over estimators on that frozen frame. TPOT is deliberately *not* called inside AutoDP's MCTS,
which would replace AutoDP's NB/LDA/RF candidate signal with a far more expensive new method.

The required ``tpot_leakfree`` preparation mode uses the shared seed-42 60/20/20 splitter:
AutoDP search, preprocessing fit, and TPOT CV all see only the outer 60% train partition. The
outer validation partition is retained but unused, matching the repository's TPOT baselines; the
outer 20% test partition remains untouched until final scoring.

Like the AutoGluon evaluator, the original target is restored by row position and dropped test
rows are charged in ``score_full`` rather than silently excluded. A train-fitted compatibility
adapter imputes numeric remnants and one-hot encodes only residual categoricals, then TPOT runs
with ``preprocessing=False`` so it cannot choose additional preprocessing operations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Callable, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from automl_aco.data.splits import split_fingerprints, split_train_val_test
from automl_aco.search.evaluation import _detect_problem_type


def _split_positions(n_rows: int, seed: int = 42):
    idx = pd.DataFrame({"_pos": np.arange(n_rows)})
    dummy = pd.Series(np.zeros(n_rows))
    x_tr, _, x_val, _, x_te, _ = split_train_val_test(idx, dummy, seed=seed)
    return x_tr["_pos"].to_numpy(), x_val["_pos"].to_numpy(), x_te["_pos"].to_numpy()


def _safe_cv_folds(y_train: pd.Series, problem_type: str, maximum: int) -> int:
    if problem_type == "regression":
        folds = min(int(maximum), len(y_train))
        if folds < 2:
            raise ValueError("TPOT regression CV requires at least two training rows")
        return folds
    counts = pd.Series(y_train).value_counts()
    if len(counts) < 2:
        raise ValueError("TPOT classification requires at least two training classes")
    for folds in range(min(int(maximum), int(counts.max())), 1, -1):
        if int((counts >= folds).sum()) >= 2:
            return folds
    raise ValueError("Fewer than two classes have enough rows for 2-fold TPOT CV")


def _tpot_ready_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize category-like columns without estimating any data statistics."""
    out = frame.copy()
    for column in out.select_dtypes(exclude=[np.number, "bool"]).columns:
        missing = out[column].isna()
        out[column] = out[column].astype(str).astype(object)
        out.loc[missing, column] = np.nan
    return out


def _minimal_adapter(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric = list(X_train.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in X_train.columns if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
            ]),
            categorical,
        ))
    if not transformers:
        raise ValueError("AutoDP produced a frame with zero usable feature columns")
    return (
        ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        ),
        [str(column) for column in numeric],
        [str(column) for column in categorical],
    )


def _default_tpot_components(problem_type: str):
    try:
        from tpot import TPOTClassifier, TPOTRegressor
        from tpot.config import get_search_space
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("TPOT 1.1.0 is required for the final evaluator") from exc
    return (TPOTClassifier if problem_type == "classification" else TPOTRegressor), get_search_space


def score_prepared(
    dataset_csv: str,
    prepared_dir: str,
    *,
    target: str = "target",
    split_seed: int = 42,
    tpot_seed: int = 1,
    max_time_mins: int = 5,
    max_eval_time_mins: int = 1,
    n_jobs: int = 2,
    memory_limit: str = "5GB",
    population_size: int = 20,
    max_cv_folds: int = 5,
    verbose: int = 2,
    estimator_factory: Optional[Callable[..., Any]] = None,
    search_space_factory: Optional[Callable[..., Any]] = None,
) -> tuple[dict, Any]:
    """Fit estimator-only TPOT on one AutoDP output and score the outer test partition."""
    with open(os.path.join(prepared_dir, "autodp_meta.json"), encoding="utf-8") as handle:
        adp_meta = json.load(handle)
    mode = adp_meta.get("mode")
    if mode != "tpot_leakfree":
        raise ValueError(
            f"TPOT evaluator requires AutoDP mode 'tpot_leakfree', got {mode!r}. "
            "Re-run the AutoDP preparation so its MCTS and TPOT see the same outer 60% train rows."
        )

    original = pd.read_csv(dataset_csv)
    if target not in original.columns:
        raise ValueError(f"Original dataset has no target column {target!r}")
    X_original = original.drop(columns=[target])
    y_original = original[target].reset_index(drop=True)
    train_rows, validation_rows, test_rows = _split_positions(len(original), seed=split_seed)
    train_set, test_set = set(train_rows.tolist()), set(test_rows.tolist())

    prepared = pd.read_csv(os.path.join(prepared_dir, "prepared.csv"))
    required = {"__adp_row__", "__adp_split__"}
    missing = required - set(prepared.columns)
    if missing:
        raise ValueError(f"AutoDP prepared frame is missing helper columns: {sorted(missing)}")
    rows = prepared["__adp_row__"].to_numpy(dtype=int)
    is_test = (prepared["__adp_split__"] == "test").to_numpy()
    train_ids, test_ids = set(rows[~is_test].tolist()), set(rows[is_test].tolist())
    unexpected_train = train_ids - train_set
    unexpected_test = test_ids - test_set
    if unexpected_train or unexpected_test:
        raise AssertionError(
            "LEAKAGE in prepared frame: "
            f"{len(unexpected_train)} non-train row(s) in TPOT training and "
            f"{len(unexpected_test)} non-test row(s) in TPOT test"
        )
    if train_ids & test_ids:
        raise AssertionError("A prepared row appears in both TPOT train and test partitions")

    feature_columns = [column for column in prepared.columns if not column.startswith("__adp_")]
    if not feature_columns:
        raise RuntimeError("AutoDP produced a frame with zero feature columns")
    X_train = prepared.loc[~is_test, feature_columns].reset_index(drop=True)
    X_test = prepared.loc[is_test, feature_columns].reset_index(drop=True)
    y_train = y_original.iloc[rows[~is_test]].reset_index(drop=True)
    y_test_kept = y_original.iloc[rows[is_test]].reset_index(drop=True)
    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError(f"empty split after AutoDP preparation (train={len(X_train)}, test={len(X_test)})")

    X_train = _tpot_ready_frame(X_train)
    X_test = _tpot_ready_frame(X_test)
    adapter, numeric_columns, categorical_columns = _minimal_adapter(X_train)
    train_matrix = np.asarray(adapter.fit_transform(X_train, y_train), dtype=np.float32)
    test_matrix = np.asarray(adapter.transform(X_test), dtype=np.float32)
    if train_matrix.ndim != 2 or train_matrix.shape[1] == 0:
        raise RuntimeError(f"TPOT compatibility adapter produced invalid train matrix {train_matrix.shape}")
    if not np.isfinite(train_matrix).all() or not np.isfinite(test_matrix).all():
        raise RuntimeError("TPOT compatibility adapter left NaN or infinity")

    problem_type, eval_metric = _detect_problem_type(y_original)
    cv_folds = _safe_cv_folds(y_train, problem_type, max_cv_folds)
    target_encoder = None
    y_train_for_tpot = y_train.to_numpy()
    if problem_type == "classification":
        target_encoder = LabelEncoder()
        y_train_for_tpot = target_encoder.fit_transform(y_train.to_numpy())
    n_classes = int(y_train.nunique()) if problem_type == "classification" else 1

    if estimator_factory is None or search_space_factory is None:
        default_estimator, default_search_space = _default_tpot_components(problem_type)
        estimator_factory = estimator_factory or default_estimator
        search_space_factory = search_space_factory or default_search_space
    group = "classifiers" if problem_type == "classification" else "regressors"
    search_space = search_space_factory(
        group,
        n_classes=n_classes,
        n_samples=int(train_matrix.shape[0]),
        n_features=int(train_matrix.shape[1]),
        random_state=int(tpot_seed),
        n_jobs=1,
    )
    model = estimator_factory(
        search_space=search_space,
        scorers=["accuracy" if problem_type == "classification" else "r2"],
        scorers_weights=[1],
        cv=cv_folds,
        preprocessing=False,
        max_time_mins=int(max_time_mins),
        max_eval_time_mins=int(max_eval_time_mins),
        n_jobs=int(n_jobs),
        memory_limit=str(memory_limit),
        validation_strategy="none",
        early_stop=5,
        verbose=int(verbose),
        random_state=int(tpot_seed),
        population_size=int(population_size),
        initial_population_size=int(population_size),
    )
    started = time.perf_counter()
    model.fit(train_matrix, y_train_for_tpot)
    predictions = np.asarray(model.predict(test_matrix))
    if target_encoder is not None:
        predictions = target_encoder.inverse_transform(predictions.astype(int))
    tpot_seconds = float(time.perf_counter() - started)

    coverage = len(X_test) / len(test_rows)
    if problem_type == "regression":
        score_kept = float(r2_score(y_test_kept, predictions))
        full_predictions = pd.Series(float(np.mean(y_train)), index=range(len(test_rows)))
        kept_positions = {int(row_id): i for i, row_id in enumerate(rows[is_test])}
        for index, row_id in enumerate(test_rows):
            if int(row_id) in kept_positions:
                full_predictions.iloc[index] = float(predictions[kept_positions[int(row_id)]])
        score_full = float(r2_score(y_original.iloc[test_rows].reset_index(drop=True), full_predictions))
        accuracy = balanced_accuracy = f1_macro = None
        r2, rmse = score_kept, float(math.sqrt(mean_squared_error(y_test_kept, predictions)))
    else:
        score_kept = float(accuracy_score(y_test_kept, predictions))
        correct = int((np.asarray(y_test_kept) == predictions).sum())
        score_full = correct / len(test_rows)
        accuracy = score_kept
        balanced_accuracy = float(balanced_accuracy_score(y_test_kept, predictions))
        f1_macro = float(f1_score(y_test_kept, predictions, average="macro", zero_division=0))
        r2 = rmse = None

    split = split_train_val_test(X_original, y_original, seed=split_seed)
    result = {
        "dataset_id": os.path.splitext(os.path.basename(dataset_csv))[0],
        "method": "autodatapre-0.1.12_ourops_tpot",
        "evaluator": type(model).__name__,
        "mode": mode,
        "autodp_status": adp_meta.get("status"),
        "autodp_pipeline": adp_meta.get("pipeline"),
        "autodp_search_seconds": adp_meta.get("search_seconds"),
        "autodp_converged": adp_meta.get("converged_default_budget"),
        "autodp_hit_cap": bool(adp_meta.get("hit_wall_clock_cap", False)),
        "tpot_eval_seconds": round(tpot_seconds, 2),
        "total_seconds": round(float(adp_meta.get("search_seconds") or 0.0) + tpot_seconds, 2),
        "problem_type": problem_type,
        "eval_metric": eval_metric,
        "score_full": float(score_full),
        "score_kept": float(score_kept),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": f1_macro,
        "r2": r2,
        "rmse": rmse,
        "test_coverage": coverage,
        "n_train_rows_expected": int(len(train_rows)),
        "n_train_rows_kept": int(len(X_train)),
        "n_validation_rows_unused": int(len(validation_rows)),
        "n_test_rows_expected": int(len(test_rows)),
        "n_test_rows_kept": int(len(X_test)),
        "n_features_autodp": int(len(feature_columns)),
        "n_features_scored": int(train_matrix.shape[1]),
        "residual_numeric_columns": numeric_columns,
        "residual_categorical_columns": categorical_columns,
        "split_fingerprints": split_fingerprints(split),
        "tpot_seed": int(tpot_seed),
        "tpot_space": group,
        "tpot_preprocessing": False,
        "cv_folds": int(cv_folds),
        "max_time_mins": int(max_time_mins),
        "max_eval_time_mins": int(max_eval_time_mins),
        "n_jobs": int(n_jobs),
        "memory_limit": str(memory_limit),
        "population_size": int(population_size),
        "selected_estimator": str(getattr(model, "fitted_pipeline_", getattr(model, "fitted_pipeline", ""))),
        "protocol": {
            "split": "seed-42 0.6/0.2/0.2 via automl_aco.data.splits.split_train_val_test",
            "autodp_search_and_fit": "outer train 60% only",
            "tpot_fit": "outer train 60% only; CV internal to TPOT",
            "validation": "outer validation 20% unused",
            "test": "outer test 20%",
            "target": "ORIGINAL y re-attached by row position",
        },
    }
    return result, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--target", default="target")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--tpot-seed", type=int, default=1)
    parser.add_argument("--max-time-mins", type=int, default=5)
    parser.add_argument("--max-eval-time-mins", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--memory-limit", default="5GB")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--max-cv-folds", type=int, default=5)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--out", default=None, help="default: <prepared-dir>/autodp_tpot_eval.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.prepared_dir, "autodp_tpot_eval.json")
    if os.path.exists(out_path) and not args.overwrite:
        with open(out_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("status", "ok") == "ok":
            print(f"[skip] already scored -> {out_path}")
            return
    result, _model = score_prepared(
        args.dataset_csv,
        args.prepared_dir,
        target=args.target,
        split_seed=args.split_seed,
        tpot_seed=args.tpot_seed,
        max_time_mins=args.max_time_mins,
        max_eval_time_mins=args.max_eval_time_mins,
        n_jobs=args.n_jobs,
        memory_limit=args.memory_limit,
        population_size=args.population_size,
        max_cv_folds=args.max_cv_folds,
        verbose=args.verbose,
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    print(
        f"[ok] {result['dataset_id']} {result['eval_metric']}: full={result['score_full']:.4f} "
        f"kept={result['score_kept']:.4f} coverage={result['test_coverage']:.3f} "
        f"AutoDP={result['autodp_search_seconds']}s TPOT={result['tpot_eval_seconds']}s -> {out_path}"
    )


if __name__ == "__main__":
    main()
