#!/usr/bin/env python3
"""Evaluate one frozen ACORec recommendation with estimator-only TPOT.

This is deliberately an experiment-side wrapper. It does not modify ACORec's
search, recommender, operator space, or performance matrix. ACORec uses the
fixed 20% validation split while searching for its preprocessing pipeline.
After selection, that pipeline and estimator-only TPOT are fitted using the
fixed 60% training split; TPOT does not reuse validation. The reported score
is computed exactly once on the fixed 20% test split.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automl_aco.data.loaders import load_gitlab_openml_dataset  # noqa: E402
from automl_aco.data.splits import split_train_val_test  # noqa: E402
from automl_aco.eval_ids import EVAL_IDS  # noqa: E402
from automl_aco.search.evaluation import _fit_pipeline, _make_preprocessor  # noqa: E402


def _safe_cv_folds(y_train: pd.Series, task_type: str, maximum: int) -> int:
    maximum = max(2, int(maximum))
    if task_type == "regression":
        folds = min(maximum, len(y_train))
        if folds < 2:
            raise ValueError("TPOT regression CV requires at least two training rows")
        return folds
    counts = pd.Series(y_train).value_counts()
    if len(counts) < 2:
        raise ValueError("TPOT classification requires at least two training classes")
    folds = min(maximum, int(counts.min()))
    if folds < 2:
        raise ValueError("A processed training class has fewer than two rows for TPOT CV")
    return folds


def _numeric_matrix(frame: Any, label: str) -> np.ndarray:
    if isinstance(frame, pd.DataFrame):
        if any(isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes):
            frame = frame.sparse.to_dense()
        non_numeric = frame.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"ACORec pipeline left non-numeric {label} columns: {non_numeric[:10]}"
            )
        values = frame.to_numpy(dtype=np.float32, copy=False)
    elif hasattr(frame, "toarray"):
        values = np.asarray(frame.toarray(), dtype=np.float32)
    else:
        values = np.asarray(frame, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Transformed {label} matrix has invalid shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Transformed {label} matrix contains NaN or infinity")
    return values


def _default_tpot_components(task_type: str):
    try:
        from tpot import TPOTClassifier, TPOTRegressor
        from tpot.config import get_search_space
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("TPOT 1.1.0 is required for the final evaluator") from exc
    estimator = TPOTClassifier if task_type == "classification" else TPOTRegressor
    return estimator, get_search_space


def evaluate_recommendation(
    dataset: Dict[str, Any],
    pipeline_config: Dict[str, Any],
    *,
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
) -> Tuple[Dict[str, Any], Any]:
    """Fit ACORec preprocessing + estimator-only TPOT and score the outer test 20%."""
    X = pd.DataFrame(dataset["X"]).copy()
    y = pd.Series(dataset["y"]).copy()
    task_type = str(dataset.get("task_type", "classification"))
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task_type {task_type!r}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        X, y, seed=int(split_seed)
    )
    preprocessor = _make_preprocessor(dict(pipeline_config))
    X_train_processed, y_train_processed = _fit_pipeline(
        preprocessor,
        X_train,
        y_train,
        X_full=X,
        y_full=y,
        prepare_mode="leakfree",
    )
    X_test_processed = preprocessor.transform(X_test)
    y_train_processed = pd.Series(y_train_processed).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)
    if len(X_train_processed) != len(y_train_processed):
        raise ValueError("Processed training X/y lengths do not match")
    if len(X_test_processed) != len(y_test):
        raise ValueError("Processed test X/y lengths do not match")

    train_matrix = _numeric_matrix(X_train_processed, "training")
    test_matrix = _numeric_matrix(X_test_processed, "test")
    cv_folds = _safe_cv_folds(y_train_processed, task_type, max_cv_folds)

    if estimator_factory is None or search_space_factory is None:
        default_estimator, default_search_space = _default_tpot_components(task_type)
        estimator_factory = estimator_factory or default_estimator
        search_space_factory = search_space_factory or default_search_space

    group = "classifiers" if task_type == "classification" else "regressors"
    n_classes = int(y_train_processed.nunique()) if task_type == "classification" else 1
    search_space = search_space_factory(
        group,
        n_classes=n_classes,
        n_samples=int(train_matrix.shape[0]),
        n_features=int(train_matrix.shape[1]),
        random_state=int(tpot_seed),
        n_jobs=1,
    )
    primary_metric = "accuracy" if task_type == "classification" else "r2"
    model = estimator_factory(
        search_space=search_space,
        scorers=[primary_metric],
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
    model.fit(train_matrix, y_train_processed.to_numpy())
    prediction = model.predict(test_matrix)
    fit_seconds = float(time.perf_counter() - started)

    result: Dict[str, Any] = {
        "status": "ok",
        "method": "acorec_tpot",
        "evaluator": type(model).__name__,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "score": None,
        "accuracy": None,
        "balanced_accuracy": None,
        "f1_macro": None,
        "r2": None,
        "rmse": None,
        "split_seed": int(split_seed),
        "tpot_seed": int(tpot_seed),
        "train_rows_raw": int(len(X_train)),
        "train_rows_processed": int(train_matrix.shape[0]),
        "validation_rows_aco_search": int(len(X_val)),
        "validation_reused_by_tpot": False,
        "test_rows": int(len(X_test)),
        "raw_features": int(X.shape[1]),
        "transformed_features": int(train_matrix.shape[1]),
        "test_fraction": 0.2,
        "tpot_space": group,
        "tpot_preprocessing": False,
        "cv_folds": int(cv_folds),
        "max_time_mins": int(max_time_mins),
        "max_eval_time_mins": int(max_eval_time_mins),
        "n_jobs": int(n_jobs),
        "memory_limit": str(memory_limit),
        "population_size": int(population_size),
        "fit_seconds": fit_seconds,
        "pipeline_config": dict(pipeline_config),
        "selected_estimator": str(
            getattr(model, "fitted_pipeline_", getattr(model, "fitted_pipeline", ""))
        ),
    }
    if task_type == "classification":
        result["accuracy"] = float(accuracy_score(y_test, prediction))
        result["balanced_accuracy"] = float(balanced_accuracy_score(y_test, prediction))
        result["f1_macro"] = float(
            f1_score(y_test, prediction, average="macro", zero_division=0)
        )
        result["score"] = result["accuracy"]
    else:
        result["r2"] = float(r2_score(y_test, prediction))
        result["rmse"] = float(math.sqrt(mean_squared_error(y_test, prediction)))
        result["score"] = result["r2"]
    return result, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendation-json", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--tpot-seed", type=int, default=1)
    parser.add_argument("--max-time-mins", type=int, default=5)
    parser.add_argument("--max-eval-time-mins", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--memory-limit", default="5GB")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--max-cv-folds", type=int, default=5)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = args.output_json or args.recommendation_json.with_name("tpot_evaluation.json")
    if output_path.exists() and not args.force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("status") == "ok":
            print(f"SKIP successful TPOT evaluation: {output_path}")
            return 0

    recommendation = json.loads(args.recommendation_json.read_text(encoding="utf-8"))
    pipeline_config = recommendation.get("pipeline_config")
    if not isinstance(pipeline_config, dict):
        raise ValueError("recommendation.json does not contain a pipeline_config object")

    dataset = load_gitlab_openml_dataset(
        args.dataset_id,
        cache_dir=str(args.data_dir),
        test_dataset_ids=[int(value) for value in EVAL_IDS],
        verbose=True,
        max_samples_if_test=int(args.max_samples),
    )
    if dataset is None:
        raise RuntimeError(f"Could not load dataset {args.dataset_id}")

    model = None
    try:
        result, model = evaluate_recommendation(
            dataset,
            pipeline_config,
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
        result.update(
            {
                "dataset_id": int(args.dataset_id),
                "dataset_name": dataset.get("name", f"D_{args.dataset_id}"),
                "dataset_backend": dataset.get("download_backend"),
                "max_samples": int(args.max_samples),
                "recommendation_json": str(args.recommendation_json),
                "aco_proxy_score": recommendation.get("recommended_performance"),
                "operator_space": recommendation.get("operator_space", "ours"),
            }
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "method": "acorec_tpot",
            "dataset_id": int(args.dataset_id),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        raise
    finally:
        del model
        gc.collect()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(
        f"TPOT outer-test {result['primary_metric']}: {float(result['score']):.6f} "
        f"({result['test_rows']} rows = 20%)"
    )
    print("Saved:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
