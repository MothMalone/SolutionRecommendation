#!/usr/bin/env python3
"""Evaluate one frozen ACORec recommendation with estimator-only TPOT.

This is deliberately an experiment-side wrapper. It does not modify ACORec's search, recommender,
operator space, or performance matrix. ACORec uses the fixed 20% validation split while searching
for its preprocessing pipeline. After selection, that pipeline and estimator-only TPOT are fitted
using the fixed 60% training split; TPOT does not reuse validation. The reported score is computed
exactly once on the fixed 20% test split.

Data source: the SAME exported ``<id>.csv`` that ``scripts/export_eval_datasets.py`` writes and that
``scripts/adp_bench.py`` / ``scripts/run_autodatapre.py`` feed to AutoDP. Both arms of the
ACORec-vs-AutoDP-under-TPOT comparison therefore read byte-identical rows in identical order, and
the seed-42 0.6/0.2/0.2 positional split lands on the same rows on both sides. The output records a
fingerprint of that CSV so the join can be verified.

The TPOT settings live in ``scripts/_tpot_eval.py`` and are shared with
``scripts/evaluate_autodp_tpot.py`` so the two evaluators cannot drift apart.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
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
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automl_aco.data.splits import split_train_val_test  # noqa: E402
from automl_aco.search.evaluation import (  # noqa: E402
    _detect_problem_type,
    _fit_pipeline,
    _make_preprocessor,
)

import _tpot_eval  # noqa: E402
from _tpot_eval import (  # noqa: E402  (kept importable under the historical names)
    TPOT_MAX_CV_FOLDS,
    TPOT_MAX_EVAL_TIME_MINS,
    TPOT_MAX_TIME_MINS,
    TPOT_MEMORY_LIMIT,
    TPOT_N_JOBS,
    TPOT_POPULATION_SIZE,
    TPOT_RANDOM_STATE,
    TPOT_SPLIT_SEED,
    _numeric_matrix,
    _safe_cv_folds,
    knob_summary,
    normalize_task_type,
    prune_rare_classes,
    to_tpot_matrix,
)


def _default_tpot_components(task_type: str):
    return _tpot_eval.default_tpot_components(task_type)


def _csv_fingerprint(csv_path: str, frame: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Cheap identity check for the exported CSV so a cross-arm join can be validated later."""
    tgt = frame[target].to_numpy()
    digest = hashlib.sha1(np.ascontiguousarray(tgt.astype("U")).tobytes()).hexdigest()[:16]
    return {
        "path": os.path.abspath(csv_path),
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "target_sha1_16": digest,
    }


def evaluate_recommendation(
    dataset: Dict[str, Any],
    pipeline_config: Dict[str, Any],
    *,
    split_seed: int = TPOT_SPLIT_SEED,
    tpot_seed: int = TPOT_RANDOM_STATE,
    max_time_mins: int = TPOT_MAX_TIME_MINS,
    max_eval_time_mins: int = TPOT_MAX_EVAL_TIME_MINS,
    n_jobs: int = TPOT_N_JOBS,
    memory_limit: str = TPOT_MEMORY_LIMIT,
    population_size: int = TPOT_POPULATION_SIZE,
    max_cv_folds: int = TPOT_MAX_CV_FOLDS,
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

    # ACORec's IQR/LOF operators can push a training class below the CV fold floor. Drop those
    # (unusable for stratified CV); test set untouched. Same handling as the AutoDP arm.
    dropped_rare_classes: list = []
    if task_type == "classification" and isinstance(X_train_processed, pd.DataFrame):
        X_train_processed, y_train_processed, dropped_rare_classes = prune_rare_classes(
            X_train_processed.reset_index(drop=True), y_train_processed, min_count=2
        )

    # DataFrame -> the No-Preprocessing compat adapter (median/mode impute + one-hot), matching
    # every other TPOT arm; already-numeric matrix (post-PCA/SVD) -> finite check only.
    train_matrix, test_matrix, adapter_meta = to_tpot_matrix(
        X_train_processed, X_test_processed, y_train_processed
    )
    cv_folds = _safe_cv_folds(y_train_processed, task_type, max_cv_folds)
    target_encoder = None
    y_train_for_tpot = y_train_processed.to_numpy()
    if task_type == "classification":
        # TPOT 1.x requires contiguous integer labels. The repository loader deliberately preserves
        # OpenML target values, which may be sparse or negative; decode predictions again before
        # computing test metrics.
        target_encoder = LabelEncoder()
        y_train_for_tpot = target_encoder.fit_transform(y_train_processed.to_numpy())

    n_classes = int(y_train_processed.nunique()) if task_type == "classification" else 1
    model, group = _tpot_eval.build_model(
        task_type,
        n_samples=int(train_matrix.shape[0]),
        n_features=int(train_matrix.shape[1]),
        n_classes=n_classes,
        cv_folds=cv_folds,
        tpot_seed=int(tpot_seed),
        max_time_mins=int(max_time_mins),
        max_eval_time_mins=int(max_eval_time_mins),
        n_jobs=int(n_jobs),
        memory_limit=str(memory_limit),
        population_size=int(population_size),
        verbose=int(verbose),
        estimator_factory=estimator_factory,
        search_space_factory=search_space_factory,
    )
    primary_metric = "accuracy" if task_type == "classification" else "r2"

    started = time.perf_counter()
    model.fit(train_matrix, y_train_for_tpot)
    prediction = model.predict(test_matrix)
    if target_encoder is not None:
        prediction = target_encoder.inverse_transform(np.asarray(prediction).astype(int))
    fit_seconds = float(time.perf_counter() - started)

    result: Dict[str, Any] = {
        "status": "ok",
        "method": "acorec_tpot",
        "evaluator": "tpot",
        "tpot_estimator": type(model).__name__,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "score": None,
        "accuracy": None,
        "balanced_accuracy": None,
        "f1_macro": None,
        "r2": None,
        "rmse": None,
        # ACORec pipelines never delete test rows, so these mirror `score` and let this row join the
        # AutoDP-under-TPOT table (which needs the coverage accounting) without holes.
        "score_full": None,
        "score_kept": None,
        "test_coverage": 1.0,
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
        "target_label_encoding": (
            "LabelEncoder_fit_on_train_inverse_before_scoring"
            if target_encoder is not None else "not_applicable"
        ),
        "cv_folds": int(cv_folds),
        "compat_adapter": adapter_meta,
        "dropped_rare_class_train_rows": [str(c) for c in dropped_rare_classes],
        "max_time_mins": int(max_time_mins),
        "max_eval_time_mins": int(max_eval_time_mins),
        "n_jobs": int(n_jobs),
        "memory_limit": str(memory_limit),
        "population_size": int(population_size),
        "fit_seconds": fit_seconds,
        "tpot_knobs": knob_summary(),
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
    result["score_full"] = result["score"]
    result["score_kept"] = result["score"]
    return result, model


def _load_exported_csv(csv_path: str, target: str) -> Dict[str, Any]:
    frame = pd.read_csv(csv_path)
    if target not in frame.columns:
        raise ValueError(f"{csv_path} has no {target!r} column (columns: {list(frame.columns)[:10]})")
    y = frame[target]
    X = frame.drop(columns=[target])
    problem_type, _metric = _detect_problem_type(y)
    return {
        "X": X,
        "y": y,
        "task_type": normalize_task_type(problem_type),
        "detected_problem_type": problem_type,
        "fingerprint": _csv_fingerprint(csv_path, frame, target),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recommendation-json", type=Path, required=True,
                        help="ACORec recommendation.json holding the frozen pipeline_config")
    parser.add_argument("--dataset-csv", type=Path, required=True,
                        help="exported <id>.csv from scripts/export_eval_datasets.py -- the SAME "
                             "file the AutoDP arm reads")
    parser.add_argument("--dataset-id", default=None,
                        help="label for the output row; defaults to the CSV basename")
    parser.add_argument("--target", default="target")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=TPOT_SPLIT_SEED)
    parser.add_argument("--tpot-seed", type=int, default=TPOT_RANDOM_STATE)
    parser.add_argument("--max-time-mins", type=int, default=TPOT_MAX_TIME_MINS)
    parser.add_argument("--max-eval-time-mins", type=int, default=TPOT_MAX_EVAL_TIME_MINS)
    parser.add_argument("--n-jobs", type=int, default=TPOT_N_JOBS)
    parser.add_argument("--memory-limit", default=TPOT_MEMORY_LIMIT)
    parser.add_argument("--population-size", type=int, default=TPOT_POPULATION_SIZE)
    parser.add_argument("--max-cv-folds", type=int, default=TPOT_MAX_CV_FOLDS)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_id = args.dataset_id or os.path.splitext(os.path.basename(str(args.dataset_csv)))[0]
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

    dataset = _load_exported_csv(str(args.dataset_csv), args.target)

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
                "dataset_id": str(dataset_id),
                "dataset_csv": dataset["fingerprint"],
                "detected_problem_type": dataset["detected_problem_type"],
                "recommendation_json": str(args.recommendation_json),
                "aco_proxy_score": recommendation.get("recommended_performance"),
                "operator_space": recommendation.get("operator_space", "ours"),
            }
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "method": "acorec_tpot",
            "evaluator": "tpot",
            "dataset_id": str(dataset_id),
            "dataset_csv": dataset["fingerprint"],
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
        f"ACORec+TPOT outer-test {result['primary_metric']}: {float(result['score']):.6f} "
        f"({result['test_rows']} rows = 20%)"
    )
    print("Saved:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
