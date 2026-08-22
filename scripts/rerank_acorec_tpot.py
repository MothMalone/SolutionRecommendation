#!/usr/bin/env python3
"""Rerank ACORec's top preprocessing pipelines with estimator-only TPOT.

Candidate TPOT models are fitted on the fixed outer-train 60% and scored on
the fixed validation 20%.  Only the selected preprocessing pipeline is then
refitted and evaluated on the untouched outer-test 20%.  This script lives in
the experiment layer and does not change ACORec's search or operator space.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from automl_aco.data.loaders import load_gitlab_openml_dataset  # noqa: E402
from automl_aco.data.splits import split_train_val_test  # noqa: E402
from automl_aco.eval_ids import EVAL_IDS  # noqa: E402
from automl_aco.search.evaluation import _fit_pipeline, _make_preprocessor  # noqa: E402
from evaluate_acorec_tpot import (  # noqa: E402
    _default_tpot_components,
    _numeric_matrix,
    _safe_cv_folds,
    evaluate_recommendation,
)


def _config_key(config: Dict[str, Any]) -> str:
    cleaned = {key: value for key, value in config.items() if key != "name"}
    return json.dumps(cleaned, sort_keys=True, default=str)


def _top_unique_candidates(recommendation: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    scored: List[Tuple[Dict[str, Any], float]] = []
    for item in recommendation.get("aco_results") or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2 or not isinstance(item[0], dict):
            continue
        try:
            score = float(item[1])
        except (TypeError, ValueError):
            score = float("-inf")
        scored.append((dict(item[0]), score))

    selected = recommendation.get("pipeline_config")
    if isinstance(selected, dict):
        try:
            selected_score = float(recommendation.get("recommended_performance"))
        except (TypeError, ValueError):
            selected_score = float("-inf")
        scored.append((dict(selected), selected_score))

    scored.sort(key=lambda item: item[1], reverse=True)
    unique: List[Dict[str, Any]] = []
    seen = set()
    for config, proxy_score in scored:
        key = _config_key(config)
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "candidate_key": key,
                "proxy_score": None if not math.isfinite(proxy_score) else proxy_score,
                "pipeline_config": config,
            }
        )
        if len(unique) >= max(1, int(top_k)):
            break
    if not unique:
        raise ValueError("No valid pipeline candidates found in recommendation.json")
    return unique


def _fit_tpot_validation(
    dataset: Dict[str, Any],
    pipeline_config: Dict[str, Any],
    *,
    split_seed: int,
    tpot_seed: int,
    max_time_mins: int,
    max_eval_time_mins: int,
    n_jobs: int,
    memory_limit: str,
    population_size: int,
    max_cv_folds: int,
    verbose: int,
) -> Dict[str, Any]:
    """Fit on outer train and score validation; never transform outer test."""
    X = pd.DataFrame(dataset["X"]).copy()
    y = pd.Series(dataset["y"]).copy()
    task_type = str(dataset.get("task_type", "classification"))
    X_train, y_train, X_val, y_val, _X_test, _y_test = split_train_val_test(
        X, y, seed=int(split_seed)
    )

    preprocessor = _make_preprocessor(dict(pipeline_config))
    X_train_p, y_train_p = _fit_pipeline(
        preprocessor,
        X_train,
        y_train,
        X_full=X,
        y_full=y,
        prepare_mode="leakfree",
    )
    X_val_p = preprocessor.transform(X_val)
    y_train_p = pd.Series(y_train_p).reset_index(drop=True)
    y_val = pd.Series(y_val).reset_index(drop=True)
    train_matrix = _numeric_matrix(X_train_p, "training")
    val_matrix = _numeric_matrix(X_val_p, "validation")
    cv_folds = _safe_cv_folds(y_train_p, task_type, max_cv_folds)

    target_encoder = None
    y_fit = y_train_p.to_numpy()
    if task_type == "classification":
        target_encoder = LabelEncoder()
        y_fit = target_encoder.fit_transform(y_fit)

    estimator_factory, search_space_factory = _default_tpot_components(task_type)
    group = "classifiers" if task_type == "classification" else "regressors"
    search_space = search_space_factory(
        group,
        n_classes=int(y_train_p.nunique()) if task_type == "classification" else 1,
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
    try:
        model.fit(train_matrix, y_fit)
        prediction = model.predict(val_matrix)
        if target_encoder is not None:
            prediction = target_encoder.inverse_transform(np.asarray(prediction).astype(int))
        if task_type == "classification":
            score = float(accuracy_score(y_val, prediction))
            rmse = None
        else:
            score = float(r2_score(y_val, prediction))
            rmse = float(math.sqrt(mean_squared_error(y_val, prediction)))
        return {
            "status": "ok",
            "validation_score": score,
            "primary_metric": primary_metric,
            "validation_rmse": rmse,
            "train_rows_raw": int(len(X_train)),
            "train_rows_processed": int(train_matrix.shape[0]),
            "validation_rows": int(len(y_val)),
            "raw_features": int(X.shape[1]),
            "transformed_features": int(train_matrix.shape[1]),
            "cv_folds": int(cv_folds),
            "fit_seconds": float(time.perf_counter() - started),
            "selected_estimator": str(
                getattr(model, "fitted_pipeline_", getattr(model, "fitted_pipeline", ""))
            ),
        }
    finally:
        del model
        gc.collect()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendation-json", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--tpot-seed", type=int, default=1)
    parser.add_argument("--selection-time-mins", type=int, default=1)
    parser.add_argument("--final-time-mins", type=int, default=1)
    parser.add_argument("--max-eval-time-mins", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--memory-limit", default="5GB")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--max-cv-folds", type=int, default=5)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Stop after validation ranking and never transform or score the outer test split.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = args.output_json or args.recommendation_json.with_name(
        "tpot_topk_rerank.json"
    )
    if output_path.exists() and not args.force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("status") == "ok" or (
            args.selection_only and existing.get("status") == "selection_complete"
        ):
            print(f"SKIP successful top-k rerank: {output_path}")
            return 0
    else:
        existing = {}

    recommendation = json.loads(args.recommendation_json.read_text(encoding="utf-8"))
    candidates = _top_unique_candidates(recommendation, args.top_k)
    dataset = load_gitlab_openml_dataset(
        args.dataset_id,
        cache_dir=str(args.data_dir),
        test_dataset_ids=[int(value) for value in EVAL_IDS],
        verbose=True,
        max_samples_if_test=int(args.max_samples),
    )
    if dataset is None:
        raise RuntimeError(f"Could not load dataset {args.dataset_id}")

    completed = {
        item.get("candidate_key"): item
        for item in existing.get("candidate_validation", [])
        if item.get("status") == "ok"
    }
    rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "status": "selecting",
        "method": "acorec_tpot_validation_topk_rerank",
        "dataset_id": int(args.dataset_id),
        "dataset_name": dataset.get("name", f"D_{args.dataset_id}"),
        "top_k_requested": int(args.top_k),
        "top_k_unique": int(len(candidates)),
        "selection_protocol": "fit outer-train 60%; rank on validation 20%; outer test untouched",
        "outer_test_seen_during_selection": False,
        "candidate_validation": rows,
    }
    for index, candidate in enumerate(candidates, start=1):
        key = candidate["candidate_key"]
        if key in completed and not args.force:
            row = completed[key]
            print(f"[{index}/{len(candidates)}] resume validation={row['validation_score']:.6f}")
        else:
            print(f"[{index}/{len(candidates)}] TPOT validation: {candidate['pipeline_config']}")
            row = dict(candidate)
            try:
                row.update(
                    _fit_tpot_validation(
                        dataset,
                        candidate["pipeline_config"],
                        split_seed=args.split_seed,
                        tpot_seed=args.tpot_seed,
                        max_time_mins=args.selection_time_mins,
                        max_eval_time_mins=args.max_eval_time_mins,
                        n_jobs=args.n_jobs,
                        memory_limit=args.memory_limit,
                        population_size=args.population_size,
                        max_cv_folds=args.max_cv_folds,
                        verbose=args.verbose,
                    )
                )
                print(f"    validation {row['primary_metric']}={row['validation_score']:.6f}")
            except Exception as exc:
                row.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"    FAILED {type(exc).__name__}: {exc}")
        rows.append(row)
        payload["candidate_validation"] = rows
        _write_json(output_path, payload)

    successful = [row for row in rows if row.get("status") == "ok"]
    if not successful:
        payload["status"] = "failed"
        payload["error"] = "All candidate TPOT validation evaluations failed"
        _write_json(output_path, payload)
        return 1
    winner = max(successful, key=lambda row: float(row["validation_score"]))
    if args.selection_only:
        payload.update(
            {
                "status": "selection_complete",
                "selected_candidate": winner,
                "outer_test_evaluations": 0,
                "selection_time_mins_per_candidate": int(args.selection_time_mins),
                "tpot_preprocessing": False,
            }
        )
        _write_json(output_path, payload)
        print(
            f"TPOT top-k validation {winner['primary_metric']}: "
            f"{winner['validation_score']:.6f}; outer test remains untouched"
        )
        print("Saved:", output_path)
        return 0
    print("Selected on validation; now opening outer test exactly once.")
    final_result, final_model = evaluate_recommendation(
        dataset,
        winner["pipeline_config"],
        split_seed=args.split_seed,
        tpot_seed=args.tpot_seed,
        max_time_mins=args.final_time_mins,
        max_eval_time_mins=args.max_eval_time_mins,
        n_jobs=args.n_jobs,
        memory_limit=args.memory_limit,
        population_size=args.population_size,
        max_cv_folds=args.max_cv_folds,
        verbose=args.verbose,
    )
    del final_model
    gc.collect()
    payload.update(
        {
            "status": "ok",
            "selected_candidate": winner,
            "final_outer_test": final_result,
            "outer_test_score": final_result.get("score"),
            "outer_test_evaluations": 1,
            "selection_time_mins_per_candidate": int(args.selection_time_mins),
            "final_time_mins": int(args.final_time_mins),
            "tpot_preprocessing": False,
        }
    )
    _write_json(output_path, payload)
    print(f"TPOT top-k outer-test {final_result['primary_metric']}: {final_result['score']:.6f}")
    print("Saved:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
