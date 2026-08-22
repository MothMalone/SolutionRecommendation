#!/usr/bin/env python3
"""Evaluate a frozen ACORec recommendation with H2O AutoML."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = Path(__file__).resolve().parent
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from automl_aco.data.loaders import load_gitlab_openml_dataset  # noqa: E402
from automl_aco.data.splits import split_train_val_test  # noqa: E402
from automl_aco.eval_ids import EVAL_IDS  # noqa: E402
from automl_aco.search.evaluation import _fit_pipeline, _make_preprocessor  # noqa: E402
from h2o_evaluator import evaluate_h2o_frames  # noqa: E402


def evaluate_recommendation(
    dataset: Dict[str, Any],
    pipeline_config: Dict[str, Any],
    *,
    split_seed: int = 42,
    h2o_preprocessing: str | None = None,
    max_runtime_secs: int = 300,
    max_runtime_secs_per_model: int = 60,
    max_models: int | None = None,
    nfolds: int = 5,
    seed: int = 42,
    nthreads: int = 1,
    max_mem_size: str = "6G",
    keep_h2o_alive: bool = False,
):
    X = pd.DataFrame(dataset["X"]).copy()
    y = pd.Series(dataset["y"]).copy()
    task_type = str(dataset.get("task_type", "classification"))
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        X, y, seed=int(split_seed)
    )
    preprocessor = _make_preprocessor(dict(pipeline_config))
    X_train_p, y_train_p = _fit_pipeline(
        preprocessor,
        X_train,
        y_train,
        prepare_mode="leakfree",
    )
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)
    result, model = evaluate_h2o_frames(
        X_train_p,
        y_train_p,
        X_val_p,
        y_val,
        X_test_p,
        y_test,
        task_type=task_type,
        h2o_preprocessing=h2o_preprocessing,
        max_runtime_secs=max_runtime_secs,
        max_runtime_secs_per_model=max_runtime_secs_per_model,
        max_models=max_models,
        nfolds=nfolds,
        seed=seed,
        nthreads=nthreads,
        max_mem_size=max_mem_size,
        keep_h2o_alive=keep_h2o_alive,
    )
    result.update(
        {
            "method": "acorec_h2o",
            "dataset_id": dataset.get("dataset_id"),
            "dataset_name": dataset.get("name"),
            "operator_space": "ours",
            "pipeline_config": dict(pipeline_config),
            "split_seed": int(split_seed),
            "validation_used_for_model_selection": True,
            "outer_test_seen_during_selection": False,
            "train_rows_raw": int(len(X_train)),
            "train_rows_processed": int(len(y_train_p)),
            "validation_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "raw_features": int(X.shape[1]),
            "transformed_features": int(X_train_p.shape[1]),
        }
    )
    return result, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendation-json", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--h2o-preprocessing", choices=["none", "target_encoding"], default="none")
    parser.add_argument("--max-runtime-secs", type=int, default=300)
    parser.add_argument("--max-runtime-secs-per-model", type=int, default=60)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--nfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--max-mem-size", default="6G")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_json.exists() and not args.force:
        existing = json.loads(args.output_json.read_text(encoding="utf-8"))
        if existing.get("status") == "ok":
            print(f"SKIP successful H2O evaluation: {args.output_json}")
            return 0
    recommendation = json.loads(args.recommendation_json.read_text(encoding="utf-8"))
    pipeline_config = recommendation.get("pipeline_config")
    if not isinstance(pipeline_config, dict):
        raise ValueError("recommendation.json does not contain pipeline_config")
    dataset = load_gitlab_openml_dataset(
        args.dataset_id,
        cache_dir=str(args.data_dir),
        test_dataset_ids=[int(value) for value in EVAL_IDS],
        verbose=True,
        max_samples_if_test=int(args.max_samples),
    )
    if dataset is None:
        raise RuntimeError(f"Could not load dataset {args.dataset_id}")
    dataset["dataset_id"] = int(args.dataset_id)
    dataset["name"] = dataset.get("name", f"D_{args.dataset_id}")
    model = None
    try:
        result, model = evaluate_recommendation(
            dataset,
            pipeline_config,
            split_seed=args.split_seed,
            h2o_preprocessing=None if args.h2o_preprocessing == "none" else args.h2o_preprocessing,
            max_runtime_secs=args.max_runtime_secs,
            max_runtime_secs_per_model=args.max_runtime_secs_per_model,
            max_models=args.max_models,
            nfolds=args.nfolds,
            seed=args.seed,
            nthreads=args.nthreads,
            max_mem_size=args.max_mem_size,
        )
        result.update(
            {
                "recommendation_json": str(args.recommendation_json),
                "aco_proxy_score": recommendation.get("recommended_performance"),
                "recommendation_protocol": recommendation.get("recommendation_protocol"),
                "max_samples": int(args.max_samples),
            }
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "method": "acorec_h2o",
            "dataset_id": int(args.dataset_id),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        raise
    finally:
        del model
        gc.collect()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"H2O outer-test score: {result['score']:.6f}")
    print("Saved:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
