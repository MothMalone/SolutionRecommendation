#!/usr/bin/env python3
"""Evaluate a frozen ACORec pipeline with AutoGluon on the shared outer split."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from automl_aco.data.loaders import load_gitlab_openml_dataset  # noqa: E402
from automl_aco.data.splits import split_train_val_test  # noqa: E402
from automl_aco.eval_ids import EVAL_IDS  # noqa: E402
from automl_aco.search.evaluation import _fit_pipeline, _make_preprocessor  # noqa: E402
from autogluon_evaluator import evaluate_autogluon_split  # noqa: E402


def evaluate_recommendation(
    dataset: Dict[str, Any],
    pipeline_config: Dict[str, Any],
    *,
    split_seed: int = 42,
    time_limit: int = 300,
    presets: str = "best_quality",
) -> dict:
    X = pd.DataFrame(dataset["X"]).copy()
    y = pd.Series(dataset["y"]).copy()
    task_type = str(dataset.get("task_type", "classification"))
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        X, y, seed=int(split_seed)
    )
    preprocessor = _make_preprocessor(dict(pipeline_config))
    X_train_processed, y_train_processed = _fit_pipeline(
        preprocessor, X_train, y_train, prepare_mode="leakfree"
    )
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    result = evaluate_autogluon_split(
        X_train_processed,
        y_train_processed,
        X_val_processed,
        y_val,
        X_test_processed,
        y_test,
        task_type=task_type,
        feature_generator="identity",
        time_limit=int(time_limit),
        presets=presets,
        seed=int(split_seed),
    )
    result.update(
        {
            "method": "acorec_autogluon",
            "dataset_id": dataset.get("dataset_id"),
            "dataset_name": dataset.get("name"),
            "operator_space": "ours",
            "pipeline_config": dict(pipeline_config),
            "split_seed": int(split_seed),
            "validation_used_for_model_selection": True,
            "outer_test_seen_during_selection": False,
            "train_rows_raw": int(len(X_train)),
            "train_rows_processed": int(len(y_train_processed)),
            "validation_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "raw_features": int(X.shape[1]),
            "transformed_features": int(X_train_processed.shape[1]),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendation-json", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--presets", default="best_quality")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_json.exists() and not args.force:
        existing = json.loads(args.output_json.read_text(encoding="utf-8"))
        if existing.get("status") == "ok":
            print("SKIP successful AutoGluon evaluation:", args.output_json)
            return 0

    started = time.perf_counter()
    try:
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
        result = evaluate_recommendation(
            dataset,
            pipeline_config,
            split_seed=args.split_seed,
            time_limit=args.time_limit,
            presets=args.presets,
        )
        result.update(
            {
                "recommendation_json": str(args.recommendation_json),
                "aco_proxy_score": recommendation.get("recommended_performance"),
                "recommendation_protocol": recommendation.get("recommendation_protocol"),
                "max_samples": int(args.max_samples),
                "acorec_and_evaluation_wall_clock_seconds": float(time.perf_counter() - started),
            }
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "method": "acorec_autogluon",
            "dataset_id": int(args.dataset_id),
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "acorec_and_evaluation_wall_clock_seconds": float(time.perf_counter() - started),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print("FAILED:", result["error"])
        return 1
    finally:
        gc.collect()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"ACORec + AutoGluon outer-test score: {result['score']:.6f}")
    print("Saved:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
