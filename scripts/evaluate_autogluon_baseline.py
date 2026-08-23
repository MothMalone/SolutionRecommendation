#!/usr/bin/env python3
"""Evaluate one canonical dataset with an AutoGluon baseline setting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from automl_aco.data.loaders import load_gitlab_openml_dataset  # noqa: E402
from automl_aco.data.splits import split_train_val_test  # noqa: E402
from automl_aco.eval_ids import EVAL_IDS  # noqa: E402
from autogluon_evaluator import evaluate_autogluon_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--setting", choices=["no_preprocessing", "default_preprocessing"], required=True)
    parser.add_argument("--feature-generator", choices=["identity", "default"], required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--presets", default="best_quality")
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_json.exists() and not args.force:
        existing = json.loads(args.output_json.read_text(encoding="utf-8"))
        if existing.get("status") == "ok":
            print("SKIP successful:", args.output_json)
            return 0

    started = time.perf_counter()
    try:
        dataset = load_gitlab_openml_dataset(
            args.dataset_id,
            cache_dir=str(args.data_dir),
            test_dataset_ids=[int(value) for value in EVAL_IDS],
            verbose=True,
            max_samples_if_test=int(args.max_samples),
        )
        if dataset is None:
            raise RuntimeError(f"Could not load dataset {args.dataset_id}")
        X = pd.DataFrame(dataset["X"]).copy()
        y = pd.Series(dataset["y"]).copy()
        X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
            X, y, seed=int(args.split_seed)
        )
        result = evaluate_autogluon_split(
            X_train, y_train, X_val, y_val, X_test, y_test,
            task_type=str(dataset.get("task_type", "classification")),
            feature_generator=args.feature_generator,
            time_limit=int(args.time_limit),
            presets=args.presets,
            seed=int(args.split_seed),
        )
        result.update(
            {
                "dataset_id": int(args.dataset_id),
                "dataset": args.dataset_name,
                "setting": args.setting,
                "split_seed": int(args.split_seed),
                "dataset_backend": dataset.get("download_backend"),
                "dataset_total_seconds": float(time.perf_counter() - started),
                "max_samples": int(args.max_samples),
            }
        )
    except Exception as error:
        result = {
            "dataset_id": int(args.dataset_id),
            "dataset": args.dataset_name,
            "setting": args.setting,
            "status": "failed",
            "split_seed": int(args.split_seed),
            "error_type": type(error).__name__,
            "error": str(error)[:4000],
            "dataset_total_seconds": float(time.perf_counter() - started),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print("FAILED:", result["error"])
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"AutoGluon {args.setting} score: {result['score']:.6f}")
    print("Saved:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
