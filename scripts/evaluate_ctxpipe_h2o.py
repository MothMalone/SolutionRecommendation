#!/usr/bin/env python3
"""Replay a frozen native CtxPipe sequence and evaluate it with H2O AutoML.

CtxPipe recommendation is produced on the outer train+validation rows.  This
adapter fits the frozen preprocessing sequence on outer train only, transforms
validation and test, lets H2O select a model on validation, and scores the
untouched outer test exactly once.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automl_aco.data.loaders import load_gitlab_openml_dataset  # noqa: E402
from automl_aco.data.splits import split_train_val_test  # noqa: E402
from automl_aco.eval_ids import EVAL_IDS  # noqa: E402
from h2o_evaluator import evaluate_h2o_frames  # noqa: E402
from evaluate_ctxpipe_tpot import replay_ctxpipe_sequence  # noqa: E402


def evaluate_recommendation(
    dataset: dict,
    sequence: list[str],
    *,
    split_seed: int = 42,
    max_runtime_secs: int = 300,
    max_runtime_secs_per_model: int = 60,
    nfolds: int = 5,
    nthreads: int = 1,
    max_mem_size: str = "6G",
) -> dict:
    if str(dataset.get("task_type", "classification")) != "classification":
        raise ValueError("Native CtxPipe H2O experiment currently supports classification only")

    X = pd.DataFrame(dataset["X"]).copy()
    y = pd.Series(dataset["y"]).copy()
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        X, y, seed=int(split_seed)
    )

    # Fit each CtxPipe transformer once on train, then transform validation and
    # test together. This avoids accidentally refitting on either holdout.
    eval_input = pd.concat([X_val, X_test], ignore_index=True)
    train_processed, eval_processed, trace = replay_ctxpipe_sequence(
        sequence,
        X_train,
        eval_input,
        y_train,
    )
    val_processed = eval_processed.iloc[: len(X_val)].reset_index(drop=True)
    test_processed = eval_processed.iloc[len(X_val) :].reset_index(drop=True)

    result, model = evaluate_h2o_frames(
        train_processed,
        y_train,
        val_processed,
        y_val,
        test_processed,
        y_test,
        task_type="classification",
        h2o_preprocessing=None,
        max_runtime_secs=int(max_runtime_secs),
        max_runtime_secs_per_model=int(max_runtime_secs_per_model),
        nfolds=int(nfolds),
        seed=42,
        nthreads=int(nthreads),
        max_mem_size=str(max_mem_size),
    )
    del model
    gc.collect()
    result.update(
        {
            "method": "ctxpipe_h2o",
            "dataset_id": int(dataset["id"]),
            "dataset_name": dataset.get("name", f"D_{dataset['id']}"),
            "split_seed": int(split_seed),
            "raw_rows": int(len(X)),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "outer_test_seen_during_ctxpipe_search": False,
            "ctxpipe_search_rows": int(len(X_train) + len(X_val)),
            "ctxpipe_sequence": [str(value) for value in sequence],
            "ctxpipe_replay": "fit_train_transform_val_test",
            "operator_trace": trace,
            "raw_features": int(X.shape[1]),
            "transformed_features": int(train_processed.shape[1]),
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctxpipe-result-json", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-runtime-secs", type=int, default=300)
    parser.add_argument("--max-runtime-secs-per-model", type=int, default=60)
    parser.add_argument("--nfolds", type=int, default=5)
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--max-mem-size", default="6G")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_json.exists() and not args.force:
        existing = json.loads(args.output_json.read_text(encoding="utf-8"))
        if existing.get("status") == "ok":
            print("SKIP successful result:", args.output_json)
            return 0

    ctxpipe_result = json.loads(args.ctxpipe_result_json.read_text(encoding="utf-8"))
    sequence = ctxpipe_result.get("sequence")
    if not isinstance(sequence, list) or len(sequence) != 6:
        raise ValueError("CtxPipe result must contain a six-operator sequence")
    dataset = load_gitlab_openml_dataset(
        args.dataset_id,
        cache_dir=str(args.data_dir),
        test_dataset_ids=[int(value) for value in EVAL_IDS],
        verbose=True,
        max_samples_if_test=int(args.max_samples),
    )
    if dataset is None:
        raise RuntimeError(f"Could not load dataset {args.dataset_id}")

    try:
        result = evaluate_recommendation(
            dataset,
            sequence,
            split_seed=args.split_seed,
            max_runtime_secs=args.max_runtime_secs,
            max_runtime_secs_per_model=args.max_runtime_secs_per_model,
            nfolds=args.nfolds,
            nthreads=args.nthreads,
            max_mem_size=args.max_mem_size,
        )
        result.update(
            {
                "dataset_backend": dataset.get("download_backend"),
                "max_samples": int(args.max_samples),
                "ctxpipe_checkpoint": ctxpipe_result.get("checkpoint"),
                "ctxpipe_native_reward": ctxpipe_result.get("native_reward"),
                "ctxpipe_official_commit": ctxpipe_result.get("official_commit"),
            }
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "method": "ctxpipe_h2o",
            "dataset_id": int(args.dataset_id),
            "ctxpipe_sequence": sequence,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        raise

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"CtxPipe + H2O outer-test accuracy: {result['score']:.6f}")
    print("Saved:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
