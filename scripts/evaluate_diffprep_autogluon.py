#!/usr/bin/env python3
"""Replay a frozen DiffPrep pipeline and score it with AutoGluon."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from automl_aco.data.splits import split_train_val_test  # noqa: E402
from autogluon_evaluator import evaluate_autogluon_split  # noqa: E402


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_saved_pipeline(repo_dir: Path, method: str, dataset_key: str):
    directory = repo_dir / "saved_pipelines" / method / dataset_key
    with (directory / "pipeline.pkl").open("rb") as handle:
        pipeline = pickle.load(handle)
    with (directory / "data_splits.pkl").open("rb") as handle:
        saved_split = pickle.load(handle)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return pipeline, saved_split, metadata, directory


def _add_diffprep_import_path(repo_dir: Path) -> Path:
    """Expose the original DiffPrep checkout before unpickling its pipeline."""
    resolved = repo_dir.resolve()
    if not (resolved / "pipeline").is_dir():
        raise FileNotFoundError(f"DiffPrep checkout has no pipeline package: {resolved}")
    path = str(resolved)
    if path not in sys.path:
        sys.path.insert(0, path)
    return resolved


def _raw_split(repo_dir: Path, dataset_key: str, split_seed: int):
    frame = pd.read_csv(repo_dir / "data" / dataset_key / "data.csv")
    target = "target"
    if target not in frame.columns:
        raise KeyError(f"DiffPrep input has no {target!r} column")
    frame = frame.loc[~frame[target].isna()].reset_index(drop=True)
    X = frame.drop(columns=[target]).copy()
    y = frame[target].copy()
    return split_train_val_test(X, y, seed=int(split_seed))


def _assert_target_alignment(raw_split, saved_split):
    for part in ("train", "val", "test"):
        raw_y = np.asarray(raw_split[1 if part == "train" else 3 if part == "val" else 5]).reshape(-1)
        saved_key = f"y_{part}"
        saved_y = _as_numpy(saved_split[saved_key]).reshape(-1)
        if raw_y.shape != saved_y.shape:
            raise RuntimeError(f"DiffPrep split mismatch in {part}: {raw_y.shape} vs {saved_y.shape}")
        raw_to_saved: dict[str, str] = {}
        saved_to_raw: dict[str, str] = {}
        for raw_label, saved_label in zip(raw_y, saved_y):
            raw_key, saved_key_value = str(raw_label), str(saved_label)
            if raw_key in raw_to_saved and raw_to_saved[raw_key] != saved_key_value:
                raise RuntimeError(f"DiffPrep label alignment mismatch in {part}")
            if saved_key_value in saved_to_raw and saved_to_raw[saved_key_value] != raw_key:
                raise RuntimeError(f"DiffPrep label alignment mismatch in {part}")
            raw_to_saved[raw_key] = saved_key_value
            saved_to_raw[saved_key_value] = raw_key


def _transform(pipeline, saved_split):
    if not pipeline.is_fitted:
        pipeline.fit(saved_split["X_train"])
    if "test" not in pipeline.pipeline[0].cache:
        pipeline.pipeline[0].pre_cache(saved_split["X_test"], "test")
    transformed = {}
    with torch.no_grad():
        for part in ("train", "val", "test"):
            array = _as_numpy(
                pipeline.transform(
                    saved_split[f"X_{part}"],
                    X_type=part,
                    max_only=True,
                    resample=False,
                )
            ).astype(np.float32, copy=False)
            if not np.isfinite(array).all():
                raise ValueError(f"DiffPrep produced NaN/inf in transformed {part}")
            transformed[part] = array
    return transformed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--method", default="diffprep_fix")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--presets", default="best_quality")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_json.exists() and not args.force:
        existing = json.loads(args.output_json.read_text(encoding="utf-8"))
        if existing.get("status") == "ok":
            print("SKIP successful:", args.output_json)
            return 0

    started = time.perf_counter()
    try:
        args.repo_dir = _add_diffprep_import_path(args.repo_dir)
        pipeline, saved_split, metadata, pipeline_dir = _load_saved_pipeline(
            args.repo_dir, args.method, args.dataset_key
        )
        raw_split = _raw_split(args.repo_dir, args.dataset_key, args.split_seed)
        _assert_target_alignment(raw_split, saved_split)
        transformed = _transform(pipeline, saved_split)
        result = evaluate_autogluon_split(
            transformed["train"], saved_split["y_train"],
            transformed["val"], saved_split["y_val"],
            transformed["test"], saved_split["y_test"],
            task_type="classification",
            feature_generator="identity",
            time_limit=int(args.time_limit),
            presets=args.presets,
            seed=int(args.train_seed),
        )
        autogluon_total_seconds = result.get("total_seconds")
        result.update(
            {
                "dataset_id": int(args.dataset_id),
                "dataset": args.dataset_name,
                "dataset_key": args.dataset_key,
                "setting": "diffprep",
                "method": "diffprep_autogluon",
                "split_seed": int(args.split_seed),
                "train_seed": int(args.train_seed),
                "diffprep_test_seen_during_search": False,
                "diffprep_pipeline_config": str(pipeline_dir / "pipeline_config.json"),
                "diffprep_internal_test_accuracy": metadata.get("original_test_acc"),
                "raw_features": int(saved_split["X_train"].shape[1]),
                "transformed_features": int(transformed["train"].shape[1]),
                "autogluon_total_seconds": autogluon_total_seconds,
                "diffprep_and_evaluation_wall_clock_seconds": float(time.perf_counter() - started),
            }
        )
    except Exception as error:
        result = {
            "dataset_id": int(args.dataset_id),
            "dataset": args.dataset_name,
            "dataset_key": args.dataset_key,
            "setting": "diffprep",
            "method": "diffprep_autogluon",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error)[:4000],
            "total_seconds": float(time.perf_counter() - started),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print("FAILED:", result["error"])
        return 1
    finally:
        gc.collect()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"DiffPrep + AutoGluon outer-test score: {result['score']:.6f}")
    print("Saved:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
