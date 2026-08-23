#!/usr/bin/env python3
"""Run/resume one leak-free ACORec similarity x search ablation on meta-dev18."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _last_history(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def _output_root(args) -> Path:
    return args.output_dir / f"sim={args.similarity_variant}__search={args.search_variant}"


def _stamp_recommendation(path: Path, args, dataset_id: int, aco_seed: int, screening: dict) -> None:
    """Attach the protocol provenance required to reproduce an individual run."""
    payload = _load_json(path)
    payload["experiment_metadata"] = {
        "protocol": "meta-dev18-leave-one-dataset-out",
        "git_commit": _git_commit(),
        "similarity_variant": args.similarity_variant,
        "search_variant": args.search_variant,
        "query_dataset_id": int(dataset_id),
        "reference_holdout_ids": [int(dataset_id)],
        "aco_seed": int(aco_seed),
        "split_seed": int(args.split_seed),
        "tpot_seed": int(screening["tpot_seed"]),
        "tpot_minutes": int(
            screening["confirmation_tpot_minutes"]
            if args.confirmation
            else screening["tpot_minutes"]
        ),
        "confirmation": bool(args.confirmation),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_one(args, dataset_id: int, aco_seed: int, variants: dict, screening: dict) -> dict:
    run_dir = _output_root(args) / f"dataset_{dataset_id}" / f"aco_seed_{aco_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    recommendation = run_dir / "recommendation.json"
    tpot_output = run_dir / "tpot_evaluation.json"
    failure_path = run_dir / "failure.json"
    if failure_path.exists() and args.force:
        failure_path.unlink()

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    try:
        # Search is intentionally resumable. --force only re-runs the final TPOT
        # evaluation; use a new output directory for a genuinely new ACO search.
        if args.stage in {"search", "both"} and not recommendation.exists():
            command = [
                sys.executable,
                str(ROOT / "scripts/run_recommend.py"),
                "--operator-space", "ours",
                "--performance-matrix", str(ROOT / "data/openml/training_performance_matrix_autogluon.csv"),
                "--metafeatures", str(ROOT / "data/openml/dataset_feats.csv"),
                "--pipeline-configs", str(ROOT / "aco/pipeline_configs.json"),
                "--dataset-source", "openml",
                "--openml-backend", "gitlab",
                "--openml-local-folder", str(args.cache_dir),
                "--dataset-ids", str(dataset_id),
                "--reference-holdout-ids", str(dataset_id),
                "--optimizer", "aco",
                "--seed", str(aco_seed),
                "--workers", "1",
                "--output-dir", str(run_dir),
                "--skip-aco-plot",
                "--no-autogluon",
                "--recommend-on-train-val",
                "--recommend-split-seed", str(args.split_seed),
                "--n-ants", str(screening["n_ants"]),
                "--n-iterations", str(screening["n_iterations"]),
                "--metric-epochs", str(args.metric_epochs),
                "--aco-lambda-smooth", "0",
                "--aco-search-fixes",
                *variants["similarity"][args.similarity_variant],
                *variants["search"][args.search_variant],
            ]
            subprocess.run(command, cwd=ROOT, env=env, check=True)

        if recommendation.exists():
            _stamp_recommendation(recommendation, args, dataset_id, aco_seed, screening)

        if args.stage in {"tpot", "both"}:
            if not recommendation.exists():
                raise FileNotFoundError(f"Missing recommendation: {recommendation}")
            if args.force or not tpot_output.exists() or _load_json(tpot_output).get("status") != "ok":
                minutes = screening["confirmation_tpot_minutes"] if args.confirmation else screening["tpot_minutes"]
                command = [
                    sys.executable,
                    str(ROOT / "scripts/evaluate_acorec_tpot.py"),
                    "--recommendation-json", str(recommendation),
                    "--dataset-id", str(dataset_id),
                    "--data-dir", str(args.cache_dir),
                    "--output-json", str(tpot_output),
                    "--max-samples", "100000",
                    "--split-seed", str(args.split_seed),
                    "--tpot-seed", str(screening["tpot_seed"]),
                    "--max-time-mins", str(minutes),
                    "--max-eval-time-mins", "1",
                    "--n-jobs", str(args.tpot_jobs),
                    "--memory-limit", args.tpot_memory,
                    "--population-size", str(args.tpot_population),
                    "--max-cv-folds", "5",
                    "--verbose", "1",
                ]
                if args.force:
                    command.append("--force")
                subprocess.run(command, cwd=ROOT, env=env, check=True)
        return {"dataset_id": dataset_id, "aco_seed": aco_seed, "status": "ok"}
    except Exception as exc:
        failure = {
            "dataset_id": dataset_id,
            "aco_seed": aco_seed,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return failure


def aggregate(args) -> int:
    root = _output_root(args)
    rows = []
    for path in root.glob("dataset_*/aco_seed_*/tpot_evaluation.json"):
        result = _load_json(path)
        run_dir = path.parent
        recommendation_path = run_dir / "recommendation.json"
        recommendation = _load_json(recommendation_path) if recommendation_path.exists() else {}
        history = _last_history(run_dir / "aco_history.csv")
        rows.append(
            {
                "dataset_id": int(path.parents[1].name.split("_")[-1]),
                "aco_seed": int(path.parent.name.split("_")[-1]),
                "similarity_variant": args.similarity_variant,
                "search_variant": args.search_variant,
                "status": result.get("status"),
                "accuracy": result.get("accuracy"),
                "balanced_accuracy": result.get("balanced_accuracy"),
                "f1_macro": result.get("f1_macro"),
                "tpot_fit_seconds": result.get("fit_seconds"),
                "tpot_minutes": result.get("max_time_mins"),
                "proxy_score": recommendation.get("recommended_performance"),
                "unique_proxy_evaluations": history.get("cache_size"),
                "proxy_evaluation_requests": history.get("cumulative_evaluation_request_count"),
                "sampling_draws": history.get("cumulative_draw_count"),
                "split_fingerprints": json.dumps(result.get("split_fingerprints", {}), sort_keys=True),
                "git_commit": recommendation.get("experiment_metadata", {}).get(
                    "git_commit", _git_commit()
                ),
                "result_path": str(path),
            }
        )
    frame = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "accuracy_results.csv", index=False)
    ok = frame[frame.status == "ok"].copy() if not frame.empty else frame
    summary = {
        "similarity_variant": args.similarity_variant,
        "search_variant": args.search_variant,
        "runs": int(len(frame)),
        "successful_runs": int(len(ok)),
        "macro_mean_accuracy": float(pd.to_numeric(ok.accuracy).mean()) if len(ok) else None,
        "median_accuracy": float(pd.to_numeric(ok.accuracy).median()) if len(ok) else None,
        "mean_tpot_fit_seconds": float(pd.to_numeric(ok.tpot_fit_seconds).mean()) if len(ok) else None,
        "mean_unique_proxy_evaluations": float(pd.to_numeric(ok.unique_proxy_evaluations).mean()) if len(ok) else None,
        "git_commit": _git_commit(),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/openml/meta_dev18.json")
    parser.add_argument("--variants", type=Path, default=ROOT / "data/openml/acorec_ablation_variants.json")
    parser.add_argument("--similarity-variant", required=True)
    parser.add_argument("--search-variant", required=True)
    parser.add_argument("--stage", choices=["search", "tpot", "both", "aggregate"], default="both")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/acorec_meta_dev_ablation")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/acorec_meta_dev_data")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--aco-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--metric-epochs", type=int, default=100)
    parser.add_argument("--tpot-jobs", type=int, default=1)
    parser.add_argument("--tpot-memory", default="5GB")
    parser.add_argument("--tpot-population", type=int, default=20)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run TPOT/failures while retaining an existing resumable ACO search",
    )
    args = parser.parse_args()
    manifest = _load_json(args.manifest)
    variants = _load_json(args.variants)
    if args.similarity_variant not in variants["similarity"]:
        parser.error("Unknown similarity variant")
    if args.search_variant not in variants["search"]:
        parser.error("Unknown search variant")
    if args.stage == "aggregate":
        return aggregate(args)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must satisfy 0 <= index < num-shards")
    screening = variants["screening"]
    seeds = args.aco_seeds or screening["aco_seeds"]
    dataset_ids = [int(value) for value in manifest["dataset_ids"]][args.shard_index :: args.num_shards]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_one(args, dataset_id, seed, variants, screening)
        for dataset_id in dataset_ids
        for seed in seeds
    ]
    root = _output_root(args)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "manifest": manifest["name"],
        "dataset_ids": dataset_ids,
        "aco_seeds": seeds,
        "split_seed": args.split_seed,
        "similarity_variant": args.similarity_variant,
        "search_variant": args.search_variant,
        "git_commit": _git_commit(),
        "results": rows,
    }
    shard_name = f"run_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    (root / shard_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0 if all(row["status"] == "ok" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
