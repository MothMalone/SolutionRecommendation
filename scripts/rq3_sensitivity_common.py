"""Shared utilities for RQ3 sensitivity ablation runs."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


def _default_root() -> str:
    return os.environ.get("ROOT", str(Path(__file__).resolve().parents[1]))


def _default_deps() -> str:
    return os.environ.get("DEPS", "")


def _default_dataset_ids() -> List[str]:
    # 20 benchmark datasets used in your experiments.
    return [
        "248",
        "1066",
        "1164",
        "1047",
        "862",
        "2",
        "40663",
        "1054",
        "1387",
        "876",
        "18",
        "1520",
        "1548",
        "184",
        "378",
        "381",
        "382",
        "993",
        "1485",
        "14",
    ]


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", default=_default_root(), help="Repo root path")
    parser.add_argument("--deps", default=_default_deps(), help="Extra PYTHONPATH prefix (e.g., /kaggle/working/acorec_deps)")
    parser.add_argument("--performance-matrix", default=None, help="Override performance matrix path")
    parser.add_argument("--metafeatures", default=None, help="Override metafeatures path")
    parser.add_argument("--pipeline-configs", default=None, help="Override pipeline configs path")
    parser.add_argument("--dataset-source", choices=["openml", "kaggle"], default="openml")
    parser.add_argument("--openml-local-folder", default=None, help="Optional local fallback folder for OpenML datasets")
    parser.add_argument("--kaggle-data-folder", default=None, help="CSV folder when --dataset-source kaggle")
    parser.add_argument("--kaggle-target-column", default="target")
    parser.add_argument("--dataset-ids", nargs="+", default=_default_dataset_ids(), help="Dataset ids to evaluate")
    parser.add_argument("--output-root", default="/kaggle/working/rq3_sensitivity", help="Root output folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--n-ants", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--k", type=int, default=5, help="Top-k similar datasets for retrieval")
    parser.add_argument("--eval-k", type=int, default=3)
    parser.add_argument("--final-autogluon-topk", type=int, default=3)
    parser.add_argument("--score-direction", choices=["higher_is_better", "lower_is_better"], default="higher_is_better")
    parser.add_argument("--require-autogluon", action="store_true", help="Require AutoGluon final eval")
    parser.add_argument("--allow-autogluon-fallback", action="store_true", help="Allow fallback if AG unavailable")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser


def resolve_default_paths(args: argparse.Namespace) -> Dict[str, str]:
    root = Path(args.root)
    performance_matrix = args.performance_matrix or str(root / "aco" / "training_performance_matrix_autogluon.csv")
    metafeatures = args.metafeatures or str(root / "data" / "openml" / "dataset_feats.csv")
    pipeline_configs = args.pipeline_configs or str(root / "aco" / "pipeline_configs.json")
    openml_local = args.openml_local_folder or str(root / "test_data_local")
    kaggle_data = args.kaggle_data_folder or str(root / "test_data_local")
    return {
        "performance_matrix": performance_matrix,
        "metafeatures": metafeatures,
        "pipeline_configs": pipeline_configs,
        "openml_local_folder": openml_local,
        "kaggle_data_folder": kaggle_data,
    }


def _build_base_command(args: argparse.Namespace, paths: Mapping[str, str]) -> List[str]:
    root = Path(args.root)
    run_script = root / "scripts" / "run_recommend.py"
    command = [
        sys.executable,
        str(run_script),
        "--performance-matrix",
        str(paths["performance_matrix"]),
        "--metafeatures",
        str(paths["metafeatures"]),
        "--pipeline-configs",
        str(paths["pipeline_configs"]),
        "--dataset-source",
        str(args.dataset_source),
        "--dataset-ids",
        *[str(x) for x in args.dataset_ids],
        "--use-aco",
        "--optimizer",
        "aco",
        "--k",
        str(int(args.k)),
        "--eval-k",
        str(int(args.eval_k)),
        "--n-ants",
        str(int(args.n_ants)),
        "--n-iterations",
        str(int(args.n_iterations)),
        "--seed",
        str(int(args.seed)),
        "--time-limit",
        str(int(args.time_limit)),
        "--final-autogluon-topk",
        str(int(args.final_autogluon_topk)),
        "--score-direction",
        str(args.score_direction),
    ]
    if args.dataset_source == "openml":
        command.extend(["--openml-local-folder", str(paths["openml_local_folder"])])
    else:
        command.extend(
            [
                "--kaggle-data-folder",
                str(paths["kaggle_data_folder"]),
                "--kaggle-target-column",
                str(args.kaggle_target_column),
            ]
        )

    if args.require_autogluon and not args.allow_autogluon_fallback:
        command.append("--require-autogluon")
    elif args.allow_autogluon_fallback:
        command.append("--allow-autogluon-fallback")

    if args.verbose:
        command.append("--verbose")
    return command


def _read_summary(output_dir: Path) -> Dict[str, Any]:
    summary_path = output_dir / "recommendations_summary.json"
    if not summary_path.exists():
        return {
            "status": "missing_summary",
            "summary_path": str(summary_path),
        }
    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    aggregate = payload.get("aggregate", {}) if isinstance(payload, dict) else {}
    return {
        "status": "ok",
        "summary_path": str(summary_path),
        "num_ok": aggregate.get("num_ok"),
        "num_failed": aggregate.get("num_failed"),
        "avg_elapsed_seconds": aggregate.get("avg_elapsed_seconds"),
        "avg_proxy_score": aggregate.get("avg_proxy_score"),
        "avg_final_score": aggregate.get("avg_final_score"),
        "avg_autogluon_score": aggregate.get("avg_autogluon_score"),
    }


def run_sensitivity_suite(
    *,
    args: argparse.Namespace,
    suite_name: str,
    variants: Sequence[Mapping[str, Any]],
) -> int:
    paths = resolve_default_paths(args)
    base_command = _build_base_command(args, paths)

    output_root = Path(args.output_root).resolve()
    suite_dir = output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, variant in enumerate(variants, start=1):
        variant_name = str(variant["name"])
        extra_flags = [str(x) for x in variant.get("flags", [])]
        variant_dir = suite_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        command = list(base_command) + extra_flags + ["--output-dir", str(variant_dir)]
        command_str = " ".join(shlex.quote(token) for token in command)
        print(f"\n[{idx}/{len(variants)}] {suite_name}:{variant_name}")
        print(command_str)

        row: Dict[str, Any] = {
            "suite": suite_name,
            "variant": variant_name,
            "output_dir": str(variant_dir),
            "command": command_str,
            "flags": " ".join(extra_flags),
        }

        if args.dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            continue

        env = os.environ.copy()
        pythonpath_parts: List[str] = []
        deps = str(args.deps).strip()
        if deps:
            pythonpath_parts.append(deps)
        pythonpath_parts.append(str(Path(args.root) / "src"))
        pythonpath_parts.append(str(args.root))
        existing = env.get("PYTHONPATH", "").strip()
        if existing:
            pythonpath_parts.append(existing)
        env["PYTHONPATH"] = ":".join(pythonpath_parts)

        proc = subprocess.run(command, env=env, check=False)
        row["return_code"] = int(proc.returncode)
        execution_status = "ok" if proc.returncode == 0 else "failed"
        row["execution_status"] = execution_status

        summary = _read_summary(variant_dir)
        summary_status = str(summary.get("status", "missing_summary"))
        row.update({k: v for k, v in summary.items() if k != "status"})
        row["summary_status"] = summary_status

        num_failed = summary.get("num_failed")
        if execution_status == "failed":
            row["status"] = "failed"
        elif summary_status == "ok":
            if isinstance(num_failed, (int, float)) and int(num_failed) > 0:
                row["status"] = "partial_failed"
            else:
                row["status"] = "ok"
        else:
            # Single-dataset runs do not emit recommendations_summary.json.
            row["status"] = "ok_no_summary"
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_csv = suite_dir / "sensitivity_results.csv"
    results_json = suite_dir / "sensitivity_results.json"
    results_df.to_csv(results_csv, index=False)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    print("\nSaved sensitivity results:")
    print(f"  {results_csv}")
    print(f"  {results_json}")

    if results_df.empty:
        return 0
    failed = int(results_df["status"].isin(["failed", "partial_failed"]).sum())
    return 1 if failed > 0 else 0
