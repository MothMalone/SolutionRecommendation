#!/usr/bin/env python3
"""Run and summarize RQ2 optimizer ablations across the 13 OpenML evaluation datasets.

Optimizers:
  Random, Greedy, Simulated Annealing (sa), Genetic (ga), MCTS (mcts), TPE (tpe), ACO (aco)

Datasets (13 OpenML test datasets):
  kc1-binary (1066), usp05 (1047), sleuth-ex2016 (862), calendarDOW (40663),
  mc2 (1054), fri-c1 (876), mfeat-morphological (18), robot-failures-lp5 (1520),
  autoUniv-au4 (1548), ipums-la-99 (378), madelon (1485), mfeat-fourier (14), colic (27)

Usage:
  # Run all missing ablation cells:
  python scripts/run_rq2_ablations.py --data-dir /kaggle/working/eval_all --out outputs/rq2_ablations.jsonl

  # Run specific optimizer or dataset:
  python scripts/run_rq2_ablations.py --optimizers ga,mcts --datasets 1047,378 --data-dir /kaggle/working/eval_all

  # Summarize results into the full RQ2 table:
  python scripts/run_rq2_ablations.py --summarize outputs/rq2_ablations.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Canonical 13 datasets for RQ2
DATASETS_13 = [
    ("kc1-binary", "1066"),
    ("usp05", "1047"),
    ("sleuth-ex2016", "862"),
    ("calendarDOW", "40663"),
    ("mc2", "1054"),
    ("fri-c1", "876"),
    ("mfeat-morphological", "18"),
    ("robot-failures-lp5", "1520"),
    ("autoUniv-au4", "1548"),
    ("ipums-la-99", "378"),
    ("madelon", "1485"),
    ("mfeat-fourier", "14"),
    ("colic", "27"),
]

DATASET_NAME_TO_ID = {name: did for name, did in DATASETS_13}
DATASET_ID_TO_NAME = {did: name for name, did in DATASETS_13}

OPTIMIZERS = [
    ("Random", "random"),
    ("Greedy", "greedy"),
    ("Simulated Annealing", "sa"),
    ("Genetic", "ga"),
    ("MCTS", "mcts"),
    ("TPE", "tpe"),
    ("ACO", "aco"),
]

OPT_NAME_TO_FLAG = {name: flag for name, flag in OPTIMIZERS}
OPT_FLAG_TO_NAME = {flag: name for name, flag in OPTIMIZERS}

# Base RQ2 results provided by user (None represents missing cells to fill)
BASE_RQ2_TABLE: Dict[str, Dict[str, Optional[float]]] = {
    "kc1-binary": {"Random": 0.828, "Greedy": 0.828, "Simulated Annealing": 0.759, "Genetic": 0.759, "MCTS": 0.828, "TPE": 0.828, "ACO": 0.862},
    "usp05": {"Random": 0.921, "Greedy": 0.921, "Simulated Annealing": 0.921, "Genetic": None, "MCTS": None, "TPE": None, "ACO": 0.921},
    "sleuth-ex2016": {"Random": 0.824, "Greedy": 0.824, "Simulated Annealing": 0.765, "Genetic": 0.824, "MCTS": 0.765, "TPE": 0.765, "ACO": 0.824},
    "calendarDOW": {"Random": 0.671, "Greedy": 0.671, "Simulated Annealing": 0.671, "Genetic": 0.671, "MCTS": 0.633, "TPE": 0.671, "ACO": 0.684},
    "mc2": {"Random": 0.719, "Greedy": 0.719, "Simulated Annealing": 0.781, "Genetic": 0.844, "MCTS": 0.719, "TPE": 0.812, "ACO": 0.813},
    "fri-c1": {"Random": 0.700, "Greedy": 0.700, "Simulated Annealing": 0.700, "Genetic": 0.700, "MCTS": 0.700, "TPE": 0.700, "ACO": 0.700},
    "mfeat-morphological": {"Random": 0.783, "Greedy": 0.785, "Simulated Annealing": 0.765, "Genetic": 0.775, "MCTS": 0.810, "TPE": 0.775, "ACO": 0.813},
    "robot-failures-lp5": {"Random": 0.688, "Greedy": 0.656, "Simulated Annealing": 0.625, "Genetic": 0.688, "MCTS": 0.594, "TPE": 0.688, "ACO": 0.688},
    "autoUniv-au4": {"Random": 0.626, "Greedy": 0.676, "Simulated Annealing": 0.606, "Genetic": 0.678, "MCTS": 0.628, "TPE": 0.670, "ACO": 0.684},
    "ipums-la-99": {"Random": 0.786, "Greedy": 0.786, "Simulated Annealing": None, "Genetic": None, "MCTS": None, "TPE": None, "ACO": 0.814},
    "madelon": {"Random": 0.825, "Greedy": 0.823, "Simulated Annealing": 0.829, "Genetic": None, "MCTS": None, "TPE": None, "ACO": 0.887},
    "mfeat-fourier": {"Random": 0.870, "Greedy": 0.878, "Simulated Annealing": None, "Genetic": None, "MCTS": None, "TPE": None, "ACO": 0.875},
    "colic": {"Random": 0.877, "Greedy": 0.877, "Simulated Annealing": 0.849, "Genetic": 0.863, "MCTS": 0.849, "TPE": 0.863, "ACO": 0.877},
}

# Standard Reference flags
ACOREC_FLAGS = [
    "--train-metric-inline",
    "--metric-loss", "pearson",
    "--metric-weight-decay", "1e-4",
    "--metric-objective", "embedding_cosine",
    "--aco-mmas-bounds",
    "--aco-weight-method", "linear",
    "--hybrid-select",
    "--final-autogluon-topk", "1",
    "--proxy-seeds", "42,52,62",
    "--cv-select-folds", "3",
    "--require-autogluon",
    "--autogluon-profile", "best_quality",
]


def _read_completed(out_path: Path) -> Dict[Tuple[str, str], float]:
    """Read completed (dataset_id, optimizer) -> score mappings."""
    done = {}
    if not out_path.exists():
        return done
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("score") is not None:
                did = str(row.get("dataset_id"))
                opt = str(row.get("optimizer"))
                done[(did, opt)] = float(row["score"])
        except Exception:
            continue
    return done


def _append_record(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def run_one(dataset_id: str, optimizer: str, data_dir: str, time_limit: int, seed: int,
            workdir: Path) -> dict:
    """Run run_recommend.py for a single (dataset_id, optimizer) pair."""
    cmd = [
        sys.executable, str(_REPO / "scripts" / "run_recommend.py"),
        "--dataset-source", "openml",
        "--openml-local-folder", str(data_dir),
        "--dataset-ids", str(dataset_id),
        "--optimizer", str(optimizer),
        "--time-limit", str(time_limit),
        "--seed", str(seed),
        "--output-dir", str(workdir),
    ] + ACOREC_FLAGS

    if optimizer == "aco":
        cmd.append("--use-aco")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_REPO))
    duration = round(time.time() - t0, 1)

    rec_path = workdir / "recommendation.json"
    if not rec_path.exists():
        # Check subfolder
        rec_path = workdir / f"dataset_{dataset_id}" / "recommendation.json"

    if proc.returncode == 0 and rec_path.exists():
        try:
            data = json.loads(rec_path.read_text())
            final = data.get("final_evaluation", {}) or {}
            score = final.get("score")
            return {
                "dataset_name": DATASET_ID_TO_NAME.get(str(dataset_id), str(dataset_id)),
                "dataset_id": str(dataset_id),
                "optimizer": str(optimizer),
                "opt_column": OPT_FLAG_TO_NAME.get(optimizer, optimizer),
                "status": "ok" if score is not None else "no_score",
                "score": score,
                "seconds": duration,
                "pipeline": data.get("pipeline_config"),
            }
        except Exception as exc:
            return {"dataset_id": str(dataset_id), "optimizer": str(optimizer),
                    "status": "parse_error", "detail": str(exc), "seconds": duration}
    return {"dataset_id": str(dataset_id), "optimizer": str(optimizer),
            "status": "failed", "returncode": proc.returncode, "seconds": duration}


def print_rq2_table(completed: Dict[Tuple[str, str], float], out_csv: Optional[str] = None) -> None:
    """Merge base table with completed results and display full Markdown + CSV format."""
    opt_cols = [name for name, _ in OPTIMIZERS]
    
    rows = []
    col_sums = {col: 0.0 for col in opt_cols}
    col_counts = {col: 0 for col in opt_cols}

    for ds_name, did in DATASETS_13:
        row = {"Dataset": ds_name}
        for col_name, opt_flag in OPTIMIZERS:
            base_val = BASE_RQ2_TABLE[ds_name].get(col_name)
            if (did, opt_flag) in completed:
                val = completed[(did, opt_flag)]
            elif base_val is not None:
                val = base_val
            else:
                val = None
            
            row[col_name] = val
            if val is not None:
                col_sums[col_name] += val
                col_counts[col_name] += 1
        rows.append(row)

    # Average row
    avg_row = {"Dataset": "Average"}
    for col_name in opt_cols:
        if col_counts[col_name] > 0:
            avg_row[col_name] = col_sums[col_name] / col_counts[col_name]
        else:
            avg_row[col_name] = None
    rows.append(avg_row)

    print("\n" + "=" * 90)
    print("RQ2 OPTIMIZER ABLATION RESULTS (13 Datasets)")
    print("=" * 90)

    # Markdown format
    header = ["Dataset"] + opt_cols
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        vals = []
        for col in header:
            v = r.get(col)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        print("| " + " | ".join(vals) + " |")

    # CSV format
    print("\n" + "-" * 40 + " CSV FORMAT " + "-" * 40)
    csv_lines = [",".join(header)]
    for r in rows:
        vals = []
        for col in header:
            v = r.get(col)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        csv_lines.append(",".join(vals))
    csv_text = "\n".join(csv_lines)
    print(csv_text)
    print("=" * 90)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(out_csv).write_text(csv_text, encoding="utf-8")
        print(f"\n[table] saved CSV to {out_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="/kaggle/working/eval_all",
                        help="Directory containing OpenML <id>.csv datasets")
    parser.add_argument("--out", default="outputs/rq2_ablations.jsonl",
                        help="Output JSONL results file")
    parser.add_argument("--out-csv", default=None,
                        help="Optional path to output the full merged CSV table")
    parser.add_argument("--summarize", action="store_true",
                        help="Summarize existing JSONL results into the full RQ2 table and exit")
    parser.add_argument("--missing-only", action="store_true", default=True,
                        help="Run only the missing cells from the RQ2 table (default: True)")
    parser.add_argument("--all", action="store_true",
                        help="Run all (dataset, optimizer) pairs, not just missing ones")
    parser.add_argument("--optimizers", default=None,
                        help="Comma-separated optimizers to run (e.g., sa,ga,mcts,tpe)")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset ids or names to run (e.g., 1047,378,1485,14)")
    parser.add_argument("--time-limit", type=int, default=120,
                        help="AutoGluon time_limit per fit in seconds (default: 120)")
    parser.add_argument("--shard", default=None,
                        help="Optional I/N shard (e.g., 1/2, 2/2) to split the run list across notebooks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratch-dir", default=None,
                        help="Temporary scratch folder (default: tempdir)")
    args = parser.parse_args()

    out_path = Path(args.out)
    completed = _read_completed(out_path)

    if args.summarize:
        print_rq2_table(completed, out_csv=args.out_csv)
        return 0

    # Determine tasks to run
    selected_opts = [o.strip() for o in args.optimizers.split(",") if o.strip()] if args.optimizers else None
    selected_ds = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets else None

    tasks: List[Tuple[str, str, str]] = []  # (ds_name, did, opt_flag)

    for ds_name, did in DATASETS_13:
        if selected_ds:
            if did not in selected_ds and ds_name not in selected_ds:
                continue
        for col_name, opt_flag in OPTIMIZERS:
            if selected_opts:
                if opt_flag not in selected_opts and col_name not in selected_opts:
                    continue
            
            # Check if this cell is missing in the base table
            is_missing_in_base = (BASE_RQ2_TABLE[ds_name].get(col_name) is None)
            
            if args.all or not args.missing_only or is_missing_in_base:
                if (did, opt_flag) not in completed:
                    tasks.append((ds_name, did, opt_flag))

    if args.shard:
        part, total = (int(x) for x in args.shard.split("/"))
        tasks = [t for i, t in enumerate(tasks) if i % total == (part - 1)]

    print(f"[rq2] planned {len(tasks)} runs ({len(completed)} already in {out_path})")
    for ds_name, did, opt in tasks:
        print(f"  - {ds_name} (id={did}) under {opt}")

    if not tasks:
        print("[rq2] All requested tasks are already complete!")
        print_rq2_table(completed, out_csv=args.out_csv)
        return 0

    scratch_base = Path(args.scratch_dir) if args.scratch_dir else (out_path.parent / "scratch_rq2")
    scratch_base.mkdir(parents=True, exist_ok=True)

    for idx, (ds_name, did, opt) in enumerate(tasks, 1):
        print(f"\n[{idx}/{len(tasks)}] Running {ds_name} (id={did}) under optimizer={opt} ...", flush=True)
        workdir = scratch_base / f"{did}_{opt}"
        workdir.mkdir(parents=True, exist_ok=True)

        res = run_one(
            dataset_id=did,
            optimizer=opt,
            data_dir=args.data_dir,
            time_limit=args.time_limit,
            seed=args.seed,
            workdir=workdir,
        )
        _append_record(out_path, res)
        if res.get("status") == "ok":
            completed[(did, opt)] = float(res["score"])
            print(f"  [ok] {ds_name} [{opt}] score={res['score']:.4f} in {res['seconds']}s", flush=True)
        else:
            print(f"  [{res.get('status')}] {ds_name} [{opt}] in {res.get('seconds', 0)}s detail={res.get('detail')}", flush=True)

    print_rq2_table(completed, out_csv=args.out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
