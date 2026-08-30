"""Build the Kaggle notebook for the extended RQ3 ant experiments."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "rq3_ant_budget_extended_kaggle.ipynb"


def _source(value: str) -> list[str]:
    return (dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def _markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def _code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(value),
    }


cells = [
    _markdown(
        """
        # RQ3: extended ant-count and cumulative-ant-budget experiments

        This notebook runs two independent experiments on the same ten RQ3 datasets:

        1. **A={5,10,15,20,25,30} x 10 iterations**: repeats the complete
           per-iteration ant-count curve, including the new A25 and A30 points.
        2. **Cumulative ant budget**: keeps the colony at 10 ants per complete
           iteration and stops after `B={5,10,15,20,25,30}` total ant draws.
           Partial final iterations are exact: B15 executes `10 + 5`, and B25
           executes `10 + 10 + 5` while retaining pheromone state.

        K=5, H=3, global-elite top-3 reinforcement, seed 42, data split and final
        evaluator settings are fixed. Search never accesses the outer test split.
        """
    ),
    _code(
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        REPO_URL = "https://github.com/MothMalone/SolutionRecommendation.git"
        BRANCH = "experiment/aco-search-ablation"
        REPO_DIR = Path("/kaggle/working/SolutionRecommendation")

        if not REPO_DIR.exists():
            subprocess.run(
                ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)],
                check=True,
            )
        else:
            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "switch", BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements-kaggle.txt")],
            check=True,
        )
        print("Repository:", REPO_DIR)
        print("Commit:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR, text=True).strip())
        """
    ),
    _code(
        """
        # Controls: this version defaults to one Kaggle run over all ten datasets.
        # Set NUM_DATASET_SHARDS=5 if the runtime is too long for one session.
        RUN_ANTS_SWEEP = True
        RUN_TOTAL_BUDGET = True
        # Set True to run the early-stopping comparison as a separate suite.
        # It is off by default because it doubles the AutoGluon workload.
        RUN_EARLY_STOP_SWEEP = False
        ANT_VALUES = "5,10,15,20,25,30"
        EARLY_STOP_ROUNDS = 2
        EARLY_STOP_MIN_IMPROVEMENT = 0.0

        DATASET_IDS = [
            "1066", "1047", "862", "1548", "378",
            "1485", "14", "1054", "1520", "876",
        ]
        NUM_DATASET_SHARDS = 1
        DATASET_SHARD_INDEX = 0

        FIXED_K = 5
        FIXED_H = 3
        SEARCH_K = 5
        TOP_K_PHEROMONE = 3
        ACO_SEED = 42
        SPLIT_SEED = 42
        EVALUATOR = "autogluon"       # keep identical to the earlier RQ3 run
        AUTOGLUON_PRESETS = "best_quality"
        FINAL_TIME_LIMIT = 300
        MAX_SAMPLES = 100_000
        RESUME = True
        FORCE = False

        DATA_DIR = Path("/kaggle/working/rq3_ant_budget_data")
        ANTS_OUTPUT_ROOT = Path("/kaggle/working/rq3_num_ants_extended")
        EARLY_OUTPUT_ROOT = Path("/kaggle/working/rq3_num_ants_early_stop")
        BUDGET_OUTPUT_ROOT = Path("/kaggle/working/rq3_total_ant_budget")

        if not 0 <= DATASET_SHARD_INDEX < NUM_DATASET_SHARDS:
            raise ValueError("DATASET_SHARD_INDEX must be in [0, NUM_DATASET_SHARDS)")
        print("Shard datasets:", DATASET_IDS[DATASET_SHARD_INDEX::NUM_DATASET_SHARDS])
        """
    ),
    _code(
        """
        import shlex

        env = os.environ.copy()
        env.update({
            "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
            "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        })

        def common_command(script, output_root, variant_values):
            command = [
                sys.executable, str(REPO_DIR / "scripts" / script),
                "--root", str(REPO_DIR),
                "--performance-matrix", str(REPO_DIR / "data/openml/training_performance_matrix_autogluon.csv"),
                "--metafeatures", str(REPO_DIR / "data/openml/dataset_feats.csv"),
                "--pipeline-configs", str(REPO_DIR / "aco/pipeline_configs.json"),
                "--dataset-ids", *DATASET_IDS,
                "--dataset-source", "openml", "--openml-backend", "gitlab",
                "--openml-local-folder", str(DATA_DIR), "--data-dir", str(DATA_DIR),
                "--output-root", str(output_root), "--variant-values", variant_values,
                "--fixed-k", str(FIXED_K), "--fixed-h", str(FIXED_H),
                "--search-k", str(SEARCH_K), "--top-k-pheromone", str(TOP_K_PHEROMONE),
                "--aco-seed", str(ACO_SEED), "--split-seed", str(SPLIT_SEED),
                "--evaluator", EVALUATOR, "--autogluon-presets", AUTOGLUON_PRESETS,
                "--final-time-limit", str(FINAL_TIME_LIMIT), "--max-samples", str(MAX_SAMPLES),
                "--dataset-shard-index", str(DATASET_SHARD_INDEX),
                "--num-dataset-shards", str(NUM_DATASET_SHARDS), "--verbose",
            ]
            if not RESUME:
                command.append("--no-resume")
            if FORCE:
                command.append("--force")
            return command

        commands = []
        if RUN_ANTS_SWEEP:
            cmd = common_command("rq3_num_ants_ablation.py", ANTS_OUTPUT_ROOT, ANT_VALUES)
            cmd += ["--n-iterations", "10", "--n-ants", "10"]
            commands.append(("Ants per iteration: " + ANT_VALUES, cmd))
        if RUN_EARLY_STOP_SWEEP:
            cmd = common_command("rq3_num_ants_ablation.py", EARLY_OUTPUT_ROOT, ANT_VALUES)
            cmd += [
                "--n-iterations", "10", "--n-ants", "10",
                "--aco-early-stop-rounds", str(EARLY_STOP_ROUNDS),
                "--aco-min-improvement", str(EARLY_STOP_MIN_IMPROVEMENT),
            ]
            commands.append((f"Ants + early stop (patience={EARLY_STOP_ROUNDS})", cmd))
        if RUN_TOTAL_BUDGET:
            cmd = common_command(
                "rq3_total_ant_budget_ablation.py", BUDGET_OUTPUT_ROOT, "5,10,15,20,25,30"
            )
            cmd += ["--n-ants", "10", "--n-iterations", "3"]
            commands.append(("Cumulative ant budgets", cmd))

        for label, command in commands:
            print("\\n===", label, "===")
            print(" ".join(shlex.quote(str(item)) for item in command))
            completed = subprocess.run(command, cwd=REPO_DIR, env=env, check=False)
            if completed.returncode not in (0, 2):
                raise RuntimeError(f"{label} failed with return code {completed.returncode}")
        """
    ),
    _code(
        """
        import shutil
        import pandas as pd

        suites = []
        if RUN_ANTS_SWEEP:
            suites.append(("num_ants", ANTS_OUTPUT_ROOT / "num_ants"))
        if RUN_EARLY_STOP_SWEEP:
            suites.append(("early_stop", EARLY_OUTPUT_ROOT / "num_ants"))
        if RUN_TOTAL_BUDGET:
            suites.append(("total_budget", BUDGET_OUTPUT_ROOT / "total_ant_budget"))

        for label, suite_dir in suites:
            print("\\n=== Results:", label, "===")
            if not suite_dir.exists():
                print("No output directory found (the corresponding run may have failed).")
                continue
            if (suite_dir / "results.csv").exists():
                frame = pd.read_csv(suite_dir / "results.csv")
                display(frame)
                report_cols = [
                    c for c in ["dataset_id", "variant", "accuracy", "f1_macro",
                                "search_best_score", "evaluator_validation_score",
                                "proxy_unique_evaluations", "search_wall_clock_seconds"]
                    if c in frame.columns
                ]
                if report_cols:
                    print("Per-dataset F1/validation report")
                    display(frame[report_cols].sort_values(["variant", "dataset_id"]))
            if (suite_dir / "summary.csv").exists():
                display(pd.read_csv(suite_dir / "summary.csv"))
            archive = shutil.make_archive(
                str(Path("/kaggle/working") / f"rq3_{label}_shard_{DATASET_SHARD_INDEX:02d}"),
                "gztar", root_dir=suite_dir,
            )
            print("Archive:", archive)

        if RUN_ANTS_SWEEP and RUN_EARLY_STOP_SWEEP:
            base_path = ANTS_OUTPUT_ROOT / "num_ants" / "results.csv"
            early_path = EARLY_OUTPUT_ROOT / "num_ants" / "results.csv"
            if base_path.exists() and early_path.exists():
                base = pd.read_csv(base_path)
                early = pd.read_csv(early_path)
                keys = ["variant", "dataset_id"]
                compare = base.merge(
                    early, on=keys, suffixes=("_full", "_early"), how="inner"
                )
                for metric in ["accuracy", "f1_macro", "search_wall_clock_seconds", "proxy_unique_evaluations"]:
                    left, right = f"{metric}_full", f"{metric}_early"
                    if left in compare and right in compare:
                        compare[f"delta_{metric}"] = compare[right] - compare[left]
                print("\\nEarly-stop minus full-search comparison")
                display(compare[[c for c in compare.columns if c in keys or c.startswith("delta_")]])
        """
    ),
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
