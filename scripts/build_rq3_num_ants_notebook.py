"""Build the Kaggle notebook for the RQ3 ``num_ants`` ablation."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "rq3_num_ants_ablation_kaggle.ipynb"


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
        # RQ3: number of ants used during ACO search

        This notebook runs the `num_ants` ablation. It varies the number of ants
        sampled in each ACO iteration: **5, 10, 15, and 20**.

        This is different from `top_k_pheromone`: the latter is kept fixed at 3
        and only controls how many elite candidates receive pheromone
        reinforcement. The transfer settings are also fixed at K=5 and H=3.

        Search uses only the externally fixed train+validation split. The frozen
        recommendation is evaluated once on the untouched outer test split.
        Each `(ant-count, dataset)` run is checkpointed independently, so the
        ten RQ3 datasets can be split across five Kaggle sessions and resumed.

        The saved results include Accuracy, macro-F1, balanced accuracy, search
        time, final evaluation time, total time, and ACO proxy diagnostics.
        """
    ),
    _code(
        """
        # Clone the experiment branch and install the evaluator dependency.
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
        # Experiment controls. Use one DATASET_SHARD_INDEX per Kaggle session.
        import json
        from pathlib import Path

        ABLATION = "num_ants"
        VARIANT_VALUES = "5,10,15,20"
        FIXED_K = 5
        FIXED_H = 3
        SEARCH_K = 5
        TOP_K_PHEROMONE = 3
        N_ITERATIONS = 10

        # The ten datasets used by the current RQ3 experiments.
        DATASET_IDS = [
            "1066", "1047", "862", "1548", "378",
            "1485", "14", "1054", "1520", "876",
        ]
        DATASET_MANIFEST = REPO_DIR / "data/openml/meta_dev18.json"

        NUM_DATASET_SHARDS = 5                 # two datasets per Kaggle session
        DATASET_SHARD_INDEX = 0                # change to 0..NUM_DATASET_SHARDS-1
        ACO_SEED = 42
        SPLIT_SEED = 42
        EVALUATOR = "autogluon"               # "autogluon" or "tpot"
        AUTOGLUON_PRESETS = "best_quality"
        FINAL_TIME_LIMIT = 300
        TPOT_TIME_MINS = 5
        MAX_SAMPLES = 100_000
        RESUME = True
        FORCE = False

        OUTPUT_DIR = Path("/kaggle/working/rq3_num_ants_ablation")
        DATA_DIR = Path("/kaggle/working/rq3_num_ants_data")
        print("Ablation:", ABLATION, "values:", VARIANT_VALUES)
        print("Fixed K/H:", FIXED_K, FIXED_H, "fixed elite count:", TOP_K_PHEROMONE)
        print("Shard:", DATASET_SHARD_INDEX, "/", NUM_DATASET_SHARDS - 1)
        print("Dataset IDs:", DATASET_IDS)
        """
    ),
    _code(
        """
        # Run/resume the independent checkpoints.
        import os
        import shlex
        import subprocess
        import sys

        command = [
            sys.executable,
            str(REPO_DIR / "scripts" / "rq3_num_ants_ablation.py"),
            "--root", str(REPO_DIR),
            "--performance-matrix", str(REPO_DIR / "data/openml/training_performance_matrix_autogluon.csv"),
            "--metafeatures", str(REPO_DIR / "data/openml/dataset_feats.csv"),
            "--pipeline-configs", str(REPO_DIR / "aco/pipeline_configs.json"),
            "--manifest", str(DATASET_MANIFEST),
            "--dataset-ids", *DATASET_IDS,
            "--dataset-source", "openml",
            "--openml-backend", "gitlab",
            "--openml-local-folder", str(DATA_DIR),
            "--data-dir", str(DATA_DIR),
            "--output-root", str(OUTPUT_DIR),
            "--variant-values", VARIANT_VALUES,
            "--fixed-k", str(FIXED_K),
            "--fixed-h", str(FIXED_H),
            "--search-k", str(SEARCH_K),
            "--top-k-pheromone", str(TOP_K_PHEROMONE),
            "--n-ants", "10",  # baseline only; num_ants values override it
            "--n-iterations", str(N_ITERATIONS),
            "--aco-seed", str(ACO_SEED),
            "--split-seed", str(SPLIT_SEED),
            "--evaluator", EVALUATOR,
            "--autogluon-presets", AUTOGLUON_PRESETS,
            "--final-time-limit", str(FINAL_TIME_LIMIT),
            "--tpot-time-mins", str(TPOT_TIME_MINS),
            "--max-samples", str(MAX_SAMPLES),
            "--dataset-shard-index", str(DATASET_SHARD_INDEX),
            "--num-dataset-shards", str(NUM_DATASET_SHARDS),
            "--verbose",
        ]
        if not RESUME:
            command.append("--no-resume")
        if FORCE:
            command.append("--force")

        env = os.environ.copy()
        env.update({
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        print(" ".join(shlex.quote(str(item)) for item in command))
        completed = subprocess.run(command, cwd=REPO_DIR, env=env, check=False)
        print("Runner return code:", completed.returncode)
        if completed.returncode not in (0, 2):
            raise RuntimeError(f"Unexpected runner return code: {completed.returncode}")
        """
    ),
    _code(
        """
        # Inspect results and archive this shard.
        import shutil
        import pandas as pd

        suite_dir = OUTPUT_DIR / ABLATION
        results_path = suite_dir / "results.csv"
        summary_path = suite_dir / "summary.csv"
        if results_path.exists():
            display(pd.read_csv(results_path))
        if summary_path.exists():
            print("\\nVariant summary")
            display(pd.read_csv(summary_path))

        archive = shutil.make_archive(
            str(Path("/kaggle/working") / f"rq3_{ABLATION}_shard_{DATASET_SHARD_INDEX:02d}"),
            "gztar",
            root_dir=suite_dir,
        )
        print("Archive:", archive)
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
