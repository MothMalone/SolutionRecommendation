"""Build the Kaggle notebook for the RQ3 pheromone-weight ablation."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "rq3_pheromone_weight_ablation_kaggle.ipynb"


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
        # RQ3: pheromone reinforcement weighting strategy

        This notebook reruns the three strategies in the existing RQ3 table:
        **exponential**, **rank**, and **uniform**. These values control
        `--aco-weight-method`, i.e. how the top sampled pipelines are weighted
        when pheromone is reinforced. They are not the separate
        `--aco-update-policy` setting.

        To match the earlier ablation, `top_k_pheromone=3`, Markov order `2`,
        and `lambda_smooth=0.7` are fixed. K=5, H=3, 10 ants, 10 iterations,
        seeds, dataset split, and final evaluator are also held constant.

        ACO search uses only train+validation. The frozen recommendation is
        evaluated once with AutoGluon on the untouched outer test split. Saved
        outputs contain Accuracy, macro-F1, balanced accuracy, and search,
        evaluation, and total wall-clock durations.

        The ten RQ3 datasets are divided into five shards, two datasets per
        Kaggle session. Each `(strategy, dataset)` checkpoint can be resumed.
        """
    ),
    _code(
        """
        # Clone/update the experiment branch and install dependencies.
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
        # Experiment controls. Change DATASET_SHARD_INDEX for each Kaggle run.
        from pathlib import Path

        ABLATION = "pheromone_weight_method"
        VARIANT_VALUES = "exponential,rank,uniform"
        FIXED_K = 5
        FIXED_H = 3
        SEARCH_K = 5
        N_ANTS = 10
        N_ITERATIONS = 10
        TOP_K_PHEROMONE = 3
        ACO_MARKOV_ORDER = 2
        ACO_LAMBDA_SMOOTH = 0.7

        DATASET_IDS = [
            "1066", "1047", "862", "1548", "378",
            "1485", "14", "1054", "1520", "876",
        ]
        DATASET_MANIFEST = REPO_DIR / "data/openml/meta_dev18.json"

        NUM_DATASET_SHARDS = 5                 # two datasets per session
        DATASET_SHARD_INDEX = 0                # change to 0..4
        ACO_SEED = 42
        SPLIT_SEED = 42
        EVALUATOR = "autogluon"
        AUTOGLUON_PRESETS = "best_quality"
        FINAL_TIME_LIMIT = 300
        MAX_SAMPLES = 100_000
        RESUME = True
        FORCE = False

        OUTPUT_DIR = Path("/kaggle/working/rq3_pheromone_weight_ablation")
        DATA_DIR = Path("/kaggle/working/rq3_pheromone_weight_data")
        print("Strategies:", VARIANT_VALUES)
        print("Shard:", DATASET_SHARD_INDEX, "/", NUM_DATASET_SHARDS - 1)
        print("Dataset IDs:", DATASET_IDS)
        """
    ),
    _code(
        """
        # Run/resume the independent strategy/dataset checkpoints.
        import os
        import shlex
        import subprocess
        import sys

        command = [
            sys.executable,
            str(REPO_DIR / "scripts" / "rq3_pheromone_weight_ablation.py"),
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
            "--n-ants", str(N_ANTS),
            "--n-iterations", str(N_ITERATIONS),
            "--top-k-pheromone", str(TOP_K_PHEROMONE),
            "--aco-markov-order", str(ACO_MARKOV_ORDER),
            "--aco-lambda-smooth", str(ACO_LAMBDA_SMOOTH),
            "--aco-seed", str(ACO_SEED),
            "--split-seed", str(SPLIT_SEED),
            "--evaluator", EVALUATOR,
            "--autogluon-presets", AUTOGLUON_PRESETS,
            "--final-time-limit", str(FINAL_TIME_LIMIT),
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
        # Inspect results and archive this shard for download.
        import shutil
        import pandas as pd

        suite_dir = OUTPUT_DIR / ABLATION
        results_path = suite_dir / "results.csv"
        summary_path = suite_dir / "summary.csv"
        if results_path.exists():
            display(pd.read_csv(results_path))
        if summary_path.exists():
            print("\\nStrategy summary")
            display(pd.read_csv(summary_path))

        archive = shutil.make_archive(
            str(Path("/kaggle/working") / f"rq3_pheromone_weight_shard_{DATASET_SHARD_INDEX:02d}"),
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
