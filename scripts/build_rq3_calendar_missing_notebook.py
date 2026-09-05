"""Build a Kaggle notebook to fill the calendarDOW gap in two RQ3 tables."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "rq3_calendar_missing_kaggle.ipynb"


def source(value: str) -> list[str]:
    return (dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(value)}


def code(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source(value)}


cells = [
    markdown(
        """
        # RQ3 completion: calendarDOW

        This notebook fills the one missing dataset in the final RQ3 dataset
        set: `calendarDOW` (OpenML ID 40663). It deliberately reproduces the
        historical settings of the two existing ablations:

        - pheromone reinforcement weights: Exponential, Rank, Uniform;
        - number of search ants: 5, 10, 15, 20, 25.

        It produces eight independent ACO + AutoGluon evaluations. Do not mix
        the output with the old `autoUniv-au4` (1548) rows; replace those rows
        with calendarDOW when constructing the final ten-dataset RQ3 tables.
        """
    ),
    code(
        """
        import subprocess
        import sys
        from pathlib import Path

        REPO_URL = "https://github.com/MothMalone/SolutionRecommendation.git"
        BRANCH = "experiment/aco-search-ablation"
        REPO_DIR = Path("/kaggle/working/SolutionRecommendation")
        if not REPO_DIR.exists():
            subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)], check=True)
        else:
            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "switch", BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements-kaggle.txt")], check=True)
        print("Commit:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR, text=True).strip())
        """
    ),
    code(
        """
        import os
        import shlex
        import subprocess
        import sys
        from pathlib import Path

        DATASET_ID = "40663"  # calendarDOW
        DATA_DIR = Path("/kaggle/working/rq3_calendar_data")
        OUTPUT_ROOT = Path("/kaggle/working/rq3_calendar_completion")
        COMMON = [
            "--root", str(REPO_DIR),
            "--performance-matrix", str(REPO_DIR / "data/openml/training_performance_matrix_autogluon.csv"),
            "--metafeatures", str(REPO_DIR / "data/openml/dataset_feats.csv"),
            "--pipeline-configs", str(REPO_DIR / "aco/pipeline_configs.json"),
            "--manifest", str(REPO_DIR / "data/openml/meta_dev18.json"),
            "--dataset-ids", DATASET_ID,
            "--dataset-source", "openml", "--openml-backend", "gitlab",
            "--openml-local-folder", str(DATA_DIR), "--data-dir", str(DATA_DIR),
            "--fixed-k", "5", "--fixed-h", "3", "--search-k", "5",
            "--top-k-pheromone", "3", "--n-iterations", "10",
            "--aco-seed", "42", "--split-seed", "42",
            "--evaluator", "autogluon", "--autogluon-presets", "best_quality",
            "--final-time-limit", "300", "--max-samples", "100000", "--verbose",
        ]
        ENV = os.environ.copy()
        ENV.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})

        jobs = [
            (
                "pheromone weights",
                [
                    sys.executable, str(REPO_DIR / "scripts" / "rq3_pheromone_weight_ablation.py"),
                    *COMMON, "--output-root", str(OUTPUT_ROOT / "pheromone"),
                    "--variant-values", "exponential,rank,uniform", "--n-ants", "10",
                    "--aco-markov-order", "2", "--aco-lambda-smooth", "0.7",
                ],
            ),
            (
                "number of ants",
                [
                    sys.executable, str(REPO_DIR / "scripts" / "rq3_num_ants_ablation.py"),
                    *COMMON, "--output-root", str(OUTPUT_ROOT / "num_ants"),
                    "--variant-values", "5,10,15,20,25", "--n-ants", "10",
                    "--aco-markov-order", "2", "--aco-lambda-smooth", "0.0",
                ],
            ),
        ]

        for label, command in jobs:
            print(f"\\n=== {label} ===")
            print(" ".join(shlex.quote(str(part)) for part in command))
            completed = subprocess.run(command, cwd=REPO_DIR, env=ENV, check=False)
            print("Return code:", completed.returncode)
            if completed.returncode not in (0, 2):
                raise RuntimeError(f"{label} failed with code {completed.returncode}")
        """
    ),
    code(
        """
        import shutil
        import pandas as pd

        for suite in ("pheromone", "num_ants"):
            print(f"\\n=== {suite} ===")
            for path in (OUTPUT_ROOT / suite).rglob("results.csv"):
                display(pd.read_csv(path))

        archive = shutil.make_archive(
            "/kaggle/working/rq3_calendar_completion",
            "gztar",
            root_dir=OUTPUT_ROOT,
        )
        print("Download this archive:", archive)
        """
    ),
]


def main() -> None:
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
