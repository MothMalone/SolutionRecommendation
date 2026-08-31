"""Build the Kaggle notebook for the RQ3 ACO alpha/beta ablation."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "rq3_alpha_beta_ablation_kaggle.ipynb"


def _source(value: str) -> list[str]:
    return (dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def _markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def _code(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(value)}


cells = [
    _markdown(
        """
        # RQ3: ACO pheromone versus heuristic influence

        This ablation tests whether transferred heuristic scores dominate ACO
        pheromone reinforcement. It evaluates five fixed configurations over
        10 datasets, sharded across five Kaggle sessions:

        | Variant | alpha | beta |
        |---|---:|---:|
        | heuristic-only | 0 | 2 |
        | pheromone-only | 1 | 0 |
        | current | 1 | 2 |
        | balanced | 1 | 1 |
        | pheromone-strong | 2 | 1 |

        Alpha controls pheromone influence and beta controls the transferred
        heuristic. K=5, H=3, rank deposits, global-elite updates, Markov
        order 2, lambda smoothing 0.7, 10 ants, and 10 iterations remain
        fixed. ACO searches only the fixed train+validation split; the frozen
        recommendation is then evaluated once by AutoGluon on the untouched
        outer test split. The results include Accuracy, Macro-F1, balanced
        accuracy, wall-clock timings, and `aco_history.csv` diagnostics.
        """
    ),
    _code(
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
    _code(
        """
        from pathlib import Path

        ABLATION = "aco_alpha_beta"
        VARIANT_VALUES = "heuristic_only,pheromone_only,current,balanced,pheromone_strong"
        DATASET_IDS = ["1066", "1047", "862", "1548", "378", "1485", "14", "1054", "1520", "876"]
        DATASET_MANIFEST = REPO_DIR / "data/openml/meta_dev18.json"

        FIXED_K, FIXED_H, SEARCH_K = 5, 3, 5
        N_ANTS, N_ITERATIONS, TOP_K_PHEROMONE = 10, 10, 3
        ACO_WEIGHT_METHOD = "rank"
        ACO_UPDATE_POLICY = "global_elite"
        ACO_MARKOV_ORDER, ACO_LAMBDA_SMOOTH = 2, 0.7
        ACO_SEED, SPLIT_SEED = 42, 42

        NUM_DATASET_SHARDS = 5
        DATASET_SHARD_INDEX = 0  # Set to 0, 1, 2, 3, or 4 in each Kaggle session.
        EVALUATOR = "autogluon"
        AUTOGLUON_PRESETS, FINAL_TIME_LIMIT = "best_quality", 300
        MAX_SAMPLES = 100_000
        RESUME, FORCE = True, False

        OUTPUT_DIR = Path("/kaggle/working/rq3_aco_alpha_beta")
        DATA_DIR = Path("/kaggle/working/rq3_aco_alpha_beta_data")
        print("Variants:", VARIANT_VALUES)
        print("Shard:", DATASET_SHARD_INDEX, "/", NUM_DATASET_SHARDS - 1)
        """
    ),
    _code(
        """
        import os
        import shlex
        import subprocess
        import sys

        command = [
            sys.executable, str(REPO_DIR / "scripts" / "rq3_alpha_beta_ablation.py"),
            "--root", str(REPO_DIR),
            "--performance-matrix", str(REPO_DIR / "data/openml/training_performance_matrix_autogluon.csv"),
            "--metafeatures", str(REPO_DIR / "data/openml/dataset_feats.csv"),
            "--pipeline-configs", str(REPO_DIR / "aco/pipeline_configs.json"),
            "--manifest", str(DATASET_MANIFEST), "--dataset-ids", *DATASET_IDS,
            "--dataset-source", "openml", "--openml-backend", "gitlab",
            "--openml-local-folder", str(DATA_DIR), "--data-dir", str(DATA_DIR),
            "--output-root", str(OUTPUT_DIR), "--variant-values", VARIANT_VALUES,
            "--fixed-k", str(FIXED_K), "--fixed-h", str(FIXED_H), "--search-k", str(SEARCH_K),
            "--n-ants", str(N_ANTS), "--n-iterations", str(N_ITERATIONS),
            "--top-k-pheromone", str(TOP_K_PHEROMONE), "--aco-weight-method", ACO_WEIGHT_METHOD,
            "--aco-update-policy", ACO_UPDATE_POLICY, "--aco-markov-order", str(ACO_MARKOV_ORDER),
            "--aco-lambda-smooth", str(ACO_LAMBDA_SMOOTH), "--aco-seed", str(ACO_SEED),
            "--split-seed", str(SPLIT_SEED), "--evaluator", EVALUATOR,
            "--autogluon-presets", AUTOGLUON_PRESETS, "--final-time-limit", str(FINAL_TIME_LIMIT),
            "--max-samples", str(MAX_SAMPLES), "--dataset-shard-index", str(DATASET_SHARD_INDEX),
            "--num-dataset-shards", str(NUM_DATASET_SHARDS), "--verbose",
        ]
        if not RESUME:
            command.append("--no-resume")
        if FORCE:
            command.append("--force")

        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
        print(" ".join(shlex.quote(str(item)) for item in command))
        completed = subprocess.run(command, cwd=REPO_DIR, env=env, check=False)
        print("Runner return code:", completed.returncode)
        if completed.returncode not in (0, 2):
            raise RuntimeError(f"Unexpected runner return code: {completed.returncode}")
        """
    ),
    _code(
        """
        import shutil
        import pandas as pd

        suite_dir = OUTPUT_DIR / ABLATION
        for name in ("results.csv", "summary.csv"):
            path = suite_dir / name
            if path.exists():
                print(f"\\n{name}")
                display(pd.read_csv(path))
        archive = shutil.make_archive(str(Path("/kaggle/working") / f"rq3_aco_alpha_beta_shard_{DATASET_SHARD_INDEX:02d}"), "gztar", root_dir=suite_dir)
        print("Archive:", archive)
        """
    ),
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
