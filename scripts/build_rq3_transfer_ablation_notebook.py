"""Build the Kaggle notebook for the two RQ3 transfer ablations."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "rq3_transfer_ablation_kaggle.ipynb"


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
        # RQ3 transfer-neighborhood ablation

        This notebook runs one of two leak-free ACORec ablations:

        - `num_retrieved_datasets`: vary K, the number of neighbors used for
          heuristic transfer, with H fixed;
        - `num_selected_pipelines`: vary H, the number of historical pipelines
          selected from each retrieved neighbor, with K fixed.

        The existing ACORec implementation is used unchanged. Search runs on the
        externally fixed 60% train + 20% validation data only. The frozen selected
        pipeline is then evaluated once on the untouched outer 20% test split.

        Each `(variant, dataset)` is checkpointed independently. Set
        `DATASET_SHARD_INDEX` to a different value in separate Kaggle Save-Version
        runs and use the same `OUTPUT_DIR` to resume.

        Accuracy remains the primary quality metric. The output additionally stores
        macro-F1, balanced accuracy, search/evaluation/total wall-clock time, proxy
        evaluation counts, and active operator assignments in the selected pipeline
        and selected ACO candidates. The operator count excludes `none`/identity;
        it is a complexity diagnostic, not a quality metric.
        """
    ),
    _code(
        """
        # Clone the experiment branch and install the final evaluator dependency.
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
        # Experiment controls. Run this notebook once per DATASET_SHARD_INDEX.
        from pathlib import Path
        import json

        ABLATION = "num_retrieved_datasets"  # or "num_selected_pipelines"
        VARIANT_VALUES = "1,3,5"              # K; use "1,2,3" for H
        FIXED_K = 5                            # used by H ablation
        FIXED_H = 3                            # used by K ablation
        SEARCH_K = 5                            # keep ACO candidate budget fixed

        DATASET_MANIFEST = REPO_DIR / "data/openml/meta_dev18.json"
        DATASET_IDS = [str(value) for value in json.loads(DATASET_MANIFEST.read_text())["dataset_ids"]]

        NUM_DATASET_SHARDS = 6                 # meta-dev18 -> 3 datasets/session
        DATASET_SHARD_INDEX = 0                # 0..NUM_DATASET_SHARDS-1
        ACO_SEED = 42
        SPLIT_SEED = 42
        N_ANTS = 10
        N_ITERATIONS = 10
        EVALUATOR = "autogluon"               # "autogluon" or "tpot"
        FINAL_TIME_LIMIT = 300                 # AutoGluon seconds per frozen pipeline
        TPOT_TIME_MINS = 5                     # used only when EVALUATOR="tpot"
        MAX_SAMPLES = 100_000
        RESUME = True
        FORCE = False

        OUTPUT_DIR = Path("/kaggle/working/rq3_transfer_ablation")
        DATA_DIR = Path("/kaggle/working/rq3_transfer_data")
        print("Ablation:", ABLATION, "values:", VARIANT_VALUES)
        print("Shard:", DATASET_SHARD_INDEX, "/", NUM_DATASET_SHARDS - 1)
        print("Datasets in manifest:", DATASET_IDS)
        """
    ),
    _code(
        """
        # Run/resume the independent dataset checkpoints.
        import os
        import shlex
        import subprocess
        import sys

        script_name = (
            "rq3_retrieved_datasets_ablation.py"
            if ABLATION == "num_retrieved_datasets"
            else "rq3_selected_pipelines_ablation.py"
        )
        command = [
            sys.executable,
            str(REPO_DIR / "scripts" / script_name),
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
            "--aco-seed", str(ACO_SEED),
            "--split-seed", str(SPLIT_SEED),
            "--evaluator", EVALUATOR,
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
        # Inspect the checkpointed result and archive this shard for download.
        import json
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
