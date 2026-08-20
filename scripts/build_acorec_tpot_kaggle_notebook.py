"""Generate the Kaggle notebook for original-space ACORec + TPOT evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "run-acorec-tpot-kaggle.ipynb"


def _lines(source: str) -> list[str]:
    return (dedent(source).strip("\n") + "\n").splitlines(keepends=True)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


cells = [
    markdown(
        """
        # ACORec (original operator space) + estimator-only TPOT

        This experiment leaves ACORec's core untouched. ACORec searches its original
        operator space with its original performance matrix and proxy evaluator. Its
        frozen top-1 preprocessing pipeline is then evaluated by TPOT with:

        - fixed seed-42 60% train / 20% ACO-search validation / 20% outer test;
        - `preprocessing=False`;
        - estimator-only `classifiers` or `regressors` search space;
        - accuracy for classification and R² for regression.

        Therefore the reported classification accuracy is computed only from TPOT's
        predictions on the fixed outer 20% test split. TPOT never evolves another
        preprocessing pipeline on top of ACORec and does not reuse ACO's validation rows.
        Classification targets are LabelEncoded for TPOT and inverse-transformed
        before scoring, which is required for OpenML labels that are not 0..K-1.
        """
    ),
    code(
        """
        # Clone/update the experiment branch and install TPOT-specific dependencies.
        import os
        import subprocess
        import sys
        from pathlib import Path

        REPO_URL = "https://github.com/MothMalone/SolutionRecommendation.git"
        BRANCH = "feature/acorec-autodp-space"
        REPO_DIR = Path("/kaggle/working/SolutionRecommendation")

        if (REPO_DIR / ".git").exists():
            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "switch", BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
        else:
            subprocess.run(
                ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)],
                check=True,
            )

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements-tpot-kaggle.txt")],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import numpy, pandas, sklearn, tpot; "
                    "from tpot import TPOTClassifier, TPOTRegressor; "
                    "print('Dependency health:', numpy.__version__, pandas.__version__, "
                    "sklearn.__version__, tpot.__version__, TPOTClassifier.__name__)"
                ),
            ],
            check=True,
        )
        os.chdir(REPO_DIR)
        print("Repo commit:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip())
        """
    ),
    code(
        """
        # Experiment controls. Run ten Save-Version jobs with SHARD_INDEX=0..9.
        import sys
        from pathlib import Path

        sys.path.insert(0, str(REPO_DIR / "src"))
        from automl_aco.eval_ids import EVAL_DATASETS

        RUN_MODE = "smoke"           # "smoke" first, then "final"
        NUM_SHARDS = 10
        SHARD_INDEX = 0               # 0..9; three datasets per final run
        WORKERS = 1

        ACO_SEED = 42
        FINAL_N_ANTS = 10
        FINAL_N_ITERATIONS = 10
        FINAL_METRIC_EPOCHS = 100

        TPOT_SPLIT_SEED = 42
        TPOT_RANDOM_STATE = 1         # matches DiffPrep + TPOT evaluator
        TPOT_MAX_TIME_MINS = 5
        TPOT_MAX_EVAL_TIME_MINS = 1
        TPOT_N_JOBS = 2
        TPOT_WORKER_MEMORY = "5GB"
        TPOT_POPULATION_SIZE = 20
        TPOT_MAX_CV_FOLDS = 5
        MAX_SAMPLES = 100_000          # same cap as the updated TPOT baselines

        if RUN_MODE not in {"smoke", "final"}:
            raise ValueError("RUN_MODE must be 'smoke' or 'final'")
        if not 0 <= SHARD_INDEX < NUM_SHARDS:
            raise ValueError("SHARD_INDEX must satisfy 0 <= SHARD_INDEX < NUM_SHARDS")

        all_ids = [int(dataset_id) for dataset_id in EVAL_DATASETS.values()]
        shard_ids = all_ids[SHARD_INDEX::NUM_SHARDS]
        run_ids = shard_ids[:1] if RUN_MODE == "smoke" else shard_ids

        CACHE_DIR = Path("/kaggle/working/acorec_tpot_eval30_data")
        OUTPUT_DIR = Path(f"/kaggle/working/acorec_tpot_{RUN_MODE}_shard_{SHARD_INDEX:02d}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Mode={RUN_MODE}; shard={SHARD_INDEX}/{NUM_SHARDS - 1}")
        print("Full shard:", shard_ids)
        print("IDs executed now:", run_ids)
        """
    ),
    code(
        """
        # Prepare the exact 17 DiffPrep snapshots, including synthetic google=100000.
        # The other 13 canonical datasets are downloaded from GitLab/DataGit on demand.
        import pandas as pd

        diffprep_names = {
            "abalone", "ada_prior", "avila", "connect-4", "eeg", "google",
            "house", "jungle_chess", "micro", "mozilla4", "obesity",
            "page-blocks", "pbcseq", "pol", "run_or_walk", "uscensus",
            "wall-robot-nav",
        }
        expected_local_ids = {int(EVAL_DATASETS[name]) for name in diffprep_names}
        input_root = Path("/kaggle/input")
        attached_google = list(input_root.glob("**/google/data.csv")) if input_root.exists() else []
        export_command = [
            sys.executable,
            str(REPO_DIR / "scripts" / "export_diffprep_datasets.py"),
            "--out-dir", str(CACHE_DIR),
        ]
        if attached_google:
            export_command += ["--diffprep-root", str(input_root)]
            print("Using attached DiffPrep input:", attached_google[0])
        else:
            export_command += ["--download"]
            print("Downloading frozen DiffPrep CSVs from GitHub.")
        subprocess.run(export_command, cwd=REPO_DIR, check=False)

        present = {int(path.stem) for path in CACHE_DIR.glob("*.csv") if path.stem.isdigit()}
        missing = sorted(expected_local_ids - present)
        if missing:
            raise RuntimeError(f"Missing DiffPrep snapshots: {missing}")
        google = pd.read_csv(CACHE_DIR / "100000.csv")
        print(f"DiffPrep snapshots ready: 17/17; Google={google.shape}, classes={google['target'].nunique()}")
        del google
        """
    ),
    code(
        """
        # Build ACORec command. --operator-space ours selects the original matrix/configs.
        # --no-autogluon means the unmodified core returns its proxy-selected top-1 pipeline;
        # the standalone TPOT evaluator below supplies the robust final score.
        import shlex

        aco_command = [
            sys.executable,
            str(REPO_DIR / "scripts" / "run_recommend.py"),
            "--operator-space", "ours",
            "--performance-matrix", str(
                REPO_DIR / "data" / "openml" / "training_performance_matrix_autogluon.csv"
            ),
            "--metafeatures", str(REPO_DIR / "data" / "openml" / "dataset_feats.csv"),
            "--pipeline-configs", str(REPO_DIR / "aco" / "pipeline_configs.json"),
            "--dataset-source", "openml",
            "--openml-backend", "gitlab",
            "--openml-local-folder", str(CACHE_DIR),
            "--dataset-ids", *[str(dataset_id) for dataset_id in run_ids],
            "--optimizer", "aco",
            "--seed", str(ACO_SEED),
            "--workers", str(WORKERS),
            "--output-dir", str(OUTPUT_DIR),
            "--skip-aco-plot",
            "--no-autogluon",
            "--verbose",
        ]
        if RUN_MODE == "smoke":
            aco_command += [
                "--n-ants", "1", "--n-iterations", "1", "--no-train-metric-inline"
            ]
        else:
            aco_command += [
                "--n-ants", str(FINAL_N_ANTS),
                "--n-iterations", str(FINAL_N_ITERATIONS),
                "--train-metric-inline",
                "--metric-epochs", str(FINAL_METRIC_EPOCHS),
            ]
        print("ACORec command:\\n", " ".join(shlex.quote(part) for part in aco_command))
        """
    ),
    code(
        """
        # Run/resume original ACORec search.
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
        subprocess.run(aco_command, cwd=REPO_DIR, env=env, check=True)
        """
    ),
    code(
        """
        # Evaluate each frozen top-1 ACORec pipeline with estimator-only TPOT.
        def dataset_output_dir(dataset_id):
            return OUTPUT_DIR if len(run_ids) == 1 else OUTPUT_DIR / f"dataset_{dataset_id}"

        tpot_minutes = 1 if RUN_MODE == "smoke" else TPOT_MAX_TIME_MINS
        for dataset_id in run_ids:
            dataset_dir = dataset_output_dir(dataset_id)
            recommendation_path = dataset_dir / "recommendation.json"
            if not recommendation_path.exists():
                raise FileNotFoundError(f"ACORec recommendation missing: {recommendation_path}")
            output_path = dataset_dir / "tpot_evaluation.json"
            command = [
                sys.executable,
                str(REPO_DIR / "scripts" / "evaluate_acorec_tpot.py"),
                "--recommendation-json", str(recommendation_path),
                "--dataset-id", str(dataset_id),
                "--data-dir", str(CACHE_DIR),
                "--output-json", str(output_path),
                "--max-samples", str(MAX_SAMPLES),
                "--split-seed", str(TPOT_SPLIT_SEED),
                "--tpot-seed", str(TPOT_RANDOM_STATE),
                "--max-time-mins", str(tpot_minutes),
                "--max-eval-time-mins", str(TPOT_MAX_EVAL_TIME_MINS),
                "--n-jobs", str(TPOT_N_JOBS),
                "--memory-limit", TPOT_WORKER_MEMORY,
                "--population-size", str(TPOT_POPULATION_SIZE),
                "--max-cv-folds", str(TPOT_MAX_CV_FOLDS),
                "--verbose", "2",
            ]
            print("\\nTPOT evaluation:", " ".join(shlex.quote(part) for part in command))
            subprocess.run(command, cwd=REPO_DIR, env=env, check=True)
        """
    ),
    code(
        """
        # Build a concise shard summary and archive all reproducible artifacts.
        import json
        import shutil

        rows = []
        for dataset_id in run_ids:
            result_path = dataset_output_dir(dataset_id) / "tpot_evaluation.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append({
                "dataset_id": dataset_id,
                "status": result.get("status"),
                "operator_space": result.get("operator_space"),
                "evaluator": result.get("evaluator"),
                "score": result.get("score"),
                "accuracy": result.get("accuracy"),
                "r2": result.get("r2"),
                "train_rows": result.get("train_rows_processed"),
                "validation_rows_aco_search": result.get("validation_rows_aco_search"),
                "validation_reused_by_tpot": result.get("validation_reused_by_tpot"),
                "test_rows": result.get("test_rows"),
                "tpot_preprocessing": result.get("tpot_preprocessing"),
                "target_label_encoding": result.get("target_label_encoding"),
            })
        summary = pd.DataFrame(rows)
        summary_path = OUTPUT_DIR / "acorec_tpot_summary.csv"
        summary.to_csv(summary_path, index=False)
        display(summary)

        archive_base = Path("/kaggle/working") / OUTPUT_DIR.name
        archive = shutil.make_archive(str(archive_base), "gztar", root_dir=OUTPUT_DIR)
        print("Summary:", summary_path)
        print("Archive:", archive)
        """
    ),
    markdown(
        """
        ## Final protocol

        After smoke succeeds, set `RUN_MODE="final"` and run ten Kaggle Save
        Versions with `SHARD_INDEX=0..9`. Each final run contains three datasets.

        A valid row must report `tpot_preprocessing=False`,
        `validation_rows_aco_search` equal to 20% of the loaded rows,
        `validation_reused_by_tpot=False`, and `test_rows` equal to the fixed outer 20%.
        Download each `.tar.gz` archive from the Kaggle Output panel.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUTPUT)
