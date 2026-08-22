"""Build the canonical No-preprocessing and H2O-default Kaggle notebook."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "reproduce-h2o-baselines.ipynb"


def source(value: str) -> list[str]:
    return (dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(value),
    }


cells = [
    markdown(
        """
        # No preprocessing and H2O Default on the canonical 30 datasets

        Both modes use exactly the same externally created 60/20/20 split.
        H2O trains only on the 60% training frame, selects the best model using
        the 20% validation frame, and predicts the untouched 20% test frame
        once. The only difference is H2O's optional target-encoding preprocessing:

        - `no_preprocessing`: `preprocessing=None`;
        - `h2o_default`: `preprocessing=["target_encoding"]`.

        In this experiment table, “H2O Default” means the agreed
        target-encoding-enabled configuration. H2O's literal API default for
        `preprocessing` is `None`, which is the separate no-preprocessing row.

        No ACORec/DiffPrep/CtxPipe code is involved in this notebook. Run five
        Save-Version jobs with `DATASET_SHARD_INDEX=0..4`.
        """
    ),
    code(
        """
        %pip install -q "h2o==3.46.0.11" "pyarrow>=15" "requests"
        """
    ),
    code(
        """
        from __future__ import annotations

        import gc
        import json
        import os
        import shutil
        import subprocess
        import sys
        import traceback
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import h2o

        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = "1"

        RUN_MODE = "smoke"       # change to final after the first dataset succeeds
        NUM_DATASET_SHARDS = 5
        DATASET_SHARD_INDEX = 0
        SPLIT_SEED = 42
        MAX_SAMPLES = 100_000
        H2O_MAX_RUNTIME_SECS = 120 if RUN_MODE == "smoke" else 300
        H2O_MAX_RUNTIME_SECS_PER_MODEL = 60
        H2O_NFOLDS = 5
        H2O_NTHREADS = 1
        H2O_MAX_MEM_SIZE = "6G"

        if RUN_MODE not in {"smoke", "final"} or not 0 <= DATASET_SHARD_INDEX < NUM_DATASET_SHARDS:
            raise ValueError("Invalid RUN_MODE or DATASET_SHARD_INDEX")

        KAGGLE = Path("/kaggle/working").exists()
        OUTPUT_DIR = Path("/kaggle/working/h2o_baselines") if KAGGLE else Path("outputs/h2o_baselines")
        CACHE_DIR = OUTPUT_DIR / "canonical_data"
        SOLUTION_DIR = Path("/kaggle/temp/SolutionRecommendation") if KAGGLE else OUTPUT_DIR / "SolutionRecommendation"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("H2O:", h2o.__version__)
        """
    ),
    code(
        """
        REPO_URL = "https://github.com/MothMalone/SolutionRecommendation.git"
        BRANCH = "feature/acorec-autodp-space"
        if (SOLUTION_DIR / ".git").exists():
            subprocess.run(["git", "-C", str(SOLUTION_DIR), "fetch", "origin", BRANCH], check=True)
            subprocess.run(["git", "-C", str(SOLUTION_DIR), "switch", BRANCH], check=True)
            subprocess.run(["git", "-C", str(SOLUTION_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
        else:
            subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(SOLUTION_DIR)], check=True)
        # Guard against an old notebook/session cache before a long run.
        evaluator_path = SOLUTION_DIR / "scripts" / "h2o_evaluator.py"
        evaluator_source = evaluator_path.read_text(encoding="utf-8")
        if "h2o.init(nthreads=int(nthreads), max_mem_size=str(max_mem_size), silent=True)" in evaluator_source:
            raise RuntimeError(
                "Stale H2O evaluator detected. Restart the Kaggle session and rerun this clone/install cell."
            )
        sys.path.insert(0, str(SOLUTION_DIR / "src"))
        sys.path.insert(0, str(SOLUTION_DIR / "scripts"))
        import importlib
        importlib.invalidate_caches()
        from automl_aco.data.loaders import load_gitlab_openml_dataset
        from automl_aco.data.splits import split_train_val_test
        from automl_aco.eval_ids import EVAL_DATASETS
        from h2o_evaluator import evaluate_h2o_frames

        # Materialize the 17 frozen DiffPrep snapshots, including synthetic Google.
        subprocess.run(
            [sys.executable, str(SOLUTION_DIR / "scripts" / "export_diffprep_datasets.py"),
             "--out-dir", str(CACHE_DIR), "--download"],
            cwd=SOLUTION_DIR,
            check=False,
        )
        print("SolutionRecommendation commit:", subprocess.check_output(["git", "-C", str(SOLUTION_DIR), "rev-parse", "HEAD"], text=True).strip())
        """
    ),
    code(
        """
        DATASETS = [{"dataset_id": int(dataset_id), "name": name} for name, dataset_id in EVAL_DATASETS.items()]
        positions = np.array_split(np.arange(len(DATASETS)), NUM_DATASET_SHARDS)
        SHARD_DATASETS = [DATASETS[int(i)] for i in positions[DATASET_SHARD_INDEX]]
        RUN_DATASETS = SHARD_DATASETS[:1] if RUN_MODE == "smoke" else SHARD_DATASETS
        print(f"Shard {DATASET_SHARD_INDEX}/{NUM_DATASET_SHARDS - 1}; datasets this run:", [x["dataset_id"] for x in RUN_DATASETS])
        """
    ),
    code(
        """
        def load_canonical(spec):
            dataset = load_gitlab_openml_dataset(
                int(spec["dataset_id"]),
                cache_dir=str(CACHE_DIR),
                test_dataset_ids=[int(value) for value in EVAL_DATASETS.values()],
                verbose=True,
                max_samples_if_test=MAX_SAMPLES,
            )
            if dataset is None:
                raise RuntimeError(f"Could not load dataset {spec['dataset_id']}")
            return dataset

        def evaluate_mode(dataset, mode):
            X = pd.DataFrame(dataset["X"]).copy()
            y = pd.Series(dataset["y"]).copy()
            X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, seed=SPLIT_SEED)
            result, model = evaluate_h2o_frames(
                X_train, y_train, X_val, y_val, X_test, y_test,
                task_type=str(dataset.get("task_type", "classification")),
                h2o_preprocessing=None if mode == "no_preprocessing" else "target_encoding",
                max_runtime_secs=H2O_MAX_RUNTIME_SECS,
                max_runtime_secs_per_model=H2O_MAX_RUNTIME_SECS_PER_MODEL,
                nfolds=H2O_NFOLDS,
                seed=42,
                nthreads=H2O_NTHREADS,
                max_mem_size=H2O_MAX_MEM_SIZE,
            )
            result.update({
                "dataset_id": int(dataset["id"]),
                "dataset": dataset.get("name", f"D_{dataset['id']}"),
                "setting": mode,
                "split_seed": SPLIT_SEED,
                "test_used_during_training_or_selection": False,
                "source": dataset.get("download_backend"),
            })
            del model
            gc.collect()
            return result
        """
    ),
    code(
        """
        RESULT_PATH = OUTPUT_DIR / f"h2o_baselines_shard_{DATASET_SHARD_INDEX:02d}_of_{NUM_DATASET_SHARDS:02d}.csv"
        rows = pd.read_csv(RESULT_PATH).to_dict("records") if RESULT_PATH.exists() else []
        completed = {(str(row.get("dataset_id")), row.get("setting")) for row in rows if row.get("status") == "ok"}

        def save_row(row):
            key = (str(row["dataset_id"]), row["setting"])
            rows[:] = [old for old in rows if (str(old.get("dataset_id")), old.get("setting")) != key]
            rows.append(row)
            pd.DataFrame(rows).to_csv(RESULT_PATH, index=False)

        for position, spec in enumerate(RUN_DATASETS, start=1):
            dataset = None
            for mode in ("no_preprocessing", "h2o_default"):
                if (str(spec["dataset_id"]), mode) in completed:
                    print(f"SKIP {spec['name']} / {mode}")
                    continue
                print(f"[{position}/{len(RUN_DATASETS)}] {spec['name']} / {mode}")
                try:
                    dataset = dataset or load_canonical(spec)
                    row = evaluate_mode(dataset, mode)
                except Exception as error:
                    traceback.print_exc()
                    row = {
                        "dataset_id": int(spec["dataset_id"]),
                        "dataset": spec["name"],
                        "setting": mode,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error)[:4000],
                    }
                save_row(row)
                print("score:", row.get("score"), "status:", row.get("status"))
            del dataset
            gc.collect()

        print("Saved:", RESULT_PATH)
        display(pd.DataFrame(rows).sort_values(["dataset_id", "setting"]))
        """
    ),
    markdown(
        """
        Download `h2o_baselines_shard_XX_of_05.csv` from the Kaggle output and
        concatenate the five shards. The two settings differ only in H2O's
        `target_encoding` preprocessing flag.
        """
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT}")
