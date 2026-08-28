"""Generate the Kaggle runner for AutoDP over ACORec's operator space with TPOT final evaluation."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "reproduce-autodp-tpot.ipynb"


def _source(text: str) -> list[str]:
    return (textwrap.dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


cells = [
    _markdown(
        """
        # AutoDP on ACORec's operator space + estimator-only TPOT

        This notebook preserves AutoDP's MCTS and its internal NB/LDA/RF candidate signal. The
        MCTS is patched to select and execute ACORec operators, using the supplied retrained
        1-NN meta-corpus. TPOT is used **only once as the final evaluator** of AutoDP's frozen
        output; it is not called inside the MCTS.

        Protocol: shared seed-42 60/20/20 split. AutoDP search and preprocessing fit use the
        outer 60% training partition; TPOT uses CV only inside that same partition; the fixed 20%
        validation partition is unused; the 20% outer test partition is evaluated once. TPOT is
        estimator-only (`preprocessing=False`), so AutoDP/ACORec remains the sole optimized
        preprocessing method.
        """
    ),
    _code(
        """
        # Clone the experiment code, install TPOT, then build AutoDP's isolated legacy environment.
        import subprocess
        import sys
        from pathlib import Path

        REPO_URL = "https://github.com/MothMalone/SolutionRecommendation.git"
        # Keep this aligned with the branch that contains adp_bench_tpot.py when publishing.
        BRANCH = "experiment/aco-search-ablation"
        REPO_DIR = Path("/kaggle/working/SolutionRecommendation")

        if (REPO_DIR / ".git").exists():
            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "switch", BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
        else:
            subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)], check=True)

# The benchmark process runs in Kaggle's main Python environment.  The repository
# loader imports requests before AutoDP is launched in its isolated legacy venv.
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "TPOT==1.1.0", "pyarrow>=15", "requests>=2.31",
], check=True)
subprocess.run(["bash", str(REPO_DIR / "scripts" / "setup_autodp_env.sh")], cwd=REPO_DIR, check=True)
adp_python = REPO_DIR / ".venv-autodp" / "bin" / "python"
subprocess.run([
    str(adp_python), "-c",
    "import requests; print('AutoDP venv requests:', requests.__version__)",
], check=True)

required = [
            REPO_DIR / "scripts" / "adp_bench_tpot.py",
            REPO_DIR / "scripts" / "eval_autodatapre_tpot.py",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                "The remote branch is stale and lacks the AutoDP-TPOT runner: " + ", ".join(missing)
            )
        print("Repo:", subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], text=True).strip())
        """
    ),
    _code(
        """
        # Controls. Attach the downloaded adp_ourops_corpus as a Kaggle Input dataset.
        import json
        import os
        from pathlib import Path
        import numpy as np
        import pandas as pd

        sys.path.insert(0, str(REPO_DIR / "src"))
        from automl_aco.eval_ids import EVAL_IDS

        RUN_MODE = "smoke"             # smoke or final
        NUM_DATASET_SHARDS = 5
        DATASET_SHARD_INDEX = 0          # zero-based: 0..4
        DATASET_IDS_OVERRIDE = None      # e.g. (1066, 1047); None = this shard

        TPOT_MAX_TIME_MINS = 5
        TPOT_MAX_EVAL_TIME_MINS = 1
        TPOT_N_JOBS = 2
        TPOT_MEMORY_LIMIT = "5GB"
        TPOT_POPULATION_SIZE = 20
        AUTODP_CAP_SECONDS = 5400        # watchdog; AutoDP otherwise uses its convergence rule

        # Edit only if the Kaggle input slug is different. The corpus files must be directly inside
        # CORPUS_DIR: Metafeature.csv and label.csv.
        CORPUS_CANDIDATES = [
            Path("/kaggle/input/adp-ourops-corpus/adp_ourops_corpus"),
            Path("/kaggle/input/adp-ourops-corpus"),
        ]
        CORPUS_DIR = next(
            (path for path in CORPUS_CANDIDATES if (path / "Metafeature.csv").exists() and (path / "label.csv").exists()),
            None,
        )
        if CORPUS_DIR is None:
            raise FileNotFoundError("Attach adp_ourops_corpus as Kaggle Input, then update CORPUS_CANDIDATES.")

        # Optional: point this at a Kaggle Input containing <OpenML-id>.csv files. Leave None with
        # Internet enabled to use the repository loader's normal download fallback.
        OPENML_LOCAL_FOLDER = None
        OUTPUT_DIR = Path("/kaggle/working/adp_ourops_tpot")
        DATA_DIR = OUTPUT_DIR / "datasets"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if DATASET_IDS_OVERRIDE is not None:
            RUN_IDS = [str(dataset_id) for dataset_id in DATASET_IDS_OVERRIDE]
        else:
            if not 0 <= DATASET_SHARD_INDEX < NUM_DATASET_SHARDS:
                raise ValueError("DATASET_SHARD_INDEX is out of range")
            pieces = np.array_split(np.asarray(EVAL_IDS, dtype=object), NUM_DATASET_SHARDS)
            RUN_IDS = [str(dataset_id) for dataset_id in pieces[DATASET_SHARD_INDEX]]
        if RUN_MODE == "smoke":
            RUN_IDS = RUN_IDS[:1]
        elif RUN_MODE != "final":
            raise ValueError("RUN_MODE must be smoke or final")

        RESULT_PATH = OUTPUT_DIR / f"adp_ourops_tpot_shard_{DATASET_SHARD_INDEX:02d}_of_{NUM_DATASET_SHARDS:02d}.jsonl"
        print("Corpus:", CORPUS_DIR)
        print("Run IDs:", RUN_IDS)
        print("Output:", RESULT_PATH)
        """
    ),
    _code(
        """
        # Run AutoDP MCTS over ACORec operators, then TPOT final evaluation. Safe to rerun: a
        # successful dataset/mode record already present in RESULT_PATH is skipped.
        command = [
            sys.executable, str(REPO_DIR / "scripts" / "adp_bench_tpot.py"),
            "--ids", *RUN_IDS,
            "--out", str(RESULT_PATH),
            "--adp-meta-corpus", str(CORPUS_DIR),
            "--adp-python", str(REPO_DIR / ".venv-autodp" / "bin" / "python"),
            "--data-dir", str(DATA_DIR),
            "--cap-seconds", str(AUTODP_CAP_SECONDS),
            "--max-time-mins", str(TPOT_MAX_TIME_MINS),
            "--max-eval-time-mins", str(TPOT_MAX_EVAL_TIME_MINS),
            "--n-jobs", str(TPOT_N_JOBS),
            "--memory-limit", TPOT_MEMORY_LIMIT,
            "--population-size", str(TPOT_POPULATION_SIZE),
        ]
        if OPENML_LOCAL_FOLDER:
            command += ["--openml-local-folder", str(OPENML_LOCAL_FOLDER)]
        print(" ".join(command))
        subprocess.run(command, cwd=REPO_DIR, check=True)
        """
    ),
    _code(
        """
        # Inspect this shard's resumable records and write a CSV companion for download.
        records_by_key = {}
        if RESULT_PATH.exists():
            for line in RESULT_PATH.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    key = (str(record.get("dataset_id")), str(record.get("mode")))
                    records_by_key[key] = record
        records = list(records_by_key.values())
        summary = pd.DataFrame(records)
        if not summary.empty:
            summary_path = RESULT_PATH.with_suffix(".csv")
            summary.to_csv(summary_path, index=False)
            display(summary.sort_values("dataset_id"))
            print("CSV:", summary_path)
        else:
            print("No records written yet:", RESULT_PATH)
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

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT}")
