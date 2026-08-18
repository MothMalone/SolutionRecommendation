"""Generate the self-contained Kaggle runner for ACORec on AutoDP36."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "run-acorec-autodp36-kaggle.ipynb"


def _lines(source: str) -> list[str]:
    text = dedent(source).strip("\n") + "\n"
    return text.splitlines(keepends=True)


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
        # ACORec on AutoDP's operator space — Kaggle runner

        This notebook clones the isolated `feature/acorec-autodp-space` branch and
        runs ACORec with `--operator-space autodp36`. By default it evaluates the
        project's canonical 30-dataset test suite; `autodp60` remains available as
        a switch. The matching 36×818 training matrix and 36 reference pipeline
        configs are selected automatically by the command runner.

        Dataset backends:

        - `gitlab` (recommended): downloads Parquet from the DataGit/OpenML mirror;
        - `openml`: uses `openml-python`, then `sklearn.fetch_openml`;
        - `auto`: tries local/OpenML first and GitLab as fallback.

        For the 30-dataset suite, the exact 17 DiffPrep CSVs are read from an attached
        Kaggle input when available, otherwise downloaded from the DiffPrep GitHub
        repository. The remaining OpenML datasets use the GitLab mirror. In particular,
        `google=100000` is a synthetic project ID and is loaded from
        `google/data.csv`, not OpenML.

        If an older version of this notebook already installed `numpy<2` in the
        current Kaggle session, use **Session > Restart Session** once before running
        this updated notebook. The current dependency pins preserve Kaggle's NumPy 2
        ABI and validate NumPy/pandas/sklearn in a clean subprocess.
        Start with `RUN_MODE = "smoke"`; change it to `"final"` after the first ID
        completes successfully.
        """
    ),
    code(
        """
        # Clone/update the exact experiment branch and install its Kaggle dependencies.
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
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements-kaggle.txt")],
            check=True,
        )
        # Verify binary compatibility in a clean interpreter before importing the repo
        # in this notebook kernel. This produces a direct installation error instead of
        # an opaque pandas/numpy ABI traceback in the following cell.
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import numpy, pandas, sklearn; "
                    "from autogluon.tabular import TabularPredictor; "
                    "print('Dependency health:', numpy.__version__, "
                    "pandas.__version__, sklearn.__version__, TabularPredictor.__name__)"
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
        # Experiment controls. SHARD_INDEX is zero-based: 0, 1, ..., NUM_SHARDS-1.
        import sys
        from pathlib import Path

        sys.path.insert(0, str(REPO_DIR / "src"))
        from automl_aco.preprocessing.autodp import (
            AUTODP_60_IDS,
            AUTODP_CLASSIFICATION_IDS,
            AUTODP_REGRESSION_IDS,
        )
        from automl_aco.eval_ids import EVAL_DATASETS

        DATASET_SUITE = "ours30"     # "ours30" (paper suite) or "autodp60"
        RUN_MODE = "smoke"          # "smoke" first, then "final"
        DOWNLOAD_BACKEND = "gitlab" # "gitlab" (recommended), "openml", or "auto"
        NUM_SHARDS = 10              # ours30: 3/run; autodp60: 6/run
        SHARD_INDEX = 0              # run 0..9 in separate Save-Version jobs
        WORKERS = 1                  # keep 1: AutoGluon + large datasets are RAM-heavy

        FINAL_N_ANTS = 10
        FINAL_N_ITERATIONS = 10
        FINAL_METRIC_EPOCHS = 100
        FINAL_AUTOGLUON_SECONDS = 300

        if DATASET_SUITE not in {"ours30", "autodp60"}:
            raise ValueError("DATASET_SUITE must be 'ours30' or 'autodp60'")
        if RUN_MODE not in {"smoke", "final"}:
            raise ValueError("RUN_MODE must be 'smoke' or 'final'")
        if DOWNLOAD_BACKEND not in {"gitlab", "openml", "auto"}:
            raise ValueError("DOWNLOAD_BACKEND must be gitlab/openml/auto")
        if not 0 <= SHARD_INDEX < NUM_SHARDS:
            raise ValueError("SHARD_INDEX must satisfy 0 <= SHARD_INDEX < NUM_SHARDS")

        ours30_ids = [int(dataset_id) for dataset_id in EVAL_DATASETS.values()]
        all_ids = ours30_ids if DATASET_SUITE == "ours30" else list(AUTODP_60_IDS)
        shard_ids = all_ids[SHARD_INDEX::NUM_SHARDS]
        run_ids = shard_ids[:1] if RUN_MODE == "smoke" else shard_ids

        CACHE_DIR = Path(f"/kaggle/working/acorec_{DATASET_SUITE}_data")
        OUTPUT_DIR = Path(
            f"/kaggle/working/acorec_autodp36_{DATASET_SUITE}_{RUN_MODE}_shard_{SHARD_INDEX:02d}"
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Suite={DATASET_SUITE}; mode={RUN_MODE}; backend={DOWNLOAD_BACKEND}")
        print(f"Shard {SHARD_INDEX}/{NUM_SHARDS - 1}: {shard_ids}")
        print("IDs executed now:", run_ids)
        print(f"Classification IDs={len(AUTODP_CLASSIFICATION_IDS)}; regression IDs={len(AUTODP_REGRESSION_IDS)}")
        """
    ),
    code(
        """
        # Materialize the exact DiffPrep half of our 30-dataset suite as <id>.csv.
        # This includes google=100000, which has no OpenML entry.
        import pandas as pd

        if DATASET_SUITE == "ours30":
            diffprep_folders = {
                "abalone", "ada_prior", "avila", "connect-4", "eeg", "google",
                "house", "jungle_chess", "micro", "mozilla4", "obesity",
                "page-blocks", "pbcseq", "pol", "run_or_walk", "uscensus",
                "wall-robot-nav",
            }
            expected_local_ids = {
                int(EVAL_DATASETS[name]) for name in diffprep_folders
            }
            input_root = Path("/kaggle/input")
            attached_google = list(input_root.glob("**/google/data.csv")) if input_root.exists() else []
            export_command = [
                sys.executable,
                str(REPO_DIR / "scripts" / "export_diffprep_datasets.py"),
                "--out-dir", str(CACHE_DIR),
            ]
            if attached_google:
                # The exporter scans recursively, so /kaggle/input is layout-independent.
                export_command += ["--diffprep-root", str(input_root)]
                print("Using attached DiffPrep Kaggle input:", attached_google[0])
            else:
                export_command += ["--download"]
                print("No DiffPrep Kaggle input found; downloading frozen CSVs from GitHub.")

            # Exit code may be 1 because the 13 OpenML CSVs are intentionally absent here;
            # those IDs are supplied by the GitLab backend below.
            export_result = subprocess.run(export_command, cwd=REPO_DIR, check=False)
            present_local_ids = {
                int(path.stem) for path in CACHE_DIR.glob("*.csv") if path.stem.isdigit()
            }
            missing_local = sorted(expected_local_ids - present_local_ids)
            if missing_local:
                raise RuntimeError(f"Missing DiffPrep CSV IDs after export: {missing_local}")
            google_frame = pd.read_csv(CACHE_DIR / "100000.csv")
            if "target" not in google_frame.columns:
                raise KeyError("Exported Google CSV does not contain normalized 'target' column")
            print(
                f"Exact DiffPrep snapshots ready: {len(expected_local_ids)}/17; "
                f"Google shape={google_frame.shape}, classes={google_frame['target'].nunique()}"
            )
            del google_frame
        """
    ),
    code(
        """
        # Lightweight connectivity probe. It reads only the first bytes, not full datasets.
        import time
        import requests
        import pandas as pd

        def probe(label, url, expected_prefix, timeout=12):
            started = time.time()
            try:
                with requests.get(
                    url,
                    stream=True,
                    timeout=(8, timeout),
                    allow_redirects=True,
                    headers={"User-Agent": "ACORec-AutoDP36-Kaggle/1.0"},
                ) as response:
                    first = next(response.iter_content(chunk_size=16), b"")
                    return {
                        "source": label,
                        "status": response.status_code,
                        "valid_prefix": response.status_code == 200 and first.startswith(expected_prefix),
                        "seconds": round(time.time() - started, 2),
                        "content_length": response.headers.get("content-length"),
                    }
            except Exception as error:
                return {
                    "source": label,
                    "status": type(error).__name__,
                    "valid_prefix": False,
                    "seconds": round(time.time() - started, 2),
                    "content_length": None,
                }

        probe_id = int(run_ids[0])
        probe_rows = [
            probe(
                "OpenML metadata",
                f"https://www.openml.org/api/v1/json/data/{probe_id}",
                b"{",
            ),
            probe(
                "GitLab metadata",
                f"https://gitlab.com/data/d/openml/{probe_id}/-/raw/master/dataset/metadata.json",
                b"{",
            ),
            probe(
                "GitLab parquet",
                f"https://gitlab.com/data/d/openml/{probe_id}/-/raw/master/dataset/tables/data.pq",
                b"PAR1",
            ),
        ]
        display(pd.DataFrame(probe_rows))
        """
    ),
    code(
        """
        # Download/load one dataset through the selected backend before launching ACORec.
        from automl_aco.data.loaders import load_gitlab_openml_dataset, load_openml_dataset

        preview_id = int(run_ids[0])
        common = dict(
            test_dataset_ids=list(all_ids),
            regression_dataset_ids=list(AUTODP_REGRESSION_IDS),
            verbose=True,
            max_samples_if_test=100_000,
        )

        if DOWNLOAD_BACKEND == "gitlab":
            preview = load_gitlab_openml_dataset(
                preview_id, cache_dir=str(CACHE_DIR), **common
            )
        else:
            preview = load_openml_dataset(
                preview_id, local_data_folder=str(CACHE_DIR), **common
            )
            if preview is None and DOWNLOAD_BACKEND == "auto":
                preview = load_gitlab_openml_dataset(
                    preview_id, cache_dir=str(CACHE_DIR), **common
                )

        if preview is None:
            raise RuntimeError(f"Dataset backend {DOWNLOAD_BACKEND!r} could not load D_{preview_id}")
        print(
            "Preflight OK:",
            {"id": preview_id, "shape": preview["X"].shape, "task": preview["task_type"],
             "backend": preview.get("download_backend", DOWNLOAD_BACKEND)},
        )
        del preview
        """
    ),
    code(
        """
        # Confirm that the final evaluator is available in this Kaggle environment.
        import autogluon
        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import IdentityFeatureGenerator
        from automl_aco.search.evaluation import _load_autogluon_components

        evaluator_predictor, evaluator_generator = _load_autogluon_components()
        assert evaluator_predictor is TabularPredictor
        assert evaluator_generator is IdentityFeatureGenerator

        print("AutoGluon version:", getattr(autogluon, "__version__", "unknown"))
        print("Final evaluator:", TabularPredictor.__name__)
        print("Feature generator after ACORec preprocessing:", IdentityFeatureGenerator.__name__)
        print("ACORec evaluator runtime preflight: PASS")
        """
    ),
    code(
        """
        # Build the exact command. Final mode uses AutoGluon because it does NOT pass --no-autogluon.
        import shlex

        command = [
            sys.executable,
            str(REPO_DIR / "scripts" / "run_recommend.py"),
            "--operator-space", "autodp36",
            "--dataset-source", "openml",
            "--openml-backend", DOWNLOAD_BACKEND,
            "--openml-local-folder", str(CACHE_DIR),
            "--dataset-ids", *[str(dataset_id) for dataset_id in run_ids],
            "--optimizer", "aco",
            "--workers", str(WORKERS),
            "--output-dir", str(OUTPUT_DIR),
            "--skip-aco-plot",
            "--verbose",
        ]

        if RUN_MODE == "smoke":
            command += [
                "--n-ants", "1",
                "--n-iterations", "1",
                "--time-limit", "60",
                "--no-autogluon",
                "--no-train-metric-inline",
            ]
        else:
            command += [
                "--n-ants", str(FINAL_N_ANTS),
                "--n-iterations", str(FINAL_N_ITERATIONS),
                "--train-metric-inline",
                "--metric-epochs", str(FINAL_METRIC_EPOCHS),
                "--time-limit", str(FINAL_AUTOGLUON_SECONDS),
                "--final-autogluon-topk", "1",
                "--autogluon-profile", "best_quality",
                "--require-autogluon",
                "--tar-outputs",
            ]

        print("Command:\\n", " ".join(shlex.quote(part) for part in command))
        """
    ),
    code(
        """
        # Run ACORec. Existing recommendation.json files are skipped, so reruns resume safely.
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
        subprocess.run(command, cwd=REPO_DIR, env=env, check=True)
        """
    ),
    code(
        """
        # Summarize outputs for download from the Kaggle Output panel.
        import json
        import pandas as pd

        rows = []
        for path in sorted(OUTPUT_DIR.rglob("recommendation.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            final_eval = payload.get("final_evaluation") or {}
            rows.append({
                "dataset_id": payload.get("dataset_id"),
                "operator_space": payload.get("operator_space"),
                "pipeline": (payload.get("pipeline_config") or {}).get("name"),
                "final_method": final_eval.get("method"),
                "final_score": final_eval.get("score", payload.get("final_performance")),
                "file": str(path),
            })

        summary = pd.DataFrame(rows)
        display(summary)
        print("Output directory:", OUTPUT_DIR)
        archives = sorted(Path("/kaggle/working").glob("acorec_autodp36_*.tar.gz"))
        print("Archives:", [str(path) for path in archives])
        """
    ),
    markdown(
        """
        ## Running the full evaluation suite

        1. Keep `NUM_SHARDS = 10`.
        2. Set `RUN_MODE = "final"`.
        3. Create ten Kaggle Save-Version runs with `SHARD_INDEX = 0, 1, ..., 9`.
        4. Download each output archive/result directory.

        With `DATASET_SUITE = "ours30"`, each final shard has three datasets. The
        shard containing ID 100000 loads the exported Google CSV automatically.

        `WORKERS = 1` is intentional. Shards divide work across separate Kaggle
        sessions; they do not launch multiple AutoGluon processes into the same RAM.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUTPUT)
