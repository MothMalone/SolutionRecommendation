#!/usr/bin/env python3
"""Generate the Kaggle notebook for native CtxPipe + estimator-only TPOT."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "reproduce-ctxpipe-tpot.ipynb"


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
        # Native CtxPipe + estimator-only TPOT (30 ACORec datasets)

        This notebook runs the **official CtxPipe operator space and official
        `ctx_32000` pretrained checkpoint**. It does not use the ACORec operator
        space and does not modify ACORec's core. CtxPipe first recommends one
        six-step preprocessing sequence; TPOT then evaluates that frozen sequence
        while searching classifiers only (`preprocessing=False`). AutoGluon is not used.

        Leakage-safe protocol:

        - fixed outer split: 60% train / 20% validation / 20% untouched test, seed 42;
        - native CtxPipe sees only outer train+validation (80%), then uses its own
          seed-0 80/20 reward split inside that subset;
        - the selected native sequence is replayed by fitting every transformer on
          outer train 60% only and transforming outer test 20%;
        - TPOT CV uses only the processed outer train 60%, never the validation or test;
        - final accuracy is computed once on the untouched outer test 20%.
        - classification targets are LabelEncoded for TPOT and inverse-transformed
          before the outer-test metric, so sparse OpenML labels are handled correctly.

        The six checkpoint files shipped by the official repository are byte-identical
        to `data/ctxpipe-3linear/ctxpipe-3linear` in the local research workspace.
        Therefore no separate weights upload is required. The GTE-large text encoder is
        a separate dependency and is downloaded from Hugging Face unless an attached
        Kaggle model directory is supplied. Use a Kaggle GPU accelerator.

        The upstream project targets Python 3.8/PyTorch 1.12/scikit-learn 0.23. This
        notebook keeps its model and operator decisions intact but applies three small
        runtime compatibility patches: CUDA fallback declaration, checkpoint
        `map_location`, and removal of a workstation-specific `pkill` command.
        """
    ),
    code(
        """
        # Clone pinned experiment sources before importing scientific packages.
        import os
        import subprocess
        import sys
        from pathlib import Path

        SOLUTION_URL = "https://github.com/MothMalone/SolutionRecommendation.git"
        SOLUTION_BRANCH = "feature/acorec-autodp-space"
        SOLUTION_DIR = Path("/kaggle/working/SolutionRecommendation")

        CTXPIPE_URL = "https://github.com/ctxpipe/ctxpipe.git"
        CTXPIPE_COMMIT = "79caaa17f17ebdeeac6ba549abe150c5b3f1381d"
        CTXPIPE_DIR = Path("/kaggle/working/ctxpipe")

        if (SOLUTION_DIR / ".git").exists():
            subprocess.run(["git", "-C", str(SOLUTION_DIR), "fetch", "origin", SOLUTION_BRANCH], check=True)
            subprocess.run(["git", "-C", str(SOLUTION_DIR), "switch", SOLUTION_BRANCH], check=True)
            subprocess.run(
                ["git", "-C", str(SOLUTION_DIR), "pull", "--ff-only", "origin", SOLUTION_BRANCH],
                check=True,
            )
        else:
            subprocess.run(
                ["git", "clone", "--branch", SOLUTION_BRANCH, "--single-branch", SOLUTION_URL, str(SOLUTION_DIR)],
                check=True,
            )

        if not (CTXPIPE_DIR / ".git").exists():
            subprocess.run(["git", "clone", CTXPIPE_URL, str(CTXPIPE_DIR)], check=True)
        subprocess.run(["git", "-C", str(CTXPIPE_DIR), "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", str(CTXPIPE_DIR), "checkout", "--detach", CTXPIPE_COMMIT], check=True)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-r",
                str(SOLUTION_DIR / "requirements-ctxpipe-tpot-kaggle.txt"),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import numpy,pandas,sklearn,torch,tpot,transformers; "
                    "print('health',numpy.__version__,pandas.__version__,sklearn.__version__,"
                    "torch.__version__,tpot.__version__,transformers.__version__)"
                ),
            ],
            check=True,
        )
        print("Solution commit:", subprocess.check_output(["git", "-C", str(SOLUTION_DIR), "rev-parse", "--short", "HEAD"], text=True).strip())
        print("CtxPipe commit:", subprocess.check_output(["git", "-C", str(CTXPIPE_DIR), "rev-parse", "HEAD"], text=True).strip())
        """
    ),
    code(
        """
        # Experiment controls. Run five Save-Version jobs with SHARD_INDEX=0..4.
        from __future__ import annotations

        import gc
        import hashlib
        import json
        import math
        import re
        import signal
        import shutil
        import time
        import traceback
        import warnings

        import numpy as np
        import pandas as pd
        import psutil
        import torch

        sys.path.insert(0, str(SOLUTION_DIR / "src"))
        from automl_aco.data.loaders import load_gitlab_openml_dataset
        from automl_aco.data.splits import split_train_val_test
        from automl_aco.eval_ids import EVAL_DATASETS, EVAL_IDS

        RUN_MODE = "smoke"          # "smoke" first, then "final"
        NUM_SHARDS = 5
        SHARD_INDEX = 0              # 0, 1, 2, 3, 4
        REQUIRE_GPU = True

        SPLIT_SEED = 42
        MAX_SAMPLES = 100_000
        MAXIMUM_MATRIX_CELLS = 200_000_000

        TPOT_RANDOM_STATE = 1
        TPOT_MAX_TIME_MINS = 5
        TPOT_MAX_EVAL_TIME_MINS = 1
        TPOT_N_JOBS = 2
        TPOT_WORKER_MEMORY = "5GB"
        TPOT_POPULATION_SIZE = 20
        TPOT_MAX_CV_FOLDS = 5
        CTXPIPE_TIMEOUT_MINS_PER_DATASET = 45

        # Set this to an attached local gte-large folder to avoid downloading it.
        ATTACHED_GTE_MODEL_DIR = None

        if RUN_MODE not in {"smoke", "final"}:
            raise ValueError("RUN_MODE must be smoke or final")
        if not 0 <= SHARD_INDEX < NUM_SHARDS:
            raise ValueError("SHARD_INDEX is outside the configured shard range")
        if REQUIRE_GPU and not torch.cuda.is_available():
            raise RuntimeError("Enable a GPU accelerator in Kaggle Notebook settings")

        all_specs = [
            {"dataset_id": int(dataset_id), "name": name}
            for name, dataset_id in EVAL_DATASETS.items()
        ]
        positions = np.array_split(np.arange(len(all_specs)), NUM_SHARDS)
        shard_specs = [all_specs[int(index)] for index in positions[SHARD_INDEX]]
        run_specs = shard_specs[:1] if RUN_MODE == "smoke" else shard_specs

        CACHE_DIR = Path("/kaggle/working/ctxpipe_tpot_data")
        OUTPUT_DIR = Path(
            f"/kaggle/working/ctxpipe_tpot_{RUN_MODE}_shard_{SHARD_INDEX:02d}"
        )
        CTXPIPE_DATA_DIR = CTXPIPE_DIR / "data" / "diffprep_dataset"
        GTE_MODEL_DIR = Path(ATTACHED_GTE_MODEL_DIR) if ATTACHED_GTE_MODEL_DIR else Path("/kaggle/working/embed/gte-large")
        for directory in (CACHE_DIR, OUTPUT_DIR, CTXPIPE_DATA_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        warnings.filterwarnings("ignore")
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        print(f"Shard {SHARD_INDEX}/{NUM_SHARDS - 1}:", run_specs)
        """
    ),
    code(
        """
        # Verify the official native checkpoint. These frozen hashes also prove that
        # the user's local ctxpipe-3linear weights are the same upstream weights.
        EXPECTED_CHECKPOINT_SHA256 = {
            "ctx_32000_encoder_model.pkl": "6bc5265ab2751cdd99767d821f43cd942e96e30ec4691d6420ecb239f5218cc5",
            "ctx_32000_fengine_model.pkl": "c4423961ad46de8fd03d2d79615028b8c529d415536b10eb222de050fb6c892e",
            "ctx_32000_fpreprocessing_model.pkl": "4c4d445a003b05ee22ef299e633e06a1a807a4f5ab4f496fff36f177080cb58b",
            "ctx_32000_fselection_model.pkl": "80c9b9eaa232099f99a9cc3a160c16a1c190087cd2766fac4120202982a210cf",
            "ctx_32000_imputernum_model.pkl": "5543afd7b45c073dda4278b70a0e1730757a10ee31b4349295ce5145a326eaea",
            "ctx_32000_logical_pipeline.pkl": "e40cd97c3dd85787bfe898959cb2ff854870d4c690213fca0be7cf3d9058d26d",
        }
        checkpoint_dir = CTXPIPE_DIR / "models" / "ctxpipe-3linear"
        observed = {}
        for filename, expected in EXPECTED_CHECKPOINT_SHA256.items():
            path = checkpoint_dir / filename
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise RuntimeError(f"CtxPipe checkpoint hash mismatch: {filename}")
            observed[filename] = digest
        print(f"Verified {len(observed)} official ctx_32000 checkpoint files")

        # Minimal compatibility patches; no operator list or model architecture changes.
        def replace_exact(path, old, new):
            path = Path(path)
            source = path.read_text(encoding="utf-8")
            if new in source:
                return
            if old not in source:
                raise RuntimeError(f"Expected upstream source was not found in {path}")
            path.write_text(source.replace(old, new), encoding="utf-8")

        replace_exact(
            CTXPIPE_DIR / "env.py",
            'DEVICE = torch.device("cuda")',
            'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
        )
        replace_exact(
            CTXPIPE_DIR / "ctxpipe" / "agent" / "dqn.py",
            "torch.load(model_path)",
            "torch.load(model_path, map_location=DEVICE, weights_only=True)",
        )
        replace_exact(
            CTXPIPE_DIR / "ctxpipe" / "env" / "pipeline.py",
            "os.system(f\\\"pkill -f '/home/{os.getlogin()}/anaconda3/envs/ctxpipe.*joblib'\\\")",
            '# Kaggle compatibility: no workstation-specific process sweep.',
        )
        print("Applied three audited Kaggle compatibility patches")
        """
    ),
    code(
        """
        # Prepare GTE-large. Native CtxPipe expects it at ../embed/gte-large.
        if not (GTE_MODEL_DIR / "config.json").exists():
            from huggingface_hub import snapshot_download

            print("Downloading thenlper/gte-large (one time in this Kaggle run)...")
            snapshot_download(
                repo_id="thenlper/gte-large",
                local_dir=str(GTE_MODEL_DIR),
            )
        if not (GTE_MODEL_DIR / "config.json").exists():
            raise FileNotFoundError(f"Invalid GTE model directory: {GTE_MODEL_DIR}")

        # Official source uses a relative path. Link only when the user supplied an
        # attached model elsewhere; this avoids duplicating the large model on disk.
        expected_gte_dir = CTXPIPE_DIR.parent / "embed" / "gte-large"
        if GTE_MODEL_DIR.resolve() != expected_gte_dir.resolve():
            expected_gte_dir.parent.mkdir(parents=True, exist_ok=True)
            if not expected_gte_dir.exists():
                expected_gte_dir.symlink_to(GTE_MODEL_DIR, target_is_directory=True)
            elif not (expected_gte_dir / "config.json").exists():
                raise RuntimeError(f"Existing GTE path is incomplete: {expected_gte_dir}")
        print("GTE model ready:", expected_gte_dir)
        """
    ),
    code(
        """
        # Prepare exact DiffPrep snapshots for the 17 datasets that use them in prior
        # experiments (including synthetic google=100000). The remaining 13 are loaded
        # from GitLab/DataGit by the repository loader.
        diffprep_names = {
            "abalone", "ada_prior", "avila", "connect-4", "eeg", "google",
            "house", "jungle_chess", "micro", "mozilla4", "obesity",
            "page-blocks", "pbcseq", "pol", "run_or_walk", "uscensus",
            "wall-robot-nav",
        }
        requested_names = [spec["name"] for spec in run_specs if spec["name"] in diffprep_names]
        attached_google = list(Path("/kaggle/input").glob("**/google/data.csv"))
        for attempt in range(3):
            missing_names = [
                name for name in requested_names
                if not (CACHE_DIR / f"{EVAL_DATASETS[name]}.csv").exists()
            ]
            if not missing_names:
                break
            export_command = [
                sys.executable,
                str(SOLUTION_DIR / "scripts" / "export_diffprep_datasets.py"),
                "--out-dir", str(CACHE_DIR),
                "--only", ",".join(missing_names),
            ]
            if attached_google:
                export_command += ["--diffprep-root", "/kaggle/input"]
            else:
                export_command += ["--download"]
            print(f"DiffPrep snapshot attempt {attempt + 1}/3: {missing_names}")
            subprocess.run(export_command, cwd=SOLUTION_DIR, check=False)
            if attempt < 2 and missing_names:
                time.sleep(3 * (attempt + 1))

        expected_ids = {int(EVAL_DATASETS[name]) for name in requested_names}
        missing_ids = sorted(
            dataset_id for dataset_id in expected_ids
            if not (CACHE_DIR / f"{dataset_id}.csv").exists()
        )
        if missing_ids:
            raise RuntimeError(
                f"Could not download required DiffPrep snapshots after 3 attempts: {missing_ids}. "
                "Inspect the FAIL lines above and retry this cell later."
            )
        print("Local CSV snapshots:", len(list(CACHE_DIR.glob("*.csv"))))
        """
    ),
    code(
        """
        # Materialize only outer train+validation for native CtxPipe. The outer test
        # rows are never written into CtxPipe's data directory.
        manifests = {}
        for spec in run_specs:
            dataset_id = int(spec["dataset_id"])
            dataset = load_gitlab_openml_dataset(
                dataset_id,
                cache_dir=str(CACHE_DIR),
                test_dataset_ids=[int(value) for value in EVAL_IDS],
                verbose=True,
                max_samples_if_test=MAX_SAMPLES,
            )
            if dataset is None:
                raise RuntimeError(f"Could not load dataset {dataset_id}")
            if dataset.get("task_type") != "classification":
                raise ValueError(f"CtxPipe native checkpoint is classification-only: {dataset_id}")

            X = pd.DataFrame(dataset["X"]).copy()
            y = pd.Series(dataset["y"]).copy()
            X_train, y_train, X_val, y_val, X_test, _y_test = split_train_val_test(
                X, y, seed=SPLIT_SEED
            )
            X_search = pd.concat([X_train, X_val], axis=0, ignore_index=True)
            y_search = pd.concat([y_train, y_val], axis=0, ignore_index=True)
            X_search.columns = [str(value) for value in X_search.columns]
            if "target" in X_search.columns:
                X_search = X_search.rename(columns={"target": "target__feature"})
            frame = X_search.reset_index(drop=True)
            frame["target"] = y_search.reset_index(drop=True)

            folder_name = f"{dataset_id}__{spec['name']}"
            dataset_dir = CTXPIPE_DATA_DIR / folder_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(dataset_dir / "data.csv", index=False)
            (dataset_dir / "info.json").write_text(
                json.dumps({"label": "target"}, indent=2), encoding="utf-8"
            )
            manifests[folder_name] = {
                "dataset_id": dataset_id,
                "dataset_name": spec["name"],
                "raw_rows": int(len(X)),
                "search_rows": int(len(frame)),
                "native_internal_train_rows": int(math.floor(0.8 * len(frame))),
                "native_internal_reward_rows": int(len(frame) - math.floor(0.8 * len(frame))),
                "outer_train_rows": int(len(X_train)),
                "outer_validation_rows": int(len(X_val)),
                "outer_test_rows": int(len(X_test)),
            }
            del dataset, X, y, X_train, y_train, X_val, y_val, X_test, frame
            gc.collect()

        (OUTPUT_DIR / "search_manifests.json").write_text(
            json.dumps(manifests, indent=2), encoding="utf-8"
        )
        display(pd.DataFrame(manifests.values()))
        """
    ),
    code(
        """
        # Run each native inference in its own process. This contains RAM from heavy
        # operators and lets the shard continue when one dataset is killed or times out.
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        # The runner file lives under OUTPUT_DIR and is invoked by absolute path;
        # explicitly expose the official CtxPipe checkout so config.py/test.py resolve.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(CTXPIPE_DIR), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        parsed = {}
        native_failures = {}
        runner_source = "\\n".join(
            [
                "import sys",
                "import config",
                "from config import default_agent_config, default_config",
                "from ctxpipe.info import Info",
                "from test import evaluate",
                "dataset_root, work_root, pipelines_path = sys.argv[1:4]",
                "default_config.pipelines_file_name = pipelines_path",
                "default_agent_config.pipelines_file_name = pipelines_path",
                "evaluate(Info(aipipe_core_prefix=work_root + '/aipipe', "
                "result_prefix=work_root + '/result', dataset_prefix=dataset_root), "
                "start=32000, end=32000)",
            ]
        ) + "\\n"
        runner_path = OUTPUT_DIR / "run_one_native_ctxpipe.py"
        runner_path.write_text(runner_source, encoding="utf-8")

        def kill_process_tree(pid):
            try:
                parent = psutil.Process(pid)
                processes = parent.children(recursive=True) + [parent]
            except psutil.NoSuchProcess:
                processes = []
            for child in reversed(processes):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(processes, timeout=10)

        for folder_name, manifest in manifests.items():
            dataset_id = manifest["dataset_id"]
            dataset_output = OUTPUT_DIR / f"dataset_{dataset_id}"
            dataset_output.mkdir(parents=True, exist_ok=True)
            isolated_root = OUTPUT_DIR / "native_inputs" / folder_name
            isolated_root.mkdir(parents=True, exist_ok=True)
            link = isolated_root / folder_name
            if not link.exists():
                link.symlink_to(CTXPIPE_DATA_DIR / folder_name, target_is_directory=True)
            pipelines_path = dataset_output / "native_pipeline.tsv"
            pipelines_path.unlink(missing_ok=True)
            native_work = dataset_output / "native_work"
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(runner_path),
                        str(isolated_root),
                        str(native_work),
                        str(pipelines_path),
                    ],
                    cwd=CTXPIPE_DIR,
                    env=env,
                    start_new_session=True,
                )
                return_code = process.wait(timeout=CTXPIPE_TIMEOUT_MINS_PER_DATASET * 60)
                if return_code != 0:
                    native_failures[folder_name] = (
                        f"CtxPipe subprocess return code {return_code}"
                    )
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            except subprocess.TimeoutExpired:
                kill_process_tree(process.pid)
                native_failures[folder_name] = (
                    f"CtxPipe exceeded {CTXPIPE_TIMEOUT_MINS_PER_DATASET} minutes"
                )

            if pipelines_path.exists():
                for line in pipelines_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split("\t", 3)
                    if len(parts) != 4:
                        continue
                    tag, emitted_name, rendered_sequence, reward = parts
                    sequence = re.findall(r"<([^>]+)>", rendered_sequence)
                    if emitted_name == folder_name and len(sequence) == 6:
                        parsed[folder_name] = {
                            **manifest,
                            "status": "ok",
                            "sequence": sequence,
                            "native_reward": float(reward),
                            "checkpoint": tag,
                            "official_commit": CTXPIPE_COMMIT,
                            "outer_test_seen": False,
                            "checkpoint_sha256": observed,
                        }
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for folder_name, manifest in manifests.items():
            dataset_dir = OUTPUT_DIR / f"dataset_{manifest['dataset_id']}"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            result = parsed.get(
                folder_name,
                {
                    **manifest,
                    "status": "failed",
                    "error": native_failures.get(
                        folder_name,
                        "Native CtxPipe did not emit a six-operator sequence",
                    ),
                    "official_commit": CTXPIPE_COMMIT,
                    "outer_test_seen": False,
                },
            )
            (dataset_dir / "ctxpipe_recommendation.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        print(f"Native recommendations: {len(parsed)}/{len(manifests)}")
        """
    ),
    code(
        """
        # Replay each frozen sequence leak-free and evaluate it with TPOT classifiers only.
        import shlex

        tpot_minutes = 1 if RUN_MODE == "smoke" else TPOT_MAX_TIME_MINS
        for spec in run_specs:
            dataset_id = int(spec["dataset_id"])
            dataset_dir = OUTPUT_DIR / f"dataset_{dataset_id}"
            recommendation_path = dataset_dir / "ctxpipe_recommendation.json"
            recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
            if recommendation.get("status") != "ok":
                print(f"SKIP TPOT {dataset_id}: native CtxPipe failed")
                continue
            output_path = dataset_dir / "tpot_evaluation.json"
            command = [
                sys.executable,
                str(SOLUTION_DIR / "scripts" / "evaluate_ctxpipe_tpot.py"),
                "--ctxpipe-result-json", str(recommendation_path),
                "--dataset-id", str(dataset_id),
                "--data-dir", str(CACHE_DIR),
                "--output-json", str(output_path),
                "--max-samples", str(MAX_SAMPLES),
                "--split-seed", str(SPLIT_SEED),
                "--tpot-seed", str(TPOT_RANDOM_STATE),
                "--max-time-mins", str(tpot_minutes),
                "--max-eval-time-mins", str(TPOT_MAX_EVAL_TIME_MINS),
                "--n-jobs", str(TPOT_N_JOBS),
                "--memory-limit", TPOT_WORKER_MEMORY,
                "--population-size", str(TPOT_POPULATION_SIZE),
                "--max-cv-folds", str(TPOT_MAX_CV_FOLDS),
                "--maximum-cells", str(MAXIMUM_MATRIX_CELLS),
                "--verbose", "2",
            ]
            print("\\n", " ".join(shlex.quote(value) for value in command))
            completed = subprocess.run(command, cwd=SOLUTION_DIR, env=env, check=False)
            if completed.returncode != 0:
                print(f"TPOT failed for {dataset_id}; failure JSON was retained")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        """
    ),
    code(
        """
        # Summarize and archive all recommendations, scores, traces, and failures.
        rows = []
        for spec in run_specs:
            dataset_id = int(spec["dataset_id"])
            dataset_dir = OUTPUT_DIR / f"dataset_{dataset_id}"
            recommendation = json.loads(
                (dataset_dir / "ctxpipe_recommendation.json").read_text(encoding="utf-8")
            )
            evaluation_path = dataset_dir / "tpot_evaluation.json"
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": spec["name"],
                    "ctxpipe_status": recommendation.get("status"),
                    "ctxpipe_sequence": " -> ".join(recommendation.get("sequence", [])),
                    "native_reward": recommendation.get("native_reward"),
                    "tpot_status": evaluation.get("status", "not_run"),
                    "accuracy": evaluation.get("accuracy"),
                    "balanced_accuracy": evaluation.get("balanced_accuracy"),
                    "f1_macro": evaluation.get("f1_macro"),
                    "target_label_encoding": evaluation.get("target_label_encoding"),
                    "error": evaluation.get("error", recommendation.get("error")),
                }
            )
        summary = pd.DataFrame(rows)
        summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
        display(summary)

        archive = shutil.make_archive(str(OUTPUT_DIR), "zip", root_dir=OUTPUT_DIR)
        print("Download:", archive)
        """
    ),
]


for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]), filename=f"cell_{index}")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUTPUT}")
