#!/usr/bin/env python3
"""Build the official AutoGluon solution notebooks."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def source(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source(text)}


def write(path: Path, cells: list[dict]) -> None:
    for index, item in enumerate(cells):
        item["id"] = f"cell-{index:02d}"
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def baseline() -> list[dict]:
    return [
        md("""
        # AutoGluon baselines: No Preprocessing and Default preprocessing

        Both settings use the canonical 30-dataset suite and the repository's
        shared `split_train_val_test` function. AutoGluon trains on outer train,
        uses outer validation for selection, and scores untouched outer test once.

        AutoGluon is limited to 300 seconds per dataset/setting in both modes;
        smoke mode only reduces the number of datasets to one per shard.
        """),
        code("""
        import os, subprocess, sys
        from pathlib import Path

        REPO_URL = "https://github.com/MothMalone/SolutionRecommendation.git"
        REPO_BRANCH = "feature/acorec-autodp-space"
        REPO_DIR = Path("/kaggle/working/SolutionRecommendation")
        if (REPO_DIR / ".git").exists():
            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "switch", REPO_BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", REPO_BRANCH], check=True)
        else:
            subprocess.run(["git", "clone", "--branch", REPO_BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements-kaggle.txt")], check=True)
        os.chdir(REPO_DIR)
        print("Commit:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip())
        """),
        code("""
        import gc, json, os, subprocess, sys, time
        from datetime import datetime, timezone
        from pathlib import Path
        import numpy as np
        import pandas as pd

        sys.path.insert(0, str(REPO_DIR / "src"))
        from automl_aco.eval_ids import EVAL_DATASETS

        RUN_MODE = "smoke"       # change to final after smoke succeeds
        NUM_DATASET_SHARDS = 5
        DATASET_SHARD_INDEX = 0
        SPLIT_SEED = 42
        TRAIN_SEED = 1
        MAX_SAMPLES = 100_000
        AG_TIME_LIMIT = 300
        AG_PRESETS = "best_quality"
        DATASETS = [{"dataset_id": int(value), "name": name} for name, value in EVAL_DATASETS.items()]
        if RUN_MODE not in {"smoke", "final"} or not 0 <= DATASET_SHARD_INDEX < NUM_DATASET_SHARDS:
            raise ValueError("Invalid RUN_MODE or shard index")
        positions = np.array_split(np.arange(len(DATASETS)), NUM_DATASET_SHARDS)
        RUN_DATASETS = [DATASETS[int(i)] for i in positions[DATASET_SHARD_INDEX]]
        if RUN_MODE == "smoke":
            RUN_DATASETS = RUN_DATASETS[:1]
        BASE_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
        OUTPUT_DIR = BASE_DIR / "autogluon_baselines"
        CACHE_DIR = BASE_DIR / "openml_datagit_cache"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True); CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Shard {DATASET_SHARD_INDEX}/{NUM_DATASET_SHARDS - 1}; limit={AG_TIME_LIMIT}s; ids={[x['dataset_id'] for x in RUN_DATASETS]}")
        """),
        code("""
        evaluator_path = REPO_DIR / "scripts" / "autogluon_evaluator.py"
        if not evaluator_path.exists():
            raise RuntimeError("Remote checkout is stale: scripts/autogluon_evaluator.py is missing. Push the AutoGluon scripts to feature/acorec-autodp-space, then restart this Kaggle session.")
        evaluator_source = evaluator_path.read_text(encoding="utf-8")
        if "def evaluate_autogluon_split" not in evaluator_source or "IdentityFeatureGenerator" not in evaluator_source:
            raise RuntimeError("Stale AutoGluon evaluator; restart the Kaggle session and rerun setup")
        import autogluon.tabular
        print("AutoGluon loaded")
        """),
        code("""
        RESULT_PATH = OUTPUT_DIR / f"autogluon_baselines_shard_{DATASET_SHARD_INDEX:02d}_of_{NUM_DATASET_SHARDS:02d}.csv"
        RESULT_DIR = OUTPUT_DIR / "per_dataset"; RESULT_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS = [("no_preprocessing", "identity"), ("default_preprocessing", "default")]
        rows = pd.read_csv(RESULT_PATH).to_dict("records") if RESULT_PATH.exists() else []
        def upsert(row):
            key = (str(row.get("dataset_id")), str(row.get("setting")))
            rows[:] = [old for old in rows if (str(old.get("dataset_id")), str(old.get("setting"))) != key]
            rows.append(row); pd.DataFrame(rows).to_csv(RESULT_PATH, index=False)
        for position, spec in enumerate(RUN_DATASETS, start=1):
            for setting, feature_generator in SETTINGS:
                if any(str(x.get("dataset_id")) == str(spec["dataset_id"]) and x.get("setting") == setting and x.get("status") == "ok" for x in rows):
                    print("SKIP successful:", spec["name"], setting); continue
                output_json = RESULT_DIR / f"{int(spec['dataset_id'])}_{setting}.json"
                started_at = datetime.now(timezone.utc).isoformat(); started = time.perf_counter()
                command = [sys.executable, str(REPO_DIR / "scripts/evaluate_autogluon_baseline.py"), "--dataset-id", str(spec["dataset_id"]), "--dataset-name", spec["name"], "--data-dir", str(CACHE_DIR), "--output-json", str(output_json), "--setting", setting, "--feature-generator", feature_generator, "--split-seed", str(SPLIT_SEED), "--time-limit", str(AG_TIME_LIMIT), "--presets", AG_PRESETS, "--max-samples", str(MAX_SAMPLES)]
                print(f"[{position}/{len(RUN_DATASETS)}] {spec['name']} / {setting}")
                process = subprocess.run(command, cwd=REPO_DIR, check=False)
                row = json.loads(output_json.read_text(encoding="utf-8")) if output_json.exists() else {"status": "failed", "error": f"exit code {process.returncode}"}
                row.update({"dataset_id": int(spec["dataset_id"]), "dataset": spec["name"], "setting": setting, "feature_generator": feature_generator, "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "notebook_wall_clock_seconds": time.perf_counter() - started})
                upsert(row); gc.collect()
        display(pd.DataFrame(rows).sort_values(["dataset_id", "setting"]))
        print("Saved:", RESULT_PATH)
        """),
        md("""
        `fit_seconds`, `prediction_seconds`, `total_seconds`, and
        `notebook_wall_clock_seconds` are stored for every dataset/setting.
        The final test is never supplied during fitting or model selection.
        """),
    ]


def diffprep() -> list[dict]:
    payload = json.loads((NB_DIR / "reproduce-diffprep-h2o.ipynb").read_text(encoding="utf-8"))
    cells = payload["cells"]
    cells[0] = md("""
    # DiffPrep + AutoGluon

    This notebook runs the pinned DiffPrep pipeline and evaluates its frozen
    transformed features with AutoGluon and `IdentityFeatureGenerator`.
    `split_train_val_test` is the shared outer splitter; the outer test stays
    untouched until final scoring. AutoGluon is limited to 300 seconds in both
    modes; smoke mode only reduces the number of datasets.
    """)
    cells[1] = code('''
    # DiffPrep plus AutoGluon; H2O and TPOT are not used here.
    %pip install -q "autogluon.tabular==1.5.0" "impyute>=0.0.8" "pyarrow>=15" "requests"
    ''')
    setup = "".join(cells[2]["source"])
    setup = setup.replace("import h2o\n", "")
    setup = setup.replace('H2O_MAX_RUNTIME_SECS = 120 if RUN_MODE == "smoke" else 300\nH2O_MAX_RUNTIME_SECS_PER_MODEL = 60\nH2O_NFOLDS = 5\nH2O_NTHREADS = 1\nH2O_MAX_MEM_SIZE = "6G"', 'AG_TIME_LIMIT = 300\nAG_PRESETS = "best_quality"')
    setup = setup.replace('OUTPUT_DIR = Path("/kaggle/working/diffprep_h2o") if KAGGLE else Path("outputs/diffprep_h2o")', 'OUTPUT_DIR = Path("/kaggle/working/diffprep_autogluon") if KAGGLE else Path("outputs/diffprep_autogluon")')
    setup = setup.replace('print("H2O:", h2o.__version__)', 'print("AutoGluon limit:", AG_TIME_LIMIT)')
    cells[2] = code(setup)
    clone = "".join(cells[3]["source"])
    start = clone.index("# Guard against an old notebook/session cache before a long run.")
    end = clone.index("# The upstream trainer evaluates X_test", start)
    guard = '''# Guard against a stale AutoGluon evaluator in the cloned checkout.
evaluator_path = SOLUTION_DIR / "scripts" / "autogluon_evaluator.py"
    if not evaluator_path.exists():
        raise RuntimeError("Remote checkout is stale: scripts/autogluon_evaluator.py is missing. Push the AutoGluon scripts to feature/acorec-autodp-space, then restart this Kaggle session.")
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
if "def evaluate_autogluon_split" not in evaluator_source or "IdentityFeatureGenerator" not in evaluator_source:
    raise RuntimeError("Stale AutoGluon evaluator detected; restart the Kaggle session and rerun this cell.")
sys.path.insert(0, str(SOLUTION_DIR / "scripts"))
sys.path.insert(0, str(SOLUTION_DIR / "src"))
import importlib
importlib.invalidate_caches()
from automl_aco.data.loaders import load_gitlab_openml_dataset
from automl_aco.eval_ids import EVAL_IDS

'''
    clone = clone[:start] + guard + clone[end:]
    cells[3] = code(clone)
    cells[6] = code('''
    # The evaluator reloads and transforms the frozen pipeline after DiffPrep
    # search. This cell intentionally does not fit an AutoGluon model.
    ''')
    cells[7] = code('''
    RESULT_PATH = OUTPUT_DIR / f"diffprep_autogluon_shard_{DATASET_SHARD_INDEX:02d}_of_{NUM_DATASET_SHARDS:02d}.csv"
    RESULT_DIR = OUTPUT_DIR / "per_dataset"; RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(RESULT_PATH).to_dict("records") if RESULT_PATH.exists() else []
    def upsert(row):
        key = (str(row.get("dataset_id")), str(row.get("setting")))
        rows[:] = [old for old in rows if (str(old.get("dataset_id")), str(old.get("setting"))) != key]
        rows.append(row); pd.DataFrame(rows).to_csv(RESULT_PATH, index=False)
    positions = np.array_split(np.arange(len(DATASETS)), NUM_DATASET_SHARDS)
    SHARD_DATASETS = [DATASETS[int(i)] for i in positions[DATASET_SHARD_INDEX]]
    RUN_DATASETS = SHARD_DATASETS[:1] if RUN_MODE == "smoke" else SHARD_DATASETS
    for position, spec in enumerate(RUN_DATASETS, start=1):
        key = (str(spec["dataset_id"]), "diffprep")
        if any((str(x.get("dataset_id")), x.get("setting"), x.get("status")) == (*key, "ok") for x in rows):
            print("SKIP successful:", spec["name"]); continue
        started = time.perf_counter(); output_json = RESULT_DIR / f"{int(spec['dataset_id'])}_diffprep.json"
        print(f"[{position}/{len(RUN_DATASETS)}] {spec['name']} / diffprep")
        try:
            dataset_key = spec.get("dataset_key", str(spec["dataset_id"]))
            dataset_key, _ = materialize_for_diffprep(spec)
            subprocess.run([sys.executable, "main.py", "--dataset", dataset_key, "--method", METHOD, "--model", "log", "--split_seed", str(SPLIT_SEED), "--train_seed", str(TRAIN_SEED)], cwd=REPO_DIR, check=True)
            subprocess.run([sys.executable, "extract_and_save_pipeline.py", "--dataset", dataset_key, "--method", METHOD, "--split_seed", str(SPLIT_SEED)], cwd=REPO_DIR, check=True)
            subprocess.run([sys.executable, "extract_pipeline_config.py", "--dataset", dataset_key, "--method", METHOD], cwd=REPO_DIR, check=True)
            command = [sys.executable, str(SOLUTION_DIR / "scripts/evaluate_diffprep_autogluon.py"), "--repo-dir", str(REPO_DIR), "--dataset-key", dataset_key, "--dataset-id", str(spec["dataset_id"]), "--dataset-name", spec["name"], "--method", METHOD, "--output-json", str(output_json), "--split-seed", str(SPLIT_SEED), "--train-seed", str(TRAIN_SEED), "--time-limit", str(AG_TIME_LIMIT), "--presets", AG_PRESETS]
            process = subprocess.run(command, cwd=SOLUTION_DIR, check=False)
            row = json.loads(output_json.read_text(encoding="utf-8")) if output_json.exists() else {"status": "failed", "error": f"exit code {process.returncode}"}
        except Exception as error:
            traceback.print_exc()
            row = {"status": "failed", "error_type": type(error).__name__, "error": str(error)[:4000]}
        row.update({"dataset_id": int(spec["dataset_id"]), "dataset": spec["name"], "setting": "diffprep", "notebook_wall_clock_seconds": time.perf_counter() - started})
        upsert(row); gc.collect()
    display(pd.DataFrame(rows).sort_values(["dataset_id", "setting"]))
    print("Saved:", RESULT_PATH)
    ''')
    cells[8] = md("""
    ## Outputs

    The result CSV includes AutoGluon fit/prediction/total runtime and the
    notebook wall clock. Successful rows must report
    `diffprep_test_seen_during_search=False`.
    """)
    return cells


def ctxpipe() -> list[dict]:
    payload = json.loads((NB_DIR / "reproduce-ctxpipe-h2o.ipynb").read_text(encoding="utf-8"))
    cells = payload["cells"]
    cells[0] = md("""
    # Native CtxPipe + AutoGluon on the 30 evaluation datasets

    Native CtxPipe search and the official checkpoint are unchanged. The
    frozen six-operator sequence is replayed leak-free and evaluated with
    AutoGluon using `IdentityFeatureGenerator`. The shared
    `split_train_val_test` function defines the outer 60/20/20 split.

    AutoGluon is limited to 300 seconds in both modes; smoke mode only runs
    one dataset per shard. Use a Kaggle GPU for native CtxPipe.
    """)
    setup = "".join(cells[1]["source"])
    setup = setup.replace("requirements-ctxpipe-h2o-kaggle.txt", "requirements-ctxpipe-autogluon-kaggle.txt")
    setup = setup.replace('"import numpy,pandas,sklearn,torch,h2o,transformers; "', '"import numpy,pandas,sklearn,torch,autogluon.tabular,transformers; "')
    setup = setup.replace('"torch.__version__,h2o.__version__,transformers.__version__)"', '"torch.__version__,autogluon.tabular.__version__,transformers.__version__)"')
    start = setup.index("evaluator_path =")
    end = setup.index('print("Solution commit:', start)
    setup = setup[:start] + '''evaluator_path = SOLUTION_DIR / "scripts" / "autogluon_evaluator.py"
    if not evaluator_path.exists():
        raise RuntimeError("Remote checkout is stale: scripts/autogluon_evaluator.py is missing. Push the AutoGluon scripts to feature/acorec-autodp-space, then restart this Kaggle session.")
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
if "def evaluate_autogluon_split" not in evaluator_source or "IdentityFeatureGenerator" not in evaluator_source:
    raise RuntimeError("Stale AutoGluon evaluator detected; restart the Kaggle session and rerun setup.")
''' + setup[end:]
    cells[1] = code(setup)
    controls = "".join(cells[2]["source"])
    controls = controls.replace('H2O_MAX_RUNTIME_SECS = 300\nH2O_MAX_RUNTIME_SECS_PER_MODEL = 60\nH2O_NFOLDS = 5\nH2O_NTHREADS = 1\nH2O_MAX_MEM_SIZE = "6G"', 'AG_TIME_LIMIT = 300\nAG_PRESETS = "best_quality"')
    controls = controls.replace("ctxpipe_h2o_data", "ctxpipe_autogluon_data").replace("ctxpipe_h2o_", "ctxpipe_autogluon_")
    cells[2] = code(controls)
    cells[8] = code('''
    # Replay each native sequence and evaluate it with AutoGluon.
    import shlex
    env = os.environ.copy()
    for spec in run_specs:
        dataset_id = int(spec["dataset_id"]); dataset_dir = OUTPUT_DIR / f"dataset_{dataset_id}"
        recommendation_path = dataset_dir / "ctxpipe_recommendation.json"
        recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
        if recommendation.get("status") != "ok":
            print(f"SKIP AutoGluon {dataset_id}: native CtxPipe failed"); continue
        output_path = dataset_dir / "autogluon_evaluation.json"
        command = [sys.executable, str(SOLUTION_DIR / "scripts/evaluate_ctxpipe_autogluon.py"), "--ctxpipe-result-json", str(recommendation_path), "--dataset-id", str(dataset_id), "--data-dir", str(CACHE_DIR), "--output-json", str(output_path), "--max-samples", str(MAX_SAMPLES), "--split-seed", str(SPLIT_SEED), "--time-limit", str(AG_TIME_LIMIT), "--presets", AG_PRESETS]
        print("AutoGluon evaluation:", " ".join(shlex.quote(value) for value in command))
        subprocess.run(command, cwd=SOLUTION_DIR, env=env, check=False)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    ''')
    cells[9] = code('''
    rows = []
    for spec in run_specs:
        dataset_id = int(spec["dataset_id"]); dataset_dir = OUTPUT_DIR / f"dataset_{dataset_id}"
        recommendation = json.loads((dataset_dir / "ctxpipe_recommendation.json").read_text(encoding="utf-8"))
        path = dataset_dir / "autogluon_evaluation.json"
        evaluation = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        rows.append({"dataset_id": dataset_id, "dataset_name": spec["name"], "ctxpipe_status": recommendation.get("status"), "ctxpipe_sequence": " -> ".join(recommendation.get("sequence", [])), "native_reward": recommendation.get("native_reward"), "autogluon_status": evaluation.get("status", "not_run"), "accuracy": evaluation.get("accuracy"), "validation_score": evaluation.get("validation_score"), "fit_seconds": evaluation.get("fit_seconds"), "total_seconds": evaluation.get("total_seconds"), "selected_model": evaluation.get("selected_model"), "error": evaluation.get("error", recommendation.get("error"))})
    summary = pd.DataFrame(rows); summary.to_csv(OUTPUT_DIR / "summary.csv", index=False); display(summary)
    archive = shutil.make_archive(str(OUTPUT_DIR), "zip", root_dir=OUTPUT_DIR); print("Download:", archive)
    ''')
    return cells


def acorec() -> list[dict]:
    return [
        md("""
        # ACORec + AutoGluon

        ACORec uses the current repository operator space (`ours`) and searches
        on outer train+validation only. The final frozen recommendation is
        evaluated by AutoGluon through `run_recommend.py`, which uses the same
        repository `split_train_val_test` implementation.

        AutoGluon is limited to 300 seconds per evaluation in both modes.
        Smoke mode only reduces the number of datasets. Runtime is recorded by
        the runner and by this notebook.
        """),
        code("""
        import os, subprocess, sys
        from pathlib import Path
        REPO_URL = "https://github.com/MothMalone/SolutionRecommendation.git"
        BRANCH = "feature/acorec-autodp-space"
        REPO_DIR = Path("/kaggle/working/SolutionRecommendation")
        if (REPO_DIR / ".git").exists():
            subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "switch", BRANCH], check=True)
            subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
        else:
            subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements-kaggle.txt")], check=True)
        os.chdir(REPO_DIR)
        print("Commit:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip())
        """),
        code("""
        from __future__ import annotations
        import json, os, shlex, shutil, subprocess, sys, time
        from pathlib import Path
        import pandas as pd
        sys.path.insert(0, str(REPO_DIR / "src"))
        from automl_aco.eval_ids import EVAL_DATASETS
        RUN_MODE = "smoke"       # change to final after smoke succeeds
        NUM_SHARDS = 10; SHARD_INDEX = 0; WORKERS = 1
        ACO_SEED = 42; SPLIT_SEED = 42; MAX_SAMPLES = 100_000
        AG_TIME_LIMIT = 300; AG_PRESETS = "best_quality"
        FINAL_N_ANTS, FINAL_N_ITERATIONS, FINAL_METRIC_EPOCHS = 10, 10, 100
        if RUN_MODE not in {"smoke", "final"} or not 0 <= SHARD_INDEX < NUM_SHARDS: raise ValueError("Invalid mode/shard")
        all_ids = [int(value) for value in EVAL_DATASETS.values()]
        run_ids = all_ids[SHARD_INDEX::NUM_SHARDS]; run_ids = run_ids[:1] if RUN_MODE == "smoke" else run_ids
        CACHE_DIR = Path("/kaggle/working/acorec_autogluon_data"); OUTPUT_DIR = Path(f"/kaggle/working/acorec_autogluon_{RUN_MODE}_shard_{SHARD_INDEX:02d}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Mode={RUN_MODE}; shard={SHARD_INDEX}; ids={run_ids}; limit={AG_TIME_LIMIT}s")
        """),
        code("""
        diffprep_names = {"abalone", "ada_prior", "avila", "connect-4", "eeg", "google", "house", "jungle_chess", "micro", "mozilla4", "obesity", "page-blocks", "pbcseq", "pol", "run_or_walk", "uscensus", "wall-robot-nav"}
        expected_ids = {int(EVAL_DATASETS[name]) for name in diffprep_names}
        command = [sys.executable, str(REPO_DIR / "scripts/export_diffprep_datasets.py"), "--out-dir", str(CACHE_DIR), "--download"]
        input_root = Path("/kaggle/input")
        attached_google = list(input_root.glob("**/google/data.csv")) if input_root.exists() else []
        if attached_google: command[-1:] = ["--diffprep-root", str(input_root)]
        subprocess.run(command, cwd=REPO_DIR, check=False)
        present = {int(path.stem) for path in CACHE_DIR.glob("*.csv") if path.stem.isdigit()}
        if expected_ids - present: raise RuntimeError(f"Missing DiffPrep snapshots: {sorted(expected_ids - present)}")
        print(f"Frozen DiffPrep snapshots ready: {len(present)}")
        """),
        code("""
        aco_command = [sys.executable, str(REPO_DIR / "scripts/run_recommend.py"), "--operator-space", "ours", "--performance-matrix", str(REPO_DIR / "data/openml/training_performance_matrix_autogluon.csv"), "--metafeatures", str(REPO_DIR / "data/openml/dataset_feats.csv"), "--pipeline-configs", str(REPO_DIR / "aco/pipeline_configs.json"), "--dataset-source", "openml", "--openml-backend", "gitlab", "--openml-local-folder", str(CACHE_DIR), "--dataset-ids", *[str(value) for value in run_ids], "--optimizer", "aco", "--seed", str(ACO_SEED), "--workers", str(WORKERS), "--output-dir", str(OUTPUT_DIR), "--skip-aco-plot", "--no-autogluon", "--recommend-on-train-val", "--recommend-split-seed", str(SPLIT_SEED), "--verbose"]
        if RUN_MODE == "smoke": aco_command += ["--n-ants", "1", "--n-iterations", "1", "--no-train-metric-inline"]
        else: aco_command += ["--n-ants", str(FINAL_N_ANTS), "--n-iterations", str(FINAL_N_ITERATIONS), "--train-metric-inline", "--metric-epochs", str(FINAL_METRIC_EPOCHS)]
        print("ACORec command:\\n", " ".join(shlex.quote(str(value)) for value in aco_command))
        """),
        code("""
        env = os.environ.copy(); env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
        started = time.perf_counter(); return_code = subprocess.run(aco_command, cwd=REPO_DIR, env=env, check=False).returncode
        run_wall_clock_seconds = time.perf_counter() - started
        if return_code != 0: raise RuntimeError(f"ACORec exited with code {return_code}")
        print(f"ACORec wall-clock seconds: {run_wall_clock_seconds:.3f}")
        for dataset_id in run_ids:
            dataset_dir = OUTPUT_DIR if len(run_ids) == 1 else OUTPUT_DIR / f"dataset_{dataset_id}"
            recommendation_path = dataset_dir / "recommendation.json"
            output_path = dataset_dir / "autogluon_evaluation.json"
            evaluation_command = [sys.executable, str(REPO_DIR / "scripts/evaluate_acorec_autogluon.py"), "--recommendation-json", str(recommendation_path), "--dataset-id", str(dataset_id), "--data-dir", str(CACHE_DIR), "--output-json", str(output_path), "--max-samples", str(MAX_SAMPLES), "--split-seed", str(SPLIT_SEED), "--time-limit", str(AG_TIME_LIMIT), "--presets", AG_PRESETS]
            print("Outer-test AutoGluon:", " ".join(shlex.quote(str(value)) for value in evaluation_command))
            subprocess.run(evaluation_command, cwd=REPO_DIR, env=env, check=False)
        """),
        code("""
        def dataset_output_dir(dataset_id): return OUTPUT_DIR if len(run_ids) == 1 else OUTPUT_DIR / f"dataset_{dataset_id}"
        rows = []
        for dataset_id in run_ids:
            dataset_dir = dataset_output_dir(dataset_id)
            result = json.loads((dataset_dir / "recommendation.json").read_text(encoding="utf-8"))
            evaluation_path = dataset_dir / "autogluon_evaluation.json"
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}
            rows.append({"dataset_id": int(dataset_id), "status": evaluation.get("status", "not_run"), "final_method": evaluation.get("method"), "score": evaluation.get("score"), "accuracy": evaluation.get("accuracy"), "aco_search_elapsed_seconds": result.get("elapsed_seconds"), "autogluon_total_seconds": evaluation.get("total_seconds"), "outer_evaluation_wall_clock_seconds": evaluation.get("acorec_and_evaluation_wall_clock_seconds"), "notebook_wall_clock_seconds": run_wall_clock_seconds, "autogluon_time_limit": AG_TIME_LIMIT, "error": evaluation.get("error", result.get("error"))})
        summary = pd.DataFrame(rows); summary.to_csv(OUTPUT_DIR / "acorec_autogluon_summary.csv", index=False); display(summary)
        archive = shutil.make_archive(str(Path("/kaggle/working") / OUTPUT_DIR.name), "gztar", root_dir=OUTPUT_DIR); print("Archive:", archive)
        """),
    ]


def main() -> None:
    write(NB_DIR / "reproduce-autogluon-baselines.ipynb", baseline())
    write(NB_DIR / "reproduce-diffprep-autogluon.ipynb", diffprep())
    write(NB_DIR / "reproduce-ctxpipe-autogluon.ipynb", ctxpipe())
    write(NB_DIR / "run-acorec-autogluon-kaggle.ipynb", acorec())
    print("Built AutoGluon notebooks")


if __name__ == "__main__":
    main()
