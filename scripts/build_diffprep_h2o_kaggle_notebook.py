"""Build Kaggle notebook for No-preprocessing and DiffPrep + H2O."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "reproduce-diffprep-tpot.ipynb"
OUTPUT = ROOT / "notebooks" / "reproduce-diffprep-h2o.ipynb"


def _source(value: str) -> list[str]:
    return (textwrap.dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def _code(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(value)}


def _markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
cells = notebook["cells"]
cells[0] = _markdown(
    """
    # No preprocessing and DiffPrep + H2O AutoML

    This notebook evaluates two settings on the exact 30-dataset ACORec test
    suite. The first sends the raw split directly to H2O. The second runs the
    original DiffPrep pipeline and sends its transformed data to H2O.

    H2O target encoding is disabled in both settings. H2O still performs its
    native categorical and missing-value handling inside individual models.
    The model is selected on validation and scored once on the outer test.
    Run five Save-Version jobs with `DATASET_SHARD_INDEX=0..4`.
    """
)
cells[1] = _code(
    """
    # Install DiffPrep dependencies and H2O; TPOT is not used here.
    %pip install -q "h2o==3.46.0.11" "impyute>=0.0.8" "pyarrow>=15" "requests"
    """
)
cells[2] = _code(
    """
    from __future__ import annotations
    import gc
    import json
    import os
    import pickle
    import shutil
    import subprocess
    import sys
    import time
    import traceback
    import warnings
    from pathlib import Path

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests
    import torch
    import h2o
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder
    warnings.filterwarnings("ignore")

    RUN_MODE = "smoke"       # change to final after the one-dataset smoke run
    NUM_DATASET_SHARDS = 5
    DATASET_SHARD_INDEX = 0
    METHOD = "diffprep_fix"
    SPLIT_SEED = 42
    TRAIN_SEED = 1
    MAX_SAMPLES = 100_000
    PARQUET_BATCH_SIZE = 4_096
    H2O_MAX_RUNTIME_SECS = 120 if RUN_MODE == "smoke" else 300
    H2O_MAX_RUNTIME_SECS_PER_MODEL = 60
    H2O_NFOLDS = 5
    H2O_NTHREADS = 1
    H2O_MAX_MEM_SIZE = "6G"

    KAGGLE = Path("/kaggle/working").exists()
    OUTPUT_DIR = Path("/kaggle/working/diffprep_h2o") if KAGGLE else Path("outputs/diffprep_h2o")
    TEMP_ROOT = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else OUTPUT_DIR / "temp"
    REPO_DIR = TEMP_ROOT / "DiffPrep"
    SOLUTION_DIR = TEMP_ROOT / "SolutionRecommendation"
    CACHE_DIR = TEMP_ROOT / "openml_datagit_cache"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if RUN_MODE not in {"smoke", "final"} or not 0 <= DATASET_SHARD_INDEX < NUM_DATASET_SHARDS:
        raise ValueError("DATASET_SHARD_INDEX out of range")
    print("H2O:", h2o.__version__)
    """
)
cells[3] = _code(
    """
    # Clone the exact DiffPrep fork and the repository containing the shared
    # H2O evaluator utility.
    if not (REPO_DIR / ".git").exists():
        subprocess.run(["git", "clone", "--branch", "kaggle-experiments", "--single-branch", "https://github.com/dangvu53/DiffPrep.git", str(REPO_DIR)], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "switch", "kaggle-experiments"], check=True)
    commit = subprocess.check_output(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True).strip()
    if not (SOLUTION_DIR / ".git").exists():
        subprocess.run(["git", "clone", "--branch", "feature/acorec-autodp-space", "--single-branch", "https://github.com/MothMalone/SolutionRecommendation.git", str(SOLUTION_DIR)], check=True)
    solution_commit = subprocess.check_output(["git", "-C", str(SOLUTION_DIR), "rev-parse", "HEAD"], text=True).strip()
    sys.path.insert(0, str(SOLUTION_DIR / "scripts"))
    sys.path.insert(0, str(SOLUTION_DIR / "src"))
    from h2o_evaluator import evaluate_h2o_frames
    from automl_aco.data.loaders import load_gitlab_openml_dataset
    from automl_aco.eval_ids import EVAL_IDS

    # The upstream trainer evaluates X_test every epoch and passes it into
    # pipeline initialization. Patch that behavior in this reproduction:
    # DiffPrep may use train/validation only; outer test is reserved for H2O.
    def patch_exact(path, old, new):
        path = Path(path)
        source = path.read_text(encoding="utf-8")
        if old in source:
            path.write_text(source.replace(old, new), encoding="utf-8")
        elif new not in source:
            raise RuntimeError(f"DiffPrep leakage patch target not found: {path}")

    for pipeline_name in ("diffprep_fix_pipeline.py", "diffprep_flex_pipeline.py"):
        pipeline_path = REPO_DIR / "pipeline" / pipeline_name
        patch_exact(
            pipeline_path,
            "return df.isnull().values.sum() > 0",
            "return df is not None and df.isnull().values.sum() > 0",
        )
        patch_exact(
            pipeline_path,
            '        first_transformer.pre_cache(X_test, "test")',
            '        if X_test is not None:' + chr(92) + 'n            first_transformer.pre_cache(X_test, "test")',
        )
    patch_exact(
        REPO_DIR / "experiment" / "diffprep_experiment.py",
        "prep_pipeline.init_parameters(X_train, X_val, X_test)",
        "prep_pipeline.init_parameters(X_train, X_val, None)",
    )
    patch_exact(
        REPO_DIR / "experiment" / "diffprep_experiment.py",
        "result, best_model = diff_prep.fit(X_train, y_train, X_val, y_val, X_test, y_test)",
        "result, best_model = diff_prep.fit(X_train, y_train, X_val, y_val, None, None)",
    )
    patch_exact(
        REPO_DIR / "trainer" / "diffprep_trainer.py",
        "            test_loss, test_acc = self.evaluate(X_test, y_test, X_type='test', max_only=False)",
        "            if X_test is None or y_test is None:" + chr(92) + "n                test_loss, test_acc = float('nan'), float('nan')" + chr(92) + "n            else:" + chr(92) + "n                test_loss, test_acc = self.evaluate(X_test, y_test, X_type='test', max_only=False)",
    )
    patch_exact(
        REPO_DIR / "extract_and_save_pipeline.py",
        "prep_pipeline.init_parameters(X_train, X_val, X_test)",
        "prep_pipeline.init_parameters(X_train, X_val, None)",
    )
    patch_exact(
        REPO_DIR / "extract_and_save_pipeline.py",
        "'original_test_acc': result['best_test_acc'],",
        "'original_test_acc': None,",
    )
    print("Patched DiffPrep: no outer-test access during pipeline search")
    os.chdir(REPO_DIR)
    print("DiffPrep commit:", commit)
    print("SolutionRecommendation commit:", solution_commit)
    """
)
# Cell 4 keeps the canonical 30-dataset list. Replace the old DiffPrep loader
# with a canonical materializer so every method sees exactly the same rows.
cells[5] = _code(
    """
    def materialize_for_diffprep(spec):
        dataset_key = spec.get("dataset_key", str(spec["dataset_id"]))
        dataset_dir = REPO_DIR / "data" / dataset_key
        data_path = dataset_dir / "data.csv"
        info_path = dataset_dir / "info.json"

        # Google is synthetic (100000), so seed the canonical loader with the
        # exact frozen DiffPrep CSV when it is not attached as a Kaggle input.
        if spec.get("source") == "kaggle_csv":
            canonical_csv = CACHE_DIR / f"{int(spec['dataset_id'])}.csv"
            source_google = REPO_DIR / "data" / dataset_key / "data.csv"
            if not canonical_csv.exists() and source_google.exists():
                shutil.copyfile(source_google, canonical_csv)

        dataset = load_gitlab_openml_dataset(
            int(spec["dataset_id"]),
            cache_dir=str(CACHE_DIR),
            test_dataset_ids=[int(value) for value in EVAL_IDS],
            verbose=True,
            max_samples_if_test=MAX_SAMPLES,
        )
        if dataset is None:
            raise RuntimeError(f"Canonical loader could not load {spec['name']}")

        frame = pd.DataFrame(dataset["X"]).copy()
        frame["target"] = pd.Series(dataset["y"]).reset_index(drop=True)
        if len(frame) < 20 or frame.shape[1] < 2:
            raise ValueError(f"Insufficient usable data: {frame.shape}")

        # DiffPrep's build_data() label-encodes the target. The canonical
        # loader already emits contiguous integer labels, so this is idempotent.
        dataset_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(data_path, index=False)
        info = {
            "label": "target",
            "dataset_id": int(spec["dataset_id"]),
            "dataset_name": spec["name"],
            "source": "canonical_acorec_loader",
            "original_rows": int(dataset.get("original_rows", len(frame))),
            "used_rows": int(len(frame)),
            "raw_features": int(frame.shape[1] - 1),
        }
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"Canonical DiffPrep input {spec['name']}: {frame.shape}")
        del dataset, frame
        gc.collect()
        return dataset_key, info
    """
)
cells[6] = _code(
    """
    def as_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def load_saved_pipeline(dataset_key):
        directory = REPO_DIR / "saved_pipelines" / METHOD / dataset_key
        with (directory / "pipeline.pkl").open("rb") as handle:
            pipeline = pickle.load(handle)
        with (directory / "data_splits.pkl").open("rb") as handle:
            split = pickle.load(handle)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        return pipeline, split, metadata, directory

    def transform_with_diffprep(pipeline, split):
        if not pipeline.is_fitted:
            pipeline.fit(split["X_train"])
        # Test rows are cached only now, after the DiffPrep search is over.
        if "test" not in pipeline.pipeline[0].cache:
            pipeline.pipeline[0].pre_cache(split["X_test"], "test")
        transformed = {}
        with torch.no_grad():
            for part in ("train", "val", "test"):
                value = pipeline.transform(split[f"X_{part}"], X_type=part, max_only=True, resample=False)
                array = as_numpy(value).astype(np.float32, copy=False)
                if not np.isfinite(array).all():
                    raise ValueError(f"DiffPrep produced NaN/inf in transformed {part}")
                transformed[part] = array
        return transformed

    def assert_split_alignment(raw_split, diffprep_split):
        # The target sequence is a compact, deterministic row-order witness.
        # It catches class filtering/sampling/order differences before H2O runs.
        for part in ("train", "val", "test"):
            raw_y = np.asarray(raw_split[f"y_{part}"]).reshape(-1)
            diff_y = as_numpy(diffprep_split[f"y_{part}"]).reshape(-1)
            raw_encoded = LabelEncoder().fit_transform(pd.Series(raw_y).astype(str))
            diff_encoded = LabelEncoder().fit_transform(pd.Series(diff_y).astype(str))
            if raw_encoded.shape != diff_encoded.shape or not np.array_equal(raw_encoded, diff_encoded):
                raise RuntimeError(
                    f"DiffPrep split mismatch in {part}: "
                    f"canonical={raw_y.shape}, diffprep={diff_y.shape}"
                )

    def build_raw_split(dataset_key, data_info):
        # Create the same deterministic raw 60/20/20 split as the baseline.
        frame = pd.read_csv(REPO_DIR / "data" / dataset_key / "data.csv")
        target = str(data_info["label"])
        frame = frame.loc[~frame[target].isna()].reset_index(drop=True)
        y = frame.pop(target).reset_index(drop=True)
        n_val = int(len(y) * 0.20)
        n_test = int(len(y) * 0.20)
        indices = np.random.RandomState(SPLIT_SEED).permutation(len(y))
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]
        return {
            "X_train": frame.iloc[train_idx].reset_index(drop=True),
            "y_train": y.iloc[train_idx].reset_index(drop=True),
            "X_val": frame.iloc[val_idx].reset_index(drop=True),
            "y_val": y.iloc[val_idx].reset_index(drop=True),
            "X_test": frame.iloc[test_idx].reset_index(drop=True),
            "y_test": y.iloc[test_idx].reset_index(drop=True),
        }

    def evaluate_setting(setting, spec, dataset_key, split, data_info, transformed=None, metadata=None):
        X_train = transformed["train"] if transformed is not None else as_numpy(split["X_train"])
        X_val = transformed["val"] if transformed is not None else as_numpy(split["X_val"])
        X_test = transformed["test"] if transformed is not None else as_numpy(split["X_test"])
        y_train = as_numpy(split["y_train"]).ravel()
        y_val = as_numpy(split["y_val"]).ravel()
        y_test = as_numpy(split["y_test"]).ravel()
        started = time.time()
        result, model = evaluate_h2o_frames(
            X_train, y_train, X_val, y_val, X_test, y_test,
            task_type="classification", h2o_preprocessing=None,
            max_runtime_secs=H2O_MAX_RUNTIME_SECS,
            max_runtime_secs_per_model=H2O_MAX_RUNTIME_SECS_PER_MODEL,
            nfolds=H2O_NFOLDS, seed=TRAIN_SEED,
            nthreads=H2O_NTHREADS, max_mem_size=H2O_MAX_MEM_SIZE,
        )
        config = {}
        if metadata is not None:
            config_path = (REPO_DIR / "saved_pipelines" / METHOD / dataset_key / "pipeline_config.json")
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
        result.update({
            "dataset_id": int(spec["dataset_id"]), "dataset": spec["name"],
            "dataset_key": dataset_key, "setting": setting,
            "method": setting if setting == "no_preprocessing" else METHOD,
            "source": data_info.get("source", "diffprep_fork_snapshot"),
            "original_rows": data_info.get("original_rows"),
            "used_rows": int(len(y_train) + len(y_val) + len(y_test)),
            "raw_features": int(split["X_train"].shape[1]),
            "transformed_features": int(X_train.shape[1]),
            "validation_rows": int(len(y_val)), "split_seed": SPLIT_SEED,
            "train_seed": TRAIN_SEED, "diffprep_commit": commit,
            "diffprep_test_seen_during_search": False,
            "solution_commit": solution_commit, "total_seconds": float(time.time() - started),
            "diffprep_internal_test_accuracy": None if metadata is None else metadata.get("original_test_acc"),
            "diffprep_pipeline_config": json.dumps(config, separators=(",", ":")),
        })
        del model
        gc.collect()
        return result
    """
)
cells[7] = _code(
    """
    RESULT_PATH = OUTPUT_DIR / f"h2o_no_preprocessing_diffprep_shard_{DATASET_SHARD_INDEX:02d}_of_{NUM_DATASET_SHARDS:02d}.csv"
    def read_rows():
        return pd.read_csv(RESULT_PATH).to_dict("records") if RESULT_PATH.exists() else []
    def upsert(rows, row):
        key = (str(row["dataset_id"]), str(row["setting"]))
        rows[:] = [old for old in rows if (str(old.get("dataset_id")), str(old.get("setting"))) != key]
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULT_PATH, index=False)

    rows = read_rows()
    completed = {(str(row.get("dataset_id")), str(row.get("setting"))) for row in rows if row.get("status") == "ok"}
    positions = np.array_split(np.arange(len(DATASETS)), NUM_DATASET_SHARDS)
    SHARD_DATASETS = [DATASETS[int(i)] for i in positions[DATASET_SHARD_INDEX]]
    RUN_DATASETS = SHARD_DATASETS[:1] if RUN_MODE == "smoke" else SHARD_DATASETS
    for position, spec in enumerate(RUN_DATASETS, start=1):
        dataset_key = spec.get("dataset_key", str(spec["dataset_id"]))
        try:
            dataset_key, data_info = materialize_for_diffprep(spec)
            subprocess.run([sys.executable, "main.py", "--dataset", dataset_key, "--method", METHOD, "--model", "log", "--split_seed", str(SPLIT_SEED), "--train_seed", str(TRAIN_SEED)], cwd=REPO_DIR, check=True)
            subprocess.run([sys.executable, "extract_and_save_pipeline.py", "--dataset", dataset_key, "--method", METHOD, "--split_seed", str(SPLIT_SEED)], cwd=REPO_DIR, check=True)
            subprocess.run([sys.executable, "extract_pipeline_config.py", "--dataset", dataset_key, "--method", METHOD], cwd=REPO_DIR, check=True)
            pipeline, split, metadata, _directory = load_saved_pipeline(dataset_key)
            transformed = transform_with_diffprep(pipeline, split)
            raw_split = build_raw_split(dataset_key, data_info)
            assert_split_alignment(raw_split, split)
            for setting, current_split, data in (("no_preprocessing", raw_split, None), ("diffprep", split, transformed)):
                if (str(spec["dataset_id"]), setting) in completed:
                    print(f"SKIP successful: {spec['name']} / {setting}")
                    continue
                print(f"[{position}/{len(RUN_DATASETS)}] {spec['name']} / {setting}")
                row = evaluate_setting(setting, spec, dataset_key, current_split, data_info, data, metadata if setting == "diffprep" else None)
                upsert(rows, row)
                print(f"H2O test accuracy: {row['accuracy']:.6f}")
        except Exception as error:
            traceback.print_exc()
            for setting in ("no_preprocessing", "diffprep"):
                if (str(spec["dataset_id"]), setting) not in completed:
                    upsert(rows, {"dataset_id": int(spec["dataset_id"]), "dataset": spec["name"], "setting": setting, "status": "failed", "error_type": type(error).__name__, "error": str(error)[:4000]})
        finally:
            gc.collect()
    print("Saved:", RESULT_PATH)
    display(pd.DataFrame(rows).sort_values(["dataset_id", "setting"]))
    """
)
cells[8] = _markdown(
    """
    ## Outputs

    Download `h2o_no_preprocessing_diffprep_shard_XX_of_05.csv` from the Kaggle
    output directory and concatenate the five shards. The `setting` column
    distinguishes `no_preprocessing` and `diffprep`.
    """
)

for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:02d}"
notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT}")
