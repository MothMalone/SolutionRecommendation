#!/usr/bin/env python3
"""Unified runner for TPOT downstream preprocessing ablations.

Supports the 4 evaluation methods:
  1) no_prep      : No preprocessing (minimal adapter: median/mode impute + one-hot so TPOT can run, TPOT preprocessing=False)
  2) default_prep : Default TPOT preprocessing (TPOT preprocessing=True)
  3) ctxpipe      : CtxPipe DQN inference (from trained checkpoint ctx_32000) -> estimator-only TPOT (preprocessing=False)
  4) acorec       : ACORec recommended pipeline -> estimator-only TPOT (preprocessing=False)

Usage:
  # Run missing cells (44956, 1119, 1471):
  python scripts/run_tpot_ablations_suite.py \
      --data-dir /kaggle/working/eval_all \
      --ctxpipe-zip ctxpipe-3linear.zip \
      --out /kaggle/working/tpot_ablations.jsonl

  # Summarize results into full 30-dataset table:
  python scripts/run_tpot_ablations_suite.py \
      --out "/kaggle/working/tpot_ablations*.jsonl" \
      --out-csv /kaggle/working/tpot_ablations_table.csv \
      --summarize
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import _tpot_eval
from automl_aco.data.splits import split_train_val_test
from automl_aco.search.evaluation import _detect_problem_type, _fit_pipeline, _make_preprocessor

# The 30 evaluation datasets in order
TABLE_DATASETS_30 = [
    (1, "1066", "kc1-binary"),
    (2, "1047", "usp05"),
    (3, "862", "sleuth-ex2016"),
    (4, "40663", "calendarDOW"),
    (5, "1054", "mc2"),
    (6, "876", "fri-c1"),
    (7, "18", "mfeat-morphological"),
    (8, "1520", "robot-failures-lp5"),
    (9, "1548", "autoUniv-au4"),
    (10, "378", "ipums-la-99"),
    (11, "1485", "madelon"),
    (12, "14", "mfeat-fourier"),
    (13, "27", "colic"),
    (14, "44956", "abalone"),
    (15, "1037", "ada_prior"),
    (16, "42932", "avila"),
    (17, "40668", "connect-4"),
    (18, "1471", "eeg"),
    (19, "100000", "google"),
    (20, "42165", "house"),
    (21, "41001", "jungle_chess"),
    (22, "41671", "micro"),
    (23, "1046", "mozilla4"),
    (24, "46597", "obesity"),
    (25, "30", "page-blocks"),
    (26, "802", "pbcseq"),
    (27, "722", "pol"),
    (28, "40922", "run_or_walk"),
    (29, "1119", "uscensus"),
    (30, "1497", "wall-robot-nav"),
]

# Base known scores from user's table (None = missing cells to run)
BASE_TPOT_TABLE: Dict[str, Dict[str, Optional[float]]] = {
    "1066": {"No Preprocessing": 0.759, "Default Preprocessing": 0.655, "DiffPrep": 0.724, "CtxPipe": 0.759, "AutoDP": 0.724, "Tool(TPOT)": 0.724},
    "1047": {"No Preprocessing": 0.000, "Default Preprocessing": 0.000, "DiffPrep": 0.868, "CtxPipe": 0.026, "AutoDP": 0.833, "Tool(TPOT)": 0.921},
    "862": {"No Preprocessing": 0.765, "Default Preprocessing": 0.765, "DiffPrep": 0.765, "CtxPipe": 0.765, "AutoDP": 0.765, "Tool(TPOT)": 0.765},
    "40663": {"No Preprocessing": 0.608, "Default Preprocessing": 0.620, "DiffPrep": 0.671, "CtxPipe": 0.595, "AutoDP": 0.620, "Tool(TPOT)": 0.633},
    "1054": {"No Preprocessing": 0.813, "Default Preprocessing": 0.781, "DiffPrep": 0.750, "CtxPipe": 0.781, "AutoDP": 0.750, "Tool(TPOT)": 0.781},
    "876": {"No Preprocessing": 0.600, "Default Preprocessing": 0.650, "DiffPrep": 0.600, "CtxPipe": 0.400, "AutoDP": 0.600, "Tool(TPOT)": 0.700},
    "18": {"No Preprocessing": 0.753, "Default Preprocessing": 0.738, "DiffPrep": 0.743, "CtxPipe": 0.770, "AutoDP": 0.753, "Tool(TPOT)": 0.750},
    "1520": {"No Preprocessing": 0.719, "Default Preprocessing": 0.656, "DiffPrep": 0.625, "CtxPipe": 0.594, "AutoDP": 0.750, "Tool(TPOT)": 0.688},
    "1548": {"No Preprocessing": 0.628, "Default Preprocessing": 0.646, "DiffPrep": 0.634, "CtxPipe": 0.492, "AutoDP": 0.622, "Tool(TPOT)": 0.658},
    "378": {"No Preprocessing": 0.809, "Default Preprocessing": 0.657, "DiffPrep": 0.818, "CtxPipe": 0.821, "AutoDP": 0.827, "Tool(TPOT)": 0.816},
    "1485": {"No Preprocessing": 0.808, "Default Preprocessing": 0.814, "DiffPrep": 0.835, "CtxPipe": 0.564, "AutoDP": 0.869, "Tool(TPOT)": 0.625},
    "14": {"No Preprocessing": 0.840, "Default Preprocessing": 0.840, "DiffPrep": 0.833, "CtxPipe": 0.768, "AutoDP": 0.848, "Tool(TPOT)": 0.835},
    "27": {"No Preprocessing": 0.877, "Default Preprocessing": 0.863, "DiffPrep": 0.890, "CtxPipe": 0.877, "AutoDP": 0.836, "Tool(TPOT)": 0.877},
    "44956": {"No Preprocessing": None, "Default Preprocessing": None, "DiffPrep": None, "CtxPipe": None, "AutoDP": 0.268, "Tool(TPOT)": None},
    "1037": {"No Preprocessing": 0.832, "Default Preprocessing": 0.833, "DiffPrep": 0.836, "CtxPipe": 0.111, "AutoDP": 0.833, "Tool(TPOT)": 0.817},
    "42932": {"No Preprocessing": 0.996, "Default Preprocessing": 0.946, "DiffPrep": 0.998, "CtxPipe": 0.652, "AutoDP": 0.967, "Tool(TPOT)": 0.921},
    "40668": {"No Preprocessing": 0.766, "Default Preprocessing": 0.657, "DiffPrep": 0.832, "CtxPipe": 0.536, "AutoDP": 0.764, "Tool(TPOT)": 0.832},
    "1471": {"No Preprocessing": 0.955, "Default Preprocessing": 0.856, "DiffPrep": 0.923, "CtxPipe": None, "AutoDP": 0.937, "Tool(TPOT)": None},
    "100000": {"No Preprocessing": 0.677, "Default Preprocessing": 0.677, "DiffPrep": 0.677, "CtxPipe": 0.593, "AutoDP": 0.696, "Tool(TPOT)": 0.660},
    "42165": {"No Preprocessing": 0.883, "Default Preprocessing": 0.851, "DiffPrep": 0.918, "CtxPipe": 0.911, "AutoDP": 0.861, "Tool(TPOT)": 0.884},
    "41001": {"No Preprocessing": 0.998, "Default Preprocessing": 0.834, "DiffPrep": 0.862, "CtxPipe": 0.861, "AutoDP": 0.980, "Tool(TPOT)": 0.860},
    "41671": {"No Preprocessing": 0.629, "Default Preprocessing": 0.613, "DiffPrep": 0.621, "CtxPipe": 0.226, "AutoDP": 0.598, "Tool(TPOT)": 0.584},
    "1046": {"No Preprocessing": 0.949, "Default Preprocessing": 0.953, "DiffPrep": 0.956, "CtxPipe": 0.937, "AutoDP": 0.942, "Tool(TPOT)": 0.955},
    "46597": {"No Preprocessing": 0.950, "Default Preprocessing": 0.941, "DiffPrep": 0.953, "CtxPipe": 0.908, "AutoDP": 0.936, "Tool(TPOT)": 0.930},
    "30": {"No Preprocessing": 0.968, "Default Preprocessing": 0.974, "DiffPrep": 0.976, "CtxPipe": 0.007, "AutoDP": 0.961, "Tool(TPOT)": 0.972},
    "802": {"No Preprocessing": 0.802, "Default Preprocessing": 0.789, "DiffPrep": 0.802, "CtxPipe": 0.748, "AutoDP": 0.802, "Tool(TPOT)": 0.761},
    "722": {"No Preprocessing": 0.983, "Default Preprocessing": 0.978, "DiffPrep": 0.984, "CtxPipe": 0.967, "AutoDP": 0.978, "Tool(TPOT)": 0.987},
    "40922": {"No Preprocessing": 0.991, "Default Preprocessing": 0.976, "DiffPrep": 0.990, "CtxPipe": 0.990, "AutoDP": 0.981, "Tool(TPOT)": 0.981},
    "1119": {"No Preprocessing": None, "Default Preprocessing": None, "DiffPrep": None, "CtxPipe": None, "AutoDP": 0.868, "Tool(TPOT)": None},
    "1497": {"No Preprocessing": 0.994, "Default Preprocessing": 0.995, "DiffPrep": 0.993, "CtxPipe": 0.016, "AutoDP": 0.996, "Tool(TPOT)": 0.992},
}

METHOD_TO_COLUMN = {
    "no_prep": "No Preprocessing",
    "default_prep": "Default Preprocessing",
    "ctxpipe": "CtxPipe",
    "acorec": "Tool(TPOT)",
}

# Standard ACORec Reference Flags (No downstream AutoGluon required; downstream is TPOT)
ACOREC_FLAGS = [
    "--train-metric-inline",
    "--metric-loss", "pearson",
    "--metric-weight-decay", "1e-4",
    "--metric-objective", "embedding_cosine",
    "--aco-mmas-bounds",
    "--aco-weight-method", "linear",
    "--hybrid-select",
    "--final-autogluon-topk", "1",
    "--proxy-seeds", "42,52,62",
    "--cv-select-folds", "3",
    "--use-aco",
    "--optimizer", "aco",
]


def _read_completed(out_spec: str) -> Dict[Tuple[str, str], float]:
    """Read completed (dataset_id, method) -> score mappings."""
    done = {}
    files = sorted(glob.glob(str(out_spec)))
    if not files and os.path.exists(str(out_spec)):
        files = [str(out_spec)]
    for f in files:
        p = Path(f)
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("status") == "ok" and row.get("score") is not None:
                    did = str(row.get("dataset_id"))
                    method = str(row.get("method"))
                    done[(did, method)] = float(row["score"])
            except Exception:
                continue
    return done


def _append_record(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_dataset(data_dir: str, dataset_id: str, target: str = "target"):
    candidates = [
        Path(data_dir) / f"{dataset_id}.csv",
        Path("/kaggle/working/eval_all") / f"{dataset_id}.csv",
        _REPO / "data" / "eval_datasets" / f"{dataset_id}.csv",
    ]
    for pattern in [f"/kaggle/input/**/{dataset_id}.csv"]:
        for p in glob.glob(pattern, recursive=True):
            candidates.append(Path(p))

    found_path = None
    for p in candidates:
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            found_path = p
            break

    if found_path is None:
        # Try to auto-export
        print(f"[data] Attempting auto-export for dataset {dataset_id} ...")
        dest_dir = Path(data_dir) if data_dir else Path("/kaggle/working/eval_all")
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            cmd = [sys.executable, str(_REPO / "scripts" / "export_eval_datasets.py"),
                   "--ids", str(dataset_id), "--out-dir", str(dest_dir)]
            subprocess.run(cmd, check=False)
        except Exception:
            pass
        auto_p = dest_dir / f"{dataset_id}.csv"
        if auto_p.exists() and auto_p.stat().st_size > 0:
            found_path = auto_p

    if found_path is None:
        raise FileNotFoundError(f"Dataset CSV for id={dataset_id} not found. Tried: {[str(c) for c in candidates]}")

    df = pd.read_csv(found_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in {found_path}")
    y = df[target]
    X = df.drop(columns=[target])
    problem_type, metric = _detect_problem_type(y)
    task_type = "classification" if problem_type in ("binary", "multiclass", "classification") else "regression"
    return X, y, task_type, problem_type


# -------------------------------------------------------------------------------------------------
# 1. No Preprocessing
# -------------------------------------------------------------------------------------------------
def run_no_prep(dataset_id: str, data_dir: str, time_limit_mins: int = 5, seed: int = 42, tpot_seed: int = 1) -> dict:
    X, y, task_type, problem_type = load_dataset(data_dir, dataset_id)
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, seed=seed)

    # Prune rare training classes (< 2 instances) for stratified CV (e.g. 44956 abalone)
    if task_type == "classification" and isinstance(X_train, pd.DataFrame):
        X_train, y_train, _ = _tpot_eval.prune_rare_classes(X_train.reset_index(drop=True), y_train.reset_index(drop=True), min_count=2)

    # Apply minimal adapter (median/mode impute + onehot) so TPOT can consume without internal preprocessing
    train_matrix, test_matrix, adapter_meta = _tpot_eval.apply_minimal_adapter(X_train, X_test, y_train)

    cv_folds = _tpot_eval.safe_cv_folds(y_train, task_type, 5)
    target_encoder = None
    y_train_arr = y_train.to_numpy()
    if task_type == "classification":
        target_encoder = LabelEncoder()
        y_train_arr = target_encoder.fit_transform(y_train_arr)

    model, _ = _tpot_eval.build_model(
        task_type,
        n_samples=int(train_matrix.shape[0]),
        n_features=int(train_matrix.shape[1]),
        n_classes=int(y_train.nunique()) if task_type == "classification" else 1,
        cv_folds=cv_folds,
        tpot_seed=tpot_seed,
        max_time_mins=time_limit_mins,
        max_eval_time_mins=1,
        n_jobs=2,
    )
    t0 = time.time()
    model.fit(train_matrix, y_train_arr)
    preds = model.predict(test_matrix)
    if target_encoder is not None:
        preds = target_encoder.inverse_transform(np.asarray(preds).astype(int))

    score = float(accuracy_score(y_test, preds) if task_type == "classification" else r2_score(y_test, preds))
    return {
        "dataset_id": str(dataset_id),
        "method": "no_prep",
        "column": METHOD_TO_COLUMN["no_prep"],
        "status": "ok",
        "score": score,
        "seconds": round(time.time() - t0, 1),
    }


# -------------------------------------------------------------------------------------------------
# 2. Default Preprocessing (TPOT preprocessing=True)
# -------------------------------------------------------------------------------------------------
def run_default_prep(dataset_id: str, data_dir: str, time_limit_mins: int = 5, seed: int = 42, tpot_seed: int = 1) -> dict:
    X, y, task_type, problem_type = load_dataset(data_dir, dataset_id)
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, seed=seed)

    # Prune rare training classes (< 2 instances) for stratified CV (e.g. 44956 abalone)
    if task_type == "classification" and isinstance(X_train, pd.DataFrame):
        X_train, y_train, _ = _tpot_eval.prune_rare_classes(X_train.reset_index(drop=True), y_train.reset_index(drop=True), min_count=2)

    # Basic OHE for categorical columns so TPOT preprocessors don't fail on string dtypes
    train_matrix, test_matrix, adapter_meta = _tpot_eval.apply_minimal_adapter(X_train, X_test, y_train)

    try:
        from tpot import TPOTClassifier, TPOTRegressor
    except ImportError as e:
        raise RuntimeError("TPOT 1.1.0 is required. Run: pip install -r requirements-tpot-kaggle.txt") from e

    cv_folds = _tpot_eval.safe_cv_folds(y_train, task_type, 5)
    target_encoder = None
    y_train_arr = y_train.to_numpy()
    if task_type == "classification":
        target_encoder = LabelEncoder()
        y_train_arr = target_encoder.fit_transform(y_train_arr)

    Estimator = TPOTClassifier if task_type == "classification" else TPOTRegressor
    model = Estimator(
        max_time_mins=time_limit_mins,
        max_eval_time_mins=1,
        population_size=20,
        cv=cv_folds,
        random_state=tpot_seed,
        n_jobs=2,
        memory_limit="5GB",
        early_stop=5,
        preprocessing=True,  # Default TPOT Preprocessing enabled
        verbose=2,
    )
    t0 = time.time()
    model.fit(train_matrix, y_train_arr)
    preds = model.predict(test_matrix)
    if target_encoder is not None:
        preds = target_encoder.inverse_transform(np.asarray(preds).astype(int))

    score = float(accuracy_score(y_test, preds) if task_type == "classification" else r2_score(y_test, preds))
    return {
        "dataset_id": str(dataset_id),
        "method": "default_prep",
        "column": METHOD_TO_COLUMN["default_prep"],
        "status": "ok",
        "score": score,
        "seconds": round(time.time() - t0, 1),
    }


# -------------------------------------------------------------------------------------------------
# 3. CtxPipe -> TPOT
# -------------------------------------------------------------------------------------------------
def setup_ctxpipe_models(ctxpipe_zip: Optional[str] = None, dest_dir: str = "/kaggle/working/models/ctxpipe-3linear"):
    dest = Path(dest_dir).resolve()
    if (dest / "ctx_32000_fengine_model.pkl").exists():
        return dest

    # 1. Search for existing extracted directory containing the pkl files
    search_dirs = [
        dest,
        _REPO / "external" / "ctxpipe" / "models" / "ctxpipe-3linear",
        _REPO / "models" / "ctxpipe-3linear",
        _REPO / "ctxpipe-3linear",
    ]
    for pattern in ["/kaggle/input/**/ctxpipe-3linear", "/kaggle/input/**/models/ctxpipe-3linear"]:
        for d in glob.glob(pattern, recursive=True):
            search_dirs.append(Path(d))

    for d in search_dirs:
        if d.exists() and (d / "ctx_32000_fengine_model.pkl").exists():
            print(f"[ctxpipe] Found existing model directory: {d}")
            if d.resolve() != dest:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    try:
                        shutil.copytree(str(d), str(dest))
                    except Exception:
                        return d
            return dest

    # 2. Search for zip file
    candidate_zips = []
    if ctxpipe_zip:
        candidate_zips.append(Path(ctxpipe_zip))
    candidate_zips.extend([
        _REPO / "ctxpipe-3linear.zip",
        _REPO / "external" / "ctxpipe" / "models" / "ctxpipe-3linear.zip",
    ])
    for pattern in ["/kaggle/input/**/ctxpipe-3linear*.zip", "/kaggle/working/**/ctxpipe-3linear*.zip"]:
        for z in glob.glob(pattern, recursive=True):
            candidate_zips.append(Path(z))

    for z in candidate_zips:
        if z.exists() and z.is_file():
            print(f"[ctxpipe] Extracting {z} -> {dest.parent} ...")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(dest.parent)
            if (dest / "ctx_32000_fengine_model.pkl").exists():
                return dest

    print(f"[ctxpipe] WARNING: Could not find ctxpipe-3linear models in candidate paths: {candidate_zips}")
    return dest


def run_ctxpipe_tpot(dataset_id: str, data_dir: str, ctxpipe_model_dir: str,
                     time_limit_mins: int = 5, seed: int = 42, tpot_seed: int = 1) -> dict:
    ctxpipe_root = str(_REPO / "external" / "ctxpipe")
    if ctxpipe_root not in sys.path:
        sys.path.insert(0, ctxpipe_root)

    model_dir_abs = os.path.abspath(ctxpipe_model_dir)
    os.environ["CTXPIPE_MODEL_DIR"] = model_dir_abs
    scratch_root = Path(ctxpipe_model_dir).parent
    os.environ["CTXPIPE_EXP_DIR"] = str(scratch_root / "exp")
    os.environ["CTXPIPE_LOG_DIR"] = str(scratch_root / "logs")

    import env
    env.init()
    from ctxpipe.agentman import AgentManager
    from ctxpipe.dataset import Dataset
    import comp
    from ctxpipe.env.primitives.imputercat import ImputerCatPrim
    from ctxpipe.env.primitives.primitive import Primitive

    csv_path = Path(data_dir) / f"{dataset_id}.csv"
    df = pd.read_csv(csv_path)
    label_idx = df.columns.get_loc("target")
    ds = Dataset(str(dataset_id), str(csv_path), label_idx)

    # 1. Run CtxPipe DQN inference
    agentman = AgentManager()
    t0 = time.time()
    seq_prims, ml_score = agentman.inference(ds, "ctx_32000")
    print(f"  [ctxpipe] Inferred pipeline: {seq_prims}")

    # 2. Replay primitives on seed-42 split
    X = df.drop(columns=["target"])
    y = df["target"]
    problem_type, _ = _detect_problem_type(y)
    task_type = "classification" if problem_type in ("binary", "multiclass", "classification") else "regression"

    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, seed=seed)

    # Prune rare classes before transform
    if task_type == "classification" and isinstance(X_train, pd.DataFrame):
        X_train, y_train, _ = _tpot_eval.prune_rare_classes(X_train.reset_index(drop=True), y_train.reset_index(drop=True), min_count=2)

    train_x = X_train.reset_index(drop=True).copy()
    train_y = y_train.reset_index(drop=True).copy()
    test_x = X_test.reset_index(drop=True).copy()

    for p in seq_prims:
        if isinstance(p, Primitive):
            p_inst = p.__class__()
            train_x, test_x = p_inst.transform(train_x, test_x, train_y)
            if not isinstance(train_x, pd.DataFrame):
                train_x = pd.DataFrame(train_x)
            if not isinstance(test_x, pd.DataFrame):
                test_x = pd.DataFrame(test_x)
            train_x = train_x.reset_index(drop=True)
            test_x = test_x.reset_index(drop=True)

    # 3. Fit estimator-only TPOT
    train_matrix, test_matrix, _ = _tpot_eval.to_tpot_matrix(train_x, test_x, train_y)
    cv_folds = _tpot_eval.safe_cv_folds(train_y, task_type, 5)
    target_encoder = None
    y_train_arr = train_y.to_numpy()
    if task_type == "classification":
        target_encoder = LabelEncoder()
        y_train_arr = target_encoder.fit_transform(y_train_arr)

    model, _ = _tpot_eval.build_model(
        task_type,
        n_samples=int(train_matrix.shape[0]),
        n_features=int(train_matrix.shape[1]),
        n_classes=int(train_y.nunique()) if task_type == "classification" else 1,
        cv_folds=cv_folds,
        tpot_seed=tpot_seed,
        max_time_mins=time_limit_mins,
        max_eval_time_mins=1,
        n_jobs=2,
    )
    model.fit(train_matrix, y_train_arr)
    preds = model.predict(test_matrix)
    if target_encoder is not None:
        preds = target_encoder.inverse_transform(np.asarray(preds).astype(int))

    score = float(accuracy_score(y_test, preds) if task_type == "classification" else r2_score(y_test, preds))
    return {
        "dataset_id": str(dataset_id),
        "method": "ctxpipe",
        "column": METHOD_TO_COLUMN["ctxpipe"],
        "pipeline": [str(p) for p in seq_prims],
        "status": "ok",
        "score": score,
        "seconds": round(time.time() - t0, 1),
    }


# -------------------------------------------------------------------------------------------------
# 4. ACORec -> TPOT
# -------------------------------------------------------------------------------------------------
def run_acorec_tpot(dataset_id: str, data_dir: str, workdir: Path,
                    time_limit_mins: int = 5, seed: int = 42, tpot_seed: int = 1) -> dict:
    t0 = time.time()
    workdir_abs = workdir.resolve()
    workdir_abs.mkdir(parents=True, exist_ok=True)

    # 1. Run ACORec recommendation search
    rec_out = workdir_abs / "acorec_rec"
    rec_out.mkdir(parents=True, exist_ok=True)
    rec_json = rec_out / "recommendation.json"

    if not rec_json.exists():
        cmd = [
            sys.executable, str(_REPO / "scripts" / "run_recommend.py"),
            "--dataset-source", "openml",
            "--openml-local-folder", str(Path(data_dir).resolve()),
            "--dataset-ids", str(dataset_id),
            "--kaggle-root", str(_REPO),
            "--time-limit", "120",
            "--seed", str(seed),
            "--output-dir", str(rec_out),
        ] + ACOREC_FLAGS
        proc = subprocess.run(cmd, cwd=str(workdir_abs), check=True)

    if not rec_json.exists():
        sub_rec = rec_out / f"dataset_{dataset_id}" / "recommendation.json"
        if sub_rec.exists():
            rec_json = sub_rec

    # 2. Evaluate recommended pipeline with estimator-only TPOT
    tpot_out_json = workdir_abs / "tpot_eval.json"
    cmd_eval = [
        sys.executable, str(_REPO / "scripts" / "evaluate_acorec_tpot.py"),
        "--recommendation-json", str(rec_json),
        "--dataset-csv", str((Path(data_dir) / f"{dataset_id}.csv").resolve()),
        "--dataset-id", str(dataset_id),
        "--output-json", str(tpot_out_json),
        "--split-seed", str(seed),
        "--tpot-seed", str(tpot_seed),
        "--max-time-mins", str(time_limit_mins),
        "--max-eval-time-mins", "1",
        "--n-jobs", "2",
        "--force",
    ]
    proc_eval = subprocess.run(cmd_eval, cwd=str(workdir_abs), check=True)

    data = json.loads(tpot_out_json.read_text(encoding="utf-8"))
    score = float(data["score"])
    return {
        "dataset_id": str(dataset_id),
        "method": "acorec",
        "column": METHOD_TO_COLUMN["acorec"],
        "status": "ok",
        "score": score,
        "pipeline": data.get("pipeline_config"),
        "seconds": round(time.time() - t0, 1),
    }


# -------------------------------------------------------------------------------------------------
# Summarize & Print Full Table
# -------------------------------------------------------------------------------------------------
def print_tpot_table(completed: Dict[Tuple[str, str], float], out_csv: Optional[str] = None):
    cols = ["No Preprocessing", "Default Preprocessing", "DiffPrep", "CtxPipe", "AutoDP", "Tool(TPOT)"]
    rows = []
    col_sums = {c: 0.0 for c in cols}
    col_counts = {c: 0 for c in cols}

    for num, did, ds_name in TABLE_DATASETS_30:
        row = {"No.": num, "Dataset_id": did, "Dataset": ds_name}
        base_vals = BASE_TPOT_TABLE.get(did, {})
        for col in cols:
            # Check completed
            method_key = next((m for m, c in METHOD_TO_COLUMN.items() if c == col), None)
            if method_key and (did, method_key) in completed:
                val = completed[(did, method_key)]
            elif base_vals.get(col) is not None:
                val = base_vals[col]
            else:
                val = None

            row[col] = val
            if val is not None:
                col_sums[col] += val
                col_counts[col] += 1
        rows.append(row)

    # Average row
    avg_row = {"No.": "", "Dataset_id": "", "Dataset": "Average"}
    for col in cols:
        avg_row[col] = (col_sums[col] / col_counts[col]) if col_counts[col] > 0 else None
    rows.append(avg_row)

    print("\n" + "=" * 96)
    print("TPOT PREPROCESSING ABLATIONS (30 Datasets)")
    print("=" * 96)

    header = ["No.", "Dataset_id", "Dataset"] + cols
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        vals = []
        for col in header:
            v = r.get(col)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        print("| " + " | ".join(vals) + " |")

    print("\n" + "-" * 42 + " CSV FORMAT " + "-" * 42)
    csv_lines = [",".join(header)]
    for r in rows:
        vals = []
        for col in header:
            v = r.get(col)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        csv_lines.append(",".join(vals))
    csv_text = "\n".join(csv_lines)
    print(csv_text)
    print("=" * 96)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(out_csv).write_text(csv_text, encoding="utf-8")
        print(f"\n[table] saved CSV to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_working = "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs"
    parser.add_argument("--data-dir", default="/kaggle/working/eval_all" if os.path.isdir("/kaggle/working") else "data/eval_datasets", help="Directory with <id>.csv datasets")
    parser.add_argument("--ctxpipe-zip", default=None, help="Path to ctxpipe-3linear.zip")
    parser.add_argument("--ctxpipe-model-dir", default=os.path.join(default_working, "models", "ctxpipe-3linear"), help="CtxPipe model directory")
    parser.add_argument("--methods", default=None, help="Comma-separated methods: no_prep, default_prep, ctxpipe, acorec")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset ids: e.g. 44956, 1119, 1471")
    parser.add_argument("--missing-only", action="store_true", default=True, help="Run only missing cells")
    parser.add_argument("--shard", default=None, help="Optional I/N shard (e.g., 1/2, 2/2) to split runs across notebooks")
    parser.add_argument("--time-limit-mins", type=int, default=5, help="TPOT max_time_mins per fit (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tpot-seed", type=int, default=1)
    parser.add_argument("--out", default=os.path.join(default_working, "tpot_ablations.jsonl"), help="Output JSONL")
    parser.add_argument("--out-csv", default=None, help="Output CSV for final table")
    parser.add_argument("--summarize", action="store_true", help="Print table and exit")
    parser.add_argument("--scratch-dir", default=os.path.join(default_working, "scratch_tpot"), help="Scratch directory")
    args = parser.parse_args()

    completed = _read_completed(args.out)
    if args.summarize:
        print_tpot_table(completed, out_csv=args.out_csv)
        return 0

    setup_ctxpipe_models(args.ctxpipe_zip or str(_REPO / "ctxpipe-3linear.zip"), args.ctxpipe_model_dir)

    selected_methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else ["no_prep", "default_prep", "ctxpipe", "acorec"]
    selected_ds = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets else ["44956", "1119", "1471"]

    tasks = []
    for did in selected_ds:
        base_vals = BASE_TPOT_TABLE.get(did, {})
        for method in selected_methods:
            col_name = METHOD_TO_COLUMN[method]
            is_missing = (base_vals.get(col_name) is None)
            if not args.missing_only or is_missing:
                if (did, method) not in completed:
                    tasks.append((did, method))

    if args.shard:
        part, total = (int(x) for x in args.shard.split("/"))
        tasks = [t for i, t in enumerate(tasks) if i % total == (part - 1)]

    print(f"[tpot-suite] Planned {len(tasks)} runs ({len(completed)} already in {args.out}):")
    for did, m in tasks:
        print(f"  - dataset {did} under {m} ({METHOD_TO_COLUMN[m]})")

    if not tasks:
        print("[tpot-suite] All tasks are already complete!")
        print_tpot_table(completed, out_csv=args.out_csv)
        return 0

    out_path = Path(args.out)
    scratch_dir = Path(args.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    for idx, (did, method) in enumerate(tasks, 1):
        print(f"\n[{idx}/{len(tasks)}] Running dataset {did} under method={method} ...", flush=True)
        workdir = scratch_dir / f"{did}_{method}"
        workdir.mkdir(parents=True, exist_ok=True)

        try:
            if method == "no_prep":
                res = run_no_prep(did, args.data_dir, args.time_limit_mins, args.seed, args.tpot_seed)
            elif method == "default_prep":
                res = run_default_prep(did, args.data_dir, args.time_limit_mins, args.seed, args.tpot_seed)
            elif method == "ctxpipe":
                res = run_ctxpipe_tpot(did, args.data_dir, args.ctxpipe_model_dir, args.time_limit_mins, args.seed, args.tpot_seed)
            elif method == "acorec":
                res = run_acorec_tpot(did, args.data_dir, workdir, args.time_limit_mins, args.seed, args.tpot_seed)
            else:
                raise ValueError(f"Unknown method {method}")

            _append_record(out_path, res)
            if res.get("status") == "ok":
                completed[(did, method)] = float(res["score"])
                print(f"  [ok] {did} [{method}] score={res['score']:.4f} in {res.get('seconds', 0)}s", flush=True)
            else:
                print(f"  [{res.get('status')}] {did} [{method}] in {res.get('seconds', 0)}s", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            err_rec = {"dataset_id": str(did), "method": method, "status": "error", "error": str(exc)}
            _append_record(out_path, err_rec)
            print(f"  [error] {did} [{method}]: {exc}", flush=True)

    print_tpot_table(completed, out_csv=args.out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
