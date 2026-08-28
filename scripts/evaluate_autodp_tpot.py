#!/usr/bin/env python3
"""Score an AutoDP-prepared dataset with estimator-only TPOT (the TPOT arm of stage 3).

Sibling of ``scripts/eval_autodatapre.py``: same seed-42 0.6/0.2/0.2 positional split, same
fit-on-train+val / predict-the-20%-test convention, same ORIGINAL-target re-attachment by row
position, same ``score_full`` / ``score_kept`` row-coverage accounting -- all of it imported from
``eval_autodatapre`` so the numbers line up with the AutoGluon and H2O arms. The only fork is the
downstream model: estimator-only TPOT instead of AutoGluon / H2O.

RUNS IN THE TPOT ENVIRONMENT (``requirements-tpot-kaggle.txt``: numpy>=1.25, scikit-learn<1.8,
TPOT==1.1.0), never in the AutoGluon/H2O env -- their dependency pins are incompatible. This is why
it reads a PREPARED frame that ``scripts/run_autodatapre.py`` already wrote in the pinned
``.venv-autodp`` env, rather than running the AutoDP search itself.

The TPOT settings are shared with ``scripts/evaluate_acorec_tpot.py`` via ``scripts/_tpot_eval.py``
so the ACORec-vs-AutoDP-under-TPOT comparison is apples-to-apples.

Two differences from the AutoGluon/H2O arms, both because TPOT with ``preprocessing=False`` hands
the frame straight to sklearn estimators:

  * Residual object columns AutoDP left in place (it often selects ``enc_null``) are one-hot
    encoded, fit on train only -- the SAME ``_encode_residual_objects`` the AutoGluon arm uses,
    recorded in ``residual_encoding_applied``.
  * A frame with NaN in it (AutoDP selected ``imp_null``) is a hard failure: imputing here would
    be downstream preprocessing, which this protocol forbids. The row is written with
    ``status: "failed"`` and an ``error`` string, and must be excluded from arm means -- the same
    discipline the ``dead_search`` rows already need.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
sys.path.insert(0, _HERE)

from automl_aco.search.evaluation import _detect_problem_type  # noqa: E402

import _tpot_eval  # noqa: E402
from _tpot_eval import (  # noqa: E402
    TPOT_MAX_CV_FOLDS,
    TPOT_MAX_EVAL_TIME_MINS,
    TPOT_MAX_TIME_MINS,
    TPOT_MEMORY_LIMIT,
    TPOT_N_JOBS,
    TPOT_POPULATION_SIZE,
    TPOT_RANDOM_STATE,
    TPOT_SPLIT_SEED,
    apply_minimal_adapter,
    knob_summary,
    normalize_task_type,
    prune_rare_classes,
    safe_cv_folds,
)
from eval_autodatapre import _split_positions  # noqa: E402


def _csv_fingerprint(csv_path: str, frame: pd.DataFrame, target: str) -> Dict[str, Any]:
    tgt = frame[target].to_numpy()
    digest = hashlib.sha1(np.ascontiguousarray(tgt.astype("U")).tobytes()).hexdigest()[:16]
    return {
        "path": os.path.abspath(csv_path),
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "target_sha1_16": digest,
    }


def score_prepared_tpot(
    dataset_csv: str,
    prepared_dir: str,
    *,
    target: str = "target",
    split_seed: int = TPOT_SPLIT_SEED,
    tpot_seed: int = TPOT_RANDOM_STATE,
    max_time_mins: int = TPOT_MAX_TIME_MINS,
    max_eval_time_mins: int = TPOT_MAX_EVAL_TIME_MINS,
    n_jobs: int = TPOT_N_JOBS,
    memory_limit: str = TPOT_MEMORY_LIMIT,
    population_size: int = TPOT_POPULATION_SIZE,
    max_cv_folds: int = TPOT_MAX_CV_FOLDS,
    verbose: int = 2,
    estimator_factory=None,
    search_space_factory=None,
) -> Tuple[Dict[str, Any], Any]:
    """Score one AutoDP-prepared dataset with estimator-only TPOT.

    Returns ``(result_dict, fitted_model)`` -- the model is handed back so ``main()`` can drop it
    and force a GC before the process exits. The split / target re-attach / coverage accounting is
    identical to ``eval_autodatapre.score_prepared``; only the downstream model differs.
    """
    with open(os.path.join(prepared_dir, "autodp_meta.json")) as f:
        adp_meta = json.load(f)

    orig = pd.read_csv(dataset_csv)
    if target not in orig.columns:
        raise ValueError(f"{dataset_csv} has no {target!r} column")
    y_orig = orig[target]
    n_rows = len(orig)
    tr, val, te = _split_positions(n_rows, seed=split_seed)
    trainval = np.concatenate([tr, val])
    test_set = set(te.tolist())

    prepared = pd.read_csv(os.path.join(prepared_dir, "prepared.csv"))
    rows = prepared["__adp_row__"].to_numpy()
    mode = adp_meta.get("mode", "native")

    if mode == "fair":
        is_test = (prepared["__adp_split__"] == "test").to_numpy()
        bad_train = set(rows[~is_test].tolist()) & test_set
        bad_test = set(rows[is_test].tolist()) - test_set
        if bad_train or bad_test:
            raise AssertionError(
                f"LEAKAGE in prepared frame: {len(bad_train)} test row(s) inside the train split, "
                f"{len(bad_test)} train row(s) inside the test split")
        train_mask, test_mask = ~is_test, is_test
    else:
        train_mask = np.isin(rows, trainval)
        test_mask = np.isin(rows, te)

    feat_cols = [c for c in prepared.columns if not c.startswith("__adp_")]
    if not feat_cols:
        raise RuntimeError("AutoDP produced a frame with zero feature columns")

    X_train = prepared.loc[train_mask, feat_cols].reset_index(drop=True)
    X_test = prepared.loc[test_mask, feat_cols].reset_index(drop=True)
    y_train = y_orig.iloc[rows[train_mask]].reset_index(drop=True)
    y_test_kept = y_orig.iloc[rows[test_mask]].reset_index(drop=True)
    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError(f"empty split after AutoDP preparation (train={len(X_train)}, test={len(X_test)})")

    problem_type, eval_metric = _detect_problem_type(y_orig)
    task_type = normalize_task_type(problem_type)

    # AutoDP's row-dropping operators (IQR/LOF/DROP) can push a training class below the CV fold
    # floor; those rows are unusable for stratified CV. Drop them (test set untouched) and record.
    dropped_rare_classes: list = []
    if task_type == "classification":
        X_train, y_train, dropped_rare_classes = prune_rare_classes(X_train, y_train, min_count=2)
        if len(X_train) == 0:
            raise RuntimeError("no training rows left after dropping rare classes")

    # The No-Preprocessing baseline's train-fitted compat adapter: median/mode impute + one-hot.
    # AutoDP frequently selects no preprocessing at all (pipeline == ['<classifier>']), leaving the
    # raw frame -- NaN and object columns TPOT (preprocessing=False) cannot consume. This is the
    # SAME adapter that baseline is scored through; AutoDP's own operators are still the only
    # preprocessing under test.
    train_matrix, test_matrix, adapter_meta = apply_minimal_adapter(X_train, X_test, y_train)

    cv_folds = safe_cv_folds(y_train, task_type, max_cv_folds)
    target_encoder = None
    y_train_for_tpot = np.asarray(y_train)
    if task_type == "classification":
        target_encoder = LabelEncoder()
        y_train_for_tpot = target_encoder.fit_transform(y_train_for_tpot)

    n_classes = int(pd.Series(y_train).nunique()) if task_type == "classification" else 1
    model, group = _tpot_eval.build_model(
        task_type,
        n_samples=int(train_matrix.shape[0]),
        n_features=int(train_matrix.shape[1]),
        n_classes=n_classes,
        cv_folds=cv_folds,
        tpot_seed=int(tpot_seed),
        max_time_mins=int(max_time_mins),
        max_eval_time_mins=int(max_eval_time_mins),
        n_jobs=int(n_jobs),
        memory_limit=str(memory_limit),
        population_size=int(population_size),
        verbose=int(verbose),
        estimator_factory=estimator_factory,
        search_space_factory=search_space_factory,
    )

    t0 = time.time()
    model.fit(train_matrix, y_train_for_tpot)
    preds = model.predict(test_matrix)
    if target_encoder is not None:
        preds = target_encoder.inverse_transform(np.asarray(preds).astype(int))
    eval_seconds = time.time() - t0

    preds = pd.Series(np.asarray(preds)).reset_index(drop=True)
    coverage = len(X_test) / len(te)
    if problem_type == "regression":
        score_kept = float(r2_score(y_test_kept, preds))
        full_pred = pd.Series(float(np.mean(y_train)), index=range(len(te)))
        kept_pos = {int(r): i for i, r in enumerate(rows[test_mask])}
        for i, row_id in enumerate(te):
            if int(row_id) in kept_pos:
                full_pred.iloc[i] = float(preds.iloc[kept_pos[int(row_id)]])
        score_full = float(r2_score(y_orig.iloc[te].reset_index(drop=True), full_pred))
    else:
        score_kept = float(accuracy_score(y_test_kept, preds))
        n_correct = int((np.asarray(y_test_kept) == np.asarray(preds)).sum())
        score_full = n_correct / len(te)  # rows AutoDP dropped count as wrong

    result: Dict[str, Any] = {
        "status": "ok",
        "dataset_id": os.path.splitext(os.path.basename(dataset_csv))[0],
        "method": "autodatapre-0.1.12",
        "evaluator": "tpot",
        "tpot_estimator": type(model).__name__,
        "mode": mode,
        "autodp_status": adp_meta.get("status"),
        "autodp_pipeline": adp_meta.get("pipeline"),
        "autodp_search_seconds": adp_meta.get("search_seconds"),
        "autodp_converged": adp_meta.get("converged_default_budget"),
        "autodp_hit_cap": bool(adp_meta.get("hit_wall_clock_cap", False)),
        "search_split": adp_meta.get("search_split"),
        "metafeature_frame": adp_meta.get("metafeature_frame"),
        "internal_scorer_seed": adp_meta.get("internal_scorer_seed"),
        "leakfree_cbe": adp_meta.get("leakfree_cbe"),
        "search_iteration_exceptions": adp_meta.get("search_iteration_exceptions"),
        "search_iteration_exception_kinds": adp_meta.get("search_iteration_exception_kinds"),
        # Carried straight through so the same filters that clean the H2O table clean this one:
        # a dead_search row is the raw frame, not an AutoDP preference, and must not be averaged in.
        "dead_search": bool(adp_meta.get("dead_search", False)),
        "dead_search_none_profit_evals": adp_meta.get("dead_search_none_profit_evals"),
        "task_type": task_type,
        "problem_type": problem_type,
        "eval_metric": eval_metric,
        "primary_metric": "accuracy" if task_type == "classification" else "r2",
        # score_full is the number directly comparable to ACORec+TPOT (ACORec never drops test rows).
        "score": score_full,
        "score_full": score_full,
        "score_kept": score_kept,
        "accuracy": score_full if task_type == "classification" else None,
        "r2": score_full if task_type == "regression" else None,
        "test_coverage": coverage,
        "n_test_rows_expected": int(len(te)),
        "n_test_rows_kept": int(len(X_test)),
        "n_train_rows": int(len(X_train)),
        "n_features_scored": int(train_matrix.shape[1]),
        "compat_adapter": adapter_meta,
        "dropped_rare_class_train_rows": [str(c) for c in dropped_rare_classes],
        "target_label_encoding": (
            "LabelEncoder_fit_on_train_inverse_before_scoring"
            if target_encoder is not None else "not_applicable"
        ),
        "tpot_space": group,
        "tpot_preprocessing": False,
        "cv_folds": int(cv_folds),
        "split_seed": int(split_seed),
        "tpot_seed": int(tpot_seed),
        "eval_seconds": round(eval_seconds, 2),
        # kept under the historical key so adp_bench.import_dir / run_arms read it back
        "autogluon_eval_seconds": round(eval_seconds, 2),
        "total_seconds": round(float(adp_meta.get("search_seconds") or 0.0) + eval_seconds, 2),
        "tpot_knobs": knob_summary(),
        "selected_estimator": str(
            getattr(model, "fitted_pipeline_", getattr(model, "fitted_pipeline", ""))
        ),
        "protocol": {
            "split": "seed-42 0.6/0.2/0.2 via automl_aco.data.splits.split_train_val_test",
            "fit": "train+val (80%), predict the 20% test split",
            "evaluator": "tpot",
            "tpot": knob_summary(),
            "target": "ORIGINAL y re-attached by row position",
        },
    }
    return result, model


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-csv", required=True,
                    help="exported <id>.csv (source of the original target) -- the SAME file the "
                         "ACORec arm reads")
    ap.add_argument("--prepared-dir", required=True,
                    help="a run_autodatapre.py output dir holding prepared.csv + autodp_meta.json")
    ap.add_argument("--target", default="target")
    ap.add_argument("--output-json", default=None, help="default: <prepared-dir>/tpot_evaluation.json")
    ap.add_argument("--dataset-id", default=None, help="label for the output row; default: CSV basename")
    ap.add_argument("--split-seed", type=int, default=TPOT_SPLIT_SEED)
    ap.add_argument("--tpot-seed", type=int, default=TPOT_RANDOM_STATE)
    ap.add_argument("--max-time-mins", type=int, default=TPOT_MAX_TIME_MINS)
    ap.add_argument("--max-eval-time-mins", type=int, default=TPOT_MAX_EVAL_TIME_MINS)
    ap.add_argument("--n-jobs", type=int, default=TPOT_N_JOBS)
    ap.add_argument("--memory-limit", default=TPOT_MEMORY_LIMIT)
    ap.add_argument("--population-size", type=int, default=TPOT_POPULATION_SIZE)
    ap.add_argument("--max-cv-folds", type=int, default=TPOT_MAX_CV_FOLDS)
    ap.add_argument("--verbose", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    dataset_id = args.dataset_id or os.path.splitext(os.path.basename(str(args.dataset_csv)))[0]
    out_path = args.output_json or os.path.join(args.prepared_dir, "tpot_evaluation.json")
    if os.path.exists(out_path) and not args.force:
        existing = json.loads(open(out_path).read())
        if existing.get("status") == "ok":
            print(f"SKIP successful TPOT evaluation: {out_path}")
            return 0

    fingerprint = None
    try:
        frame = pd.read_csv(args.dataset_csv)
        fingerprint = _csv_fingerprint(str(args.dataset_csv), frame, args.target)
    except Exception:
        pass

    model = None
    try:
        result, model = score_prepared_tpot(
            args.dataset_csv, args.prepared_dir, target=args.target,
            split_seed=args.split_seed, tpot_seed=args.tpot_seed,
            max_time_mins=args.max_time_mins, max_eval_time_mins=args.max_eval_time_mins,
            n_jobs=args.n_jobs, memory_limit=args.memory_limit,
            population_size=args.population_size, max_cv_folds=args.max_cv_folds,
            verbose=args.verbose,
        )
        result["dataset_id"] = str(dataset_id)
        result["dataset_csv"] = fingerprint
    except Exception as exc:
        result = {
            "status": "failed",
            "dataset_id": str(dataset_id),
            "method": "autodatapre-0.1.12",
            "evaluator": "tpot",
            "dataset_csv": fingerprint,
            "prepared_dir": os.path.abspath(args.prepared_dir),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[fail] {dataset_id}: {type(exc).__name__}: {exc} -> {out_path}")
        raise
    finally:
        del model
        gc.collect()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(
        f"[ok  ] {result['dataset_id']} ({result['mode']}) [tpot] {result['eval_metric']}: "
        f"full={result['score_full']:.4f} kept={result['score_kept']:.4f} "
        f"coverage={result['test_coverage']:.3f} search={result['autodp_search_seconds']}s "
        f"eval={result['eval_seconds']}s -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
