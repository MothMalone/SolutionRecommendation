#!/usr/bin/env python3
"""Score an AutoDP-prepared dataset with OUR AutoGluon protocol (stage 3).

Runs in the MAIN environment (the one with AutoGluon), reading the prepared frame stage 2 wrote in
the pinned AutoDP environment.

The protocol is the one every number in our results tables uses, reusing the same code paths:
  * ``split_train_val_test`` -> seed-42 0.6/0.2/0.2 split, taken positionally over the exported row
    order, so AutoDP is scored on exactly the rows our own pipelines are scored on;
  * fit on train+val (80%), predict the 20% test split -- the ``fit_include_val=True`` protocol;
  * ``_fit_predict_with_autogluon`` -> the same TabularPredictor settings, presets, DyStack
    handling, IdentityFeatureGenerator and XGB-retry fallback;
  * ``_detect_problem_type`` -> the same problem type and metric (accuracy / R2), derived from the
    ORIGINAL target.

Two deliberate adaptations, both recorded in the output JSON:

1. TARGET. AutoDP's ``read_dataset`` LabelEncodes the target, which destroys continuous targets.
   We ignore whatever target column its output carries and re-attach the ORIGINAL y by row
   position. Its preprocessing of the FEATURES is what is under test.

2. ROW COVERAGE. Several AutoDP operators delete rows from the TEST split (``DROP`` imputation,
   ``ZSB``/``IQR``/``LOF`` outlier removal, ``ED``/``AD`` dedup). Our own evaluator forbids that
   (``evaluate_candidates_autogluon`` discards any candidate whose transform changes the test row
   count). Rather than silently score AutoDP on a smaller, easier test set, we report both:
     score_kept -- metric over the test rows AutoDP retained. Generous to AutoDP, and the
                   convention its own internal evaluation uses (its classifiers ``dropna`` the test
                   split before scoring). Undefined-vs-ours when coverage < 100%.
     score_full -- metric over ALL of our test rows. Rows AutoDP dropped still need a prediction:
                   classification counts them wrong, regression falls back to the training-target
                   mean (the best constant predictor). This is the number directly comparable to
                   ours.
   With coverage == 1.0 the two are identical, and the comparison is exact.

Residual encoding: if AutoDP's chosen pipeline leaves non-numeric columns (it often selects
``enc_null``), they are one-hot encoded with the SAME ``OneHotEncoder(handle_unknown="ignore")``
our ``no_preprocessing`` baseline uses, fit on the training rows only. Without it AutoGluon under
IdentityFeatureGenerator cannot consume the frame at all; applying it to the baseline as well as to
AutoDP keeps the floor identical. ``residual_encoding_applied`` records when this fired.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder

from automl_aco.data.splits import split_train_val_test
from automl_aco.search.evaluation import _detect_problem_type, _fit_predict_with_autogluon


def _split_positions(n_rows: int, seed: int = 42):
    idx = pd.DataFrame({"_pos": np.arange(n_rows)})
    dummy = pd.Series(np.zeros(n_rows))
    x_tr, _, x_val, _, x_te, _ = split_train_val_test(idx, dummy, seed=seed)
    return (x_tr["_pos"].to_numpy(), x_val["_pos"].to_numpy(), x_te["_pos"].to_numpy())


def _encode_residual_objects(train: pd.DataFrame, test: pd.DataFrame):
    """One-hot any leftover categorical columns, fit on train only. Returns (train, test, applied)."""
    obj_cols = list(train.select_dtypes(include=["object", "category", "bool"]).columns)
    if not obj_cols:
        return train, test, []
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    tr_mat = enc.fit_transform(train[obj_cols].astype(str))
    te_mat = enc.transform(test[obj_cols].astype(str))
    names = [f"{c}__{v}" for c, vals in zip(obj_cols, enc.categories_) for v in vals]
    tr_out = pd.concat([train.drop(columns=obj_cols).reset_index(drop=True),
                        pd.DataFrame(tr_mat, columns=names)], axis=1)
    te_out = pd.concat([test.drop(columns=obj_cols).reset_index(drop=True),
                        pd.DataFrame(te_mat, columns=names)], axis=1)
    return tr_out, te_out, obj_cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-csv", required=True, help="the exported <id>.csv (source of the original target)")
    ap.add_argument("--prepared-dir", required=True, help="stage-2 output dir holding prepared.csv")
    ap.add_argument("--target", default="target")
    ap.add_argument("--time-limit", type=int, default=300, help="AutoGluon time_limit, as in run_recommend.py")
    ap.add_argument("--autogluon-profile", choices=["best_quality", "medium_quality", "local_rf_xt"],
                    default="best_quality")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="default: <prepared-dir>/autodp_eval.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.prepared_dir, "autodp_eval.json")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] already scored -> {out_path}")
        return

    with open(os.path.join(args.prepared_dir, "autodp_meta.json")) as f:
        adp_meta = json.load(f)

    orig = pd.read_csv(args.dataset_csv)
    y_orig = orig[args.target]
    n_rows = len(orig)
    tr, val, te = _split_positions(n_rows, seed=args.seed)
    trainval = np.concatenate([tr, val])
    trainval_set, test_set = set(trainval.tolist()), set(te.tolist())

    prepared = pd.read_csv(os.path.join(args.prepared_dir, "prepared.csv"))
    rows = prepared["__adp_row__"].to_numpy()
    mode = adp_meta.get("mode", "native")

    if mode == "fair":
        # Stage 2 already partitioned by our split; verify AutoDP's row deletions did not smuggle a
        # test row into the training frame (or vice versa) before anything is fit.
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

    X_train, X_test, residual_cols = _encode_residual_objects(X_train, X_test)

    problem_type, eval_metric = _detect_problem_type(y_orig)
    train_df = X_train.copy()
    train_df[args.target] = y_train

    from autogluon.tabular import TabularPredictor
    from autogluon.features.generators import IdentityFeatureGenerator

    ag_kwargs = dict(
        TabularPredictor=TabularPredictor, IdentityFeatureGenerator=IdentityFeatureGenerator,
        train_df=train_df, test_df=X_test, target_column=args.target, problem_type=problem_type,
        eval_metric=eval_metric, time_limit_per_model=int(args.time_limit),
        verbosity=2 if args.verbose else 0, autogluon_profile=args.autogluon_profile,
    )
    t0 = time.time()
    try:
        preds, _sel, issue = _fit_predict_with_autogluon(**ag_kwargs)
    except Exception as exc:
        if "xgbclassifier" in str(exc).lower() and "n_classes_" in str(exc).lower():
            print("  ! XGBoost compatibility issue, retrying without XGB models")
            preds, _sel, issue = _fit_predict_with_autogluon(excluded_model_types=["XGB"], **ag_kwargs)
        else:
            raise
    eval_seconds = time.time() - t0
    if issue == "no_models_fitted":
        raise RuntimeError("AutoGluon fitted no models on the AutoDP-prepared data")

    preds = pd.Series(np.asarray(preds)).reset_index(drop=True)
    coverage = len(X_test) / len(te)
    if problem_type == "regression":
        score_kept = float(r2_score(y_test_kept, preds))
        # Dropped rows still need a prediction: fall back to the best constant predictor.
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

    result = {
        "dataset_id": os.path.splitext(os.path.basename(args.dataset_csv))[0],
        "method": "autodatapre-0.1.12",
        "mode": mode,
        "autodp_status": adp_meta.get("status"),
        "autodp_pipeline": adp_meta.get("pipeline"),
        "autodp_search_seconds": adp_meta.get("search_seconds"),
        "autodp_converged": adp_meta.get("converged_default_budget"),
        "autodp_hit_cap": bool(adp_meta.get("hit_wall_clock_cap", False)),
        "autogluon_eval_seconds": round(eval_seconds, 2),
        "total_seconds": round(float(adp_meta.get("search_seconds") or 0.0) + eval_seconds, 2),
        "problem_type": problem_type,
        "eval_metric": eval_metric,
        "score_full": score_full,
        "score_kept": score_kept,
        "test_coverage": coverage,
        "n_test_rows_expected": int(len(te)),
        "n_test_rows_kept": int(len(X_test)),
        "n_train_rows": int(len(X_train)),
        "n_features_scored": int(X_train.shape[1]),
        "residual_encoding_applied": residual_cols,
        "protocol": {
            "split": "seed-42 0.6/0.2/0.2 via automl_aco.data.splits.split_train_val_test",
            "fit": "train+val (80%), predict the 20% test split",
            "autogluon_profile": args.autogluon_profile,
            "time_limit": int(args.time_limit),
            "target": "ORIGINAL y re-attached by row position",
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[ok  ] {result['dataset_id']} ({mode}) {eval_metric}: full={score_full:.4f} "
          f"kept={score_kept:.4f} coverage={coverage:.3f} "
          f"search={adp_meta.get('search_seconds')}s eval={eval_seconds:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
