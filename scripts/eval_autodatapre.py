#!/usr/bin/env python3
"""Score an AutoDP-prepared dataset with OUR downstream-AutoML protocol (stage 3).

Runs in the MAIN environment (the one with AutoGluon / H2O), reading the prepared frame stage 2
wrote in the pinned AutoDP environment.

``evaluator`` selects the downstream AutoML that produces the reported score:
  * ``autogluon`` (default) -- the protocol every number in our results tables uses.
  * ``h2o`` -- identical protocol to ``autogluon``, only the downstream model swapped. Same
    seed-42 0.6/0.2/0.2 positional split (``automl_aco.data.splits.split_train_val_test``), same
    fit-on-train+val / predict-the-20%-test, same original-target re-attachment, same
    ``score_full`` / ``score_kept`` row-coverage accounting -- all of it the shared code below; the
    only fork is ``_fit_predict_with_h2o`` in place of ``_fit_predict_with_autogluon``. H2O AutoML
    with downstream preprocessing OFF (``preprocessing=None``), ``max_runtime_secs`` = the same
    ``--time-limit`` the AutoGluon path gets, and ``max_runtime_secs_per_model=60`` / ``nfolds=5``
    (the two knobs the ACORec H2O run sets explicitly). Leftover categoricals go to H2O as native
    ``enum`` factors rather than one-hot encoded -- OHE would be downstream preprocessing.

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


# H2O AutoML knobs. Only two are set explicitly (matching the ACORec H2O run); the rest are
# H2OAutoML's own defaults -- StackedEnsemble on, sort_metric AUTO. max_runtime_secs comes from
# --time-limit, seed from --seed, preprocessing is forced off.
H2O_RUNTIME_PER_MODEL = 60
H2O_NFOLDS = 5


def _fit_predict_with_h2o(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    target: str,
    problem_type: str,
    time_limit: int,
    seed: int,
    verbose: bool,
):
    """H2O AutoML, downstream preprocessing OFF. Returns ``(preds, meta)``.

    Drop-in replacement for ``_fit_predict_with_autogluon`` -- same inputs (train frame + test
    features), same output (a prediction per test row). Object columns are converted to H2O
    ``enum`` factors rather than one-hot encoded upstream. The JVM cluster is started here and
    shut down in ``finally`` so a six-dataset shard does not accumulate H2O memory.
    """
    import h2o
    from h2o.automl import H2OAutoML

    is_classification = problem_type in ("binary", "multiclass")
    h2o.init(nthreads=-1, max_mem_size="6G")
    try:
        h2o.no_progress()
        train_df = X_train.copy()
        train_df.columns = [str(c) for c in train_df.columns]
        train_df[target] = np.asarray(y_train)
        test_df = X_test.copy()
        test_df.columns = [str(c) for c in test_df.columns]

        htrain = h2o.H2OFrame(train_df)
        htest = h2o.H2OFrame(test_df)
        x_cols = [c for c in htrain.columns if c != target]
        if is_classification:
            htrain[target] = htrain[target].asfactor()

        aml = H2OAutoML(
            max_runtime_secs=int(time_limit),
            max_runtime_secs_per_model=H2O_RUNTIME_PER_MODEL,
            nfolds=H2O_NFOLDS,
            seed=int(seed),
            preprocessing=None,
            verbosity=("info" if verbose else None),
        )
        aml.train(x=x_cols, y=target, training_frame=htrain)
        pred_df = aml.leader.predict(htest).as_data_frame(use_multi_thread=True)
        preds = pred_df["predict"].to_numpy()
        if is_classification:
            # H2O returns factor levels as strings; re-cast to the original label dtype so the
            # accuracy comparison is not "0" != 0 on every row.
            try:
                preds = preds.astype(np.asarray(y_train).dtype)
            except (ValueError, TypeError):
                pass
        meta = {
            "h2o_version": h2o.__version__,
            "h2o_leader": str(getattr(aml.leader, "model_id", None)),
            "h2o_n_models": int(aml.leaderboard.nrows),
            "max_runtime_secs": int(time_limit),
            "max_runtime_secs_per_model": H2O_RUNTIME_PER_MODEL,
            "nfolds": H2O_NFOLDS,
        }
        return preds, meta
    finally:
        try:
            h2o.cluster().shutdown()
        except Exception:
            pass


def score_prepared(
    dataset_csv: str,
    prepared_dir: str,
    *,
    target: str = "target",
    time_limit: int = 300,
    autogluon_profile: str = "best_quality",
    seed: int = 42,
    verbose: bool = False,
    evaluator: str = "autogluon",
) -> dict:
    """Score one AutoDP-prepared dataset with our downstream-AutoML protocol. Returns the result dict.

    Shared by this CLI and by scripts/adp_bench.py, so both produce identical numbers.

    ``evaluator`` is ``autogluon`` (default) or ``h2o``. Only the downstream AutoML differs; the
    split, the fit/predict convention, the target re-attachment and the score_full/score_kept
    accounting are identical for both.
    """
    if evaluator not in ("autogluon", "h2o"):
        raise ValueError(f"evaluator must be 'autogluon' or 'h2o', got {evaluator!r}")
    with open(os.path.join(prepared_dir, "autodp_meta.json")) as f:
        adp_meta = json.load(f)

    orig = pd.read_csv(dataset_csv)
    y_orig = orig[target]
    n_rows = len(orig)
    tr, val, te = _split_positions(n_rows, seed=seed)
    trainval = np.concatenate([tr, val])
    test_set = set(te.tolist())

    prepared = pd.read_csv(os.path.join(prepared_dir, "prepared.csv"))
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

    problem_type, eval_metric = _detect_problem_type(y_orig)
    evaluator_meta: dict = {}

    if evaluator == "autogluon":
        # AutoGluon under IdentityFeatureGenerator cannot consume object columns, so any residual
        # categoricals AutoDP left in place are one-hot encoded (fit on train only).
        X_train, X_test, residual_cols = _encode_residual_objects(X_train, X_test)
        train_df = X_train.copy()
        train_df[target] = y_train

        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import IdentityFeatureGenerator

        ag_kwargs = dict(
            TabularPredictor=TabularPredictor, IdentityFeatureGenerator=IdentityFeatureGenerator,
            train_df=train_df, test_df=X_test, target_column=target, problem_type=problem_type,
            eval_metric=eval_metric, time_limit_per_model=int(time_limit),
            verbosity=2 if verbose else 0, autogluon_profile=autogluon_profile,
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
    else:
        # H2O consumes object columns natively as enum factors -- no upstream OHE (that would be
        # downstream preprocessing, which this evaluator deliberately leaves off).
        residual_cols = []
        t0 = time.time()
        preds, evaluator_meta = _fit_predict_with_h2o(
            X_train=X_train, y_train=y_train, X_test=X_test, target=target,
            problem_type=problem_type, time_limit=int(time_limit), seed=seed, verbose=verbose,
        )
        eval_seconds = time.time() - t0

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

    return {
        "dataset_id": os.path.splitext(os.path.basename(dataset_csv))[0],
        "method": "autodatapre-0.1.12",
        "mode": mode,
        "autodp_status": adp_meta.get("status"),
        "autodp_pipeline": adp_meta.get("pipeline"),
        "autodp_search_seconds": adp_meta.get("search_seconds"),
        "autodp_converged": adp_meta.get("converged_default_budget"),
        "autodp_hit_cap": bool(adp_meta.get("hit_wall_clock_cap", False)),
        # Protocol fields from scripts/autodp_protocol.py, so a reported number can be traced to
        # what produced it. search_iteration_exceptions in particular: their MCTS loop swallows
        # every exception in a bare `except:`, so a search that failed on EVERY iteration still
        # "converges" and reports a pipeline that was never actually evaluated -- most likely on
        # small frames, where a classifier's internal k_folds guard returns None on a batch shrunk
        # by Is_BatchTraining, and drop_unpromising's sort() cannot compare None to float. A row
        # with this near its iteration count is not a real preference and should be flagged, not
        # averaged in as if the search had succeeded.
        "search_split": adp_meta.get("search_split"),
        "metafeature_frame": adp_meta.get("metafeature_frame"),
        "internal_scorer_seed": adp_meta.get("internal_scorer_seed"),
        "leakfree_cbe": adp_meta.get("leakfree_cbe"),
        "search_iteration_exceptions": adp_meta.get("search_iteration_exceptions"),
        "search_iteration_exception_kinds": adp_meta.get("search_iteration_exception_kinds"),
        "evaluator": evaluator,
        "evaluator_meta": evaluator_meta,
        # Kept under the historical key for back-compat (adp_bench.import_dir reads it); it is the
        # downstream-AutoML wall time regardless of which evaluator ran.
        "autogluon_eval_seconds": round(eval_seconds, 2),
        "eval_seconds": round(eval_seconds, 2),
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
            "evaluator": evaluator,
            "autogluon_profile": autogluon_profile if evaluator == "autogluon" else None,
            "h2o": ({"preprocessing": None, "max_runtime_secs": int(time_limit),
                     "max_runtime_secs_per_model": H2O_RUNTIME_PER_MODEL, "nfolds": H2O_NFOLDS,
                     "seed": int(seed),
                     "defaults": "StackedEnsemble/sort_metric H2OAutoML defaults"}
                    if evaluator == "h2o" else None),
            "time_limit": int(time_limit),
            "target": "ORIGINAL y re-attached by row position",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-csv", required=True, help="the exported <id>.csv (source of the original target)")
    ap.add_argument("--prepared-dir", required=True, help="stage-2 output dir holding prepared.csv")
    ap.add_argument("--target", default="target")
    ap.add_argument("--time-limit", type=int, default=300, help="AutoGluon time_limit, as in run_recommend.py")
    ap.add_argument("--autogluon-profile", choices=["best_quality", "medium_quality", "local_rf_xt"],
                    default="best_quality")
    ap.add_argument("--evaluator", choices=["autogluon", "h2o"], default="autogluon",
                    help="downstream AutoML that produces the score. h2o = H2O AutoML at its "
                         "defaults, downstream preprocessing off, max_runtime_secs = --time-limit.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="default: <prepared-dir>/autodp_eval.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.prepared_dir, "autodp_eval.json")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] already scored -> {out_path}")
        return

    result = score_prepared(
        args.dataset_csv, args.prepared_dir, target=args.target, time_limit=args.time_limit,
        autogluon_profile=args.autogluon_profile, seed=args.seed, verbose=args.verbose,
        evaluator=args.evaluator,
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[ok  ] {result['dataset_id']} ({result['mode']}) [{result['evaluator']}] "
          f"{result['eval_metric']}: "
          f"full={result['score_full']:.4f} kept={result['score_kept']:.4f} "
          f"coverage={result['test_coverage']:.3f} search={result['autodp_search_seconds']}s "
          f"eval={result['eval_seconds']}s -> {out_path}")


if __name__ == "__main__":
    main()
