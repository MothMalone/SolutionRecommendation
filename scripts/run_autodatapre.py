#!/usr/bin/env python3
"""Run the AutoDP baseline (pypi ``autodatapre==0.1.12``) on an exported evaluation dataset.

RUNS IN THE PINNED ``.venv-autodp`` ENVIRONMENT (python 3.10, numpy 1.23, pandas 1.5, sklearn
1.1), never in the main env: AutoDP's code uses ``np.bool``, ``DataFrame.append`` and positional
``drop(..., 1)``, all removed in the versions AutoGluon needs. See requirements-autodp.txt.

This stage only produces a PREPARED DATASET. Scoring is stage 3 (scripts/eval_autodatapre.py),
which runs the identical AutoGluon protocol our own pipelines are scored with.

AutoDP's method is left intact: the MCTS search, its pretrained meta-learner that picks the
search-space ordering from its own Metafeature.csv neighbours, and its internal NB/LDA/RF scoring
signal all run exactly as published. We only take its OUTPUT (the prepared frame) and score that
with AutoGluon, which is the same thing we do to our own pipelines.

Two protocols, because ``AutoDP.Classifier(df, ...)`` has no fit/transform split:

  fair    THE REPORTED PATH, and the default. Their MCTS scores candidate pipelines on OUR seed-42
          0.6 train / 0.2 val, and their evaluation layer is moved onto ours by
          ``scripts/autodp_protocol.py`` -- see that module for the full list and for the line
          between what is moved and what stays theirs. The winning chain is then applied to
          {train: our 80%, test: our 20%}. Our test rows reach neither the search's scoring dict
          nor its family-order prior. This is the protocol our own method is held to.
  native  NOT REPORTED. A literal reproduction of the published API, deliberately left unpatched:
          the MCTS searches over the FULL dataset and its internal scorer holds out its own
          UNSEEDED random 20%, so our test rows are visible to the search. Kept only so the
          deviation is inspectable.

Faithfulness notes (each one is a deliberate, documented deviation):
  * ``AutoDP.Classifier`` does not return the winning pipeline, so native mode replicates its body
    (``read_dataset`` -> ``CLA_*`` -> apply -> ``merge_datasets`` -> ``dropna``) in order to record
    it. The one omission is the final ``getAcc`` classifier refit, which does not mutate the
    prepared frame and only burns time.
  * AutoDP's operators transform train and test INDEPENDENTLY (e.g. ``ZS`` z-scores test with
    test's own mean/std) and several DELETE ROWS from the test split (``DROP``, ``ZSB``/``IQR``/
    ``LOF``, ``ED``/``AD``). Both behaviours are theirs and are preserved; stage 3 reports test
    coverage so the effect is visible rather than silently inflating their score. The ONE
    exception is ``CBE``, which consumed the labels of the rows it was encoding -- that is
    protocol, not operator semantics, and ``autodp_protocol.install_leakfree_cbe`` refits it on
    the train block only.
  * ``read_dataset`` runs LabelEncoder on the target, which destroys continuous targets on
    regression datasets. We keep that inside the search (it is their signal) but stage 3 re-attaches
    the ORIGINAL y by row index, so the reported R2 means what it says.
  * Their public entry points wrap everything in a bare ``except:`` and silently return the RAW
    frame on failure. We detect that instead of hiding it: ``status`` records whether the prepared
    frame is genuinely prepared, and stage 3 still scores whatever their method returned.

Output per dataset, under ``<out-dir>/<mode>/dataset_<id>/``:
    prepared.csv    features + ``__adp_row__`` (original row position) + ``__adp_split__``
    autodp_meta.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")  # AutoDP draws flowcharts/progress bars unconditionally; keep it headless.

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

# Reused verbatim so the row split is bit-identical to the one our own evaluation uses.
from automl_aco.data.splits import split_train_val_test  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autodp_protocol  # noqa: E402  (moves their evaluation layer onto ours)

HELPER_COLS = ("New_ID", "row")  # scratch columns AutoDP's dedup operators leave behind


def _split_positions(n_rows: int, seed: int = 42):
    """Original-row positions of the seed-42 0.6/0.2/0.2 split, via the shared splitter."""
    idx = pd.DataFrame({"_pos": np.arange(n_rows)})
    dummy = pd.Series(np.zeros(n_rows))
    x_tr, _, x_val, _, x_te, _ = split_train_val_test(idx, dummy, seed=seed)
    return (x_tr["_pos"].to_numpy(), x_val["_pos"].to_numpy(), x_te["_pos"].to_numpy())


def _task_type(target: pd.Series) -> str:
    """Same rule as automl_aco.search.evaluation._detect_problem_type / the OpenML loader."""
    return "regression" if (target.nunique() > 50 and target.dtype.kind in "iufc") else "classification"


def _strip_helpers(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in HELPER_COLS if c in df.columns]
    return df.drop(columns=drop) if drop else df


_ADAPTER = None  # set to scripts/autodp_our_space when --operator-space ours


def _apply_pipeline(dataset: dict, order, mctsdata) -> None:
    """Apply AutoDP's winning operator chain in place.

    Mirrors ``MCTS_DATA.getAcc`` minus the final classifier refit (which reads the dict but never
    mutates it). ``order[0]`` is the classifier/regressor choice, so operators start at index 1.
    """
    if _ADAPTER is not None:
        # Our operator codes live in mctsdata.listN too, so the dispatch below would route them to
        # AutoDP's own operator classes, which do not understand them.
        _ADAPTER._apply_pipeline(dataset, order)
        return
    for op in order[1:]:
        if op in mctsdata.list1:
            mctsdata.choose_imputer(dataset, op)
        elif op in mctsdata.list2:
            mctsdata.choose_encoding(dataset, op)
        elif op in mctsdata.list3:
            mctsdata.choose_normalizer(dataset, op)
        elif op in mctsdata.list4:
            mctsdata.choose_feature(dataset, op)
        elif op in mctsdata.list5:
            mctsdata.choose_duolicate(dataset, op)
        elif op in mctsdata.list6:
            mctsdata.choose_outlier(dataset, op)


def _search(df_search: pd.DataFrame, target: str, task: str, runtime, mctsdata, MCTS,
            dataset=None, df_metafeatures: pd.DataFrame = None):
    """Run AutoDP's MCTS and return (its internal dataset dict, pipeline, curves).

    ``runtime=None`` selects AutoDP's default: run until its own convergence rule fires (20
    consecutive iterations improving by less than 0.001).

    Two frames, chosen independently, because MCTS uses them for different things:

    * ``dataset`` -- the {train,target,test,target_test} dict the search SCORES on. When the caller
      supplies one (``_run_fair``), it is our seed-42 0.6 train / 0.2 val, and their unseeded
      ``read_dataset`` is bypassed entirely. MCTS never calls ``read_dataset`` itself -- the dict is
      a parameter -- so no patching is involved. ``dataset=None`` falls back to their behaviour and
      is only reachable from the unreported ``native`` path.
    * ``df_metafeatures`` -- passed as their ``df`` argument, whose ONLY use is
      ``get_CLA_meta_task_order(df)``, the metafeature family-order prior. We hand it the FULL
      frame: ACORec reads its query metafeatures from a precomputed full-dataset table, so
      full-frame metafeatures on both sides is the symmetric choice. Restricting AutoDP to
      train+val here would be an asymmetry in OUR favour.
    """
    df_search = df_search.copy()
    if dataset is None:
        dataset = mctsdata.read_dataset(df_search, target)  # their unseeded internal 80/20
    df_mf = df_search if df_metafeatures is None else df_metafeatures.copy()
    if task == "classification":
        fn = MCTS.CLA_With_TimeBudget if runtime else MCTS.CLA_Without_TimeBudget
        args = (df_mf, dataset, runtime, target) if runtime else (df_mf, dataset, target)
    else:
        fn = MCTS.REG_With_TimeBudget if runtime else MCTS.REG_Without_TimeBudget
        args = (df_mf, dataset, runtime, target) if runtime else (df_mf, dataset, target)
    t0 = time.time()
    times, scores, pipeline = fn(*args)
    # Stage markers: when the wall-clock cap kills this process, the last line in the log says
    # whether it died in the MCTS search or in the apply step (some operators -- AD dedup is O(n^2)
    # string comparisons, LOF, MICE -- can run for hours on a 5000-row frame, and AutoDP only checks
    # its time budget BETWEEN search iterations, never inside one).
    print(f"[stage] search finished in {time.time() - t0:.0f}s, pipeline={list(pipeline)}; "
          f"applying it now", flush=True)
    return dataset, list(pipeline), list(times), list(scores)


def _run_native(df: pd.DataFrame, target: str, task: str, runtime, mctsdata, MCTS, seed: int = 42):
    """Published-API protocol: search AND prepare on the full dataset.

    NOT A REPORTED PATH. Kept only as a literal reproduction of what their released API does, so
    the deviation is inspectable. It is deliberately left UNPATCHED -- their unseeded
    ``read_dataset`` still runs here, and its internal "test" is drawn from the full frame and so
    contains our held-out rows. Giving it our split would produce a protocol nobody published and
    nobody asked for; use ``--mode fair``, which is the default, for every number that is reported.
    """
    from autodatapre.Pipeline_Generation.MCTS import merge_datasets

    dataset, pipeline, times, scores = _search(df, target, task, runtime, mctsdata, MCTS)

    status = "ok"
    try:
        _apply_pipeline(dataset, pipeline, mctsdata)
        prepared = merge_datasets(dataset)
        prepared = prepared.dropna(how="all", subset=list(prepared.columns[:-1]))
        prepared = _strip_helpers(prepared).drop(columns=[target], errors="ignore")
    except Exception:
        # AutoDP's own entry point swallows this and returns the raw frame; record it instead.
        status = "apply_failed_returned_raw"
        prepared = df.drop(columns=[target]).copy()
        print("[warn] applying the winning pipeline failed; AutoDP falls back to the RAW frame:\n"
              + traceback.format_exc(), flush=True)

    out = prepared.copy()
    out["__adp_row__"] = out.index
    out["__adp_split__"] = ""  # stage 3 assigns rows to splits by __adp_row__
    return out, pipeline, times, scores, status


def _run_fair(df: pd.DataFrame, target: str, task: str, runtime, mctsdata, MCTS, seed: int = 42):
    """Leak-free protocol, on OUR split.

    The search scores on our 0.6 train / 0.2 val (``autodp_protocol.build_search_dataset``); the
    winning chain is then applied to {train+val 80%, test 20%}. Our test rows reach neither the
    search's scoring dict nor its family-order prior.

    This replaces their ``read_dataset``, whose ``train_test_split(X, Y, test_size=0.2)`` carried
    no ``random_state``: the search signal was a different random 20% on every run, so the same
    dataset and seed could return different pipelines.
    """
    from sklearn.preprocessing import LabelEncoder

    tr, val, te = _split_positions(len(df), seed=seed)
    trainval = np.concatenate([tr, val])  # same order evaluation.py uses when fit_include_val=True

    search_dataset = autodp_protocol.build_search_dataset(df, target, tr, val)
    dataset, pipeline, times, scores = _search(
        df.iloc[trainval], target, task, runtime, mctsdata, MCTS,
        dataset=search_dataset, df_metafeatures=df,
    )
    del dataset  # searched only to obtain the pipeline; its frames are train/val, not train/test
    prepared, status = _apply_fair(df, target, pipeline, mctsdata, seed=seed)
    return prepared, pipeline, times, scores, status


def _apply_fair(df: pd.DataFrame, target: str, pipeline, mctsdata, seed: int = 42):
    """The apply half of the fair protocol: winning chain -> {train+val 80%, test 20%}.

    Shared by the normal path (_run_fair) and the checkpoint-salvage path (_salvage_worker), so
    a salvaged pipeline goes through byte-identical apply logic.
    """
    from sklearn.preprocessing import LabelEncoder

    tr, val, te = _split_positions(len(df), seed=seed)
    trainval = np.concatenate([tr, val])

    # The apply step reuses the same LabelEncoding the search ran under, so the operators that read
    # labels see a consistent target. Stage 3 re-attaches the original y regardless.
    y_enc = pd.Series(LabelEncoder().fit_transform(df[target]), index=df.index, name=target)
    feats = df.drop(columns=[target])
    d = {
        "train": feats.iloc[trainval].copy(),
        "target": y_enc.iloc[trainval].to_frame(),
        "test": feats.iloc[te].copy(),
        # Real held-out labels, and their CBE would fit on them -- autodp_protocol's leak-free CBE
        # (installed in _worker) fits on the train block only, which is what makes this safe.
        "target_test": y_enc.iloc[te].to_frame(),
    }
    autodp_protocol.assert_dict_aligned(d)

    status = "ok"
    try:
        _apply_pipeline(d, pipeline, mctsdata)
        p_train = _strip_helpers(d["train"])
        p_test = _strip_helpers(d["test"])
        missing = [c for c in p_train.columns if c not in p_test.columns]
        if missing:
            raise RuntimeError(f"prepared test split is missing train columns: {missing[:10]}")
        p_test = p_test[list(p_train.columns)]  # operators can reorder columns by dtype
    except Exception:
        status = "apply_failed_returned_raw"
        p_train, p_test = feats.iloc[trainval].copy(), feats.iloc[te].copy()
        print("[warn] applying the winning pipeline failed; falling back to the RAW frame:\n"
              + traceback.format_exc(), flush=True)

    p_train = p_train.copy()
    p_train["__adp_row__"] = p_train.index
    p_train["__adp_split__"] = "trainval"
    p_test = p_test.copy()
    p_test["__adp_row__"] = p_test.index
    p_test["__adp_split__"] = "test"
    return pd.concat([p_train, p_test], axis=0, ignore_index=True), status


def _worker(csv_path: str, target: str, mode: str, runtime, seed: int, out_dir: str,
            operator_space: str = "theirs", meta_corpus=None,
            family_order: str = "prior") -> None:
    """Body of one dataset run. Executed in a child process so a wall-clock cap can kill it."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

    if operator_space == "ours":
        # Arm "their method, our operators": patches AutoDP in memory so its MCTS searches ACORec's
        # operator space, executed by ACORec's own implementations. Their search, their meta-learner
        # and their scoring signal are untouched. See scripts/autodp_our_space.py for the three
        # patch points and the disclosure about pca/svd being unrepresentable in their value model.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import autodp_our_space
        autodp_our_space.install(verbose=True, retrained_dir=meta_corpus,
                                 family_order=family_order)
        global _ADAPTER
        _ADAPTER = autodp_our_space

    from autodatapre.Pipeline_Generation import MCTS_DATA as mctsdata
    from autodatapre.Pipeline_Generation import MCTS

    # Move their EVALUATION layer onto ours, leaving their search untouched. `native` is the
    # literal published API and is deliberately excluded (see _run_native's docstring).
    exc_counter = autodp_protocol.ExceptionCounter()
    if mode == "fair":
        autodp_protocol.install_leakfree_cbe(verbose=True)
        autodp_protocol.install_scorer_patches(seed=seed, verbose=True)
        # Best-so-far checkpoint, written per NODE EVALUATION so a kill can be salvaged apply-only.
        # Iteration granularity was not enough: on 378/722 no iteration ever completed, because
        # their scorer's profit=None poisons drop_unpromising and the loop spins (see
        # SearchCheckpoint's docstring).
        os.makedirs(out_dir, exist_ok=True)
        checkpoint = autodp_protocol.SearchCheckpoint(_checkpoint_path(out_dir))
        checkpoint.install(verbose=True)
        # The counter needs the checkpoint to tell a DEAD loop (iterations raising with no node
        # evaluated between them) from a merely slow one, and to abort the former immediately
        # rather than spinning until the wall-clock cap.
        exc_counter = autodp_protocol.ExceptionCounter(checkpoint=checkpoint)
        exc_counter.install(verbose=True)

    df = pd.read_csv(csv_path)
    task = _task_type(df[target])
    n_rows_in = len(df)

    t0 = time.time()
    runner = _run_native if mode == "native" else _run_fair
    prepared, pipeline, times, scores, status = runner(df, target, task, runtime, mctsdata, MCTS,
                                                       seed=seed)
    elapsed = time.time() - t0

    os.makedirs(out_dir, exist_ok=True)
    prepared.to_csv(os.path.join(out_dir, "prepared.csv"), index=False)

    feat_cols = [c for c in prepared.columns if not c.startswith("__adp_")]
    meta = {
        "dataset_csv": os.path.abspath(csv_path),
        "mode": mode,
        "operator_space": operator_space,
        "meta_corpus": str(meta_corpus) if meta_corpus else None,
        "task_type": task,
        "status": status,
        "autodp_version": "0.1.12",
        "runtime_budget_seconds": runtime,
        "converged_default_budget": runtime is None,
        "search_seconds": round(elapsed, 2),
        "seed": seed,
        # The protocol fields. A number without these cannot be traced to what produced it.
        "search_split": (autodp_protocol.SEARCH_SPLIT_TAG if mode == "fair"
                         else "theirs-unseeded-80/20-of-full-frame"),
        "metafeature_frame": "full" if mode == "fair" else "full",
        "internal_scorer_seed": seed if mode == "fair" else None,
        "leakfree_cbe": mode == "fair",
        **exc_counter.report(),
        "pipeline": pipeline,
        "search_curve_times": times,
        "search_curve_scores": scores,
        "n_rows_input": int(n_rows_in),
        "n_rows_prepared": int(len(prepared)),
        "n_features_input": int(df.shape[1] - 1),
        "n_features_prepared": len(feat_cols),
        "prepared_feature_columns": feat_cols,
        "n_object_cols_prepared": int(prepared[feat_cols].select_dtypes(include=["object", "category"]).shape[1]),
    }
    with open(os.path.join(out_dir, "autodp_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[ok  ] mode={mode} status={status} pipeline={pipeline} "
          f"rows {n_rows_in}->{len(prepared)} feats {df.shape[1] - 1}->{len(feat_cols)} "
          f"in {elapsed:.1f}s", flush=True)


def _checkpoint_path(out_dir: str) -> str:
    return os.path.join(out_dir, "search_checkpoint.json")


def _salvage_worker(csv_path: str, target: str, mode: str, seed: int, out_dir: str,
                    operator_space: str = "theirs", meta_corpus=None,
                    family_order: str = "prior") -> None:
    """Apply-only child: take the checkpointed best-so-far pipeline and produce prepared.csv.

    Run after the watchdog killed the search at the wall-clock cap. The search already paid for
    the pipeline; this only replays the fair-protocol apply step, which is one pass instead of
    hundreds of iterations. `fair` mode only -- `native` is not a reported path and keeps the old
    kill-means-timeout behaviour.
    """
    random.seed(seed)
    np.random.seed(seed)

    ckpt = autodp_protocol.SearchCheckpoint.read(_checkpoint_path(out_dir))
    if ckpt is None:
        raise SystemExit("no checkpoint to salvage")
    pipeline = list(ckpt["pipeline"])

    if operator_space == "ours":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import autodp_our_space
        autodp_our_space.install(verbose=True, retrained_dir=meta_corpus, family_order=family_order)
        global _ADAPTER
        _ADAPTER = autodp_our_space

    from autodatapre.Pipeline_Generation import MCTS_DATA as mctsdata

    autodp_protocol.install_leakfree_cbe(verbose=True)
    autodp_protocol.install_scorer_patches(seed=seed, verbose=True)

    df = pd.read_csv(csv_path)
    task = _task_type(df[target])
    t0 = time.time()
    prepared, status = _apply_fair(df, target, pipeline, mctsdata, seed=seed)
    elapsed = time.time() - t0

    os.makedirs(out_dir, exist_ok=True)
    prepared.to_csv(os.path.join(out_dir, "prepared.csv"), index=False)
    feat_cols = [c for c in prepared.columns if not c.startswith("__adp_")]
    meta = {
        "dataset_csv": os.path.abspath(csv_path),
        "mode": mode,
        "operator_space": operator_space,
        "meta_corpus": str(meta_corpus) if meta_corpus else None,
        "task_type": task,
        "status": status,
        "autodp_version": "0.1.12",
        "runtime_budget_seconds": None,
        "converged_default_budget": False,
        "hit_wall_clock_cap": True,
        # DISCLOSE WHEN REPORTING: the search was killed at the cap; this pipeline is the best
        # of the iterations that completed, not of a converged search.
        "salvaged_from_checkpoint": True,
        "checkpoint_node_evals": int(ckpt.get("node_evals_completed") or 0),
        "checkpoint_none_profit_evals": int(ckpt.get("none_profit_evals") or 0),
        "checkpoint_depth": ckpt.get("depth"),
        "checkpoint_internal_profit": ckpt.get("profit"),
        "search_seconds": None,
        "apply_seconds": round(elapsed, 2),
        "seed": seed,
        "search_split": autodp_protocol.SEARCH_SPLIT_TAG,
        "metafeature_frame": "full",
        "internal_scorer_seed": seed,
        "leakfree_cbe": True,
        "search_iteration_exceptions": None,  # counted in the killed process, not recoverable
        "search_iteration_exception_kinds": None,
        "search_iteration_first_traceback": None,
        "pipeline": pipeline,
        "search_curve_times": [],
        "search_curve_scores": [],
        "n_rows_input": int(len(df)),
        "n_rows_prepared": int(len(prepared)),
        "n_features_input": int(df.shape[1] - 1),
        "n_features_prepared": len(feat_cols),
        "prepared_feature_columns": feat_cols,
        "n_object_cols_prepared": int(prepared[feat_cols].select_dtypes(include=["object", "category"]).shape[1]),
    }
    with open(os.path.join(out_dir, "autodp_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[ok  ] SALVAGED mode={mode} status={status} pipeline={pipeline} "
          f"({meta['checkpoint_node_evals']} node evaluations completed before the abort)",
          flush=True)


def _dead_search_worker(csv_path: str, target: str, mode: str, seed: int, out_dir: str,
                        operator_space: str = "ours") -> None:
    """Apply-nothing child: the search crashed on every iteration and scored zero nodes, so there
    is no checkpointed pipeline to salvage. Produce the RAW frame as an empty pipeline instead of
    a `no prepared dataset` failure, and label it dead_search so the reported row is 'AutoDP was
    given the operator space and its search collapsed', not 'AutoDP evaluated its options and
    chose no preprocessing'. The adapter is deliberately not installed -- an empty chain touches
    no operator -- but `operator_space` is still recorded as provenance. `fair` mode only.
    """
    random.seed(seed)
    np.random.seed(seed)

    marker_path = autodp_protocol.dead_search_marker_path(out_dir)
    marker = {}
    if os.path.exists(marker_path):
        with open(marker_path) as f:
            marker = json.load(f)

    from autodatapre.Pipeline_Generation import MCTS_DATA as mctsdata

    df = pd.read_csv(csv_path)
    task = _task_type(df[target])
    t0 = time.time()
    prepared, status = _apply_fair(df, target, [], mctsdata, seed=seed)
    elapsed = time.time() - t0

    os.makedirs(out_dir, exist_ok=True)
    prepared.to_csv(os.path.join(out_dir, "prepared.csv"), index=False)
    feat_cols = [c for c in prepared.columns if not c.startswith("__adp_")]
    meta = {
        "dataset_csv": os.path.abspath(csv_path),
        "mode": mode,
        "operator_space": operator_space,
        "task_type": task,
        "status": "dead_search_raw_frame",
        "autodp_version": "0.1.12",
        "runtime_budget_seconds": None,
        "converged_default_budget": False,
        "hit_wall_clock_cap": False,
        # DISCLOSE WHEN REPORTING: AutoDP's MCTS raised on every iteration and scored no candidate,
        # so this row is the untouched frame, not a search result. dead_search + empty_pipeline
        # both flag it; the counts are how much spinning happened before the abort.
        "dead_search": True,
        "empty_pipeline": True,
        "salvaged_from_checkpoint": False,
        "dead_search_spin_iterations": marker.get("spin_iterations"),
        "dead_search_node_evals": marker.get("node_evals_completed"),
        "dead_search_none_profit_evals": marker.get("none_profit_evals"),
        "search_seconds": None,
        "apply_seconds": round(elapsed, 2),
        "seed": seed,
        "search_split": autodp_protocol.SEARCH_SPLIT_TAG,
        "metafeature_frame": "full",
        "internal_scorer_seed": seed,
        "leakfree_cbe": True,
        "search_iteration_exceptions": marker.get("search_iteration_exceptions"),
        "search_iteration_exception_kinds": marker.get("search_iteration_exception_kinds"),
        "search_iteration_first_traceback": marker.get("first_traceback"),
        "pipeline": [],
        "search_curve_times": [],
        "search_curve_scores": [],
        "n_rows_input": int(len(df)),
        "n_rows_prepared": int(len(prepared)),
        "n_features_input": int(df.shape[1] - 1),
        "n_features_prepared": len(feat_cols),
        "prepared_feature_columns": feat_cols,
        "n_object_cols_prepared": int(prepared[feat_cols].select_dtypes(include=["object", "category"]).shape[1]),
    }
    with open(os.path.join(out_dir, "autodp_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[ok  ] DEAD-SEARCH SALVAGE mode={mode} status={status} pipeline=[] raw frame "
          f"({marker.get('search_iteration_exceptions')} swallowed exceptions, "
          f"{marker.get('none_profit_evals')}/{marker.get('node_evals_completed')} nodes scored None)",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-csv", required=True, help="an exported <id>.csv from export_eval_datasets.py")
    ap.add_argument("--dataset-id", default=None, help="defaults to the csv basename")
    ap.add_argument("--target", default="target")
    ap.add_argument("--mode", choices=["native", "fair"], required=True)
    ap.add_argument("--runtime", type=float, default=None,
                    help="AutoDP runTime budget in seconds; omit for its default run-to-convergence")
    ap.add_argument("--cap-seconds", type=float, default=3600.0,
                    help="wall-clock watchdog. On expiry the run is killed and retried with an "
                         "explicit runTime equal to the cap, so a non-converging dataset still "
                         "yields a prepared frame instead of nothing.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adp-family-order", choices=["prior", "all"], default="prior",
                    help="prior (default) = transfer their shipped task-order prior through "
                         "operator aliasing. all = give the search all six families in ACORec's "
                         "canonical order with no transferred prior, which is the symmetric "
                         "comparison since ACORec's ACO searches all six every run.")
    ap.add_argument("--adp-meta-corpus", default=None,
                    help="Corpus dir from scripts/build_adp_meta_corpus.py. Retrains AutoDP's "
                         "1-NN meta-learner over ACORec's operators instead of aliasing onto "
                         "their shipped label.csv. Only meaningful with --operator-space ours.")
    ap.add_argument("--operator-space", choices=["theirs", "ours"], default="theirs",
                    help="theirs = AutoDP's own operators (unmodified). ours = its MCTS searches ACORec's operator space via scripts/autodp_our_space.py, for the same-operator-space arm.")
    ap.add_argument("--out-dir", default="outputs/autodp")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    did = args.dataset_id or os.path.splitext(os.path.basename(args.dataset_csv))[0]
    space_tag = args.mode if args.operator_space == "theirs" else f"{args.mode}_ourops"
    out_dir = os.path.join(args.out_dir, space_tag, f"dataset_{did}")
    done = os.path.join(out_dir, "autodp_meta.json")
    if os.path.exists(done) and not args.overwrite:
        print(f"[skip] {did} ({args.mode}) already done -> {done}")
        return

    ctx = mp.get_context("spawn")  # torch + sklearn are not fork-safe on macOS

    def _try_salvage() -> bool:
        """After a cap kill: apply the checkpointed best-so-far pipeline instead of giving up.

        The search wrote its running best to search_checkpoint.json after every completed
        iteration (autodp_protocol.SearchCheckpoint), so a kill only loses the iterations that
        never happened, not the ones already paid for. Returns True when prepared.csv +
        autodp_meta.json exist, i.e. the dataset now has a scoreable result.
        """
        if args.mode != "fair":
            return False
        ckpt = autodp_protocol.SearchCheckpoint.read(_checkpoint_path(out_dir))
        if ckpt is None:
            return False
        print(f"[warn] {did} ({args.mode}): killed at the cap after "
              f"{ckpt.get('node_evals_completed')} node evaluations; salvaging the "
              f"checkpointed pipeline {ckpt.get('pipeline')} apply-only", flush=True)
        sp = ctx.Process(target=_salvage_worker,
                         args=(args.dataset_csv, args.target, args.mode, args.seed, out_dir,
                               args.operator_space, args.adp_meta_corpus, args.adp_family_order))
        sp.start()
        # The apply step gets its OWN budget, not the cap the search just exhausted: it is a
        # single pass over the frame rather than a search loop, so the cap that was too small for
        # hundreds of iterations says nothing about it. Floored so a deliberately tiny --cap-seconds
        # (smoke tests) cannot sabotage the salvage it is meant to trigger.
        sp.join(timeout=max(float(args.cap_seconds), 600.0))
        if sp.is_alive():
            sp.terminate()
            sp.join(30)
            if sp.is_alive():
                sp.kill()
                sp.join()
            print(f"[warn] {did}: the apply step itself exceeded the cap; salvage abandoned",
                  flush=True)
            return False
        return sp.exitcode == 0 and os.path.exists(done)

    def _try_dead_search_salvage() -> bool:
        """After a DEAD-search abort with no checkpointed pipeline: score the raw frame.

        ``ExceptionCounter`` writes ``dead_search.json`` when it kills a search that raised on
        every iteration and never scored a node. There is no pipeline to apply, so the honest
        downstream input is the untouched frame -- produced here as an empty pipeline, labelled
        ``dead_search`` so the row is not read as a genuine AutoDP no-preprocessing preference.
        """
        if args.mode != "fair":
            return False
        if not os.path.exists(autodp_protocol.dead_search_marker_path(out_dir)):
            return False
        print(f"[warn] {did} ({args.mode}): search collapsed with no scored node; "
              f"salvaging the raw frame as an empty pipeline", flush=True)
        sp = ctx.Process(target=_dead_search_worker,
                         args=(args.dataset_csv, args.target, args.mode, args.seed, out_dir,
                               args.operator_space))
        sp.start()
        sp.join(timeout=max(float(args.cap_seconds), 600.0))
        if sp.is_alive():
            sp.terminate()
            sp.join(30)
            if sp.is_alive():
                sp.kill()
                sp.join()
            print(f"[warn] {did}: the raw-frame apply itself overran the cap; salvage abandoned",
                  flush=True)
            return False
        return sp.exitcode == 0 and os.path.exists(done)

    # The retry budget is HALF the cap, not the cap itself: AutoDP still has to apply the winning
    # pipeline and merge the frames after its search loop ends, and on a large dataset that tail
    # can exceed whatever slack is left, which just burns the cap a second time. Worst case per
    # dataset is therefore ~1.5x cap_seconds rather than ~2.25x.
    retry_runtime = args.cap_seconds / 2.0
    for attempt, runtime in enumerate([args.runtime, retry_runtime]):
        if attempt == 1:
            print(f"[warn] {did} ({args.mode}) exceeded the {args.cap_seconds:.0f}s cap; "
                  f"retrying with an explicit runTime={retry_runtime:.0f}s budget", flush=True)
        proc = ctx.Process(target=_worker, args=(args.dataset_csv, args.target, args.mode, runtime,
                                                 args.seed, out_dir, args.operator_space,
                                                 args.adp_meta_corpus,
                                                 args.adp_family_order))
        proc.start()
        proc.join(timeout=args.cap_seconds)
        if proc.is_alive():
            proc.terminate()
            proc.join(30)
            if proc.is_alive():
                proc.kill()
                proc.join()
            # A checkpointed pipeline beats both alternatives from here: the retry re-searches
            # from scratch (and on the datasets that hit the cap once, it reliably hits it again
            # -- dataset 722 burned 2x cap for nothing), and the failure row records nothing.
            if _try_salvage() or _try_dead_search_salvage():
                return
            if attempt == 0 and args.runtime is None:
                continue  # convergence mode never terminated -> fall back to the capped budget
            # Record the timeout so a rerun does not spend the same hours again, and so the report
            # can show the dataset as "AutoDP did not finish" rather than silently omitting it.
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "autodp_failed.json"), "w") as f:
                json.dump({
                    "dataset_id": did, "mode": args.mode, "status": "timeout",
                    "cap_seconds": args.cap_seconds, "retry_runtime": retry_runtime,
                    "detail": "killed at the wall-clock cap in both the convergence attempt and the "
                              "explicit-budget retry, and checkpoint salvage produced no result "
                              "(no completed iteration, or the apply step itself overran the cap); "
                              "AutoDP checks its budget only between search iterations, so a single "
                              "slow operator (AD dedup is O(n^2), LOF, MICE) can overrun any budget",
                }, f, indent=2)
            print(f"[FAIL] {did} ({args.mode}): killed at the wall-clock cap; recorded as a timeout. "
                  f"Rerun this dataset alone with a larger --cap-seconds to chase a real number.")
            sys.exit(2)
        if proc.exitcode != 0:
            # A crashed search (segfault in a native op, OOM kill) is salvageable the same way a
            # capped one is: any completed iteration left a checkpoint behind. Exit 3 is the
            # dead-search abort -- if it scored at least one node _try_salvage handles it, and if
            # it scored none _try_dead_search_salvage produces the raw frame.
            if _try_salvage() or _try_dead_search_salvage():
                return
            print(f"[FAIL] {did} ({args.mode}): worker exited {proc.exitcode}")
            sys.exit(proc.exitcode or 1)
        if runtime is not None and args.runtime is None:
            # Record that the headline convergence budget did not finish on its own.
            with open(done) as f:
                meta = json.load(f)
            meta["converged_default_budget"] = False
            meta["hit_wall_clock_cap"] = True
            with open(done, "w") as f:
                json.dump(meta, f, indent=2, default=str)
        return


if __name__ == "__main__":
    main()
