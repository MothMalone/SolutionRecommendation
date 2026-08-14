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

  native  Faithful to the published API: the MCTS searches over the FULL dataset, so our 20% test
          rows are visible to the search (its internal scorer holds out its own unseeded random
          20%). Transductive, and generous to AutoDP.
  fair    The search sees ONLY our 80% train+val. The winning operator chain is then applied to
          {train: our 80%, test: our 20%} using AutoDP's OWN operator classes, so our test rows
          never inform the search. This is the protocol our method is held to.

Faithfulness notes (each one is a deliberate, documented deviation):
  * ``AutoDP.Classifier`` does not return the winning pipeline, so native mode replicates its body
    (``read_dataset`` -> ``CLA_*`` -> apply -> ``merge_datasets`` -> ``dropna``) in order to record
    it. The one omission is the final ``getAcc`` classifier refit, which does not mutate the
    prepared frame and only burns time.
  * AutoDP's operators transform train and test INDEPENDENTLY (e.g. ``ZS`` z-scores test with
    test's own mean/std) and several DELETE ROWS from the test split (``DROP``, ``ZSB``/``IQR``/
    ``LOF``, ``ED``/``AD``). Both behaviours are theirs and are preserved; stage 3 reports test
    coverage so the effect is visible rather than silently inflating their score.
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


def _search(df_search: pd.DataFrame, target: str, task: str, runtime, mctsdata, MCTS):
    """Run AutoDP's MCTS on ``df_search`` and return (its internal dataset dict, pipeline, curves).

    ``runtime=None`` selects AutoDP's default: run until its own convergence rule fires (20
    consecutive iterations improving by less than 0.001).
    """
    df_search = df_search.copy()
    dataset = mctsdata.read_dataset(df_search, target)  # their unseeded internal 80/20 + LabelEncoder
    if task == "classification":
        fn = MCTS.CLA_With_TimeBudget if runtime else MCTS.CLA_Without_TimeBudget
        args = (df_search, dataset, runtime, target) if runtime else (df_search, dataset, target)
    else:
        fn = MCTS.REG_With_TimeBudget if runtime else MCTS.REG_Without_TimeBudget
        args = (df_search, dataset, runtime, target) if runtime else (df_search, dataset, target)
    t0 = time.time()
    times, scores, pipeline = fn(*args)
    # Stage markers: when the wall-clock cap kills this process, the last line in the log says
    # whether it died in the MCTS search or in the apply step (some operators -- AD dedup is O(n^2)
    # string comparisons, LOF, MICE -- can run for hours on a 5000-row frame, and AutoDP only checks
    # its time budget BETWEEN search iterations, never inside one).
    print(f"[stage] search finished in {time.time() - t0:.0f}s, pipeline={list(pipeline)}; "
          f"applying it now", flush=True)
    return dataset, list(pipeline), list(times), list(scores)


def _run_native(df: pd.DataFrame, target: str, task: str, runtime, mctsdata, MCTS):
    """Published-API protocol: search AND prepare on the full dataset."""
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


def _run_fair(df: pd.DataFrame, target: str, task: str, runtime, mctsdata, MCTS):
    """Leak-free protocol: search on our 80%, then apply the winner to our 80%/20% with their ops."""
    from sklearn.preprocessing import LabelEncoder

    tr, val, te = _split_positions(len(df))
    trainval = np.concatenate([tr, val])  # same order evaluation.py uses when fit_include_val=True

    dataset, pipeline, times, scores = _search(df.iloc[trainval], target, task, runtime, mctsdata, MCTS)
    del dataset  # searched only to obtain the pipeline; its frames are their internal sub-split

    # Their search ran on a LabelEncoder'd target, so the apply step gets the same encoding for the
    # operators that read labels (feature selection). Stage 3 re-attaches the original y regardless.
    y_enc = pd.Series(LabelEncoder().fit_transform(df[target]), index=df.index, name=target)
    feats = df.drop(columns=[target])
    d = {
        "train": feats.iloc[trainval].copy(),
        "target": y_enc.iloc[trainval].to_frame(),
        "test": feats.iloc[te].copy(),
        "target_test": y_enc.iloc[te].to_frame(),
    }

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
    return pd.concat([p_train, p_test], axis=0, ignore_index=True), pipeline, times, scores, status


def _worker(csv_path: str, target: str, mode: str, runtime, seed: int, out_dir: str,
            operator_space: str = "theirs") -> None:
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
        autodp_our_space.install(verbose=True)
        global _ADAPTER
        _ADAPTER = autodp_our_space

    from autodatapre.Pipeline_Generation import MCTS_DATA as mctsdata
    from autodatapre.Pipeline_Generation import MCTS

    df = pd.read_csv(csv_path)
    task = _task_type(df[target])
    n_rows_in = len(df)

    t0 = time.time()
    runner = _run_native if mode == "native" else _run_fair
    prepared, pipeline, times, scores, status = runner(df, target, task, runtime, mctsdata, MCTS)
    elapsed = time.time() - t0

    os.makedirs(out_dir, exist_ok=True)
    prepared.to_csv(os.path.join(out_dir, "prepared.csv"), index=False)

    feat_cols = [c for c in prepared.columns if not c.startswith("__adp_")]
    meta = {
        "dataset_csv": os.path.abspath(csv_path),
        "mode": mode,
        "operator_space": operator_space,
        "task_type": task,
        "status": status,
        "autodp_version": "0.1.12",
        "runtime_budget_seconds": runtime,
        "converged_default_budget": runtime is None,
        "search_seconds": round(elapsed, 2),
        "seed": seed,
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
                                                 args.seed, out_dir, args.operator_space))
        proc.start()
        proc.join(timeout=args.cap_seconds)
        if proc.is_alive():
            proc.terminate()
            proc.join(30)
            if proc.is_alive():
                proc.kill()
                proc.join()
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
                              "explicit-budget retry; AutoDP checks its budget only between search "
                              "iterations, so a single slow operator (AD dedup is O(n^2), LOF, MICE) "
                              "can overrun any budget on a large frame",
                }, f, indent=2)
            print(f"[FAIL] {did} ({args.mode}): killed at the wall-clock cap; recorded as a timeout. "
                  f"Rerun this dataset alone with a larger --cap-seconds to chase a real number.")
            sys.exit(2)
        if proc.exitcode != 0:
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
