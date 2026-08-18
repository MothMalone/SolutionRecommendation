#!/usr/bin/env python3
"""Build a retrained meta-learner corpus for AutoDP over ACORec's operator space.

WHY
---
AutoDP's `get_CLA_meta_task_order` is a deterministic 1-NN: it computes 7 metafeatures for the
query dataset, finds the closest row of `Metafeature.csv`, takes that neighbour's best-scoring
pipeline from `label.csv`, and returns the ORDER of operator families in it. Shipped, both CSVs
describe AutoDP's own operators, so on ACORec's space every operator had to be aliased
(`pca`/`svd` both collapsing onto `TB`). This script regenerates the two CSVs over OUR operators,
so their search selects a task order learned in the space it is actually searching.

WHAT IS NOT RETRAINED
---------------------
`model_CLA.pickle` (the value estimator) is deliberately left alone. Its input is the output of a
`MultiHeadAttention` that `Estimate_after_profit.get_Estimate` constructs with fresh random
weights on EVERY call and never persists -- there is no torch.save/load anywhere in the package.
Measured on one dataset, 4 pipelines x 40 calls: between-pipeline sd 0.0235 vs within-pipeline sd
0.3114, a signal-to-noise ratio of 0.076. Refitting the MLP would fit one random projection and
then be queried through a different one, so it cannot be made coherent without replacing their
architecture -- which would stop the arm from being "their search". See docs/DATASET_CHANGE_AND_RQ3.md.

LEAKAGE
-------
This corpus is a fitting step, so the 30 evaluation datasets must not appear in it. Dataset
selection runs through eval_ids.assert_disjoint and the run aborts if any survives.

USAGE
-----
    python scripts/build_adp_meta_corpus.py --out-dir data/adp_ourops_corpus \
        --n-datasets 200 --pipelines-per-dataset 10

Resumable: completed datasets are read back from progress.jsonl and skipped. Shardable with
--shard I/N for parallel Kaggle notebooks; merge with --merge afterwards.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from automl_aco.eval_ids import EVAL_ID_SET, assert_disjoint, normalize_id  # noqa: E402
from automl_aco.adp_metafeatures import as_matrix, batch_dataset_vectors  # noqa: E402
from automl_aco.search.evaluation import evaluate_candidates_simple  # noqa: E402
from autodp_our_space import STEP_OPERATORS, _code  # noqa: E402

# Their pipeline is 7 slots: 6 preprocessing families then a model. Our space has exactly 6
# families, so the shape is preserved and n_features_in_ stays 14 -- the arm varies the
# vocabulary, not the architecture.
FAMILY_ORDER = ["imputation", "encoding", "scaling", "feature_selection",
                "dimensionality_reduction", "outlier_removal"]
MODEL_SLOT = "RF"          # constant: our proxy is LogReg; the slot exists only to keep 7 slots.
NULL = "none"


def sample_pipeline(rng: random.Random) -> tuple:
    """(ordered slot codes, config dict). Order is sampled too -- it is what the meta-learner learns."""
    order = FAMILY_ORDER[:]
    rng.shuffle(order)
    cfg, slots = {}, []
    for fam in order:
        # A family is off ~25% of the time, mirroring the *_null slots in their label.csv.
        op = NULL if rng.random() < 0.25 else rng.choice(STEP_OPERATORS[fam])
        cfg[fam] = op
        slots.append(_code(fam, op))
    return slots + [MODEL_SLOT], cfg


def load_table(dataset_id: str, local_dirs, target_column: str):
    for d in local_dirs:
        p = Path(d) / f"{dataset_id}.csv"
        if p.exists():
            df = pd.read_csv(p)
            if target_column in df.columns:
                return df
            df = df.rename(columns={df.columns[-1]: target_column})
            return df
    from automl_aco.data.loaders import load_openml_dataset
    ds = None
    for d in local_dirs:                      # let the loader see the folders too
        ds = load_openml_dataset(dataset_id, local_data_folder=str(d), prefer_local=True)
        if ds is not None:
            break
    if ds is None:
        ds = load_openml_dataset(dataset_id, prefer_local=True)
    if ds is None:
        # It returns None rather than raising, so without this the caller dies on a
        # TypeError that says nothing about which id or why.
        raise RuntimeError(
            f"could not load dataset {dataset_id}: no local <id>.csv in {list(local_dirs)} "
            "and the OpenML fetch returned nothing"
        )
    df = ds["X"].copy()
    df[target_column] = ds["y"]
    return df


def available_locally(local_dirs) -> set:
    """Dataset ids with a ready-made <id>.csv in any of the supplied directories."""
    found = set()
    for d in local_dirs:
        base = Path(d)
        if base.is_dir():
            for f in base.rglob("*.csv"):
                stem = f.stem
                if stem.isdigit():
                    found.add(normalize_id(stem))
    return found


def choose_ids(args, local_dirs=()) -> list:
    if args.ids:
        chosen = [normalize_id(i) for i in args.ids.split(",") if i.strip()]
        assert_disjoint(chosen, context="adp meta-corpus --ids")
        return chosen
    feats = pd.read_csv(REPO / "data" / "openml" / "dataset_feats.csv", index_col=0)
    pool = [normalize_id(i) for i in feats.index]
    pool = [i for i in pool if i not in EVAL_ID_SET]
    assert_disjoint(pool, context="adp meta-corpus candidate pool")

    # Prefer datasets that are already on disk. Sampling the library blind and relying on an
    # OpenML fetch fails wholesale in offline/partially-offline environments (observed on Kaggle:
    # 200/200 instant failures), and the corpus does not need any PARTICULAR datasets -- only
    # enough non-evaluation tables to learn a task order from.
    if not args.allow_download:
        have = available_locally(local_dirs)
        usable = [i for i in pool if i in have]
        if len(usable) < args.n_datasets:
            print(f"[corpus] note: {len(usable)} of {args.n_datasets} requested datasets are "
                  f"available locally in {[str(d) for d in local_dirs]}.")
            if not usable:
                raise SystemExit(
                    "[corpus] no usable datasets on disk. Point --local-dir at a folder of "
                    "<id>.csv files (e.g. the mathurinache/openml Kaggle mount), or pass "
                    "--allow-download to fetch from OpenML."
                )
        pool = usable
        print(f"[corpus] {len(pool)} non-evaluation datasets available locally")

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    chosen = pool[: args.n_datasets]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        chosen = [d for k, d in enumerate(chosen) if k % n == (i - 1) % n]
    assert_disjoint(chosen, context="adp meta-corpus selected datasets")
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-datasets", type=int, default=200)
    ap.add_argument("--ids", default="",
                    help="comma-separated dataset ids, overriding the sampled selection")
    ap.add_argument("--pipelines-per-dataset", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", default="", help="I/N round-robin over the selected datasets")
    ap.add_argument("--local-dir", action="append", default=[],
                    help="directory of ready-made <id>.csv; repeatable")
    ap.add_argument("--target-column", default="target")
    ap.add_argument("--adp-python", default=str(REPO / ".venv-autodp" / "bin" / "python"))
    ap.add_argument("--allow-download", action="store_true",
                    help="Sample from the whole library and fetch missing tables from OpenML. "
                         "Off by default: selection is restricted to <id>.csv already on disk, "
                         "so the build cannot fail wholesale on a blocked or partial network.")
    ap.add_argument("--merge", action="store_true",
                    help="skip generation; assemble Metafeature.csv/label.csv from progress.jsonl")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "datasets"
    cache.mkdir(exist_ok=True)
    progress = out / "progress.jsonl"

    done = {}
    if progress.exists():
        for line in progress.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["dataset_id"]] = r

    local_dirs = [Path(d) for d in args.local_dir] + [cache]
    if not args.merge:
        ids = choose_ids(args, local_dirs)
        todo = [d for d in ids if d not in done]
        print(f"[corpus] {len(ids)} datasets selected, {len(done)} already done, {len(todo)} to run")

        with progress.open("a") as fh:
            for k, ds in enumerate(todo, 1):
                t0 = time.time()
                try:
                    df = load_table(ds, local_dirs, args.target_column)
                    df.to_csv(cache / f"{ds}.csv", index=False)

                    rng = random.Random(f"{args.seed}:{ds}")
                    slots, cfgs = [], []
                    for _ in range(args.pipelines_per_dataset):
                        s, c = sample_pipeline(rng)
                        slots.append(s)
                        cfgs.append(c)

                    _, _, results, _ = evaluate_candidates_simple(
                        df, args.target_column, cfgs,
                        proxy_settings={"model": "logreg", "split_seeds": [42]},
                    )
                    by_cfg = {json.dumps(c, sort_keys=True): sc for c, sc in results}
                    scores = [by_cfg.get(json.dumps(c, sort_keys=True)) for c in cfgs]
                    if all(s is None or not np.isfinite(s) for s in scores):
                        raise RuntimeError("every pipeline failed to score")

                    row = {"dataset_id": ds, "status": "ok",
                           "shape": f"{df.shape[0]}*{df.shape[1]}",
                           "pipelines": [",".join(s) for s in slots],
                           "scores": [None if s is None else float(s) for s in scores],
                           "seconds": round(time.time() - t0, 2)}
                except Exception as exc:
                    row = {"dataset_id": ds, "status": "fail",
                           "error": f"{type(exc).__name__}: {exc}",
                           "seconds": round(time.time() - t0, 2)}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                done[ds] = row
                mark = "ok  " if row["status"] == "ok" else "FAIL"
                print(f"  [{k}/{len(todo)}] {mark} {ds}  {row.get('error','')} ({row['seconds']}s)")

    # ---- assemble ----
    ok = [r for r in done.values() if r.get("status") == "ok"]
    if not ok:
        print("[corpus] nothing succeeded; not writing CSVs")
        return 1
    ok.sort(key=lambda r: r["dataset_id"])
    assert_disjoint([r["dataset_id"] for r in ok], context="adp meta-corpus output")

    files = [cache / f"{r['dataset_id']}.csv" for r in ok]
    missing = [f for f in files if not f.exists()]
    if missing:
        print(f"[corpus] {len(missing)} cached CSV(s) missing, e.g. {missing[0]}; rerun without --merge")
        return 1
    print(f"[corpus] computing AutoDP metafeatures for {len(files)} datasets ...")
    meta = as_matrix(batch_dataset_vectors(files, adp_python=args.adp_python), files)
    pd.DataFrame(meta).to_csv(out / "Metafeature.csv", index=False)

    rows, pid = [], 0
    k = args.pipelines_per_dataset
    for r in ok:
        # Exactly k rows per dataset, always. Their reader slices df.iloc[k*minid : k*minid+k],
        # which silently misaligns on their own shipped label.csv (group sizes 10/6/4/11/9).
        # Padding to a fixed k is what makes that indexing correct here.
        pipes, scores = r["pipelines"][:k], r["scores"][:k]
        while len(pipes) < k:
            pipes.append(pipes[-1])
            scores.append(scores[-1])
        for p, s in zip(pipes, scores):
            pid += 1
            rows.append({"Id": pid, "DatasetName": r["dataset_id"], "Target": "target",
                         "Pipeline": p,
                         "EvaluationMetric": 0.0 if s is None else float(s),
                         "Time": 0.0, "Size": r.get("shape", ""), "Website": "acorec-retrained"})
    pd.DataFrame(rows).to_csv(out / "label.csv", index=False)

    print(f"[corpus] wrote {out/'Metafeature.csv'}  ({meta.shape[0]} x {meta.shape[1]})")
    print(f"[corpus] wrote {out/'label.csv'}        ({len(rows)} rows, {k} per dataset)")
    print(f"[corpus] failures: {sum(1 for r in done.values() if r.get('status')!='ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
