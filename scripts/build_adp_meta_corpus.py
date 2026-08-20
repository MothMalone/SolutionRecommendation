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
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd

# The proxy fits thousands of models; sklearn emits ConvergenceWarning per lbfgs fit and
# "Features [...] are constant" per feature-selection call on PCA'd input. Left on, they bury the
# per-dataset progress lines under thousands of lines of noise and make the run unreadable.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=DataConversionWarning)
except Exception:
    pass
np.seterr(divide="ignore", invalid="ignore")

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
class _NoSignal(Exception):
    """Dataset scored fine but every pipeline tied; carries the row to record."""

    def __init__(self, row):
        super().__init__(row.get("dataset_id", "?"))
        self.row = row


FAMILY_ORDER = ["imputation", "encoding", "scaling", "feature_selection",
                "dimensionality_reduction", "outlier_removal"]
MODEL_SLOT = "RF"          # constant: our proxy is LogReg; the slot exists only to keep 7 slots.
NULL = "none"


def sample_pipeline(rng: random.Random, *, has_missing: bool = False,
                    has_categorical: bool = False) -> tuple:
    """(ordered slot codes, config dict). Order is sampled too -- it is what the meta-learner learns.

    Turning a family off is only offered where the data allows it. The proxy rejects
    `imputation: none` on a frame with missing values ("missing values require non-'none'
    imputation") and cannot fit a model on unencoded categoricals, so sampling those blind wasted
    ~50% of every dataset's evaluations AND -- because a rejected pipeline was written as 0.0 --
    taught the meta-learner that a perfectly good pipeline was the worst one available.
    """
    order = FAMILY_ORDER[:]
    rng.shuffle(order)
    cfg, slots = {}, []
    for fam in order:
        may_skip = True
        if fam == "imputation" and has_missing:
            may_skip = False
        if fam == "encoding" and has_categorical:
            may_skip = False
        # A family is off ~25% of the time, mirroring the *_null slots in their label.csv.
        op = NULL if (may_skip and rng.random() < 0.25) else rng.choice(STEP_OPERATORS[fam])
        cfg[fam] = op
        slots.append(_code(fam, op))
    return slots + [MODEL_SLOT], cfg


def subsample_preserving_classes(df: pd.DataFrame, target_column: str, n: int,
                                 seed: int) -> pd.DataFrame:
    """Subsample to ~n rows without starving any class.

    A plain df.sample() drops rare classes below the proxy's 3-member minimum, at which point it
    returns no results at all and the dataset is lost -- observed on 184 (18 classes, 28k rows):
    "only 0/10 pipelines scored" in 0.03s, because nothing was ever evaluated. Take a floor of 3
    rows per class first, then fill the remainder at random.
    """
    if n <= 0 or len(df) <= n:
        return df
    y = df[target_column]
    counts = y.value_counts()
    floor = pd.concat([
        df[y == cls].sample(n=min(3, int(cnt)), random_state=seed)
        for cls, cnt in counts.items()
    ])
    if len(floor) >= n:
        return floor                       # more classes than the budget; keep the floor intact
    remainder = df.drop(index=floor.index)
    extra = remainder.sample(n=min(n - len(floor), len(remainder)), random_state=seed)
    return pd.concat([floor, extra]).sample(frac=1.0, random_state=seed)


def describe_frame(df: pd.DataFrame, target_column: str) -> dict:
    X = df.drop(columns=[target_column], errors="ignore")
    return {
        "has_missing": bool(X.isna().to_numpy().any()),
        "has_categorical": bool(any(X[c].dtype == object or str(X[c].dtype) == "category"
                                    for c in X.columns)),
    }


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
    # Keep classification datasets only. The corpus feeds their CLASSIFICATION meta-learner, and
    # 1466 of the library's 2560 rows (57%) are regression -- sampling blind spent well over half
    # of every build discovering that per dataset, at a download and a load each.
    if "NumberOfClasses" in feats.columns:
        before = len(feats)
        feats = feats[feats["NumberOfClasses"] >= 2]
        print(f"[corpus] pool restricted to classification: {before} -> {len(feats)} datasets")
    # Drop extreme-width frames. --score-max-rows caps rows, but the proxy's cost is driven by
    # COLUMNS: dataset 4136 (dexter, ~20k features) burned 1444s and scored nothing. 11 of a
    # 120-dataset sample sat above 5k features, up to 54,614 -- some hours of guaranteed waste.
    # They also cannot help: the widest evaluation dataset is madelon at 500 features, so a
    # 50k-feature corpus row is never the nearest neighbour of anything we evaluate.
    if args.max_features and "NumberOfFeatures" in feats.columns:
        before = len(feats)
        feats = feats[feats["NumberOfFeatures"] <= args.max_features]
        print(f"[corpus] pool restricted to <= {args.max_features} features: "
              f"{before} -> {len(feats)} datasets")
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
    ap.add_argument("--split-seeds", type=lambda v: [int(x) for x in v.split(",")],
                    default=[42, 52, 62],
                    help="proxy train/val splits to average per pipeline. More seeds cost time but "
                         "give finer resolution -- a single small split ties many pipelines at the "
                         "same accuracy, which carries no signal.")
    ap.add_argument("--min-distinct-scores", type=int, default=3,
                    help="drop a dataset whose sampled pipelines produce fewer distinct scores "
                         "than this; it cannot teach a task order.")
    ap.add_argument("--max-sample-rounds", type=int, default=2,
                    help="oversampling rounds allowed to reach --pipelines-per-dataset valid "
                         "scores. Each round costs a full batch of proxy fits, and failures are "
                         "what trigger it -- onehot on a high-cardinality column can widen a frame "
                         "to hundreds of features, after which outlier removal deletes every row. "
                         "Two rounds bounds the damage; a dataset that still cannot fill its block "
                         "is recorded as a failure rather than chased.")
    ap.add_argument("--max-features", type=int, default=1000,
                    help="skip library datasets wider than this. Proxy cost scales with columns, "
                         "and frames far wider than any evaluation dataset cannot be retrieved as "
                         "a neighbour anyway. 0 disables.")
    ap.add_argument("--score-max-rows", type=int, default=1500,
                    help="subsample to at most this many rows for the PROXY SCORING only; "
                         "metafeatures still come from the full cached frame. 0 disables.")
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="stop starting new datasets after this many seconds and assemble what "
                         "finished. Use it to fit the build inside a fixed session.")
    ap.add_argument("--allow-download", action="store_true",
                    help="Sample from the whole library and fetch missing tables from OpenML. "
                         "Off by default: selection is restricted to <id>.csv already on disk, "
                         "so the build cannot fail wholesale on a blocked or partial network.")
    ap.add_argument("--merge", action="store_true",
                    help="skip generation; assemble Metafeature.csv/label.csv from progress.jsonl")
    ap.add_argument("--merge-from", action="append", default=[],
                    help="additional corpus dir(s) whose progress.jsonl and datasets/ to fold in. "
                         "Repeatable. Use when shards ran in separate notebooks, each having "
                         "written its own progress.jsonl.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "datasets"
    cache.mkdir(exist_ok=True)
    progress = out / "progress.jsonl"

    # Fold in sibling shard dirs first, so their rows are visible both to --merge and to the
    # resume check. Later sources win only if the earlier row was not a success.
    done = {}
    extra_caches = []
    for src in list(args.merge_from) + [str(out)]:
        src_dir = Path(src)
        extra_caches.append(src_dir / "datasets")
        pfile = src_dir / "progress.jsonl"
        if not pfile.exists():
            if src_dir != out:
                print(f"[corpus] warning: no progress.jsonl in {src_dir}")
            continue
        n = 0
        for line in pfile.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            prev = done.get(r["dataset_id"])
            if prev is None or prev.get("status") != "ok":
                done[r["dataset_id"]] = r
            n += 1
        if src_dir != out:
            print(f"[corpus] folded in {n} row(s) from {pfile}")

    local_dirs = [Path(d) for d in args.local_dir] + [cache]
    if not args.merge:
        ids = choose_ids(args, local_dirs)
        todo = [d for d in ids if d not in done]
        print(f"[corpus] {len(ids)} datasets selected, {len(done)} already done, {len(todo)} to run")

        budget_t0 = time.time()
        with progress.open("a") as fh:
            for i, ds in enumerate(todo, 1):
                if args.time_budget and (time.time() - budget_t0) > args.time_budget:
                    print(f"[corpus] time budget of {args.time_budget:.0f}s reached after "
                          f"{i - 1} dataset(s); assembling what completed. The corpus is smaller "
                          f"than requested but valid -- rerun with a larger --time-budget to grow it.")
                    break
                t0 = time.time()
                try:
                    df = load_table(ds, local_dirs, args.target_column)
                    df.to_csv(cache / f"{ds}.csv", index=False)

                    # The corpus feeds their CLASSIFICATION meta-learner (label.csv); regression
                    # has its own label_reg.csv. A continuous target makes every pipeline fail with
                    # "Unknown label type: continuous" -- two full sampling rounds spent to learn
                    # what the target column already says.
                    # Ask sklearn, do not guess. A float target with as few as 12 distinct values
                    # is "continuous" to type_of_target -- the exact function that raises inside
                    # the proxy -- so a dtype+cardinality heuristic lets those through and every
                    # sampled pipeline then dies with "Unknown label type: continuous".
                    from sklearn.utils.multiclass import type_of_target
                    try:
                        target_kind = type_of_target(df[args.target_column])
                    except Exception:
                        target_kind = "unknown"
                    if target_kind.startswith("continuous"):
                        row = {"dataset_id": ds, "status": "regression",
                               "shape": f"{df.shape[0]}*{df.shape[1]}",
                               "seconds": round(time.time() - t0, 2)}
                        raise _NoSignal(row)

                    rng = random.Random(f"{args.seed}:{ds}")
                    # Score on a subsample when the frame is large. Metafeatures are computed
                    # later from the CACHED FULL csv, so row count -- their metafeature #1 -- stays
                    # honest; only the proxy fit is cheapened. Without this a single 4000x60
                    # dataset costs ~390s and a 12h session cannot hold both the corpus and the arm.
                    scoring_df = subsample_preserving_classes(
                        df, args.target_column, args.score_max_rows, args.seed)
                    shape = describe_frame(scoring_df, args.target_column)
                    k = args.pipelines_per_dataset

                    # Oversample and keep the first k that actually score. A rejected pipeline is
                    # dropped, never written as 0.0 -- a failed evaluation is missing information,
                    # not evidence that the pipeline is bad.
                    keep_slots, keep_scores, attempts = [], [], 0
                    while len(keep_slots) < k and attempts < args.max_sample_rounds:
                        attempts += 1
                        want = k - len(keep_slots)
                        batch = [sample_pipeline(rng, **shape) for _ in range(want * 2)]
                        _, _, results, _ = evaluate_candidates_simple(
                            scoring_df, args.target_column, [c for _, c in batch],
                            proxy_settings={"model": "logreg",
                                            "split_seeds": args.split_seeds},
                        )
                        by_cfg = {json.dumps(c, sort_keys=True): sc for c, sc in results}
                        gained = 0
                        for slot, cfg in batch:
                            sc = by_cfg.get(json.dumps(cfg, sort_keys=True))
                            if sc is not None and np.isfinite(sc) and len(keep_slots) < k:
                                keep_slots.append(slot)
                                keep_scores.append(float(sc))
                                gained += 1
                        if gained == 0:
                            # Nothing scored at all. The causes are structural, not random -- a
                            # sparse/wide frame the proxy cannot reshape ("Shape of passed values
                            # is (9, 1), indices imply (9, 1024)"), or one that needs
                            # with_mean=False. Another identical round will fail identically, so
                            # stop instead of paying for it twice.
                            break

                    if len(keep_slots) < k:
                        raise RuntimeError(
                            f"only {len(keep_slots)}/{k} pipelines scored after "
                            f"{attempts} sampling round(s)")
                    # A dataset whose pipelines all tie teaches the meta-learner nothing: their
                    # 1-NN takes idxmax of the neighbour's block, so with every score equal the
                    # "best" pipeline is just whichever came first. Exclude it rather than let it
                    # contribute an arbitrary task order.
                    if len(set(round(x, 6) for x in keep_scores)) < args.min_distinct_scores:
                        row = {"dataset_id": ds, "status": "no_signal",
                               "shape": f"{df.shape[0]}*{df.shape[1]}",
                               "distinct_scores": len(set(round(x, 6) for x in keep_scores)),
                               "seconds": round(time.time() - t0, 2)}
                        raise _NoSignal(row)

                    row = {"dataset_id": ds, "status": "ok",
                           "shape": f"{df.shape[0]}*{df.shape[1]}",
                           "pipelines": [",".join(sl) for sl in keep_slots],
                           "scores": keep_scores,
                           "distinct_scores": len(set(round(x, 6) for x in keep_scores)),
                           "seconds": round(time.time() - t0, 2)}
                except _NoSignal as skip:
                    row = skip.row
                except Exception as exc:
                    row = {"dataset_id": ds, "status": "fail",
                           "error": f"{type(exc).__name__}: {exc}",
                           "seconds": round(time.time() - t0, 2)}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                done[ds] = row
                mark = {"ok": "ok  ", "no_signal": "SKIP", "regression": "SKIP",
                        "fail": "FAIL"}[row["status"]]
                extra = row.get("error", "")
                if row["status"] == "regression":
                    extra = "continuous target -- classification corpus only, excluded"
                elif row["status"] == "no_signal":
                    extra = f"all {row['distinct_scores']} distinct score(s) -- no signal, excluded"
                elif row["status"] == "ok":
                    extra = f"{row['distinct_scores']} distinct scores"
                print(f"  [{i}/{len(todo)}] {mark} {ds}  {extra} ({row['seconds']}s)")

    # ---- assemble ----
    ok = [r for r in done.values() if r.get("status") == "ok"]
    if not ok:
        print("[corpus] nothing succeeded; not writing CSVs")
        return 1
    ok.sort(key=lambda r: r["dataset_id"])
    assert_disjoint([r["dataset_id"] for r in ok], context="adp meta-corpus output")

    def _cached(dsid: str) -> Path:
        for base in [cache] + extra_caches:
            cand = base / f"{dsid}.csv"
            if cand.exists():
                return cand
        return cache / f"{dsid}.csv"        # reported as missing below

    files = [_cached(r["dataset_id"]) for r in ok]
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
        if len(pipes) != k:
            raise ValueError(f"{r['dataset_id']}: {len(pipes)} pipelines, expected exactly {k}")
        for p, s in zip(pipes, scores):
            pid += 1
            rows.append({"Id": pid, "DatasetName": r["dataset_id"], "Target": "target",
                         "Pipeline": p,
                         "EvaluationMetric": float(s),
                         "Time": 0.0, "Size": r.get("shape", ""), "Website": "acorec-retrained"})
    pd.DataFrame(rows).to_csv(out / "label.csv", index=False)

    # Every shard notebook builds this corpus independently -- artifacts cannot be passed
    # between Kaggle notebooks here. If two shards end up with different corpora, their arm
    # results are not comparable and nothing downstream would notice. The fingerprint is over the
    # dataset ids and their pipeline/score blocks, so a differing corpus is caught by eye.
    import hashlib

    h = hashlib.sha256()
    for r in ok:
        h.update(r["dataset_id"].encode())
        for pipe, sc in zip(r["pipelines"][:k], r["scores"][:k]):
            h.update(pipe.encode())
            h.update(f"{sc:.6f}".encode())
    fingerprint = h.hexdigest()[:16]
    (out / "corpus_fingerprint.txt").write_text(fingerprint + "\n")

    print(f"[corpus] wrote {out/'Metafeature.csv'}  ({meta.shape[0]} x {meta.shape[1]})")
    print(f"[corpus] wrote {out/'label.csv'}        ({len(rows)} rows, {k} per dataset)")
    n_fail = sum(1 for r in done.values() if r.get("status") == "fail")
    n_skip = sum(1 for r in done.values()
                 if r.get("status") in ("no_signal", "regression"))
    print(f"[corpus] {len(ok)} datasets usable | {n_fail} failed | {n_skip} dropped (no signal)")
    print()
    print(f"CORPUS FINGERPRINT  {fingerprint}  ({len(ok)} datasets x {k} pipelines)")
    print("Every shard must print this same value, or its arm results are not comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
