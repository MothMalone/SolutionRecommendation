#!/usr/bin/env python3
"""Head-to-head: REF vs the multi-fidelity ladder, on reference-holdout datasets.

THIS IS THE VALIDATION THE LADDER NEEDS BEFORE IT GOES NEAR THE EVALUATION SETS.
`docs/SIGNAL_DIAGNOSIS.md` argues the ladder from a simulation. A simulation is a model of the
selection process, not a measurement of it, and its 3-rung numbers are explicitly an upper bound
(the rungs there are independent observers; in the implementation they share a validation split).
This script measures the thing for real.

PROTOCOL, and the parts that make the number mean something:

  * Datasets come from ``data/adp_ourops_corpus/datasets`` -- verified disjoint from both EVAL_IDS
    and THEIR_DATASETS. Nothing here touches the 30 evaluation datasets, so the result is a
    legitimate basis for choosing a configuration.
  * EVERY dataset in the comparison is passed to ``--holdout-ids`` on EVERY run, not just the one
    being scored. These datasets ARE in the reference library, so without that the inline Siamese
    would train on the performance profile of the very dataset it is about to recommend for.
  * Both arms run the SAME seed, the SAME splits and the SAME AutoGluon profile. The only thing
    that varies is the selection ladder, which is the whole point of an ablation.
  * ``--require-autogluon`` is on: a fallback to the proxy evaluator would silently produce a
    number that is not an AutoGluon score, and the two arms would stop being comparable.

Local-machine defaults: ``local_rf_xt`` (best_quality segfaults under macOS OpenMP), short time
limits, one thread. These make the ABSOLUTE scores lower than a Kaggle run would give; the
comparison between the two arms is what transfers, not the level.

    python scripts/validate_ladder.py --datasets 853,1510,1452 --out outputs/ladder_val.jsonl
    python scripts/validate_ladder.py --summarize outputs/ladder_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "data" / "adp_ourops_corpus" / "datasets"

sys.path.insert(0, str(REPO / "src"))
from automl_aco.eval_ids import EVAL_IDS  # noqa: E402

THEIR_DATASETS = {"42493", "43723", "8335", "40945", "1461", "31", "42178", "184", "40701", "1590"}

# The deployed configuration, minus the AutoGluon profile and budget, which the caller sets so the
# same script can run locally and on Kaggle. Mirrors ACOREC_REF_FLAGS in run_arms.py.
#
# The metric flags are NOT here: run_recommend rejects --metric-path together with
# --train-metric-inline, and this script trains the Siamese ONCE and reuses it (see
# train_shared_metric). run_recommend itself builds the recommender once before its dataset loop,
# so a batched production run pays that cost once too -- it is one subprocess per dataset here that
# would otherwise repeat it, which is an artifact of the harness rather than of ACORec, and it
# would put ~3 identical minutes into both arms' runtimes and make the cost comparison meaningless.
REF_FLAGS = [
    "--metric-loss", "pearson", "--metric-weight-decay", "1e-4",
    "--metric-objective", "embedding_cosine", "--aco-mmas-bounds", "--aco-weight-method", "linear",
    "--hybrid-select", "--proxy-seeds", "42,52,62", "--cv-select-folds", "3",
    "--optimizer", "aco", "--require-autogluon",
]


def train_shared_metric(holdout: list, args) -> str:
    """Train the Siamese ONCE against this comparison's holdout, and reuse it for every run.

    Safe precisely because the holdout is identical for every run in the grid: all the datasets
    being compared are excluded from the reference on all of them, so one metric is the same
    object each run would have trained for itself. Both arms then load byte-identical weights,
    which also removes metric-training variance as a confound between them.
    """
    metric_path = Path(args.scratch) / "shared_metric.pt"
    if metric_path.exists() and not args.retrain_metric:
        print(f"[validate] reusing metric {metric_path}")
        return str(metric_path)

    probe = holdout[0]
    cmd = [
        sys.executable, str(REPO / "scripts" / "run_recommend.py"),
        "--dataset-source", "csv",
        "--dataset-csv", str(CORPUS / f"{probe}.csv"),
        "--target-column", "target", "--dataset-id", str(probe),
        "--holdout-ids", ",".join(holdout),
        "--operator-space", "ours",
        "--train-metric-inline",
        "--metric-loss", "pearson", "--metric-weight-decay", "1e-4",
        "--metric-objective", "embedding_cosine",
        "--seed", str(args.seed),
        "--save-trained-metric", str(metric_path),
        # Cheapest possible run that still reaches the save: the recommendation is thrown away,
        # only the trained weights are wanted.
        "--baseline-only", "no_preprocessing",
        "--autogluon-profile", "local_rf_xt", "--time-limit", "15",
        "--output-dir", str(Path(args.scratch) / "_metric_train"),
    ]
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    print(f"[validate] training the shared Siamese metric (once) ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=3600)
    if not metric_path.exists():
        raise SystemExit(f"metric training produced no file:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    print(f"[validate] metric trained in {time.time() - t0:.0f}s -> {metric_path}")
    return str(metric_path)


def arm_flags(arm: str, args) -> list:
    """The ONLY difference between arms. Everything else is shared by construction.

    `ladder` changes TWO things at once -- the selection ladder (wider search, screening rung,
    K=5) and the no-search arm's neighbour aggregation (5/1 instead of the hardcoded 1/1). The
    first partial results made that a problem rather than a detail: on dataset 49 both arms
    selected `no_search_retrieval` and still scored differently (0.8000 vs 0.8333), which ONLY
    the aggregation change can explain. A `ladder - ref` delta therefore cannot be attributed.

    `screen_only` and `retrieval_only` split it:

        retrieval_only - ref   = the neighbour aggregation alone
        screen_only    - ref   = the wider search + screening rung + K=5 alone
        ladder         - ref   = both, and whether they add up
    """
    ladder_search = [
        "--n-ants", str(args.ladder_ants), "--n-iterations", str(args.ladder_iters),
        "--final-autogluon-topk", str(args.ladder_topk),
        "--screen-topk", str(args.screen_topk),
        "--screen-profile", args.screen_profile,
        "--screen-time-limit", str(args.screen_time_limit),
    ]
    ref_search = ["--n-ants", str(args.ref_ants), "--n-iterations", str(args.ref_iters),
                  "--final-autogluon-topk", "1"]
    agg_5 = ["--hybrid-no-search-neighbor-k", "5", "--hybrid-no-search-top-l", "1"]
    agg_1 = ["--hybrid-no-search-neighbor-k", "1", "--hybrid-no-search-top-l", "1"]

    return {
        "ref":            ref_search + agg_1,
        "ladder":         ladder_search + agg_5,
        "screen_only":    ladder_search + agg_1,
        "retrieval_only": ref_search + agg_5,
    }[arm]


def run_one(arm: str, did: str, holdout: list, args, metric_path: str) -> dict:
    workdir = Path(args.scratch) / f"{arm}_{did}"
    cmd = [
        sys.executable, str(REPO / "scripts" / "run_recommend.py"),
        "--dataset-source", "csv",
        "--dataset-csv", str(CORPUS / f"{did}.csv"),
        "--target-column", "target",
        "--dataset-id", str(did),
        # Every dataset in the comparison, not just this one: they are all in the reference
        # library, and the inline Siamese would otherwise train on their performance profiles.
        "--holdout-ids", ",".join(holdout),
        "--operator-space", "ours",
        "--use-aco",
        "--prepare-mode", "leakfree",
        "--time-limit", str(args.time_limit),
        "--seed", str(args.seed),
        "--autogluon-profile", args.autogluon_profile,
        "--metric-path", metric_path,   # identical weights in both arms; see train_shared_metric
        "--output-dir", str(workdir),
    ] + REF_FLAGS + arm_flags(arm, args)

    env = dict(os.environ)
    # AutoGluon + OpenMP on macOS segfaults under thread contention; this is what makes a local
    # run finish instead of dying halfway through the grid.
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=args.per_run_timeout)
    elapsed = round(time.time() - t0, 1)

    row = {"arm": arm, "dataset_id": did, "seconds": elapsed, "returncode": proc.returncode}
    rec = workdir / "recommendation.json"
    if proc.returncode != 0 or not rec.exists():
        row.update(status="failed",
                   error=(proc.stderr or proc.stdout or "")[-600:])
        return row

    data = json.loads(rec.read_text())
    final = data.get("final_evaluation", {}) or {}
    cfg = dict(data.get("pipeline_config", {}) or {})
    name = str(cfg.get("name", "") or "")
    steps = {k: v for k, v in cfg.items() if k not in ("name", "step_order")}
    active = {k: v for k, v in steps.items() if str(v) != "none" and k != "encoding"}
    if name.startswith("no_search_retrieval"):
        selected = "no_search_retrieval"
    elif name.startswith("light_"):
        selected = name
    elif not active:
        selected = "no_preprocessing"
    else:
        selected = "aco"
    row.update(
        status="ok",
        score=final.get("score"),
        eval_method=final.get("method"),
        selected_candidate=selected,
        n_gate_candidates=len(data.get("ag_candidate_scores") or {}) or None,
        pipeline={k: v for k, v in steps.items()},
    )
    return row


def summarize(path: str) -> int:
    import statistics as st
    from collections import Counter

    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(str(r["dataset_id"]), {})[r["arm"]] = r
    arms = [a for a in ("ref", "retrieval_only", "screen_only", "ladder")
            if any(a in v for v in by.values())]
    others = [a for a in arms if a != "ref"]

    hdr = f"{'dataset':>9} | {'ref':>8}"
    for a in others:
        hdr += f" | {a[:14]:>14} {'delta':>8}"
    print(hdr)
    print("-" * len(hdr))

    deltas = {a: [] for a in others}
    for did in sorted(by, key=lambda s: (len(s), s)):
        base = by[did].get("ref")
        if not base or not isinstance(base.get("score"), float):
            continue
        line = f"{did:>9} | {base['score']:>8.4f}"
        for a in others:
            r = by[did].get(a)
            if r and isinstance(r.get("score"), float):
                d = r["score"] - base["score"]
                deltas[a].append(d)
                line += f" | {r['score']:>14.4f} {d:>+8.4f}"
            else:
                line += f" | {'--':>14} {'--':>8}"
        print(line)
    print("-" * len(hdr))

    print("\n=== paired deltas vs REF ===")
    for a in others:
        d = deltas[a]
        if not d:
            continue
        mean = sum(d) / len(d)
        w = sum(1 for x in d if x > 1e-9)
        l = sum(1 for x in d if x < -1e-9)
        t = len(d) - w - l
        line = (f"  {a:15} n={len(d):<3} mean={mean:+.4f} median={st.median(d):+.4f} "
                f"W/L/T={w}/{l}/{t}")
        if len(d) > 1:
            se = st.stdev(d) / (len(d) ** 0.5)
            line += f"  t={mean/se if se else float('nan'):+.2f}"
        print(line)
    print("\n  (t is a paired t-statistic. |t| < 2 means the difference is NOT separable from "
          "noise on\n   this many datasets -- report it that way rather than as a win.)")

    if "ladder" in deltas and "screen_only" in deltas and "retrieval_only" in deltas:
        ml = sum(deltas["ladder"]) / max(len(deltas["ladder"]), 1)
        ms = sum(deltas["screen_only"]) / max(len(deltas["screen_only"]), 1)
        mr = sum(deltas["retrieval_only"]) / max(len(deltas["retrieval_only"]), 1)
        print(f"\n=== attribution ===")
        print(f"  retrieval aggregation alone : {mr:+.4f}")
        print(f"  search ladder alone         : {ms:+.4f}")
        print(f"  both together               : {ml:+.4f}   (sum of parts {ms + mr:+.4f})")
        if abs(ml) > 1e-9 and abs(mr) > abs(ml) * 0.6:
            print("  -> most of the combined effect is the RETRIEVAL AGGREGATION, a one-line "
                  "config change,\n     not the screening ladder that cost the extra runtime.")

    print("\n=== runtime ===")
    for a in arms:
        ts = [by[d][a].get("seconds", 0) for d in by if a in by[d]]
        rt = [by[d]["ref"].get("seconds", 0) for d in by if a in by[d] and "ref" in by[d]]
        if ts:
            ratio = sum(ts) / max(sum(rt), 1e-9) if rt else 1.0
            print(f"  {a:15} mean={sum(ts)/len(ts):>6.0f}s   {ratio:.2f}x REF")

    print("\n=== what the gate actually selected ===")
    for a in arms:
        c = Counter(by[d][a].get("selected_candidate") for d in by
                    if a in by[d] and by[d][a].get("status") == "ok")
        print(f"  {a:15} " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    print("  (if 'aco' is rare, the SEARCH is not what is winning the gate -- the transfer arm "
          "and\n   the floor are, and widening the search cannot help that.)")

    bad = [r for r in rows if r.get("status") == "ok" and r.get("eval_method") != "autogluon"]
    if bad:
        print(f"\n!! {len(bad)} row(s) are NOT AutoGluon scores "
              f"({sorted({str(r.get('eval_method')) for r in bad})}) -- not comparable")
    failed = [r for r in rows if r.get("status") != "ok"]
    if failed:
        print(f"\n!! {len(failed)} failed run(s): "
              + ", ".join(f"{r['arm']}/{r['dataset_id']}" for r in failed))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="", help="comma-separated corpus dataset ids")
    ap.add_argument("--out", default="outputs/ladder_validation.jsonl")
    ap.add_argument("--summarize", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-limit", type=int, default=60, help="AutoGluon seconds per fit")
    ap.add_argument("--autogluon-profile", default="local_rf_xt",
                    choices=["best_quality", "medium_quality", "local_rf_xt"],
                    help="local_rf_xt by default: best_quality segfaults under macOS OpenMP")
    ap.add_argument("--ref-ants", type=int, default=4)
    ap.add_argument("--ref-iters", type=int, default=3)
    ap.add_argument("--ladder-ants", type=int, default=8)
    ap.add_argument("--ladder-iters", type=int, default=6)
    ap.add_argument("--ladder-topk", type=int, default=5)
    ap.add_argument("--screen-topk", type=int, default=20)
    ap.add_argument("--screen-profile", default="local_rf_xt")
    ap.add_argument("--screen-time-limit", type=int, default=20)
    ap.add_argument("--per-run-timeout", type=int, default=3600)
    ap.add_argument("--scratch", default="/tmp/ladder_val")
    ap.add_argument("--arms", default="ref,ladder")
    ap.add_argument("--retrain-metric", action="store_true",
                    help="retrain the shared Siamese even if the cached file exists")
    args = ap.parse_args()

    if args.summarize:
        return summarize(args.summarize)

    ids = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not ids:
        ap.error("--datasets is required")
    bad = [d for d in ids if d in {str(e) for e in EVAL_IDS} or d in THEIR_DATASETS]
    if bad:
        ap.error(f"{bad} are evaluation datasets. Tuning on them is exactly what this script "
                 f"exists to avoid.")
    missing = [d for d in ids if not (CORPUS / f"{d}.csv").exists()]
    if missing:
        ap.error(f"no CSV in {CORPUS} for {missing}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.scratch).mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    done.add((r["arm"], str(r["dataset_id"])))
            except Exception:
                pass

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"[validate] {len(ids)} dataset(s) x {len(arms)} arm(s), "
          f"profile={args.autogluon_profile}, time_limit={args.time_limit}s")
    print(f"[validate] holdout covers all {len(ids)} datasets on every run")

    metric_path = train_shared_metric(ids, args)

    for did in ids:
        for arm in arms:
            if (arm, did) in done:
                print(f"  skip {arm}/{did} (already done)")
                continue
            print(f"  {arm}/{did} ...", flush=True)
            try:
                row = run_one(arm, did, ids, args, metric_path)
            except subprocess.TimeoutExpired:
                row = {"arm": arm, "dataset_id": did, "status": "timeout",
                       "seconds": args.per_run_timeout}
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"    -> {row.get('status')} score={row.get('score')} "
                  f"sel={row.get('selected_candidate')} {row.get('seconds')}s", flush=True)

    print(f"\n[validate] wrote {out}")
    return summarize(str(out))


if __name__ == "__main__":
    raise SystemExit(main())
