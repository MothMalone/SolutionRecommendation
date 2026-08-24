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
REF_FLAGS = [
    "--train-metric-inline", "--metric-loss", "pearson", "--metric-weight-decay", "1e-4",
    "--metric-objective", "embedding_cosine", "--aco-mmas-bounds", "--aco-weight-method", "linear",
    "--hybrid-select", "--proxy-seeds", "42,52,62", "--cv-select-folds", "3",
    "--optimizer", "aco", "--require-autogluon",
]


def arm_flags(arm: str, args) -> list:
    """The ONLY difference between the two arms. Everything else is shared by construction."""
    if arm == "ref":
        return ["--n-ants", str(args.ref_ants), "--n-iterations", str(args.ref_iters),
                "--final-autogluon-topk", "1"]
    return [
        "--n-ants", str(args.ladder_ants), "--n-iterations", str(args.ladder_iters),
        "--final-autogluon-topk", str(args.ladder_topk),
        "--screen-topk", str(args.screen_topk),
        "--screen-profile", args.screen_profile,
        "--screen-time-limit", str(args.screen_time_limit),
        "--hybrid-no-search-neighbor-k", "5", "--hybrid-no-search-top-l", "1",
    ]


def run_one(arm: str, did: str, holdout: list, args) -> dict:
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
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(str(r["dataset_id"]), {})[r["arm"]] = r

    print(f"{'dataset':>9} | {'REF':>8} {'sel':>18} {'s':>6} | {'LADDER':>8} {'sel':>18} {'s':>6} | {'delta':>7}")
    print("-" * 96)
    deltas, wins, losses, ties = [], 0, 0, 0
    for did in sorted(by, key=lambda s: (len(s), s)):
        a, b = by[did].get("ref"), by[did].get("ladder")
        if not a or not b:
            continue
        sa, sb = a.get("score"), b.get("score")
        ta = f"{sa:.4f}" if isinstance(sa, float) else str(a.get("status"))
        tb = f"{sb:.4f}" if isinstance(sb, float) else str(b.get("status"))
        if isinstance(sa, float) and isinstance(sb, float):
            d = sb - sa
            deltas.append(d)
            wins += d > 1e-9
            losses += d < -1e-9
            ties += abs(d) <= 1e-9
            dt = f"{d:+.4f}"
        else:
            dt = "--"
        print(f"{did:>9} | {ta:>8} {str(a.get('selected_candidate'))[:18]:>18} {a.get('seconds',0):>6.0f} "
              f"| {tb:>8} {str(b.get('selected_candidate'))[:18]:>18} {b.get('seconds',0):>6.0f} | {dt:>7}")

    if deltas:
        import statistics as st
        mean = sum(deltas) / len(deltas)
        print("-" * 96)
        print(f"\nn={len(deltas)}  mean delta={mean:+.4f}  median={st.median(deltas):+.4f}")
        print(f"ladder wins {wins}, loses {losses}, ties {ties}")
        if len(deltas) > 1:
            sd = st.stdev(deltas)
            se = sd / (len(deltas) ** 0.5)
            print(f"sd={sd:.4f}  se={se:.4f}  mean/se={mean/se if se else float('nan'):+.2f}")
            print("\n(mean/se is a paired t-statistic. |t| < 2 on this many datasets means the "
                  "difference is\n not separable from noise -- report it as such rather than as a win.)")
        # Cost is half the claim: the ladder is only worth it if it stays inside the time budget.
        ref_t = [by[d]["ref"].get("seconds", 0) for d in by if "ref" in by[d]]
        lad_t = [by[d]["ladder"].get("seconds", 0) for d in by if "ladder" in by[d]]
        if ref_t and lad_t:
            print(f"\nmean runtime: REF {sum(ref_t)/len(ref_t):.0f}s  "
                  f"LADDER {sum(lad_t)/len(lad_t):.0f}s  "
                  f"({sum(lad_t)/max(sum(ref_t), 1e-9):.2f}x)")
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

    for did in ids:
        for arm in arms:
            if (arm, did) in done:
                print(f"  skip {arm}/{did} (already done)")
                continue
            print(f"  {arm}/{did} ...", flush=True)
            try:
                row = run_one(arm, did, ids, args)
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
