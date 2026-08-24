#!/usr/bin/env python3
"""Measure the CV gate's real fidelity: Spearman between its validation and test scores.

This is the number the ladder's K setting hangs on, and until now it was ASSUMED.

`docs/SIGNAL_DIAGNOSIS.md` shows the gain from letting more candidates reach the gate (K) rising
with K -- but only because the simulation holds gate fidelity constant in K. It is not constant.
The gate selects on a validation split, so ranking more candidates on it is the same winner's-curse
mechanism the proxy axis suffers from. Sweeping the assumed fidelity showed the optimum moving:
K~20 at rho=0.90, K~5 at 0.55, K~3 at 0.40, with K=10 falling BELOW K=5 in the low cases.

So the honest question is: what IS the gate's rho, and how does it vary with validation-set size?
That is measurable without touching a single evaluation dataset -- run a spread of candidates
through the gate on reference-holdout datasets and correlate the val scores it selects on against
the test scores it reports.

Rows are per (dataset, n_candidates_available). A dataset with 87 rows has a ~17-row validation
block; if rho collapses there, the ladder needs to be size-gated rather than applied uniformly.

    python scripts/measure_gate_fidelity.py --datasets 786,853,1452 --out outputs/gate_rho.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "data" / "adp_ourops_corpus" / "datasets"
sys.path.insert(0, str(REPO / "src"))

from automl_aco.eval_ids import EVAL_IDS  # noqa: E402

THEIR_DATASETS = {"42493", "43723", "8335", "40945", "1461", "31", "42178", "184", "40701", "1590"}


def candidate_grid(n: int, seed: int = 42) -> list:
    """A spread of genuinely different pipelines.

    Deliberately diverse rather than ACO-like: the question is how well the gate ORDERS
    candidates, which needs candidates whose true scores actually differ. A grid of near-identical
    pipelines would measure tie-breaking noise instead.
    """
    imput = ["none", "mean", "median", "most_frequent"]
    scale = ["none", "standard", "minmax", "robust"]
    fsel = ["none", "variance_threshold", "mutual_info"]
    dred = ["none", "pca", "svd"]
    outl = ["none", "zscore", "iqr"]
    combos = list(itertools.product(imput, scale, fsel, dred, outl))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(combos), size=min(n, len(combos)), replace=False)
    out = []
    for j, i in enumerate(idx):
        im, sc, fs, dr, ou = combos[i]
        out.append({
            "name": f"grid{j}", "imputation": im, "scaling": sc, "encoding": "onehot",
            "feature_selection": fs, "dimensionality_reduction": dr, "outlier_removal": ou,
            "step_order": ["imputation", "encoding", "scaling", "feature_selection",
                           "outlier_removal", "dimensionality_reduction"],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--out", default="outputs/gate_fidelity.jsonl")
    ap.add_argument("--n-candidates", type=int, default=10)
    ap.add_argument("--time-limit", type=int, default=30)
    ap.add_argument("--autogluon-profile", default="local_rf_xt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.summarize:
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        ok = [r for r in rows if r.get("spearman") is not None]
        if not ok:
            print("no usable rows")
            return 1
        print(f"{'dataset':>9} {'rows':>7} {'val_rows':>9} {'n_cand':>7} {'spearman':>9} {'top1_regret':>12}")
        print("-" * 62)
        for r in sorted(ok, key=lambda r: r["n_rows"]):
            print(f"{r['dataset_id']:>9} {r['n_rows']:>7} {r['val_rows']:>9} "
                  f"{r['n_scored']:>7} {r['spearman']:>+9.3f} {r['top1_regret']:>12.4f}")
        rho = np.array([r["spearman"] for r in ok], dtype=float)
        print("-" * 62)
        print(f"\nmean gate rho = {np.nanmean(rho):+.3f}   median = {np.nanmedian(rho):+.3f}")
        small = [r for r in ok if r["val_rows"] < 120]
        large = [r for r in ok if r["val_rows"] >= 120]
        for lbl, grp in (("val_rows < 120", small), ("val_rows >= 120", large)):
            if grp:
                v = np.array([r["spearman"] for r in grp], dtype=float)
                print(f"  {lbl:18} n={len(grp):<3} mean rho={np.nanmean(v):+.3f}")
        print("\nCompare against docs/SIGNAL_DIAGNOSIS.md section 6: the optimal K was ~20 at "
              "rho=0.90,\n~5 at 0.55 and ~3 at 0.40. That mapping is what turns this number into "
              "a K setting.")
        return 0

    ids = [d.strip() for d in args.datasets.split(",") if d.strip()]
    bad = [d for d in ids if d in {str(e) for e in EVAL_IDS} or d in THEIR_DATASETS]
    if bad:
        ap.error(f"{bad} are evaluation datasets; this must run on reference-holdout data only")

    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    from scipy.stats import spearmanr

    from automl_aco.search.evaluation import evaluate_candidates_autogluon

    out.parent.mkdir(parents=True, exist_ok=True)
    cands = candidate_grid(args.n_candidates, seed=args.seed)

    for did in ids:
        df = pd.read_csv(CORPUS / f"{did}.csv")
        n = len(df)
        print(f"\n=== {did}: {n} rows x {df.shape[1]-1} features ===", flush=True)
        t0 = time.time()
        try:
            # select_on_val=True gives back results ordered by VAL score, carrying TEST scores --
            # exactly the two series whose agreement defines gate fidelity.
            _best, _bs, results, unsorted = evaluate_candidates_autogluon(
                dataset=df, target_column="target", candidate_configs=cands,
                time_limit_per_model=args.time_limit,
                autogluon_profile=args.autogluon_profile,
                select_on_val=True, prepare_mode="leakfree", verbose=False,
            )
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            with out.open("a") as fh:
                fh.write(json.dumps({"dataset_id": did, "n_rows": n, "status": "failed",
                                     "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            continue

        # `results` is val-ordered; its position IS the val rank. Test scores ride along.
        val_rank = list(range(len(results)))
        test_scores = [s for _c, s in results]
        rho = None
        if len(results) >= 3:
            # Negate: position 0 is the BEST val score, so rank ascends as quality descends.
            rho = float(spearmanr([-r for r in val_rank], test_scores).statistic)
        best_test = max(test_scores) if test_scores else float("nan")
        top1_regret = (best_test - test_scores[0]) if test_scores else float("nan")

        row = {
            "dataset_id": did, "n_rows": n, "n_features": int(df.shape[1] - 1),
            "val_rows": int(round(0.2 * n)), "n_scored": len(results),
            "spearman": rho, "top1_regret": top1_regret,
            "best_test": best_test, "gate_pick_test": test_scores[0] if test_scores else None,
            "seconds": round(time.time() - t0, 1),
            "status": "ok",
        }
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"  scored {len(results)}/{len(cands)}  gate rho={rho if rho is None else round(rho,3)}  "
              f"top1_regret={top1_regret:.4f}  ({row['seconds']}s)", flush=True)

    print(f"\nwrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
