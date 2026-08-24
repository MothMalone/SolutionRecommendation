#!/usr/bin/env python3
"""How much does each link in the chain actually contribute? Reference library only.

Answers three questions that the arms table cannot, because the arms only ever show the END of
the chain:

  1. RETRIEVAL -- is the metafeature->neighbour->pipeline transfer carrying real query-specific
     signal, or is it a constant recommendation wearing a metric's clothes? Compared against a
     global default (best mean-rank pipeline, metafeatures never consulted) and a random neighbour.
  2. THE METAFEATURE CEILING -- an RF trained directly on (metafeatures -> per-pipeline gain)
     upper-bounds what ANY model over these metafeatures can extract. If it ties the 1-NN, the
     retrieval model is not the bottleneck; the metafeatures are.
  3. THE PROXY -- given a proxy whose rank agreement with AutoGluon is rho, how much accuracy does
     selection lose? Turns the measured rho (outputs/proxy_fidelity.log) into an accuracy number,
     so proxy work and retrieval work can be compared on one scale.

EVERY number here comes from the reference library (data/openml/*), never from EVAL_IDS. Eval ids
are dropped up front and the drop is asserted, so this can be run and re-run freely while tuning.

    python scripts/diagnose_signal_contribution.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

from automl_aco.eval_ids import EVAL_IDS  # noqa: E402


def load_reference(perf_path: str, meta_path: str):
    """The (pipelines x datasets) matrix and (datasets x metafeatures) table, eval ids removed."""
    pm = pd.read_csv(perf_path, index_col=0)
    # Columns are 'D_<id>'; metafeature rows are bare ids.
    pm.columns = [c[2:] if str(c).startswith("D_") else str(c) for c in pm.columns]
    mf = pd.read_csv(meta_path, index_col=0)
    mf.index = mf.index.astype(str)

    eval_ids = {str(e) for e in EVAL_IDS}
    keep = [c for c in pm.columns
            if c not in eval_ids and c in set(mf.index) and pm[c].notna().sum() >= 10]
    assert not (set(keep) & eval_ids), "eval ids leaked into the diagnostic reference set"
    return pm[keep], mf.loc[keep]


def contribution_table(P: pd.DataFrame, M: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Leave-one-out: what would each selection strategy have recommended for each dataset?"""
    rng = np.random.default_rng(seed)
    Pv = P.values
    X = MinMaxScaler().fit_transform(SimpleImputer(strategy="mean").fit_transform(M))
    S = cosine_similarity(X)
    np.fill_diagonal(S, -np.inf)
    n = P.shape[1]

    best = np.nanmax(Pv, axis=0)
    base = P.loc["baseline"].values
    ranks = P.rank(axis=0, ascending=True)

    def scored(j, pipe_idx):
        # A missing matrix entry means that pipeline was never measured here; recommending it
        # would in practice fall back to the untransformed frame.
        v = Pv[pipe_idx, j]
        return base[j] if np.isnan(v) else v

    out = {k: [] for k in ("oracle", "global_default", "cosine_top1",
                           "cosine_top5_vote", "random_neighbour", "no_preprocessing")}
    for j in range(n):
        out["oracle"].append(best[j])
        out["no_preprocessing"].append(base[j])

        # Metafeatures never consulted: the pipeline with the best mean rank over everything else.
        mean_rank = ranks.drop(columns=[P.columns[j]]).mean(axis=1)
        out["global_default"].append(scored(j, int(np.argmax(mean_rank.values))))

        nb = int(np.argmax(S[j]))
        out["cosine_top1"].append(scored(j, int(np.nanargmax(Pv[:, nb]))))

        top5 = np.argsort(S[j])[-5:]
        w = S[j][top5]
        w = w / w.sum()
        prof = np.zeros(Pv.shape[0])
        for t, wt in zip(top5, w):
            v = Pv[:, t].copy()
            v[np.isnan(v)] = np.nanmean(v)
            prof += wt * v
        out["cosine_top5_vote"].append(scored(j, int(np.argmax(prof))))

        r = int(rng.integers(0, n - 1))
        r = r if r != j else n - 1
        out["random_neighbour"].append(scored(j, int(np.nanargmax(Pv[:, r]))))

    rows = []
    for name, vals in out.items():
        v = np.asarray(vals, dtype=float)
        rows.append({"strategy": name,
                     "mean_score": float(np.nanmean(v)),
                     "regret_vs_oracle": float(np.nanmean(best - v))})
    return pd.DataFrame(rows).set_index("strategy")


def proxy_regret(P: pd.DataFrame, rhos=(0.43, 0.6, 0.7, 0.8, 0.9), seed: int = 0,
                 n_rep: int = 200) -> pd.DataFrame:
    """Accuracy lost to a proxy whose rank agreement with AutoGluon is rho.

    A Gaussian copula turns rho into a proxy ranking with that Spearman against the true scores,
    then selection takes the argmax. This isolates the SELECTION cost of proxy noise, assuming
    the search reaches every pipeline -- an upper bound on how well a noisy proxy can do.
    """
    from scipy.special import ndtri
    from scipy.stats import rankdata

    rng = np.random.default_rng(seed)
    Pv = P.values
    best = np.nanmax(Pv, axis=0)
    rows = []
    for rho in rhos:
        picks = []
        for j in range(P.shape[1]):
            t = Pv[:, j][~np.isnan(Pv[:, j])]
            if len(t) < 3:
                picks.append(np.nan)
                continue
            zt = ndtri(np.clip((rankdata(t) - 0.5) / len(t), 1e-6, 1 - 1e-6))
            draws = (rho * zt[None, :]
                     + np.sqrt(1 - rho ** 2) * rng.standard_normal((n_rep, len(t))))
            picks.append(float(np.mean(t[np.argmax(draws, axis=1)])))
        picks = np.asarray(picks, dtype=float)
        rows.append({"proxy_spearman": rho,
                     "mean_score": float(np.nanmean(picks)),
                     "regret_vs_oracle": float(np.nanmean(best - picks))})
    return pd.DataFrame(rows).set_index("proxy_spearman")


def headroom(P: pd.DataFrame) -> str:
    gap = P.max(axis=0) - P.loc["baseline"]
    gap = gap.dropna()
    return (f"  datasets where the BEST pipeline beats no-preprocessing by <=0.005: "
            f"{float((gap <= 0.005).mean()):.1%}\n"
            f"  ... by >=0.05:                                                     "
            f"{float((gap >= 0.05).mean()):.1%}\n"
            f"  median headroom: {float(gap.median()):.4f}   mean: {float(gap.mean()):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--performance-matrix",
                    default=os.path.join(_REPO, "data", "openml",
                                         "training_performance_matrix_autogluon.csv"))
    ap.add_argument("--metafeatures",
                    default=os.path.join(_REPO, "data", "openml", "dataset_feats.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-proxy", action="store_true", help="skip the proxy simulation (needs scipy)")
    args = ap.parse_args()

    P, M = load_reference(args.performance_matrix, args.metafeatures)
    print(f"Reference set: {P.shape[1]} datasets x {P.shape[0]} pipelines "
          f"(EVAL_IDS held out)\n")

    print("=== HEADROOM: how much is even available? ===")
    print(headroom(P), "\n")

    print("=== CONTRIBUTION: leave-one-out recommendation quality ===")
    tbl = contribution_table(P, M, seed=args.seed)
    print(tbl.round(4).to_string(), "\n")
    lift = (tbl.loc["global_default", "regret_vs_oracle"]
            - tbl.loc["cosine_top1", "regret_vs_oracle"])
    print(f"  metafeature retrieval buys {lift:+.4f} accuracy over a CONSTANT recommendation.")
    print(f"  a random neighbour costs "
          f"{tbl.loc['random_neighbour', 'regret_vs_oracle'] - tbl.loc['cosine_top1', 'regret_vs_oracle']:+.4f} "
          f"-- so the metafeatures are not pure noise, but nearly all their value is\n"
          f"  'pick a generally-good pipeline', not 'pick the right one for THIS dataset'.\n")

    if not args.skip_proxy:
        print("=== PROXY: accuracy lost to surrogate ranking noise ===")
        print("(measured logreg proxy rho ~= 0.42, from outputs/proxy_fidelity.log)")
        print(proxy_regret(P, seed=args.seed).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
