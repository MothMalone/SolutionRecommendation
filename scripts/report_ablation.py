#!/usr/bin/env python3
"""Aggregate ACORec headline runs into RQ3 ablation tables (Markdown).

Reads dataset_<id>/recommendation.json from one or more run dirs. Each run's recommendation.json
now carries `ag_candidate_scores` (per-candidate AutoGluon test score) plus `final_evaluation.score`
(the chosen pipeline). From those, two ablations fall out for free:

  RQ3-B  no-search vs search : `no_search_retrieval` score vs the chosen ACO pipeline score.
  RQ3-A  cross-feature space : compare the chosen score ACROSS runs (our-space vs their-space, etc.).

Usage:
  # no-search vs search, from one run:
  python scripts/report_ablation.py --run NAME=path/to/run [--run NAME2=path2 ...] --out report.md
"""
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional


def _load_run(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(glob.glob(os.path.join(path, "**", "recommendation.json"), recursive=True)):
        try:
            rec = json.load(open(p))
        except Exception:
            continue
        did = str(rec.get("dataset_id"))
        cs = rec.get("ag_candidate_scores") or {}
        chosen = (rec.get("final_evaluation") or {}).get("score")
        chosen_name = (rec.get("pipeline_config") or {}).get("name")
        out[did] = {"scores": cs, "chosen": chosen, "chosen_name": chosen_name}
    return out


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.4f}"


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=path (repeatable)")
    ap.add_argument("--out", default="ablation_report.md")
    args = ap.parse_args()

    runs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs[name] = _load_run(path)

    all_ids = sorted({d for r in runs.values() for d in r}, key=lambda x: (len(x), x))
    lines: List[str] = []

    # ---- RQ3-B: no-search vs search (per run that has both) ----
    lines.append("# RQ3-B — No-search (transfer-only) vs Search (ACO)\n")
    lines.append("`no_search` = the transfer pipeline's AG test score; `search` = the chosen ACO "
                 "pipeline's score. Δ>0 means ACO search improved on pure transfer (the heuristic "
                 "was a good start *and* search added value); Δ≈0 means search added nothing beyond "
                 "transfer; Δ<0 flags where the full→per-operator signal was too weak to help.\n")
    for name, run in runs.items():
        lines.append(f"\n## run: {name}\n")
        lines.append("| Dataset | no_search | search (chosen) | Δ(search−nosearch) |")
        lines.append("|---|---|---|---|")
        d_ns, d_se = [], []
        for did in sorted(run, key=lambda x: (len(x), x)):
            sc = run[did]["scores"]
            ns = sc.get("no_search_retrieval")
            se = run[did]["chosen"]
            d = (se - ns) if (ns is not None and se is not None) else None
            if ns is not None:
                d_ns.append(ns)
            if se is not None:
                d_se.append(se)
            lines.append(f"| {did} | {_fmt(ns)} | {_fmt(se)} | {('%+.4f' % d) if d is not None else '—'} |")
        mns, mse = _mean(d_ns), _mean(d_se)
        md = (mse - mns) if (mns is not None and mse is not None) else None
        lines.append(f"| **mean** | **{_fmt(mns)}** | **{_fmt(mse)}** | **{('%+.4f' % md) if md is not None else '—'}** |")

    # ---- RQ3-A: cross-feature space comparison (chosen score across runs) ----
    if len(runs) > 1:
        lines.append("\n\n# RQ3-A — Operator-space comparison (chosen score per run)\n")
        lines.append("Compare the chosen ACORec score across operator spaces. our-space − their-space "
                     "(= dropping feature_selection + dimensionality_reduction) quantifies the "
                     "cross-feature operator benefit DiffPrep's per-feature space cannot capture.\n")
        names = list(runs.keys())
        lines.append("| Dataset | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        col_means = {n: [] for n in names}
        for did in all_ids:
            row = [did]
            for n in names:
                v = runs[n].get(did, {}).get("chosen")
                if v is not None:
                    col_means[n].append(v)
                row.append(_fmt(v))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("| **mean** | " + " | ".join(f"**{_fmt(_mean(col_means[n]))}**" for n in names) + " |")

    open(args.out, "w").write("\n".join(lines))
    print(f"Wrote {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
