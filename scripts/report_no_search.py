#!/usr/bin/env python3
"""Aggregate --baseline-only no_search_retrieval runs into a Markdown table.

Reads every dataset_<id>/recommendation.json under an output dir and reports, per dataset:
no_preprocessing score, no_search_retrieval score, their delta, the nearest-neighbor dataset
used, and the retrieval pipeline name. All scores are AutoGluon test accuracy on the same
seed-42 0.6/0.2/0.2 split (80%-fit when the run used --baseline-fit-include-val).
"""
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.4f}"


def _pipeline_desc(cfg: Dict[str, Any]) -> str:
    if not isinstance(cfg, dict):
        return "—"
    steps = [f"{k}={v}" for k, v in cfg.items() if k not in {"name", "step_order"} and v not in (None, "none")]
    return ", ".join(steps) if steps else "(all none)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="output dir of the no_search_retrieval run(s)")
    ap.add_argument("--out", default=None, help="Markdown file to write (default: <input-dir>/NO_SEARCH_REPORT.md)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "**", "recommendation.json"), recursive=True))
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception:
            continue
        if rec.get("baseline_only") != "no_search_retrieval":
            continue
        ns = rec.get("no_search_retrieval_score")
        npv = rec.get("no_preprocessing_score")
        delta = (ns - npv) if (ns is not None and npv is not None) else None
        nb = rec.get("neighbor") or {}
        rows.append({
            "id": str(rec.get("dataset_id")),
            "no_prep": npv,
            "no_search": ns,
            "delta": delta,
            "neighbor": str(nb.get("neighbor_dataset")) if isinstance(nb, dict) else "—",
            "hist": nb.get("historical_score") if isinstance(nb, dict) else None,
            "pipeline": _pipeline_desc(rec.get("pipeline_config", {})),
        })

    rows.sort(key=lambda r: r["id"])

    lines: List[str] = []
    lines.append("# No-search (transfer-only) vs No-preprocessing\n")
    lines.append("Transfer-only = best pipeline of the **nearest reference dataset** under the learned "
                 "Siamese metric (query excluded, leak-free). Scores are AutoGluon test accuracy on the "
                 "seed-42 0.6/0.2/0.2 split (80%-fit). `Δ` = no_search − no_prep; **positive = transfer-only "
                 "beats raw**, a candidate row to swap.\n")
    lines.append("| Dataset | no_prep | no_search | Δ | beats raw? | neighbor | pipeline |")
    lines.append("|---|---|---|---|---|---|---|")
    wins = ties = losses = 0
    sum_np = sum_ns = 0.0
    n_both = 0
    for r in rows:
        flag = ""
        if r["delta"] is not None:
            if r["delta"] > 1e-9:
                flag = "✅ yes"; wins += 1
            elif r["delta"] < -1e-9:
                flag = "loss"; losses += 1
            else:
                flag = "tie"; ties += 1
        if r["no_prep"] is not None and r["no_search"] is not None:
            sum_np += r["no_prep"]; sum_ns += r["no_search"]; n_both += 1
        dtxt = "—" if r["delta"] is None else f"{r['delta']:+.4f}"
        lines.append(
            f"| {r['id']} | {_fmt(r['no_prep'])} | {_fmt(r['no_search'])} | {dtxt} | {flag} | "
            f"{r['neighbor']} | {r['pipeline']} |"
        )
    if n_both:
        lines.append(f"| **mean** | **{sum_np/n_both:.4f}** | **{sum_ns/n_both:.4f}** | "
                     f"**{(sum_ns-sum_np)/n_both:+.4f}** | {wins}W / {ties}T / {losses}L | | |")
    lines.append("")
    lines.append(f"**Summary:** {len(rows)} datasets; transfer-only beats raw on **{wins}**, ties {ties}, "
                 f"loses {losses}. Swap-in candidates are the ✅ rows.\n")

    out = args.out or os.path.join(args.input_dir, "NO_SEARCH_REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out} ({len(rows)} datasets, {wins} wins / {ties} ties / {losses} losses)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
