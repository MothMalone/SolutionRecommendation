#!/usr/bin/env python3
"""Post-hoc analysis for CtxPipe inference outputs.

Produces summary JSON + PNG charts for:
- Ctx reward vs AutoGluon final score alignment
- Operator usage and blank dominance
- Pipeline complexity vs final score
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_sequence(sequence_raw: str) -> List[str]:
    names = re.findall(r"<([^>]+)>", sequence_raw)
    if names:
        return [name.strip() for name in names if name.strip()]

    try:
        value = ast.literal_eval(sequence_raw)
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
    except Exception:
        pass

    return []


def _load_ctx_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing pipelines.tsv: {path}")

    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        tag, dataset, sequence_raw, reward_raw = parts[:4]
        try:
            reward = float(reward_raw)
        except Exception:
            reward = np.nan
        rows.append(
            {
                "tag": tag,
                "dataset": dataset,
                "sequence_raw": sequence_raw,
                "sequence": _parse_sequence(sequence_raw),
                "ctx_reward": reward,
            }
        )
    return pd.DataFrame(rows)


def _select_best_ctx(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    def _tag_num(tag: str) -> int:
        m = re.search(r"(\d+)$", str(tag))
        return int(m.group(1)) if m else -1

    tmp = df.copy()
    tmp["tag_num"] = tmp["tag"].map(_tag_num)
    tmp["ctx_reward_filled"] = tmp["ctx_reward"].fillna(-np.inf)
    tmp = tmp.sort_values(["dataset", "ctx_reward_filled", "tag_num"], ascending=[True, False, False])
    best = tmp.groupby("dataset", as_index=False).head(1).drop(columns=["tag_num", "ctx_reward_filled"])
    return best.reset_index(drop=True)


def _count_non_blank(seq: List[str]) -> int:
    return sum(1 for x in seq if x.lower() not in {"blank", "primitive", "none"})


def _operator_counts(seqs: List[List[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for seq in seqs:
        for op in seq:
            counts[op] = counts.get(op, 0) + 1
    return counts


def _plot_reward_vs_final(merged: pd.DataFrame, out_dir: Path, title: str) -> None:
    df = merged[(merged["status"] == "ok") & merged["ctx_reward"].notna() & merged["ag_score"].notna()].copy()
    plt.figure(figsize=(8, 6))
    if not df.empty:
        plt.scatter(df["ctx_reward"], df["ag_score"], alpha=0.85)
        lo = min(df["ctx_reward"].min(), df["ag_score"].min())
        hi = max(df["ctx_reward"].max(), df["ag_score"].max())
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("CTXPipe internal reward")
    plt.ylabel("AutoGluon re-eval score")
    plt.title(f"{title} - CTX reward vs final")
    plt.tight_layout()
    plt.savefig(out_dir / "ctx_reward_vs_ag_score.png", dpi=160)
    plt.close()


def _plot_operator_usage(best: pd.DataFrame, out_dir: Path, title: str) -> None:
    counts = _operator_counts(best["sequence"].tolist())
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    labels = [x[0] for x in items]
    values = [x[1] for x in items]

    plt.figure(figsize=(9, 5))
    if items:
        plt.barh(labels[::-1], values[::-1])
    plt.xlabel("Count across selected pipeline steps")
    plt.title(f"{title} - Suggested operator usage (top 10)")
    plt.tight_layout()
    plt.savefig(out_dir / "operator_usage_top10.png", dpi=160)
    plt.close()


def _plot_complexity_vs_score(merged: pd.DataFrame, out_dir: Path, title: str) -> None:
    df = merged[(merged["status"] == "ok") & merged["ag_score"].notna()].copy()
    if df.empty:
        return

    grouped = [g["ag_score"].values for _, g in df.groupby("n_non_blank")]
    labels = [str(k) for k, _ in df.groupby("n_non_blank")]

    plt.figure(figsize=(8, 5))
    plt.boxplot(grouped, labels=labels)
    plt.xlabel("Non-blank operators in selected pipeline")
    plt.ylabel("AutoGluon re-eval score")
    plt.title(f"{title} - Pipeline complexity vs final score")
    plt.tight_layout()
    plt.savefig(out_dir / "complexity_vs_score.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pipelines-tsv", required=True)
    p.add_argument("--ag-results-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--title", default="CtxPipe")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipelines_tsv = Path(args.pipelines_tsv)
    ag_results_csv = Path(args.ag_results_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_ctx = _load_ctx_rows(pipelines_tsv)
    best_ctx = _select_best_ctx(raw_ctx)
    best_ctx["n_non_blank"] = best_ctx["sequence"].map(_count_non_blank)

    ag = pd.read_csv(ag_results_csv)
    merged = best_ctx.merge(ag, on="dataset", how="left", suffixes=("", "_ag"))

    _plot_reward_vs_final(merged, out_dir, args.title)
    _plot_operator_usage(best_ctx, out_dir, args.title)
    _plot_complexity_vs_score(merged, out_dir, args.title)

    ok = merged[(merged["status"] == "ok") & merged["ag_score"].notna()]
    if not ok.empty and ok["ctx_reward"].notna().sum() >= 2:
        corr = ok[["ctx_reward", "ag_score"]].corr(method="spearman").iloc[0, 1]
        corr = float(corr) if pd.notna(corr) else np.nan
    else:
        corr = np.nan

    counts = _operator_counts(best_ctx["sequence"].tolist())
    summary = {
        "n_rows_in_pipelines_tsv": int(len(raw_ctx)),
        "n_selected_datasets": int(len(best_ctx)),
        "n_ag_ok": int((merged["status"] == "ok").sum()) if "status" in merged else 0,
        "n_ag_error": int((merged["status"] == "error").sum()) if "status" in merged else 0,
        "mean_ctx_reward": float(best_ctx["ctx_reward"].mean()) if not best_ctx.empty else np.nan,
        "mean_ag_score_ok": float(ok["ag_score"].mean()) if not ok.empty else np.nan,
        "spearman_ctx_reward_vs_ag_score": corr,
        "blank_only_fraction": float((best_ctx["n_non_blank"] == 0).mean()) if not best_ctx.empty else np.nan,
        "operator_usage_top10": sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "files": {
            "ctx_reward_vs_ag_score_png": str(out_dir / "ctx_reward_vs_ag_score.png"),
            "operator_usage_top10_png": str(out_dir / "operator_usage_top10.png"),
            "complexity_vs_score_png": str(out_dir / "complexity_vs_score.png"),
            "merged_csv": str(out_dir / "merged_results.csv"),
        },
    }

    merged.to_csv(out_dir / "merged_results.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
