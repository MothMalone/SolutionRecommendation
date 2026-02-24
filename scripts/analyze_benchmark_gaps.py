#!/usr/bin/env python3
"""Analyze benchmark performance gaps and generate evidence plots.

This script uses the benchmark table shared in chat and optionally merges
CTXPipe eval output (ctx_reward vs ag_score) for proxy-vs-final metric analysis.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon


def _shared_table_df() -> pd.DataFrame:
    # Rows are in the exact order provided by the user.
    rows: List[Dict[str, float]] = [
        {
            "ctxpipe_ctx_internal": 0.5409,
            "ctxpipe_ctx_space": 0.7605,
            "difffix_diffprep": 0.7732,
            "ours_diffprep": 0.7653,
            "ours_no_warmup": 0.7731,
            "ours_our_space": 0.7751,
            "full_autogluon": 0.7773,
            "baseline_autogluon": 0.5590,
        },
        {
            "ctxpipe_ctx_internal": 0.7241,
            "ctxpipe_ctx_space": 0.7586,
            "difffix_diffprep": 0.6897,
            "ours_diffprep": 0.7931,
            "ours_no_warmup": 0.6897,
            "ours_our_space": 0.7586,
            "full_autogluon": 0.6897,
            "baseline_autogluon": 0.7241,
        },
        {
            "ctxpipe_ctx_internal": 0.6486,
            "ctxpipe_ctx_space": 0.7297,
            "difffix_diffprep": 0.6757,
            "ours_diffprep": 0.7297,
            "ours_no_warmup": 0.7838,
            "ours_our_space": 0.7838,
            "full_autogluon": 0.7297,
            "baseline_autogluon": 0.7297,
        },
        {
            "ctxpipe_ctx_internal": 0.7895,
            "ctxpipe_ctx_space": 0.8684,
            "difffix_diffprep": 0.9474,
            "ours_diffprep": 0.8421,
            "ours_no_warmup": 0.8684,
            "ours_our_space": 0.8947,
            "full_autogluon": 0.7895,
            "baseline_autogluon": 0.8421,
        },
        {
            "ctxpipe_ctx_internal": 0.8235,
            "ctxpipe_ctx_space": 0.7059,
            "difffix_diffprep": 0.7647,
            "ours_diffprep": 0.8235,
            "ours_no_warmup": 0.7647,
            "ours_our_space": 0.7647,
            "full_autogluon": 0.7647,
            "baseline_autogluon": 0.7647,
        },
        {
            "ctxpipe_ctx_internal": 0.8045,
            "ctxpipe_ctx_space": 0.8715,
            "difffix_diffprep": 0.9162,
            "ours_diffprep": 0.9832,
            "ours_no_warmup": 0.9553,
            "ours_our_space": 0.9553,
            "full_autogluon": 0.9609,
            "baseline_autogluon": 0.8994,
        },
        {
            "ctxpipe_ctx_internal": 0.6456,
            "ctxpipe_ctx_space": 0.6329,
            "difffix_diffprep": 0.5063,
            "ours_diffprep": 0.6709,
            "ours_no_warmup": 0.5823,
            "ours_our_space": 0.6709,
            "full_autogluon": 0.6582,
            "baseline_autogluon": 0.6329,
        },
        {
            "ctxpipe_ctx_internal": 0.7188,
            "ctxpipe_ctx_space": 0.7500,
            "difffix_diffprep": 0.6875,
            "ours_diffprep": 0.8438,
            "ours_no_warmup": 0.7500,
            "ours_our_space": 0.7500,
            "full_autogluon": 0.8438,
            "baseline_autogluon": 0.7812,
        },
        {
            "ctxpipe_ctx_internal": 0.7652,
            "ctxpipe_ctx_space": 0.8665,
            "difffix_diffprep": 0.8633,
            "ours_diffprep": 0.8691,
            "ours_no_warmup": 0.8670,
            "ours_our_space": 0.8687,
            "full_autogluon": 0.8665,
            "baseline_autogluon": 0.0000,
        },
        {
            "ctxpipe_ctx_internal": 0.3500,
            "ctxpipe_ctx_space": 0.4000,
            "difffix_diffprep": 0.7000,
            "ours_diffprep": 0.5500,
            "ours_no_warmup": 0.4500,
            "ours_our_space": 0.6500,
            "full_autogluon": 0.4500,
            "baseline_autogluon": 0.5500,
        },
        {
            "ctxpipe_ctx_internal": 0.7900,
            "ctxpipe_ctx_space": 0.8150,
            "difffix_diffprep": 0.7675,
            "ours_diffprep": 0.7675,
            "ours_no_warmup": 0.7975,
            "ours_our_space": 0.7950,
            "full_autogluon": 0.7550,
            "baseline_autogluon": 0.7525,
        },
        {
            "ctxpipe_ctx_internal": 0.5404,
            "ctxpipe_ctx_space": 0.8250,
            "difffix_diffprep": 0.8423,
            "ours_diffprep": 0.8846,
            "ours_no_warmup": 0.8519,
            "ours_our_space": 0.8712,
            "full_autogluon": 0.8750,
            "baseline_autogluon": 0.0000,
        },
        {
            "ctxpipe_ctx_internal": 0.7800,
            "ctxpipe_ctx_space": 0.8650,
            "difffix_diffprep": 0.8275,
            "ours_diffprep": 0.8450,
            "ours_no_warmup": 0.8950,
            "ours_our_space": 0.8850,
            "full_autogluon": 0.8550,
            "baseline_autogluon": 0.8450,
        },
        {
            "ctxpipe_ctx_internal": 0.6862,
            "ctxpipe_ctx_space": 0.7576,
            "difffix_diffprep": 0.7663,
            "ours_diffprep": 0.7975,
            "ours_no_warmup": 0.7714,
            "ours_our_space": 0.8018,
            "full_autogluon": 0.7704,
            "baseline_autogluon": 0.6216,
        },
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "dataset_row", np.arange(1, len(df) + 1))
    return df


def _format_method_label(method: str) -> str:
    labels = {
        "ctxpipe_ctx_internal": "CTXPipe internal",
        "ctxpipe_ctx_space": "CTXPipe space",
        "difffix_diffprep": "DiffFix (DiffPrep)",
        "ours_diffprep": "Ours (DiffPrep)",
        "ours_no_warmup": "Ours (No warmup)",
        "ours_our_space": "Ours (Our space)",
        "full_autogluon": "Full AutoGluon",
        "baseline_autogluon": "Baseline AG only",
    }
    return labels[method]


def _bh_fdr(pvals: List[float]) -> List[float]:
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = min(prev, ranked[i] * n / rank)
        adjusted[i] = adj
        prev = adj
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out.tolist()


def _plot_means(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plot_df = summary_df.sort_values("mean", ascending=False)
    sns.barplot(data=plot_df, x="label", y="mean", color="#4C78A8")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("")
    plt.title("Average Accuracy Across Shared Benchmark Rows")
    for i, row in enumerate(plot_df.itertuples(index=False)):
        plt.text(i, row.mean + 0.005, f"{row.mean:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0.0, 1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_delta_vs_ours(pairwise_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    plot_df = pairwise_df.sort_values("mean_delta_vs_ours")
    sns.barplot(data=plot_df, x="mean_delta_vs_ours", y="label", color="#72B7B2")
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("Mean (Ours - Method) Accuracy Delta")
    plt.ylabel("")
    plt.title("Per-Method Gap vs Ours (Our Space)")
    for i, row in enumerate(plot_df.itertuples(index=False)):
        plt.text(
            row.mean_delta_vs_ours + (0.001 if row.mean_delta_vs_ours >= 0 else -0.001),
            i,
            f"{row.mean_delta_vs_ours:+.3f}",
            va="center",
            ha="left" if row.mean_delta_vs_ours >= 0 else "right",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_ctx_proxy_gap(ctx_df: pd.DataFrame, out_path: Path) -> None:
    clean = ctx_df.dropna(subset=["ctx_reward", "ag_score"]).copy()
    plt.figure(figsize=(6.2, 6.0))
    sns.scatterplot(data=clean, x="ctx_reward", y="ag_score", hue="status", palette="deep", s=60)
    lim_min = min(clean["ctx_reward"].min(), clean["ag_score"].min()) - 0.05
    lim_max = max(clean["ctx_reward"].max(), clean["ag_score"].max()) + 0.05
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="gray", linewidth=1)
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.xlabel("CTXPipe internal reward")
    plt.ylabel("AutoGluon re-eval score")
    plt.title("CTXPipe Proxy Metric vs Final Metric")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_ctx_operator_usage(operator_counts: Counter, out_path: Path) -> None:
    top = operator_counts.most_common(10)
    labels = [x[0] for x in top]
    values = [x[1] for x in top]
    plt.figure(figsize=(8.2, 4.2))
    sns.barplot(x=values, y=labels, color="#E45756")
    plt.xlabel("Count Across Suggested Pipeline Steps")
    plt.ylabel("")
    plt.title("CTXPipe Suggested Operator Usage (Top 10)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _plot_ctx_nonblank_vs_ag(ctx_nonblank: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6.8, 4.6))
    sns.boxplot(data=ctx_nonblank, x="nonblank_ops", y="ag_score", color="#54A24B")
    sns.stripplot(data=ctx_nonblank, x="nonblank_ops", y="ag_score", color="black", alpha=0.55, size=4)
    plt.xlabel("Non-blank operators in suggested pipeline")
    plt.ylabel("AutoGluon re-eval score")
    plt.title("CTXPipe Pipeline Complexity vs Final Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    out_dir = Path("analysis/perf_gap")
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = _shared_table_df()
    method_cols = [
        "ctxpipe_ctx_internal",
        "ctxpipe_ctx_space",
        "difffix_diffprep",
        "ours_diffprep",
        "ours_no_warmup",
        "ours_our_space",
        "full_autogluon",
        "baseline_autogluon",
    ]

    summary_rows = []
    for m in method_cols:
        summary_rows.append(
            {
                "method": m,
                "label": _format_method_label(m),
                "mean": float(scores[m].mean()),
                "std": float(scores[m].std(ddof=1)),
                "median": float(scores[m].median()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    ranks = scores[method_cols].rank(axis=1, method="average", ascending=False)
    rank_df = pd.DataFrame(
        {
            "method": method_cols,
            "label": [_format_method_label(x) for x in method_cols],
            "mean_rank": [float(ranks[m].mean()) for m in method_cols],
            "best_or_tied_count": [int((scores[m] == scores[method_cols].max(axis=1)).sum()) for m in method_cols],
        }
    ).sort_values("mean_rank")

    # Non-parametric omnibus test across all methods.
    friedman_stat, friedman_p = friedmanchisquare(*[scores[m].values for m in method_cols])

    ours = "ours_our_space"
    pairwise_rows = []
    raw_pvals = []
    methods_for_pvals = []
    for m in method_cols:
        if m == ours:
            continue
        delta = scores[ours] - scores[m]
        greater = int((delta > 0).sum())
        lower = int((delta < 0).sum())
        equal = int((delta == 0).sum())
        if np.allclose(delta.values, 0.0):
            pval = 1.0
        else:
            _, pval = wilcoxon(delta.values, alternative="two-sided", zero_method="zsplit")
        pairwise_rows.append(
            {
                "method": m,
                "label": _format_method_label(m),
                "mean_delta_vs_ours": float(delta.mean()),
                "median_delta_vs_ours": float(delta.median()),
                "wins": greater,
                "losses": lower,
                "ties": equal,
                "wilcoxon_p": float(pval),
            }
        )
        raw_pvals.append(float(pval))
        methods_for_pvals.append(m)

    adjusted = _bh_fdr(raw_pvals)
    pmap = {m: p for m, p in zip(methods_for_pvals, adjusted)}
    for row in pairwise_rows:
        row["wilcoxon_p_fdr"] = pmap[row["method"]]
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values("mean_delta_vs_ours", ascending=False)

    summary_df.to_csv(out_dir / "summary_stats.csv", index=False)
    rank_df.to_csv(out_dir / "rank_stats.csv", index=False)
    pairwise_df.to_csv(out_dir / "pairwise_vs_ours.csv", index=False)

    _plot_means(summary_df, out_dir / "mean_accuracy.png")
    _plot_delta_vs_ours(pairwise_df, out_dir / "delta_vs_ours.png")

    # Optional CTXPipe proxy-vs-final evidence from provided output file.
    ctx_file = Path("/Users/khoatran/Downloads/results(2).csv")
    ctx_summary = None
    ctx_usage_summary = None
    ctx_nonblank_summary = None
    if ctx_file.exists():
        ctx_df = pd.read_csv(ctx_file)
        if {"ctx_reward", "ag_score", "status"}.issubset(ctx_df.columns):
            clean_ctx = ctx_df.dropna(subset=["ctx_reward", "ag_score"]).copy()
            clean_ctx["gap_ag_minus_ctx"] = clean_ctx["ag_score"] - clean_ctx["ctx_reward"]
            ctx_summary = {
                "rows_total": int(len(ctx_df)),
                "rows_with_both_scores": int(len(clean_ctx)),
                "mean_gap_ag_minus_ctx": float(clean_ctx["gap_ag_minus_ctx"].mean()),
                "median_gap_ag_minus_ctx": float(clean_ctx["gap_ag_minus_ctx"].median()),
                "positive_gap_rate": float((clean_ctx["gap_ag_minus_ctx"] > 0).mean()),
                "negative_gap_rate": float((clean_ctx["gap_ag_minus_ctx"] < 0).mean()),
            }
            clean_ctx.to_csv(out_dir / "ctxpipe_proxy_gap_rows.csv", index=False)
            _plot_ctx_proxy_gap(ctx_df, out_dir / "ctxpipe_proxy_vs_final.png")

        if "sequence" in ctx_df.columns:
            op_counts: Counter = Counter()
            nonblank_counts: List[int] = []
            nonblank_ag_rows = []
            total_steps = 0
            for _, row in ctx_df.iterrows():
                raw = row.get("sequence")
                if pd.isna(raw):
                    continue
                try:
                    seq = ast.literal_eval(raw)
                except Exception:
                    continue
                nb = 0
                for op in seq:
                    op_counts[op] += 1
                    total_steps += 1
                    if op != "blank":
                        nb += 1
                nonblank_counts.append(nb)
                if not pd.isna(row.get("ag_score")):
                    nonblank_ag_rows.append({"nonblank_ops": nb, "ag_score": float(row["ag_score"])})
            if total_steps > 0 and nonblank_counts:
                ctx_usage_summary = {
                    "total_steps": int(total_steps),
                    "blank_rate": float(op_counts["blank"] / total_steps),
                    "nonblank_mean": float(np.mean(nonblank_counts)),
                    "nonblank_median": float(np.median(nonblank_counts)),
                    "pipelines_with_le1_nonblank": int(sum(x <= 1 for x in nonblank_counts)),
                    "pipeline_count": int(len(nonblank_counts)),
                }
                pd.DataFrame(op_counts.items(), columns=["operator", "count"]).sort_values(
                    "count", ascending=False
                ).to_csv(out_dir / "ctxpipe_operator_usage.csv", index=False)
                _plot_ctx_operator_usage(op_counts, out_dir / "ctxpipe_operator_usage.png")

            if nonblank_ag_rows:
                nb_df = pd.DataFrame(nonblank_ag_rows)
                corr = float(nb_df["nonblank_ops"].corr(nb_df["ag_score"]))
                summary_table = (
                    nb_df.groupby("nonblank_ops")["ag_score"].agg(["count", "mean", "median"]).reset_index()
                )
                summary_table.to_csv(out_dir / "ctxpipe_nonblank_vs_ag_summary.csv", index=False)
                _plot_ctx_nonblank_vs_ag(nb_df, out_dir / "ctxpipe_nonblank_vs_ag.png")
                ctx_nonblank_summary = {"corr": corr}

    # Build markdown report.
    report_lines = [
        "# Performance Gap Analysis",
        "",
        "## Data Scope",
        f"- Shared benchmark rows analyzed: **{len(scores)}**",
        f"- Methods compared: **{len(method_cols)}**",
        "",
        "## Omnibus Test",
        f"- Friedman statistic: **{friedman_stat:.4f}**",
        f"- Friedman p-value: **{friedman_p:.4g}**",
        "",
        "## Mean Accuracy",
    ]
    for row in summary_df.sort_values("mean", ascending=False).itertuples(index=False):
        report_lines.append(f"- {row.label}: {row.mean:.4f} (std {row.std:.4f})")

    report_lines.extend(["", "## Mean Rank (lower is better)"])
    for row in rank_df.itertuples(index=False):
        report_lines.append(
            f"- {row.label}: rank {row.mean_rank:.3f}, best-or-tied on {row.best_or_tied_count}/{len(scores)} rows"
        )

    report_lines.extend(["", "## Pairwise vs Ours (Our space)"])
    for row in pairwise_df.itertuples(index=False):
        report_lines.append(
            "- "
            + f"{row.label}: mean delta {row.mean_delta_vs_ours:+.4f}, "
            + f"wins/losses/ties={row.wins}/{row.losses}/{row.ties}, "
            + f"Wilcoxon p={row.wilcoxon_p:.4g}, FDR p={row.wilcoxon_p_fdr:.4g}"
        )

    if ctx_summary is not None:
        report_lines.extend(
            [
                "",
                "## CTXPipe Proxy-vs-Final Metric Gap",
                f"- Rows in CTXPipe output: {ctx_summary['rows_total']}",
                f"- Rows with both ctx_reward and ag_score: {ctx_summary['rows_with_both_scores']}",
                f"- Mean(ag_score - ctx_reward): {ctx_summary['mean_gap_ag_minus_ctx']:+.4f}",
                f"- Median(ag_score - ctx_reward): {ctx_summary['median_gap_ag_minus_ctx']:+.4f}",
                f"- Positive gap rate: {ctx_summary['positive_gap_rate']:.1%}",
                f"- Negative gap rate: {ctx_summary['negative_gap_rate']:.1%}",
            ]
        )
    if ctx_usage_summary is not None:
        report_lines.extend(
            [
                "",
                "## CTXPipe Operator Usage Concentration",
                f"- Total suggested steps parsed: {ctx_usage_summary['total_steps']}",
                f"- Blank-step rate: {ctx_usage_summary['blank_rate']:.1%}",
                f"- Non-blank operators per pipeline (mean): {ctx_usage_summary['nonblank_mean']:.3f}",
                f"- Non-blank operators per pipeline (median): {ctx_usage_summary['nonblank_median']:.3f}",
                "- Pipelines with <=1 non-blank operator: "
                + f"{ctx_usage_summary['pipelines_with_le1_nonblank']}/{ctx_usage_summary['pipeline_count']}",
            ]
        )
    if ctx_nonblank_summary is not None:
        report_lines.extend(
            [
                "",
                "## CTXPipe Complexity-vs-Score Signal",
                f"- Correlation(non-blank operator count, ag_score): {ctx_nonblank_summary['corr']:+.3f}",
            ]
        )

    (out_dir / "report.md").write_text("\n".join(report_lines))
    print(f"Wrote analysis artifacts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
