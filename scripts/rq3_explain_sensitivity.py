#!/usr/bin/env python3
"""Explain RQ3 sensitivity outputs with per-dataset diagnostics.

Given a suite directory (e.g., outputs/ablation_budget/rq3_budget), this script
reads each variant's recommendation artifacts and emits:
  - ablation_explain_rows.csv: one row per (variant, dataset)
  - ablation_explain_summary.csv: one row per variant with win/tie/loss stats

It is designed to make ablation tables explainable by labeling frequent causes of
"weird" behavior:
  - proxy/fallback metric instead of AutoGluon final metric
  - coarse test-set quantization (small denominator)
  - differences within 1 test sample of baseline
  - stale history artifact (legacy one-point history files)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

BASELINE_BY_SUITE: Dict[str, List[str]] = {
    "rq3_budget": ["ants10_iter10"],
    "rq3_aco_dynamics": ["a1_b2_r0p2"],
    "rq3_transfer_weighting": ["sim_T1_floor0p05"],
    "rq3_transfer_neighbors": ["K5_L3"],
    # Support both old and new naming conventions.
    "rq3_pheromone_update": ["w_rank_k3_m2_l0p7", "k3_rank_m2_l0p7"],
}


@dataclass
class RecRow:
    suite: str
    variant: str
    dataset_id: str
    recommendation_path: str
    final_method: str
    final_score: Optional[float]
    proxy_score: Optional[float]
    final_error: Optional[str]
    history_len: int
    history_max_best: Optional[float]
    n_candidates: int
    implied_denominator: Optional[int]
    implied_hits: Optional[int]
    quantization_step: Optional[float]


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    return f


def _iter_variant_dirs(suite_dir: Path) -> Iterable[Path]:
    for p in sorted(suite_dir.iterdir()):
        if p.is_dir():
            yield p


def _find_recommendation_files(variant_dir: Path) -> List[Tuple[str, Path]]:
    matches: List[Tuple[str, Path]] = []

    single = variant_dir / "recommendation.json"
    if single.exists():
        matches.append(("unknown", single))

    for dataset_dir in sorted(variant_dir.glob("dataset_*")):
        if not dataset_dir.is_dir():
            continue
        rec = dataset_dir / "recommendation.json"
        if rec.exists():
            dataset_id = dataset_dir.name.replace("dataset_", "", 1)
            matches.append((dataset_id, rec))

    return matches


def _infer_denominator(score: Optional[float], max_denominator: int) -> Tuple[Optional[int], Optional[int]]:
    if score is None:
        return None, None
    frac = Fraction(str(score)).limit_denominator(max_denominator)
    denom = int(frac.denominator)
    num = int(frac.numerator)
    return denom, num


def _parse_recommendation(
    *,
    suite_name: str,
    variant_name: str,
    dataset_id: str,
    rec_path: Path,
    max_denominator: int,
) -> RecRow:
    with rec_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    final_eval = payload.get("final_evaluation", {}) if isinstance(payload, dict) else {}
    final_method = str(final_eval.get("method", "unknown"))
    final_score = _safe_float(final_eval.get("score", payload.get("final_performance")))
    proxy_score = _safe_float(payload.get("recommended_performance"))
    final_error = final_eval.get("error")
    if final_error is not None:
        final_error = str(final_error)

    aco_history = payload.get("aco_history", []) if isinstance(payload, dict) else []
    history_len = len(aco_history) if isinstance(aco_history, list) else 0
    history_max_best = None
    if isinstance(aco_history, list):
        best_vals = [_safe_float(row.get("best_score")) for row in aco_history if isinstance(row, dict)]
        best_vals = [v for v in best_vals if v is not None]
        if best_vals:
            history_max_best = float(max(best_vals))

    aco_results = payload.get("aco_results", []) if isinstance(payload, dict) else []
    n_candidates = len(aco_results) if isinstance(aco_results, list) else 0

    denom, num = _infer_denominator(final_score, max_denominator=max_denominator)
    step = (1.0 / float(denom)) if denom and denom > 0 else None

    return RecRow(
        suite=suite_name,
        variant=variant_name,
        dataset_id=str(dataset_id),
        recommendation_path=str(rec_path),
        final_method=final_method,
        final_score=final_score,
        proxy_score=proxy_score,
        final_error=final_error,
        history_len=history_len,
        history_max_best=history_max_best,
        n_candidates=n_candidates,
        implied_denominator=denom,
        implied_hits=num,
        quantization_step=step,
    )


def _choose_baseline(suite_name: str, variants: List[str], explicit: Optional[str]) -> Optional[str]:
    if explicit and explicit in variants:
        return explicit
    candidates = BASELINE_BY_SUITE.get(suite_name, [])
    for c in candidates:
        if c in variants:
            return c
    if variants:
        return sorted(variants)[0]
    return None


def _label_row(
    row: pd.Series,
    *,
    stale_history_threshold: int,
) -> str:
    labels: List[str] = []

    method = str(row.get("final_method", ""))
    if method != "autogluon":
        labels.append("non_autogluon_final_metric")

    err = str(row.get("final_error") or "")
    if "No candidate produced valid AutoGluon evaluation results" in err:
        labels.append("autogluon_no_valid_candidates")

    den = row.get("implied_denominator")
    if pd.notna(den):
        den_i = int(den)
        if den_i <= 50:
            labels.append("coarse_quantization_small_test_set")

    delta_hits = row.get("delta_hits_vs_baseline")
    if pd.notna(delta_hits):
        if abs(float(delta_hits)) <= 1.0:
            labels.append("within_1_test_sample_of_baseline")

    hist_len = int(row.get("history_len") or 0)
    if hist_len <= stale_history_threshold:
        labels.append("legacy_or_truncated_history")

    if not labels:
        labels.append("material_difference_or_needs_manual_check")
    return ";".join(labels)


def build_explanation_table(
    *,
    suite_dir: Path,
    baseline_variant: Optional[str],
    max_denominator: int,
    stale_history_threshold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    suite_name = suite_dir.name
    rows: List[RecRow] = []

    variant_dirs = list(_iter_variant_dirs(suite_dir))
    for variant_dir in variant_dirs:
        variant = variant_dir.name
        rec_files = _find_recommendation_files(variant_dir)
        for dataset_id, rec_path in rec_files:
            try:
                row = _parse_recommendation(
                    suite_name=suite_name,
                    variant_name=variant,
                    dataset_id=dataset_id,
                    rec_path=rec_path,
                    max_denominator=max_denominator,
                )
                rows.append(row)
            except Exception as exc:
                rows.append(
                    RecRow(
                        suite=suite_name,
                        variant=variant,
                        dataset_id=str(dataset_id),
                        recommendation_path=str(rec_path),
                        final_method="parse_failed",
                        final_score=None,
                        proxy_score=None,
                        final_error=f"parse_failed: {exc}",
                        history_len=0,
                        history_max_best=None,
                        n_candidates=0,
                        implied_denominator=None,
                        implied_hits=None,
                        quantization_step=None,
                    )
                )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return df, df

    variants = sorted(df["variant"].dropna().unique().tolist())
    baseline = _choose_baseline(suite_name=suite_name, variants=variants, explicit=baseline_variant)
    df["baseline_variant"] = baseline

    if baseline is not None:
        base = (
            df[df["variant"] == baseline][["dataset_id", "final_score", "implied_denominator", "implied_hits"]]
            .rename(
                columns={
                    "final_score": "baseline_score",
                    "implied_denominator": "baseline_denominator",
                    "implied_hits": "baseline_hits",
                }
            )
            .drop_duplicates(subset=["dataset_id"], keep="last")
        )
        df = df.merge(base, on="dataset_id", how="left")
    else:
        df["baseline_score"] = np.nan
        df["baseline_denominator"] = np.nan
        df["baseline_hits"] = np.nan

    df["delta_score_vs_baseline"] = df["final_score"] - df["baseline_score"]

    def _delta_hits(r: pd.Series) -> Optional[float]:
        den = r.get("implied_denominator")
        bden = r.get("baseline_denominator")
        score = r.get("final_score")
        bscore = r.get("baseline_score")
        if pd.isna(den) or pd.isna(bden) or pd.isna(score) or pd.isna(bscore):
            return None
        if int(den) != int(bden) or int(den) <= 0:
            return None
        return float(round((float(score) - float(bscore)) * int(den)))

    df["delta_hits_vs_baseline"] = df.apply(_delta_hits, axis=1)
    df["used_autogluon_final"] = df["final_method"].eq("autogluon")
    df["explanation_labels"] = df.apply(
        lambda r: _label_row(r, stale_history_threshold=stale_history_threshold),
        axis=1,
    )

    def _safe_mean(s: pd.Series) -> Optional[float]:
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if vals.empty:
            return None
        return float(vals.mean())

    def _safe_std(s: pd.Series) -> Optional[float]:
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if vals.empty:
            return None
        return float(vals.std(ddof=0))

    # Win/tie/loss vs baseline by dataset.
    score_eps = 1e-12
    df["cmp_vs_baseline"] = np.where(
        df["delta_score_vs_baseline"].isna(),
        "no_baseline",
        np.where(
            df["delta_score_vs_baseline"] > score_eps,
            "win",
            np.where(df["delta_score_vs_baseline"] < -score_eps, "loss", "tie"),
        ),
    )

    summary_rows: List[Dict[str, Any]] = []
    for variant, g in df.groupby("variant", sort=True):
        total = int(len(g))
        wins = int((g["cmp_vs_baseline"] == "win").sum())
        ties = int((g["cmp_vs_baseline"] == "tie").sum())
        losses = int((g["cmp_vs_baseline"] == "loss").sum())
        summary_rows.append(
            {
                "suite": suite_name,
                "variant": variant,
                "baseline_variant": baseline,
                "n_datasets": total,
                "mean_final_score": _safe_mean(g["final_score"]),
                "std_final_score": _safe_std(g["final_score"]),
                "mean_delta_vs_baseline": _safe_mean(g["delta_score_vs_baseline"]),
                "mean_delta_hits_vs_baseline": _safe_mean(g["delta_hits_vs_baseline"]),
                "win_count": wins,
                "tie_count": ties,
                "loss_count": losses,
                "win_rate": (wins / total) if total > 0 else None,
                "autogluon_final_rate": float(g["used_autogluon_final"].mean()) if total > 0 else None,
                "non_autogluon_count": int((~g["used_autogluon_final"]).sum()),
                "legacy_or_truncated_history_count": int(
                    g["explanation_labels"].str.contains("legacy_or_truncated_history", na=False).sum()
                ),
                "within_one_sample_count": int(
                    g["explanation_labels"].str.contains("within_1_test_sample_of_baseline", na=False).sum()
                ),
                "coarse_quantization_count": int(
                    g["explanation_labels"].str.contains("coarse_quantization_small_test_set", na=False).sum()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)
    df = df.sort_values(["dataset_id", "variant"]).reset_index(drop=True)
    return df, summary_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain RQ3 sensitivity run outputs")
    parser.add_argument("--suite-dir", required=True, help="Path to a suite dir (e.g., outputs/ablation_budget/rq3_budget)")
    parser.add_argument("--baseline-variant", default=None, help="Optional explicit baseline variant name")
    parser.add_argument("--max-denominator", type=int, default=4000, help="Max denominator when inferring quantization")
    parser.add_argument(
        "--stale-history-threshold",
        type=int,
        default=1,
        help="History length <= threshold is labeled legacy/truncated",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output file prefix; default is <suite-dir>/ablation_explain",
    )
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    if not suite_dir.exists() or not suite_dir.is_dir():
        raise FileNotFoundError(f"Suite dir not found: {suite_dir}")

    rows_df, summary_df = build_explanation_table(
        suite_dir=suite_dir,
        baseline_variant=args.baseline_variant,
        max_denominator=max(1, int(args.max_denominator)),
        stale_history_threshold=max(0, int(args.stale_history_threshold)),
    )

    if args.output_prefix:
        prefix = Path(args.output_prefix).resolve()
    else:
        prefix = suite_dir / "ablation_explain"

    rows_path = Path(f"{prefix}_rows.csv")
    summary_path = Path(f"{prefix}_summary.csv")
    rows_path.parent.mkdir(parents=True, exist_ok=True)

    rows_df.to_csv(rows_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved: {rows_path}")
    print(f"Saved: {summary_path}")
    if summary_df.empty:
        print("No recommendation rows found under suite directory.")
    else:
        print("\nVariant summary preview:")
        cols = [
            "variant",
            "n_datasets",
            "mean_final_score",
            "mean_delta_vs_baseline",
            "win_count",
            "tie_count",
            "loss_count",
            "autogluon_final_rate",
            "within_one_sample_count",
        ]
        print(summary_df[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
