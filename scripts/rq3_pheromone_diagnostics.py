#!/usr/bin/env python3
"""Diagnostics for RQ3 pheromone-update sensitivity (analysis only, no reruns).

Given an existing rq3_pheromone_update suite directory, this script extracts
defensible diagnostics from recommendation artifacts:
  - per-(variant,dataset) row metrics
  - variant-level summary
  - fixed-rank (k sweep) summary
  - fixed-k (weight method sweep) summary
  - per-dataset sensitivity summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _normalize_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if np.isfinite(f) and abs(f - round(f)) <= 1e-9:
            return str(int(round(f)))
        return str(value).strip()
    s = str(value).strip()
    m = re.fullmatch(r"([0-9]+)\.0+", s)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(?i)(?:d|dataset|openml)[_\-: ]*([0-9]+)", s)
    if m:
        return m.group(1)
    return s


def _discover_recommendations(suite_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for variant_dir in sorted(suite_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        # single-run layout
        single = variant_dir / "recommendation.json"
        if single.exists():
            rows.append(
                {
                    "variant": variant,
                    "dataset_id": "unknown",
                    "recommendation_path": str(single),
                    "dataset_dir": str(variant_dir),
                }
            )
        # batch layout
        for ds_dir in sorted(variant_dir.glob("dataset_*")):
            if not ds_dir.is_dir():
                continue
            rec = ds_dir / "recommendation.json"
            if not rec.exists():
                continue
            ds = ds_dir.name.replace("dataset_", "", 1)
            rows.append(
                {
                    "variant": variant,
                    "dataset_id": str(ds),
                    "recommendation_path": str(rec),
                    "dataset_dir": str(ds_dir),
                }
            )
    return rows


def _cfg_key(cfg: Mapping[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def _cfg_hash(cfg: Mapping[str, Any]) -> str:
    return hashlib.sha1(_cfg_key(cfg).encode("utf-8")).hexdigest()[:12]


def _pipeline_fingerprint(cfg: Mapping[str, Any]) -> str:
    ordered_steps = [
        "imputation",
        "scaling",
        "encoding",
        "feature_selection",
        "outlier_removal",
        "dimensionality_reduction",
    ]
    parts: List[str] = []
    for step in ordered_steps:
        if step in cfg:
            parts.append(f"{step}={cfg.get(step)}")
    if "step_order" in cfg:
        parts.append(f"step_order={cfg.get('step_order')}")
    return "|".join(parts)


def _to_result_pairs(raw_results: Any) -> List[Tuple[Dict[str, Any], float]]:
    out: List[Tuple[Dict[str, Any], float]] = []
    if not isinstance(raw_results, list):
        return out
    for item in raw_results:
        if (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and isinstance(item[0], dict)
        ):
            score = _safe_float(item[1])
            if score is None:
                continue
            out.append((dict(item[0]), float(score)))
    return out


def _parse_variant_params(
    payload: Mapping[str, Any],
    variant: str,
) -> Dict[str, Any]:
    hp = payload.get("search_hyperparams", {}) if isinstance(payload, dict) else {}
    k = hp.get("top_k_pheromone")
    method = hp.get("weight_method")
    m = hp.get("markov_order")
    lam = hp.get("lambda_smooth")

    try:
        k_val = int(k) if k is not None else None
    except Exception:
        k_val = None
    method_val = str(method) if method is not None else None
    try:
        m_val = int(m) if m is not None else None
    except Exception:
        m_val = None
    lam_val = _safe_float(lam)

    # fallback from variant name
    if k_val is None:
        km = re.search(r"(?:^|_)k(\d+)(?:_|$)", variant)
        if km:
            k_val = int(km.group(1))
    if method_val is None:
        if "_rank_" in variant or variant.startswith("w_rank_"):
            method_val = "rank"
        elif "_uniform_" in variant or variant.startswith("w_uniform_"):
            method_val = "uniform"
        elif "_exp_" in variant or "_exponential_" in variant or variant.startswith("w_exp_"):
            method_val = "exponential"
    if m_val is None:
        mm = re.search(r"(?:^|_)m(\d+)(?:_|$)", variant)
        if mm:
            m_val = int(mm.group(1))
    if lam_val is None:
        lm = re.search(r"(?:^|_)l([0-9]+p[0-9]+)(?:_|$)", variant)
        if lm:
            lam_val = float(lm.group(1).replace("p", "."))

    return {
        "top_k_pheromone": k_val,
        "weight_method": method_val,
        "markov_order": m_val,
        "lambda_smooth": lam_val,
    }


def _reinforcement_weights(method: str, sorted_scores: Sequence[float]) -> np.ndarray:
    if len(sorted_scores) == 0:
        return np.zeros(0, dtype=float)
    scores = np.asarray(sorted_scores, dtype=float)
    if method == "linear" and len(scores) > 1:
        w = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8) + 1e-3
    elif method == "exponential" and len(scores) > 1:
        scaled = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        w = np.exp(scaled)
    elif method == "rank":
        n = len(scores)
        w = np.arange(n, 0, -1, dtype=float)
    elif method == "reciprocal":
        n = len(scores)
        w = 1.0 / np.arange(1, n + 1, dtype=float)
    elif method == "power_rank":
        p = 4.0
        n = len(scores)
        ranks = np.arange(1, n + 1, dtype=float)
        w = 1.0 / (ranks**p)
    elif method == "uniform":
        w = np.ones(len(scores), dtype=float)
    else:
        w = np.ones(len(scores), dtype=float)
    s = float(w.sum())
    if s <= 0 or not np.isfinite(s):
        return np.ones(len(scores), dtype=float) / max(1, len(scores))
    return w / s


def _entropy_metrics(weights: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if weights.size == 0:
        return None, None, None
    if weights.size == 1:
        return 0.0, 1.0, 1.0
    w = np.clip(weights.astype(float), 1e-15, None)
    w = w / w.sum()
    h = float(-(w * np.log(w)).sum())
    h_norm = float(h / np.log(len(w)))
    n_eff = float(np.exp(h))
    top_share = float(np.max(w))
    return h_norm, n_eff, top_share


def _history_metrics(history_path: Path) -> Dict[str, Any]:
    if not history_path.exists():
        return {
            "history_points": np.nan,
            "history_first": np.nan,
            "history_last": np.nan,
            "history_net_progress": np.nan,
            "history_improve_steps": np.nan,
            "history_plateau_from_iter": np.nan,
        }
    try:
        hist = pd.read_csv(history_path)
    except Exception:
        return {
            "history_points": np.nan,
            "history_first": np.nan,
            "history_last": np.nan,
            "history_net_progress": np.nan,
            "history_improve_steps": np.nan,
            "history_plateau_from_iter": np.nan,
        }
    if "best_score" not in hist.columns:
        return {
            "history_points": int(len(hist)),
            "history_first": np.nan,
            "history_last": np.nan,
            "history_net_progress": np.nan,
            "history_improve_steps": np.nan,
            "history_plateau_from_iter": np.nan,
        }
    seq = pd.to_numeric(hist["best_score"], errors="coerce")
    valid = seq[np.isfinite(seq.to_numpy(dtype=float))]
    if valid.empty:
        return {
            "history_points": int(len(hist)),
            "history_first": np.nan,
            "history_last": np.nan,
            "history_net_progress": np.nan,
            "history_improve_steps": np.nan,
            "history_plateau_from_iter": np.nan,
        }
    arr = valid.to_numpy(dtype=float)
    first = float(arr[0])
    last = float(arr[-1])
    diffs = np.diff(arr)
    improve_steps = int((diffs > 1e-12).sum())
    plateau_from = np.nan
    best_so_far = -np.inf
    last_improve_iter = 1
    for idx, val in enumerate(arr, start=1):
        if val > best_so_far + 1e-12:
            best_so_far = val
            last_improve_iter = idx
    if last_improve_iter < len(arr):
        plateau_from = float(last_improve_iter + 1)

    return {
        "history_points": int(len(hist)),
        "history_first": first,
        "history_last": last,
        "history_net_progress": float(last - first),
        "history_improve_steps": improve_steps,
        "history_plateau_from_iter": plateau_from,
    }


def _wl(delta: Any) -> str:
    d = _safe_float(delta)
    if d is None:
        return "na"
    if d > 1e-12:
        return "win"
    if d < -1e-12:
        return "loss"
    return "tie"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose RQ3 pheromone-update sensitivity from existing outputs")
    parser.add_argument("--suite-dir", required=True, help="Path to rq3_pheromone_update suite directory")
    parser.add_argument("--baseline-variant", default="w_rank_k3_m2_l0p7", help="Baseline variant for deltas/win-loss")
    parser.add_argument("--output-prefix", default=None, help="Output prefix (without extension)")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    if not suite_dir.exists():
        raise FileNotFoundError(f"Suite dir not found: {suite_dir}")
    out_prefix = Path(args.output_prefix).resolve() if args.output_prefix else (suite_dir / "pheromone_diagnostics")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    discovered = _discover_recommendations(suite_dir)
    if not discovered:
        raise RuntimeError(f"No recommendation.json found under {suite_dir}")

    rows: List[Dict[str, Any]] = []
    for item in discovered:
        variant = str(item["variant"])
        dataset_id = _normalize_id(item["dataset_id"])
        rec_path = Path(item["recommendation_path"])
        ds_dir = Path(item["dataset_dir"])
        try:
            payload = json.loads(rec_path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append(
                {
                    "variant": variant,
                    "dataset_id": dataset_id,
                    "recommendation_path": str(rec_path),
                    "parse_error": str(exc),
                }
            )
            continue

        final_eval = payload.get("final_evaluation", {}) if isinstance(payload, dict) else {}
        final_method = str(final_eval.get("method", "unknown"))
        final_score = _safe_float(final_eval.get("score", payload.get("final_performance")))
        proxy_score = _safe_float(payload.get("recommended_performance"))
        proxy_minus_final = (
            float(proxy_score - final_score)
            if proxy_score is not None and final_score is not None
            else np.nan
        )

        params = _parse_variant_params(payload, variant=variant)
        top_k = int(params["top_k_pheromone"]) if params["top_k_pheromone"] is not None else 3
        method = str(params["weight_method"] or "rank")

        result_pairs = _to_result_pairs(payload.get("aco_results"))
        n_candidates = len(result_pairs)
        unique_map: Dict[str, Tuple[Dict[str, Any], float]] = {}
        for cfg, score in result_pairs:
            key = _cfg_key(cfg)
            prev = unique_map.get(key)
            if prev is None or score > prev[1]:
                unique_map[key] = (cfg, score)
        unique_pairs = list(unique_map.values())
        unique_scores = np.asarray([sc for _cfg, sc in unique_pairs], dtype=float) if unique_pairs else np.asarray([], dtype=float)
        sorted_scores = np.sort(unique_scores)[::-1] if unique_scores.size else np.asarray([], dtype=float)
        top_scores = sorted_scores[: max(1, top_k)] if sorted_scores.size else np.asarray([], dtype=float)

        unique_count = int(len(unique_pairs))
        diversity_ratio = float(unique_count / n_candidates) if n_candidates > 0 else np.nan

        score_best = float(np.max(unique_scores)) if unique_scores.size else np.nan
        score_mean = float(np.mean(unique_scores)) if unique_scores.size else np.nan
        score_std = float(np.std(unique_scores)) if unique_scores.size else np.nan
        score_iqr = float(np.percentile(unique_scores, 75) - np.percentile(unique_scores, 25)) if unique_scores.size else np.nan
        score_top1_topk_gap = (
            float(top_scores[0] - top_scores[-1])
            if top_scores.size >= 2
            else 0.0 if top_scores.size == 1 else np.nan
        )

        weights = _reinforcement_weights(method=method, sorted_scores=top_scores.tolist())
        w_entropy, w_eff, w_top_share = _entropy_metrics(weights)

        hist_metrics = _history_metrics(ds_dir / "aco_history.csv")

        final_cfg = payload.get("pipeline_config") if isinstance(payload.get("pipeline_config"), dict) else {}
        final_fp = _pipeline_fingerprint(final_cfg) if final_cfg else ""
        final_hash = _cfg_hash(final_cfg) if final_cfg else ""
        best_proxy_hash = ""
        if unique_pairs:
            best_proxy_cfg = max(unique_pairs, key=lambda x: x[1])[0]
            best_proxy_hash = _cfg_hash(best_proxy_cfg)

        rows.append(
            {
                "variant": variant,
                "dataset_id": dataset_id,
                "recommendation_path": str(rec_path),
                "final_method": final_method,
                "final_score": final_score,
                "proxy_score": proxy_score,
                "proxy_minus_final": proxy_minus_final,
                "top_k_pheromone": top_k,
                "weight_method": method,
                "markov_order": params["markov_order"],
                "lambda_smooth": params["lambda_smooth"],
                "n_candidates": n_candidates,
                "n_unique_candidates": unique_count,
                "unique_ratio": diversity_ratio,
                "candidate_best_score": score_best,
                "candidate_mean_score": score_mean,
                "candidate_std_score": score_std,
                "candidate_iqr_score": score_iqr,
                "candidate_top1_topk_gap": score_top1_topk_gap,
                "reinforce_entropy_norm": w_entropy,
                "reinforce_effective_count": w_eff,
                "reinforce_top_share": w_top_share,
                "final_pipeline_hash": final_hash,
                "final_pipeline_fingerprint": final_fp,
                "best_proxy_pipeline_hash": best_proxy_hash,
                **hist_metrics,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No analyzable rows produced.")

    base = (
        df[df["variant"] == args.baseline_variant][["dataset_id", "final_score", "final_pipeline_hash"]]
        .drop_duplicates(subset=["dataset_id"], keep="last")
        .rename(
            columns={
                "final_score": "baseline_score",
                "final_pipeline_hash": "baseline_pipeline_hash",
            }
        )
    )
    df = df.merge(base, on="dataset_id", how="left")
    df["delta_vs_baseline"] = df["final_score"] - df["baseline_score"]
    df["wl_vs_baseline"] = df["delta_vs_baseline"].map(_wl)
    df["changed_pipeline_vs_baseline"] = (
        df["final_pipeline_hash"].notna()
        & df["baseline_pipeline_hash"].notna()
        & (df["final_pipeline_hash"] != df["baseline_pipeline_hash"])
    )
    df["proxy_rank_gap"] = df["candidate_best_score"] - df["final_score"]

    variant_summary = (
        df.groupby("variant", dropna=False)
        .agg(
            n_datasets=("dataset_id", "nunique"),
            mean_final_score=("final_score", "mean"),
            mean_proxy_minus_final=("proxy_minus_final", "mean"),
            mean_n_unique=("n_unique_candidates", "mean"),
            mean_unique_ratio=("unique_ratio", "mean"),
            mean_score_std=("candidate_std_score", "mean"),
            mean_top1_topk_gap=("candidate_top1_topk_gap", "mean"),
            mean_reinforce_entropy=("reinforce_entropy_norm", "mean"),
            mean_reinforce_top_share=("reinforce_top_share", "mean"),
            mean_history_points=("history_points", "mean"),
            mean_history_net_progress=("history_net_progress", "mean"),
            mean_delta_vs_baseline=("delta_vs_baseline", "mean"),
            win_count=("wl_vs_baseline", lambda s: int((s == "win").sum())),
            tie_count=("wl_vs_baseline", lambda s: int((s == "tie").sum())),
            loss_count=("wl_vs_baseline", lambda s: int((s == "loss").sum())),
            changed_pipeline_count=("changed_pipeline_vs_baseline", lambda s: int(pd.Series(s).fillna(False).sum())),
        )
        .reset_index()
        .sort_values("mean_final_score", ascending=False)
    )

    fixed_rank = (
        df[
            (df["weight_method"] == "rank")
            & (pd.to_numeric(df["markov_order"], errors="coerce") == 2)
            & (np.isclose(pd.to_numeric(df["lambda_smooth"], errors="coerce"), 0.7, equal_nan=False))
        ]
        .groupby("top_k_pheromone", dropna=False)
        .agg(
            n=("dataset_id", "nunique"),
            mean_final_score=("final_score", "mean"),
            mean_proxy_minus_final=("proxy_minus_final", "mean"),
            mean_unique_ratio=("unique_ratio", "mean"),
            mean_reinforce_entropy=("reinforce_entropy_norm", "mean"),
            mean_reinforce_top_share=("reinforce_top_share", "mean"),
            mean_history_net_progress=("history_net_progress", "mean"),
        )
        .reset_index()
        .sort_values("top_k_pheromone")
    )

    fixed_k3 = (
        df[
            (pd.to_numeric(df["top_k_pheromone"], errors="coerce") == 3)
            & (pd.to_numeric(df["markov_order"], errors="coerce") == 2)
            & (np.isclose(pd.to_numeric(df["lambda_smooth"], errors="coerce"), 0.7, equal_nan=False))
        ]
        .groupby("weight_method", dropna=False)
        .agg(
            n=("dataset_id", "nunique"),
            mean_final_score=("final_score", "mean"),
            mean_proxy_minus_final=("proxy_minus_final", "mean"),
            mean_unique_ratio=("unique_ratio", "mean"),
            mean_reinforce_entropy=("reinforce_entropy_norm", "mean"),
            mean_reinforce_top_share=("reinforce_top_share", "mean"),
            mean_history_net_progress=("history_net_progress", "mean"),
        )
        .reset_index()
        .sort_values("mean_final_score", ascending=False)
    )

    per_dataset_sensitivity = (
        df.groupby("dataset_id", dropna=False)
        .agg(
            n_variants=("variant", "nunique"),
            final_min=("final_score", "min"),
            final_max=("final_score", "max"),
            final_range=("final_score", lambda s: float(np.nanmax(s.to_numpy(dtype=float)) - np.nanmin(s.to_numpy(dtype=float)))),
            mean_proxy_minus_final=("proxy_minus_final", "mean"),
            changed_pipeline_rate=("changed_pipeline_vs_baseline", lambda s: float(pd.Series(s).fillna(False).mean())),
        )
        .reset_index()
        .sort_values("final_range", ascending=False)
    )

    rows_path = out_prefix.with_name(out_prefix.name + "_rows.csv")
    variant_path = out_prefix.with_name(out_prefix.name + "_variant_summary.csv")
    fixed_rank_path = out_prefix.with_name(out_prefix.name + "_fixed_rank_m2_l0p7_k_summary.csv")
    fixed_k3_path = out_prefix.with_name(out_prefix.name + "_fixed_k3_m2_l0p7_weight_summary.csv")
    sens_path = out_prefix.with_name(out_prefix.name + "_dataset_sensitivity.csv")

    df.to_csv(rows_path, index=False)
    variant_summary.to_csv(variant_path, index=False)
    fixed_rank.to_csv(fixed_rank_path, index=False)
    fixed_k3.to_csv(fixed_k3_path, index=False)
    per_dataset_sensitivity.to_csv(sens_path, index=False)

    print(f"Saved: {rows_path}")
    print(f"Saved: {variant_path}")
    print(f"Saved: {fixed_rank_path}")
    print(f"Saved: {fixed_k3_path}")
    print(f"Saved: {sens_path}")
    print("\nVariant summary:")
    print(variant_summary.to_string(index=False))
    print("\nFixed-rank (m=2, lambda=0.7) K sweep:")
    print(fixed_rank.to_string(index=False))
    print("\nFixed-k=3 (m=2, lambda=0.7) weight sweep:")
    print(fixed_k3.to_string(index=False))
    print("\nPer-dataset sensitivity:")
    print(per_dataset_sensitivity.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
