#!/usr/bin/env python3
"""Diagnostics for RQ3 transfer-neighbor (top-K / top-L) sensitivity.

This script analyzes existing recommendation artifacts (no rerun) and produces:
  - per-row diagnostics for each (variant, dataset)
  - variant-level summary
  - fixed-L and fixed-K trend summaries

Key diagnostics:
  - similarity spread among top-K neighbors
  - similarity-weight entropy / effective-neighbor count
  - top-L pipeline overlap (pairwise Jaccard) and duplication ratio
  - normalized eta concentration margins per preprocessing step
  - final-score win/tie/loss vs baseline variant
"""
from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from automl_aco.metalearning.recommender import MetaPipelineRecommender
from automl_aco.search.heuristics import (
    aggregate_operator_heuristics,
    build_transfer_candidates,
    compute_similarity_weights,
    initialize_aco_with_transferred_eta,
    select_top_k_neighbors,
    select_top_l_pipelines_per_neighbor,
)
from automl_aco.utils.operator_spec import base_operator_name


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _normalize_id(value: Any) -> str:
    if pd.isna(value):
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


def _meta_overlap_count(meta_index_like: pd.Series, perf_df: pd.DataFrame) -> int:
    perf_norm = {_normalize_id(c) for c in perf_df.columns}
    vals = meta_index_like.astype(str).map(_normalize_id)
    return len(set(vals) & perf_norm)


def _maybe_set_meta_index(meta_df: pd.DataFrame, perf_df: pd.DataFrame, explicit_col: Optional[str]) -> pd.DataFrame:
    best_overlap = _meta_overlap_count(pd.Series(meta_df.index.astype(str)), perf_df)
    best_source: Tuple[str, Optional[str]] = ("index", None)

    candidate_cols = list(meta_df.columns)
    if explicit_col:
        if explicit_col not in meta_df.columns:
            raise ValueError(f"--metafeatures-id-column={explicit_col!r} not found")
        candidate_cols = [explicit_col]
    else:
        prioritized = ["dataset_id", "did", "openml_id", "id", "Dataset", "dataset", "Unnamed: 0"]
        candidate_cols = [c for c in prioritized if c in meta_df.columns] + [
            c for c in meta_df.columns if c not in prioritized
        ]

    for col in candidate_cols:
        overlap = _meta_overlap_count(meta_df[col], perf_df)
        if overlap > best_overlap:
            best_overlap = overlap
            best_source = ("column", col)

    if best_source[0] == "column" and best_source[1] is not None and best_overlap > 0:
        return meta_df.set_index(str(best_source[1]))
    return meta_df


def _infer_pipeline_config_from_name(pipeline_name: str) -> Dict[str, Any]:
    token = str(pipeline_name).strip().lower()
    cfg: Dict[str, Any] = {
        "name": str(pipeline_name),
        "imputation": "none",
        "scaling": "none",
        "encoding": "onehot",
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "none",
    }
    if "knn" in token:
        cfg["imputation"] = "knn"
    elif "mostfreq" in token or "most_frequent" in token:
        cfg["imputation"] = "most_frequent"
    elif "constant" in token:
        cfg["imputation"] = "constant"
    elif "median" in token:
        cfg["imputation"] = "median"
    elif "mean" in token:
        cfg["imputation"] = "mean"

    if "no_scale" in token:
        cfg["scaling"] = "none"
    elif "robust" in token:
        cfg["scaling"] = "robust"
    elif "minmax" in token or "uniform" in token or "quantile" in token:
        cfg["scaling"] = "minmax"
    elif "maxabs" in token:
        cfg["scaling"] = "maxabs"
    elif "standard" in token:
        cfg["scaling"] = "standard"

    if "mutualinfo" in token or "mutual_info" in token:
        cfg["feature_selection"] = "mutual_info"
    elif "kbest" in token or "k_best" in token:
        cfg["feature_selection"] = "k_best"
    elif "variance" in token:
        cfg["feature_selection"] = "variance_threshold"

    if "iforest" in token or "isolation" in token:
        cfg["outlier_removal"] = "isolation_forest"
    elif "zscore" in token:
        cfg["outlier_removal"] = "zscore"
    elif "iqr" in token:
        cfg["outlier_removal"] = "iqr"
    elif "lof" in token:
        cfg["outlier_removal"] = "lof"

    if "pca" in token:
        cfg["dimensionality_reduction"] = "pca"
    elif "svd" in token:
        cfg["dimensionality_reduction"] = "svd"
    return cfg


def _discover_recommendations(suite_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant_dir in sorted(suite_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name

        # Single-run layout: <variant>/recommendation.json
        single = variant_dir / "recommendation.json"
        if single.exists():
            rows.append(
                {
                    "variant": variant,
                    "dataset_id": "unknown",
                    "recommendation_path": str(single),
                }
            )

        # Batch layout: <variant>/dataset_<id>/recommendation.json
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
                }
            )
    return rows


def _jaccard(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return None
    return float(len(sa & sb) / len(sa | sb))


def _entropy_from_weights(weights: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(weights) == 0:
        return None, None
    if len(weights) == 1:
        return 0.0, 1.0
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-15, None)
    w = w / w.sum()
    h = float(-(w * np.log(w)).sum())
    h_norm = float(h / np.log(len(w)))
    n_eff = float(np.exp(h))
    return h_norm, n_eff


def _parse_k_l(rec_payload: Mapping[str, Any], variant_name: str) -> Tuple[Optional[int], Optional[int]]:
    hp = rec_payload.get("search_hyperparams", {}) if isinstance(rec_payload, dict) else {}
    k = hp.get("heuristic_top_k")
    l = hp.get("heuristic_top_l")
    try:
        k_i = int(k) if k is not None else None
    except Exception:
        k_i = None
    try:
        l_i = int(l) if l is not None else None
    except Exception:
        l_i = None
    if k_i is not None and l_i is not None:
        return k_i, l_i

    m = re.fullmatch(r"K(\d+)_L(\d+)", str(variant_name))
    if m:
        return int(m.group(1)), int(m.group(2))
    return k_i, l_i


def _prepare_recommender(
    *,
    perf_path: Path,
    meta_path: Path,
    pipeline_configs_path: Path,
    meta_id_col: Optional[str],
) -> MetaPipelineRecommender:
    perf = pd.read_csv(perf_path, index_col=0)
    meta_raw = pd.read_csv(meta_path)
    meta = _maybe_set_meta_index(meta_raw, perf, explicit_col=meta_id_col)
    with pipeline_configs_path.open("r", encoding="utf-8") as f:
        pipeline_configs: List[Dict[str, Any]] = json.load(f)
    existing_names = {str(cfg.get("name")) for cfg in pipeline_configs if isinstance(cfg, dict) and "name" in cfg}
    missing = [str(name) for name in perf.index if str(name) not in existing_names]
    if missing:
        pipeline_configs = list(pipeline_configs) + [_infer_pipeline_config_from_name(name) for name in missing]
    return MetaPipelineRecommender(perf, meta, pipeline_configs, verbose=False)


def _build_new_mf_scaled(recommender: MetaPipelineRecommender, dataset_id: str) -> Optional[np.ndarray]:
    ds = _normalize_id(dataset_id)
    if ds not in recommender.metafeatures_df.index:
        return None
    row = recommender.metafeatures_df.loc[ds]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    new_mf_df = pd.DataFrame([row.to_dict()]).reindex(columns=recommender.metafeatures_df.columns, fill_value=0)
    new_mf_imputed = recommender.imputer.transform(new_mf_df)
    new_mf_scaled = recommender.scaler.transform(new_mf_imputed).ravel()
    return np.asarray(new_mf_scaled, dtype=float)


def _diagnose_one(
    *,
    recommender: MetaPipelineRecommender,
    dataset_id: str,
    options: Mapping[str, Sequence[str]],
    top_k: int,
    top_l: int,
    dataset_weighting: str,
    sim_temperature: float,
    eta_floor: float,
) -> Dict[str, Any]:
    new_mf_scaled = _build_new_mf_scaled(recommender, dataset_id=dataset_id)
    if new_mf_scaled is None:
        return {"diag_error": f"dataset_id {dataset_id} missing in metafeatures after alignment"}

    sims = recommender._compute_dataset_similarities(new_mf_scaled)
    top_neighbors = select_top_k_neighbors(
        dataset_similarities=sims,
        top_k=max(1, int(top_k)),
        query_dataset_id=dataset_id,
    )
    weights_map = compute_similarity_weights(
        top_k_neighbors=top_neighbors,
        dataset_weighting=dataset_weighting,
        similarity_temperature=float(sim_temperature),
    )
    weights = [float(weights_map.get(ds, 0.0)) for ds, _ in top_neighbors]
    sim_vals = [float(s) for _ds, s in top_neighbors]
    sim_spread = float(max(sim_vals) - min(sim_vals)) if sim_vals else np.nan
    sim_gap12 = float(sim_vals[0] - sim_vals[1]) if len(sim_vals) >= 2 else np.nan
    weight_entropy_norm, effective_neighbors = _entropy_from_weights(weights)

    top_l_pipelines = select_top_l_pipelines_per_neighbor(
        performance_matrix=recommender.performance_matrix_imputed,
        top_k_neighbors=top_neighbors,
        top_l=max(1, int(top_l)),
        score_direction="higher_is_better",
    )
    pipeline_lists: List[List[str]] = []
    for ds, _sim in top_neighbors:
        rows = top_l_pipelines.get(ds, [])
        pipeline_lists.append([str(r.get("pipeline")) for r in rows if isinstance(r, dict)])

    jaccards: List[float] = []
    for a, b in combinations(pipeline_lists, 2):
        j = _jaccard(a, b)
        if j is not None:
            jaccards.append(float(j))
    mean_jaccard = float(np.mean(jaccards)) if jaccards else np.nan

    all_pipes = [p for lst in pipeline_lists for p in lst]
    unique_pipes = len(set(all_pipes))
    total_slots = len(all_pipes)
    dup_ratio = float(1.0 - unique_pipes / total_slots) if total_slots > 0 else np.nan

    transfer_candidates = build_transfer_candidates(
        performance_matrix=recommender.performance_matrix_imputed,
        top_k_neighbors=top_neighbors,
        top_l_pipelines=top_l_pipelines,
        similarity_weights=weights_map,
        score_direction="higher_is_better",
    )
    raw_eta = aggregate_operator_heuristics(
        transfer_candidates=transfer_candidates,
        pipeline_configs=recommender.pipeline_configs,
        options=options,
    )
    eta_norm = initialize_aco_with_transferred_eta(raw_eta=raw_eta, eta_floor=float(eta_floor))

    step_rows: Dict[str, Any] = {}
    margins: List[float] = []
    for step, op_list in options.items():
        vals = np.asarray(eta_norm.get(step, np.ones(len(op_list), dtype=float)), dtype=float)
        if vals.size == 0:
            continue
        idx = int(np.argmax(vals))
        top_val = float(vals[idx])
        second_val = float(np.partition(vals, -2)[-2]) if vals.size >= 2 else top_val
        margin = float(top_val - second_val)
        margins.append(margin)
        step_rows[f"eta_topop_{step}"] = str(op_list[idx]) if idx < len(op_list) else ""
        step_rows[f"eta_margin_{step}"] = margin
        step_rows[f"eta_range_{step}"] = float(np.max(vals) - np.min(vals))

    return {
        "neighbor_count": len(top_neighbors),
        "sim_spread": sim_spread,
        "sim_gap12": sim_gap12,
        "weight_entropy_norm": weight_entropy_norm,
        "effective_neighbors": effective_neighbors,
        "topl_pairwise_jaccard": mean_jaccard,
        "topl_dup_ratio": dup_ratio,
        "topl_unique_pipelines": unique_pipes,
        "topl_total_slots": total_slots,
        "eta_margin_mean": float(np.mean(margins)) if margins else np.nan,
        **step_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose RQ3 transfer-neighbor sensitivity from existing outputs")
    parser.add_argument("--suite-dir", required=True, help="Path to rq3_transfer_neighbors suite directory")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Repo root path")
    parser.add_argument("--performance-matrix", default=None, help="Override performance matrix path")
    parser.add_argument("--metafeatures", default=None, help="Override metafeatures path")
    parser.add_argument("--pipeline-configs", default=None, help="Override pipeline configs path")
    parser.add_argument("--metafeatures-id-column", default=None, help="Optional metafeatures id column")
    parser.add_argument("--baseline-variant", default="K5_L3", help="Baseline variant for delta/win-loss")
    parser.add_argument("--output-prefix", default=None, help="Output prefix (without extension)")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    if not suite_dir.exists():
        raise FileNotFoundError(f"Suite dir not found: {suite_dir}")

    root = Path(args.root).resolve()
    perf_path = Path(args.performance_matrix) if args.performance_matrix else (root / "data/openml/training_performance_matrix_autogluon.csv")
    meta_path = Path(args.metafeatures) if args.metafeatures else (root / "data/openml/dataset_feats.csv")
    pipeline_configs_path = Path(args.pipeline_configs) if args.pipeline_configs else (root / "aco/pipeline_configs.json")
    out_prefix = Path(args.output_prefix).resolve() if args.output_prefix else (suite_dir / "transfer_neighbor_diagnostics")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    recommender = _prepare_recommender(
        perf_path=perf_path,
        meta_path=meta_path,
        pipeline_configs_path=pipeline_configs_path,
        meta_id_col=args.metafeatures_id_column,
    )

    discovered = _discover_recommendations(suite_dir)
    if not discovered:
        raise RuntimeError(f"No recommendation.json found under {suite_dir}")

    rows: List[Dict[str, Any]] = []
    for item in discovered:
        variant = str(item["variant"])
        dataset_id = str(item["dataset_id"])
        rec_path = Path(str(item["recommendation_path"]))
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
        final_score = _safe_float(final_eval.get("score", payload.get("final_performance")))
        proxy_score = _safe_float(payload.get("recommended_performance"))
        final_method = str(final_eval.get("method", "unknown"))
        proxy_minus_final = (
            float(proxy_score - final_score)
            if proxy_score is not None and final_score is not None
            else np.nan
        )

        k_val, l_val = _parse_k_l(payload, variant_name=variant)
        hp = payload.get("search_hyperparams", {}) if isinstance(payload, dict) else {}
        dataset_weighting = str(hp.get("dataset_weighting", "similarity"))
        sim_temperature = _safe_float(hp.get("heuristic_similarity_temperature"))
        eta_floor = _safe_float(hp.get("heuristic_eta_floor"))
        options = payload.get("search_options", {})
        if not isinstance(options, dict) or not options:
            rows.append(
                {
                    "variant": variant,
                    "dataset_id": dataset_id,
                    "recommendation_path": str(rec_path),
                    "final_method": final_method,
                    "final_score": final_score,
                    "proxy_score": proxy_score,
                    "proxy_minus_final": proxy_minus_final,
                    "heuristic_top_k": k_val,
                    "heuristic_top_l": l_val,
                    "diag_error": "missing_search_options",
                }
            )
            continue

        if k_val is None:
            k_val = 5
        if l_val is None:
            l_val = 3
        if sim_temperature is None:
            sim_temperature = 1.0
        if eta_floor is None:
            eta_floor = 0.05

        diag = _diagnose_one(
            recommender=recommender,
            dataset_id=dataset_id,
            options=options,
            top_k=int(k_val),
            top_l=int(l_val),
            dataset_weighting=dataset_weighting,
            sim_temperature=float(sim_temperature),
            eta_floor=float(eta_floor),
        )
        row = {
            "variant": variant,
            "dataset_id": dataset_id,
            "recommendation_path": str(rec_path),
            "final_method": final_method,
            "final_score": final_score,
            "proxy_score": proxy_score,
            "proxy_minus_final": proxy_minus_final,
            "heuristic_top_k": int(k_val),
            "heuristic_top_l": int(l_val),
            "dataset_weighting": dataset_weighting,
            "sim_temperature": float(sim_temperature),
            "eta_floor": float(eta_floor),
        }
        row.update(diag)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No analyzable rows produced.")

    # Baseline deltas and win/tie/loss
    base = (
        df[df["variant"] == args.baseline_variant][["dataset_id", "final_score"]]
        .drop_duplicates(subset=["dataset_id"], keep="last")
        .rename(columns={"final_score": "baseline_score"})
    )
    df = df.merge(base, on="dataset_id", how="left")
    df["delta_vs_baseline"] = df["final_score"] - df["baseline_score"]
    df["wl_vs_baseline"] = np.where(
        df["delta_vs_baseline"] > 1e-12,
        "win",
        np.where(df["delta_vs_baseline"] < -1e-12, "loss", "tie"),
    )

    # Variant summary
    variant_summary = (
        df.groupby("variant", dropna=False)
        .agg(
            n_datasets=("dataset_id", "nunique"),
            mean_final_score=("final_score", "mean"),
            mean_proxy_minus_final=("proxy_minus_final", "mean"),
            mean_sim_spread=("sim_spread", "mean"),
            mean_sim_gap12=("sim_gap12", "mean"),
            mean_weight_entropy=("weight_entropy_norm", "mean"),
            mean_effective_neighbors=("effective_neighbors", "mean"),
            mean_topl_jaccard=("topl_pairwise_jaccard", "mean"),
            mean_topl_dup_ratio=("topl_dup_ratio", "mean"),
            mean_eta_margin=("eta_margin_mean", "mean"),
            win_count=("wl_vs_baseline", lambda s: int((s == "win").sum())),
            tie_count=("wl_vs_baseline", lambda s: int((s == "tie").sum())),
            loss_count=("wl_vs_baseline", lambda s: int((s == "loss").sum())),
        )
        .reset_index()
        .sort_values("mean_final_score", ascending=False)
    )

    # Fixed-L and Fixed-K trend summaries (when available)
    fixed_l = (
        df[df["heuristic_top_l"] == 3]
        .groupby("heuristic_top_k", dropna=False)
        .agg(
            n=("dataset_id", "nunique"),
            mean_final_score=("final_score", "mean"),
            mean_weight_entropy=("weight_entropy_norm", "mean"),
            mean_topl_jaccard=("topl_pairwise_jaccard", "mean"),
            mean_eta_margin=("eta_margin_mean", "mean"),
        )
        .reset_index()
        .sort_values("heuristic_top_k")
    )
    fixed_k = (
        df[df["heuristic_top_k"] == 5]
        .groupby("heuristic_top_l", dropna=False)
        .agg(
            n=("dataset_id", "nunique"),
            mean_final_score=("final_score", "mean"),
            mean_weight_entropy=("weight_entropy_norm", "mean"),
            mean_topl_jaccard=("topl_pairwise_jaccard", "mean"),
            mean_eta_margin=("eta_margin_mean", "mean"),
        )
        .reset_index()
        .sort_values("heuristic_top_l")
    )

    rows_path = out_prefix.with_name(out_prefix.name + "_rows.csv")
    variant_path = out_prefix.with_name(out_prefix.name + "_variant_summary.csv")
    fixed_l_path = out_prefix.with_name(out_prefix.name + "_fixedL3_summary.csv")
    fixed_k_path = out_prefix.with_name(out_prefix.name + "_fixedK5_summary.csv")

    df.to_csv(rows_path, index=False)
    variant_summary.to_csv(variant_path, index=False)
    fixed_l.to_csv(fixed_l_path, index=False)
    fixed_k.to_csv(fixed_k_path, index=False)

    print(f"Saved: {rows_path}")
    print(f"Saved: {variant_path}")
    print(f"Saved: {fixed_l_path}")
    print(f"Saved: {fixed_k_path}")
    print("\nVariant summary:")
    print(variant_summary.to_string(index=False))
    print("\nFixed L=3 trend:")
    print(fixed_l.to_string(index=False))
    print("\nFixed K=5 trend:")
    print(fixed_k.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
