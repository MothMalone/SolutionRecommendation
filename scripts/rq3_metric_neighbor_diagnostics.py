"""Diagnose Siamese metric neighbor collapse.

This script does not evaluate pipelines or run AutoGluon. It only checks whether
nearest-neighbor retrieval collapses to one dataset under the learned metric.

It reports three retrieval modes:
1. cosine_scaled: existing fallback cosine over imputed/minmax metafeatures.
2. siamese_scaled_production: production learned metric path over
   imputed/minmax metafeatures.
3. siamese_raw_unscaled_probe: the same learned metric evaluated on raw-filled
   metafeatures as a negative-control probe.

If mode 2 collapses to a single top neighbor, the learned metric itself has a
hubness/collapse problem. If only mode 3 collapses, raw unscaled metafeatures are
unsafe for this metric.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automl_aco.metalearning.recommender import MetaPipelineRecommender
from automl_aco.search.heuristics import select_top_k_neighbors


def _normalize_id(val: object) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        f = float(val)
        if np.isfinite(f) and abs(f - round(f)) <= 1e-9:
            return str(int(round(f)))
        return str(val).strip()
    text = str(val).strip()
    float_like = re.fullmatch(r"([0-9]+)\.0+", text)
    if float_like:
        return float_like.group(1)
    prefixed = re.fullmatch(r"(?i)(?:d|dataset|openml)[_\-: ]*([0-9]+)", text)
    if prefixed:
        return prefixed.group(1)
    return text


def _maybe_set_meta_index(meta_df: pd.DataFrame, perf_df: pd.DataFrame, explicit_col: Optional[str]) -> pd.DataFrame:
    perf_norm = {_normalize_id(c) for c in perf_df.columns}

    def overlap(values: Iterable[object]) -> int:
        return len({_normalize_id(v) for v in values} & perf_norm)

    if explicit_col:
        if explicit_col not in meta_df.columns:
            raise ValueError(f"--metafeatures-id-column={explicit_col!r} not found")
        return meta_df.set_index(explicit_col)

    best_col: Optional[str] = None
    best_overlap = overlap(meta_df.index)
    prioritized = ["dataset_id", "did", "openml_id", "id", "Dataset", "dataset", "Unnamed: 0"]
    candidate_cols = [c for c in prioritized if c in meta_df.columns] + [c for c in meta_df.columns if c not in prioritized]
    for col in candidate_cols:
        cur = overlap(meta_df[col])
        if cur > best_overlap:
            best_overlap = cur
            best_col = str(col)
    if best_col is not None and best_overlap > 0:
        return meta_df.set_index(best_col)
    return meta_df


def _parse_dataset_ids(raw: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for token in raw or []:
        for piece in str(token).split(","):
            text = piece.strip()
            if text:
                out.append(_normalize_id(text))
    return out


def _dummy_configs(perf: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "name": str(name),
            "imputation": "none",
            "scaling": "none",
            "encoding": "onehot",
            "feature_selection": "none",
            "outlier_removal": "none",
            "dimensionality_reduction": "none",
        }
        for name in perf.index
    ]


def _lookup_raw_meta(recommender: MetaPipelineRecommender, dataset_id: str) -> np.ndarray:
    ds_norm = _normalize_id(dataset_id)
    for idx in recommender.metafeatures_df.index:
        if _normalize_id(idx) == ds_norm:
            return recommender.metafeatures_df.loc[idx].fillna(0).to_numpy(dtype=np.float32, copy=False)
    raise KeyError(f"Dataset {dataset_id} absent from aligned metafeatures")


def _lookup_scaled_meta(recommender: MetaPipelineRecommender, dataset_id: str) -> np.ndarray:
    raw = pd.DataFrame([recommender.metafeatures_df.loc[dataset_id]]).reindex(
        columns=recommender.metafeatures_df.columns,
        fill_value=0,
    )
    return recommender.scaler.transform(recommender.imputer.transform(raw)).ravel()


def _cosine_scaled_sims(recommender: MetaPipelineRecommender, dataset_id: str) -> List[Tuple[Any, float]]:
    new_scaled = _lookup_scaled_meta(recommender, dataset_id)
    old_metric_type = recommender.metric_type
    old_embedder = recommender.embedder
    old_projector = recommender.projector
    try:
        recommender.metric_type = None
        recommender.embedder = None
        recommender.projector = None
        return recommender._compute_dataset_similarities(new_scaled)
    finally:
        recommender.metric_type = old_metric_type
        recommender.embedder = old_embedder
        recommender.projector = old_projector


def _siamese_scaled_production_sims(recommender: MetaPipelineRecommender, dataset_id: str) -> List[Tuple[Any, float]]:
    new_scaled = _lookup_scaled_meta(recommender, dataset_id)
    return recommender._compute_dataset_similarities(new_scaled)


def _siamese_raw_unscaled_sims(recommender: MetaPipelineRecommender, dataset_id: str) -> List[Tuple[Any, float]]:
    if recommender.metric_type != "regression" or recommender.embedder is None or recommender.projector is None:
        raise RuntimeError("No trained/loaded Siamese regression metric available")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch required for Siamese diagnostic") from exc

    known_raw = recommender.metafeatures_df.fillna(0).to_numpy(dtype=np.float32, copy=False)
    new_raw = _lookup_raw_meta(recommender, dataset_id).reshape(1, -1)
    with torch.no_grad():
        known_tensor = torch.tensor(known_raw, dtype=torch.float32)
        new_tensor = torch.tensor(new_raw, dtype=torch.float32)
        emb_known = recommender.embedder(known_tensor)
        emb_new = recommender.embedder(new_tensor).squeeze(0)
        emb_known = emb_known / (emb_known.norm(dim=1, keepdim=True) + 1e-8)
        emb_new = emb_new / (emb_new.norm() + 1e-8)
        sims: List[Tuple[Any, float]] = []
        objective = "projector_product"
        if isinstance(recommender.metric_params, dict):
            objective = str(recommender.metric_params.get("metric_objective", objective))
        for ds_id, h_known in zip(recommender.metafeatures_df.index, emb_known):
            if objective == "embedding_cosine":
                sim = float((emb_new * h_known).sum().item())
            else:
                inter = (emb_new * h_known).unsqueeze(0)
                sim = float(recommender.projector(inter).item())
            sims.append((ds_id, sim))
        return sims


def _summarize_scores(sims: Sequence[Tuple[Any, float]], query_id: str, top_k: int) -> Dict[str, Any]:
    finite = np.asarray([float(s) for _ds, s in sims if np.isfinite(float(s))], dtype=float)
    selected = select_top_k_neighbors(sims, top_k=top_k, query_dataset_id=query_id)
    row: Dict[str, Any] = {
        "top1_neighbor": str(selected[0][0]) if selected else "",
        "top1_similarity": float(selected[0][1]) if selected else np.nan,
        "top5_neighbors": ";".join(str(ds) for ds, _sim in selected[:5]),
        "score_mean": float(np.mean(finite)) if finite.size else np.nan,
        "score_std": float(np.std(finite)) if finite.size else np.nan,
        "score_min": float(np.min(finite)) if finite.size else np.nan,
        "score_max": float(np.max(finite)) if finite.size else np.nan,
        "score_range": float(np.max(finite) - np.min(finite)) if finite.size else np.nan,
    }
    if len(selected) >= 2:
        row["top1_top2_margin"] = float(selected[0][1] - selected[1][1])
    else:
        row["top1_top2_margin"] = np.nan
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Siamese neighbor collapse")
    parser.add_argument("--performance-matrix", required=True)
    parser.add_argument("--metafeatures", required=True)
    parser.add_argument("--metafeatures-id-column", default=None)
    parser.add_argument("--dataset-ids", nargs="+", default=None)
    parser.add_argument("--metric-path", default=None)
    parser.add_argument("--train-metric-inline", action="store_true")
    parser.add_argument("--metric-hidden-dim", type=int, default=32)
    parser.add_argument("--metric-embed-dim", type=int, default=32)
    parser.add_argument("--metric-epochs", type=int, default=100)
    parser.add_argument("--metric-lr", type=float, default=1e-3)
    parser.add_argument(
        "--metric-objective",
        choices=["embedding_cosine", "projector_product"],
        default="embedding_cosine",
        help=(
            "embedding_cosine trains the embedding space directly; projector_product "
            "keeps the older projector(emb_i * emb_j) objective."
        ),
    )
    parser.add_argument(
        "--metric-similarity-target",
        choices=["rank_cosine", "row_zscore_cosine", "row_minmax_cosine", "legacy_global_zscore_cosine"],
        default="legacy_global_zscore_cosine",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="/kaggle/working/rq3_metric_neighbor_diagnostics")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    perf = pd.read_csv(args.performance_matrix, index_col=0)
    raw_meta = pd.read_csv(args.metafeatures)
    meta = _maybe_set_meta_index(raw_meta, perf, args.metafeatures_id_column)
    recommender = MetaPipelineRecommender(perf, meta, _dummy_configs(perf), verbose=bool(args.verbose))

    if args.metric_path:
        recommender.load_metric(args.metric_path)
        metric_mode = "loaded"
    elif args.train_metric_inline:
        metric_mode = "trained_inline"
        if args.verbose:
            print(
                "Training metric: "
                f"hidden_dim={args.metric_hidden_dim}, embed_dim={args.metric_embed_dim}, "
                f"epochs={args.metric_epochs}, target={args.metric_similarity_target}, "
                f"objective={args.metric_objective}"
            )
        recommender.train_metric(
            method="regression",
            hidden_dim=int(args.metric_hidden_dim),
            embed_dim=int(args.metric_embed_dim),
            epochs=int(args.metric_epochs),
            lr=float(args.metric_lr),
            seed=int(args.seed),
            similarity_target=str(args.metric_similarity_target),
            score_direction="higher_is_better",
            metric_objective=str(args.metric_objective),
        )
    else:
        metric_mode = "none"

    requested = _parse_dataset_ids(args.dataset_ids)
    if requested:
        dataset_ids = [ds for ds in requested if ds in {_normalize_id(x) for x in recommender.metafeatures_df.index}]
    else:
        dataset_ids = [str(x) for x in recommender.metafeatures_df.index]

    # Map normalized IDs back to exact aligned index labels.
    norm_to_idx = {_normalize_id(idx): str(idx) for idx in recommender.metafeatures_df.index}
    dataset_ids = [norm_to_idx.get(_normalize_id(ds), str(ds)) for ds in dataset_ids]

    rows: List[Dict[str, Any]] = []
    modes = ["cosine_scaled"]
    if recommender.metric_type == "regression":
        modes.extend(["siamese_scaled_production", "siamese_raw_unscaled_probe"])

    for dataset_id in dataset_ids:
        for mode in modes:
            try:
                if mode == "cosine_scaled":
                    sims = _cosine_scaled_sims(recommender, dataset_id)
                elif mode == "siamese_scaled_production":
                    sims = _siamese_scaled_production_sims(recommender, dataset_id)
                else:
                    sims = _siamese_raw_unscaled_sims(recommender, dataset_id)
                row = _summarize_scores(sims, dataset_id, int(args.top_k))
                row.update({"dataset_id": str(dataset_id), "mode": mode, "status": "ok", "error": ""})
            except Exception as exc:
                row = {"dataset_id": str(dataset_id), "mode": mode, "status": "failed", "error": str(exc)}
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metric_neighbor_diagnostics.csv", index=False)

    count_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "metric_mode": metric_mode,
        "n_query_datasets": int(len(dataset_ids)),
        "modes": modes,
        "metric_objective": str(args.metric_objective) if recommender.metric_type == "regression" else None,
        "training_preprocessing_note": (
            "Metric training now uses the same imputed/minmax-scaled metafeatures as production "
            "inference. The raw_unscaled probe intentionally feeds raw fillna(0) metafeatures as a "
            "negative-control check for preprocessing sensitivity."
        ),
    }
    for mode, group in df[df["status"] == "ok"].groupby("mode"):
        counts = Counter(group["top1_neighbor"].astype(str))
        n = int(len(group))
        top_neighbor, top_count = counts.most_common(1)[0] if counts else ("", 0)
        summary[f"{mode}_dominant_top1_neighbor"] = top_neighbor
        summary[f"{mode}_dominant_top1_fraction"] = float(top_count / max(1, n))
        summary[f"{mode}_mean_score_std"] = float(pd.to_numeric(group["score_std"], errors="coerce").mean())
        summary[f"{mode}_mean_top1_top2_margin"] = float(pd.to_numeric(group["top1_top2_margin"], errors="coerce").mean())
        for neighbor, count in counts.most_common(20):
            count_rows.append({"mode": mode, "top1_neighbor": neighbor, "count": int(count), "fraction": float(count / max(1, n))})
    pd.DataFrame(count_rows).to_csv(out_dir / "metric_neighbor_top1_counts.csv", index=False)
    with (out_dir / "metric_neighbor_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Saved metric neighbor diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
