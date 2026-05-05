"""Heuristic transfer utilities for ACO (Phase 2)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logging import get_logger
from ..utils.operator_spec import base_operator_name

logger = get_logger(__name__)

EPS = 1e-8
HEURISTIC_TRANSFER_METHODS = {"weighted_topk_topl", "legacy_weighted_average"}
SCORE_DIRECTION_CHOICES = {"higher_is_better", "lower_is_better"}


def _normalize_dataset_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if np.isfinite(f) and abs(f - round(f)) <= 1e-9:
            return str(int(round(f)))
        return str(value).strip()

    text = str(value).strip()
    float_like = re.fullmatch(r"([0-9]+)\.0+", text)
    if float_like:
        return float_like.group(1)

    prefixed = re.fullmatch(r"(?i)(?:d|dataset|openml)[_\-: ]*([0-9]+)", text)
    if prefixed:
        return prefixed.group(1)
    return text


def _to_higher_is_better(values: np.ndarray, score_direction: str) -> np.ndarray:
    direction = str(score_direction).strip().lower()
    if direction not in SCORE_DIRECTION_CHOICES:
        raise ValueError(
            f"Unsupported score_direction={score_direction!r}. "
            f"Expected one of {sorted(SCORE_DIRECTION_CHOICES)}."
        )
    arr = np.asarray(values, dtype=float)
    if direction == "lower_is_better":
        arr = -arr
    return arr


def compute_dataset_similarities(
    metafeatures_df: pd.DataFrame,
    new_metafeatures: np.ndarray,
    metafeatures_scaled: Optional[np.ndarray] = None,
    dataset_similarity_scores: Optional[Mapping[Any, float]] = None,
) -> List[Tuple[Any, float]]:
    if metafeatures_df.shape[0] == 0:
        return []

    if dataset_similarity_scores is not None:
        normalized_scores: Dict[str, float] = {}
        for dataset_id, score in dataset_similarity_scores.items():
            value = float(score)
            if np.isfinite(value):
                normalized_scores[_normalize_dataset_id(dataset_id)] = value
        sims = [
            (dataset_id, float(normalized_scores.get(_normalize_dataset_id(dataset_id), 0.0)))
            for dataset_id in metafeatures_df.index
        ]
    else:
        known = metafeatures_scaled if metafeatures_scaled is not None else metafeatures_df.fillna(0).values
        sims_arr = cosine_similarity(known, np.asarray(new_metafeatures, dtype=float).reshape(1, -1)).ravel()
        sims = list(zip(metafeatures_df.index, sims_arr))

    sims.sort(key=lambda item: (-float(item[1]), str(item[0])))
    return [(dataset_id, float(score)) for dataset_id, score in sims if np.isfinite(float(score))]


def select_top_k_neighbors(
    dataset_similarities: Sequence[Tuple[Any, float]],
    top_k: Optional[int] = 10,
    query_dataset_id: Optional[Any] = None,
) -> List[Tuple[Any, float]]:
    query_norm = _normalize_dataset_id(query_dataset_id) if query_dataset_id is not None else None
    filtered: List[Tuple[Any, float]] = []
    for dataset_id, sim in dataset_similarities:
        if query_norm is not None and _normalize_dataset_id(dataset_id) == query_norm:
            continue
        if np.isfinite(float(sim)):
            filtered.append((dataset_id, float(sim)))
    filtered.sort(key=lambda item: (-item[1], str(item[0])))

    if top_k is None:
        return filtered
    k = max(1, int(top_k))
    return filtered[: min(k, len(filtered))]


def compute_similarity_weights(
    top_k_neighbors: Sequence[Tuple[Any, float]],
    dataset_weighting: str = "similarity",
    similarity_temperature: float = 1.0,
) -> Dict[Any, float]:
    if len(top_k_neighbors) == 0:
        return {}

    dataset_ids = [dataset_id for dataset_id, _ in top_k_neighbors]
    if str(dataset_weighting).strip().lower() != "similarity":
        uniform = 1.0 / float(len(dataset_ids))
        return {dataset_id: uniform for dataset_id in dataset_ids}

    similarities = np.asarray([sim for _dataset_id, sim in top_k_neighbors], dtype=float)
    if len(similarities) == 1 or np.allclose(similarities, similarities[0], atol=1e-12, rtol=1e-9):
        uniform = 1.0 / float(len(dataset_ids))
        return {dataset_id: uniform for dataset_id in dataset_ids}

    temp = max(float(similarity_temperature), EPS)
    shifted = (similarities / temp) - np.max(similarities / temp)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= EPS or not np.isfinite(total) or not np.isfinite(weights).all():
        uniform = 1.0 / float(len(dataset_ids))
        return {dataset_id: uniform for dataset_id in dataset_ids}
    weights = weights / total
    return {dataset_id: float(weight) for dataset_id, weight in zip(dataset_ids, weights)}


def select_top_l_pipelines_per_neighbor(
    performance_matrix: pd.DataFrame,
    top_k_neighbors: Sequence[Tuple[Any, float]],
    top_l: int = 3,
    score_direction: str = "higher_is_better",
) -> Dict[Any, List[Dict[str, Any]]]:
    selected: Dict[Any, List[Dict[str, Any]]] = {}
    l = max(1, int(top_l))

    for dataset_id, _sim in top_k_neighbors:
        if dataset_id not in performance_matrix.columns:
            continue
        scores = pd.to_numeric(performance_matrix[dataset_id], errors="coerce").dropna()
        if scores.empty:
            continue

        oriented = _to_higher_is_better(scores.to_numpy(dtype=float, copy=False), score_direction=score_direction)
        ranking = pd.DataFrame(
            {
                "pipeline": scores.index.astype(str),
                "historical_score": scores.to_numpy(dtype=float, copy=False),
                "oriented_score": oriented,
            }
        )
        ranking = ranking.sort_values(
            by=["oriented_score", "pipeline"],
            ascending=[False, True],
            kind="mergesort",
        )
        ranking = ranking.head(min(l, len(ranking)))

        records: List[Dict[str, Any]] = []
        for rank, row in enumerate(ranking.itertuples(index=False), start=1):
            records.append(
                {
                    "pipeline": str(row.pipeline),
                    "historical_score": float(row.historical_score),
                    "oriented_score": float(row.oriented_score),
                    "pipeline_rank_within_neighbor": rank,
                }
            )
        selected[dataset_id] = records
    return selected


def compute_relative_pipeline_quality(
    performance_matrix: pd.DataFrame,
    source_dataset: Any,
    score_direction: str = "higher_is_better",
) -> pd.Series:
    if source_dataset not in performance_matrix.columns:
        return pd.Series(dtype=float)

    scores = pd.to_numeric(performance_matrix[source_dataset], errors="coerce").dropna()
    if scores.empty:
        return pd.Series(dtype=float)

    oriented_scores = _to_higher_is_better(scores.to_numpy(dtype=float, copy=False), score_direction=score_direction)
    oriented = pd.Series(oriented_scores, index=scores.index, dtype=float)

    min_score = float(oriented.min())
    max_score = float(oriented.max())
    if max_score - min_score <= EPS:
        return pd.Series(1.0, index=oriented.index, dtype=float)

    quality = (oriented - min_score) / (max_score - min_score + EPS)
    return quality.clip(lower=0.0, upper=1.0)


def build_transfer_candidates(
    performance_matrix: pd.DataFrame,
    top_k_neighbors: Sequence[Tuple[Any, float]],
    top_l_pipelines: Mapping[Any, Sequence[Mapping[str, Any]]],
    similarity_weights: Mapping[Any, float],
    score_direction: str = "higher_is_better",
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for source_dataset, similarity_score in top_k_neighbors:
        pipelines = top_l_pipelines.get(source_dataset, [])
        if not pipelines:
            continue

        relative_quality = compute_relative_pipeline_quality(
            performance_matrix=performance_matrix,
            source_dataset=source_dataset,
            score_direction=score_direction,
        )
        sim_weight = float(similarity_weights.get(source_dataset, 0.0))

        for row in pipelines:
            pipeline_name = str(row["pipeline"])
            quality = float(relative_quality.get(pipeline_name, 0.0))
            if not np.isfinite(quality):
                quality = 0.0
            candidate_weight = float(sim_weight * quality)
            candidates.append(
                {
                    "source_dataset": source_dataset,
                    "pipeline": pipeline_name,
                    "historical_score": float(row["historical_score"]),
                    "relative_pipeline_quality": quality,
                    "similarity_score": float(similarity_score),
                    "similarity_weight": sim_weight,
                    "pipeline_rank_within_neighbor": int(row["pipeline_rank_within_neighbor"]),
                    "candidate_weight": candidate_weight,
                }
            )
    return candidates


def aggregate_operator_heuristics(
    transfer_candidates: Sequence[Mapping[str, Any]],
    pipeline_configs: Sequence[Mapping[str, Any]],
    options: Mapping[str, Sequence[str]],
) -> Dict[str, np.ndarray]:
    cfg_map = {str(cfg["name"]): cfg for cfg in pipeline_configs if "name" in cfg}
    raw_eta: Dict[str, np.ndarray] = {}

    for step, operators in options.items():
        step_values = np.full(len(operators), np.nan, dtype=float)

        for idx, operator in enumerate(operators):
            target_base = base_operator_name(operator)
            numerator = 0.0
            denominator = 0.0
            for candidate in transfer_candidates:
                pipeline_name = str(candidate.get("pipeline"))
                cfg = cfg_map.get(pipeline_name)
                if cfg is None or step not in cfg:
                    continue
                if base_operator_name(cfg.get(step)) != target_base:
                    continue

                cand_weight = float(candidate.get("candidate_weight", 0.0))
                rel_quality = float(candidate.get("relative_pipeline_quality", 0.0))
                if cand_weight <= 0 or not np.isfinite(cand_weight) or not np.isfinite(rel_quality):
                    continue

                numerator += cand_weight * rel_quality
                denominator += cand_weight

            if denominator > EPS:
                step_values[idx] = numerator / denominator

        observed = step_values[np.isfinite(step_values)]
        if observed.size > 0:
            # Missing operators fall back to this step's weakest observed signal.
            fallback = float(np.min(observed))
            step_values[~np.isfinite(step_values)] = fallback
        else:
            step_values[:] = 1.0

        raw_eta[step] = step_values
    return raw_eta


def normalize_eta_with_floor(raw_eta: Mapping[str, np.ndarray], eta_floor: float = 0.05) -> Dict[str, np.ndarray]:
    floor = float(eta_floor)
    if not np.isfinite(floor):
        floor = 0.05
    floor = min(max(floor, 0.0), 0.95)

    normalized: Dict[str, np.ndarray] = {}
    for step, raw_values in raw_eta.items():
        arr = np.asarray(raw_values, dtype=float).copy()
        if arr.size == 0:
            normalized[step] = arr
            continue

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            normalized[step] = np.ones_like(arr)
            continue

        fill_value = float(np.min(finite))
        arr[~np.isfinite(arr)] = fill_value

        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val - min_val <= EPS:
            norm = np.ones_like(arr)
        else:
            norm = floor + (1.0 - floor) * (arr - min_val) / (max_val - min_val + EPS)
            norm = np.clip(norm, floor, 1.0)

        if not np.isfinite(norm).all():
            norm[~np.isfinite(norm)] = max(floor, EPS)
        normalized[step] = norm
    return normalized


def initialize_aco_with_transferred_eta(raw_eta: Mapping[str, np.ndarray], eta_floor: float = 0.05) -> Dict[str, np.ndarray]:
    return normalize_eta_with_floor(raw_eta=raw_eta, eta_floor=eta_floor)


def _compute_aco_heuristic_legacy(
    performance_matrix: pd.DataFrame,
    metafeatures_df: pd.DataFrame,
    pipeline_configs: List[Dict[str, Any]],
    options: Mapping[str, List[str]],
    new_metafeatures: np.ndarray,
    dataset_weighting: str = "equality",
    top_k: Optional[int] = 10,
    use_top_pipelines_from_metric: bool = True,
    recommend_func: Optional[Callable[..., Dict[str, Any]]] = None,
    recommend_kwargs: Optional[Dict[str, Any]] = None,
    metafeatures_scaled: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    perf_subset = performance_matrix.copy()

    if use_top_pipelines_from_metric and recommend_func is not None:
        rec_args = recommend_kwargs or {}
        rec_result = recommend_func(**rec_args)
        top_pipelines: List[str] = []
        if isinstance(rec_result, dict):
            pipeline_cfg = rec_result.get("pipeline_config")
            if isinstance(pipeline_cfg, dict):
                name = pipeline_cfg.get("name")
                if name in performance_matrix.index:
                    top_pipelines = [name]
        if top_pipelines:
            perf_subset = performance_matrix.loc[performance_matrix.index.isin(top_pipelines)]
        else:
            logger.debug("No valid pipeline from metric; using full performance matrix")

    if str(dataset_weighting).strip().lower() == "similarity" and metafeatures_df.shape[0] > 0:
        try:
            known = metafeatures_scaled if metafeatures_scaled is not None else metafeatures_df.fillna(0).values
            sims = cosine_similarity(known, np.asarray(new_metafeatures, dtype=float).reshape(1, -1)).ravel()

            if top_k is not None and top_k < len(sims):
                top_idx = np.argsort(sims)[-int(top_k):]
                sims_masked = np.zeros_like(sims)
                sims_masked[top_idx] = sims[top_idx]
                sims = sims_masked

            sims = sims - sims.min()
            if sims.sum() <= 0:
                sims = np.ones_like(sims)
            dataset_weights = sims / (sims.sum() + EPS)
        except Exception:
            dataset_weights = np.ones(metafeatures_df.shape[0], dtype=float) / max(1, metafeatures_df.shape[0])
    else:
        dataset_weights = np.ones(metafeatures_df.shape[0], dtype=float) / max(1, metafeatures_df.shape[0])

    common_names = [name for name in metafeatures_df.index if name in perf_subset.columns]
    if len(common_names) == 0:
        raise ValueError("No common datasets between metafeatures and performance matrix")

    perf_tbl = perf_subset.loc[:, common_names].fillna(0)
    weight_series = pd.Series(dataset_weights, index=metafeatures_df.index, dtype=float)
    aligned_weights = weight_series.reindex(common_names).fillna(0.0).to_numpy(dtype=float, copy=False)
    if aligned_weights.sum() <= EPS:
        aligned_weights = np.ones(len(common_names), dtype=float) / max(1, len(common_names))
    else:
        aligned_weights = aligned_weights / (aligned_weights.sum() + EPS)

    if perf_tbl.shape[1] == 0:
        pipeline_perf_mean = pd.Series(0.0, index=perf_subset.index, dtype=float)
    else:
        pipeline_perf_mean = pd.Series(np.dot(perf_tbl.values, aligned_weights), index=perf_tbl.index, dtype=float)

    eta: Dict[str, np.ndarray] = {}
    cfg_map = {str(cfg["name"]): cfg for cfg in pipeline_configs if "name" in cfg}

    for step, values in options.items():
        arr = np.full(len(values), EPS, dtype=float)
        for i, val in enumerate(values):
            val_base = base_operator_name(val)
            matched = [
                perf_val
                for pname, perf_val in pipeline_perf_mean.items()
                if cfg_map.get(str(pname)) is not None and base_operator_name(cfg_map[str(pname)].get(step)) == val_base
            ]
            if matched:
                arr[i] = float(np.mean(matched)) + EPS

        nonzero_vals = arr[arr > EPS * 2]
        if nonzero_vals.size > 0:
            min_val = float(nonzero_vals.min())
            arr[arr <= EPS] = min_val * 0.8
        else:
            arr[:] = 1.0 / len(arr)

        arr = arr / (arr.max() + EPS) if arr.sum() > 0 else np.ones_like(arr) / len(arr)
        arr = np.clip(arr, EPS, None)
        eta[step] = arr
    return eta


def _emit_phase2_logs(
    *,
    verbose: bool,
    options: Mapping[str, Sequence[str]],
    top_k_neighbors: Sequence[Tuple[Any, float]],
    similarity_weights: Mapping[Any, float],
    top_l_pipelines: Mapping[Any, Sequence[Mapping[str, Any]]],
    transfer_candidates: Sequence[Mapping[str, Any]],
    raw_eta: Mapping[str, np.ndarray],
    normalized_eta: Mapping[str, np.ndarray],
) -> None:
    logger.debug("Phase2 top-K neighbors: %s", top_k_neighbors)
    logger.debug("Phase2 similarity weights: %s", dict(similarity_weights))
    logger.debug("Phase2 transfer candidate count: %s", len(transfer_candidates))
    logger.debug("Phase2 raw eta: %s", {step: vals.tolist() for step, vals in raw_eta.items()})
    logger.debug("Phase2 normalized eta: %s", {step: vals.tolist() for step, vals in normalized_eta.items()})
    if not verbose:
        return

    print("Phase 2 transfer: top-K neighbors")
    for dataset_id, similarity_score in top_k_neighbors:
        print(
            f"  - {dataset_id}: similarity={float(similarity_score):.6f}, "
            f"weight={float(similarity_weights.get(dataset_id, 0.0)):.6f}"
        )

    print("Phase 2 transfer: top-L pipelines per neighbor")
    for dataset_id, pipelines in top_l_pipelines.items():
        for row in pipelines:
            pipeline_name = str(row.get("pipeline"))
            candidate = next(
                (
                    c
                    for c in transfer_candidates
                    if c.get("source_dataset") == dataset_id and str(c.get("pipeline")) == pipeline_name
                ),
                None,
            )
            quality = float(candidate.get("relative_pipeline_quality", 0.0)) if candidate is not None else 0.0
            cand_weight = float(candidate.get("candidate_weight", 0.0)) if candidate is not None else 0.0
            print(
                f"  - {dataset_id} | rank={int(row.get('pipeline_rank_within_neighbor', 0))} "
                f"| pipeline={pipeline_name} | S={float(row.get('historical_score', 0.0)):.6f} "
                f"| q={quality:.6f} | w={cand_weight:.6f}"
            )

    print("Phase 2 transfer: raw eta")
    for step, operators in options.items():
        vals = raw_eta.get(step, np.ones(len(operators), dtype=float))
        joined = ", ".join(f"{operator}={float(vals[idx]):.6f}" for idx, operator in enumerate(operators))
        print(f"  - {step}: {joined}")

    print("Phase 2 transfer: normalized eta")
    for step, operators in options.items():
        vals = normalized_eta.get(step, np.ones(len(operators), dtype=float))
        joined = ", ".join(f"{operator}={float(vals[idx]):.6f}" for idx, operator in enumerate(operators))
        print(f"  - {step}: {joined}")


def compute_aco_heuristic(
    performance_matrix: pd.DataFrame,
    metafeatures_df: pd.DataFrame,
    pipeline_configs: List[Dict[str, Any]],
    options: Mapping[str, List[str]],
    new_metafeatures: np.ndarray,
    dataset_weighting: str = "similarity",
    top_k: Optional[int] = 10,
    use_top_pipelines_from_metric: bool = True,
    recommend_func: Optional[Callable[..., Dict[str, Any]]] = None,
    recommend_kwargs: Optional[Dict[str, Any]] = None,
    metafeatures_scaled: Optional[np.ndarray] = None,
    dataset_similarity_scores: Optional[Mapping[Any, float]] = None,
    top_l: int = 3,
    similarity_temperature: float = 1.0,
    eta_floor: float = 0.05,
    heuristic_transfer_method: str = "weighted_topk_topl",
    score_direction: str = "higher_is_better",
    query_dataset_id: Optional[Any] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    method = str(heuristic_transfer_method).strip().lower()
    if method not in HEURISTIC_TRANSFER_METHODS:
        raise ValueError(
            f"Unsupported heuristic_transfer_method={heuristic_transfer_method!r}. "
            f"Expected one of {sorted(HEURISTIC_TRANSFER_METHODS)}."
        )

    if method == "legacy_weighted_average":
        return _compute_aco_heuristic_legacy(
            performance_matrix=performance_matrix,
            metafeatures_df=metafeatures_df,
            pipeline_configs=pipeline_configs,
            options=options,
            new_metafeatures=new_metafeatures,
            dataset_weighting=dataset_weighting,
            top_k=top_k,
            use_top_pipelines_from_metric=use_top_pipelines_from_metric,
            recommend_func=recommend_func,
            recommend_kwargs=recommend_kwargs,
            metafeatures_scaled=metafeatures_scaled,
        )

    dataset_similarities = compute_dataset_similarities(
        metafeatures_df=metafeatures_df,
        new_metafeatures=new_metafeatures,
        metafeatures_scaled=metafeatures_scaled,
        dataset_similarity_scores=dataset_similarity_scores,
    )
    top_k_neighbors = select_top_k_neighbors(
        dataset_similarities=dataset_similarities,
        top_k=top_k,
        query_dataset_id=query_dataset_id,
    )
    similarity_weights = compute_similarity_weights(
        top_k_neighbors=top_k_neighbors,
        dataset_weighting=dataset_weighting,
        similarity_temperature=similarity_temperature,
    )
    top_l_pipelines = select_top_l_pipelines_per_neighbor(
        performance_matrix=performance_matrix,
        top_k_neighbors=top_k_neighbors,
        top_l=top_l,
        score_direction=score_direction,
    )
    transfer_candidates = build_transfer_candidates(
        performance_matrix=performance_matrix,
        top_k_neighbors=top_k_neighbors,
        top_l_pipelines=top_l_pipelines,
        similarity_weights=similarity_weights,
        score_direction=score_direction,
    )
    raw_eta = aggregate_operator_heuristics(
        transfer_candidates=transfer_candidates,
        pipeline_configs=pipeline_configs,
        options=options,
    )
    normalized_eta = initialize_aco_with_transferred_eta(raw_eta=raw_eta, eta_floor=eta_floor)

    _emit_phase2_logs(
        verbose=verbose,
        options=options,
        top_k_neighbors=top_k_neighbors,
        similarity_weights=similarity_weights,
        top_l_pipelines=top_l_pipelines,
        transfer_candidates=transfer_candidates,
        raw_eta=raw_eta,
        normalized_eta=normalized_eta,
    )
    return normalized_eta
