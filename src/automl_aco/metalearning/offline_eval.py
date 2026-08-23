"""Leak-free offline diagnostics for dataset-similarity retrieval."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from ..utils.operator_spec import base_operator_name


def retrieval_metrics(
    predicted_scores: Mapping[Any, float],
    target_scores: Mapping[Any, float],
    ks: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Compute NDCG and top-k overlap on a shared reference set."""
    common = sorted(set(predicted_scores) & set(target_scores), key=str)
    if not common:
        raise ValueError("No common reference datasets for retrieval evaluation")
    predicted = np.asarray([float(predicted_scores[key]) for key in common], dtype=float)
    target = np.asarray([float(target_scores[key]) for key in common], dtype=float)
    # sklearn NDCG requires non-negative relevance; ordering is preserved by shifting.
    relevance = target - np.nanmin(target)
    if np.allclose(relevance, 0.0):
        relevance = np.ones_like(relevance)
    result: dict[str, float] = {}
    predicted_order = np.argsort(-predicted, kind="mergesort")
    target_order = np.argsort(-target, kind="mergesort")
    for raw_k in ks:
        k = min(max(1, int(raw_k)), len(common))
        result[f"ndcg_at_{raw_k}"] = float(
            ndcg_score(relevance.reshape(1, -1), predicted.reshape(1, -1), k=k)
        )
        overlap = len(set(predicted_order[:k]) & set(target_order[:k])) / float(k)
        result[f"overlap_at_{raw_k}"] = float(overlap)
    return result


def warm_start_regret(
    query_scores: pd.Series,
    reference_performance: pd.DataFrame,
    predicted_neighbor_ids: Sequence[Any],
    top_l: int = 3,
) -> float:
    """Oracle minus best query score among top-L pipelines of predicted neighbors."""
    query = pd.to_numeric(query_scores, errors="coerce").dropna()
    if query.empty:
        return float("nan")
    candidates: set[str] = set()
    for dataset_id in predicted_neighbor_ids:
        if dataset_id not in reference_performance.columns:
            continue
        scores = pd.to_numeric(reference_performance[dataset_id], errors="coerce").dropna()
        candidates.update(str(name) for name in scores.nlargest(max(1, int(top_l))).index)
    candidate_scores = query.reindex(sorted(candidates)).dropna()
    if candidate_scores.empty:
        return float("nan")
    return float(query.max() - candidate_scores.max())


def eta_operator_spearman(
    eta: Mapping[str, np.ndarray],
    query_scores: pd.Series,
    pipeline_configs: Sequence[Mapping[str, Any]],
    options: Mapping[str, Sequence[str]],
) -> float:
    """Mean per-step rank correlation between transferred eta and held-out quality."""
    cfg_by_name = {str(cfg.get("name")): cfg for cfg in pipeline_configs if cfg.get("name") is not None}
    query = pd.to_numeric(query_scores, errors="coerce")
    correlations: list[float] = []
    for step, operators in options.items():
        predicted = np.asarray(eta.get(step, []), dtype=float)
        if predicted.size != len(operators):
            continue
        actual: list[float] = []
        for operator in operators:
            target = base_operator_name(operator)
            values = [
                float(query.get(name))
                for name, cfg in cfg_by_name.items()
                if base_operator_name(cfg.get(step, "none")) == target
                and pd.notna(query.get(name))
            ]
            actual.append(float(np.mean(values)) if values else float("nan"))
        actual_arr = np.asarray(actual, dtype=float)
        mask = np.isfinite(predicted) & np.isfinite(actual_arr)
        if mask.sum() < 2 or np.allclose(predicted[mask], predicted[mask][0]):
            continue
        corr = spearmanr(predicted[mask], actual_arr[mask]).statistic
        if np.isfinite(corr):
            correlations.append(float(corr))
    return float(np.mean(correlations)) if correlations else float("nan")


def paired_accuracy_summary(
    candidate: Sequence[float],
    baseline: Sequence[float],
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Macro accuracy difference with a paired dataset bootstrap interval."""
    cand = np.asarray(candidate, dtype=float)
    base = np.asarray(baseline, dtype=float)
    mask = np.isfinite(cand) & np.isfinite(base)
    differences = cand[mask] - base[mask]
    if differences.size == 0:
        raise ValueError("No paired finite accuracy values")
    rng = np.random.RandomState(seed)
    draws = rng.choice(differences, size=(max(1, int(bootstrap_samples)), differences.size), replace=True)
    means = draws.mean(axis=1)
    return {
        "datasets": int(differences.size),
        "candidate_mean_accuracy": float(cand[mask].mean()),
        "baseline_mean_accuracy": float(base[mask].mean()),
        "mean_accuracy_delta": float(differences.mean()),
        "median_accuracy_delta": float(np.median(differences)),
        "wins": int(np.sum(differences > 0.0)),
        "ties": int(np.sum(np.isclose(differences, 0.0))),
        "losses": int(np.sum(differences < 0.0)),
        "bootstrap_ci95_low": float(np.quantile(means, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(means, 0.975)),
    }
