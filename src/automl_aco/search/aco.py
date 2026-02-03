"""k-order Markov conditional ACO search."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def search_pipelines_aco(
    options: Mapping[str, List[str]],
    evaluate_fn: Callable[[List[Dict[str, Any]]], Tuple[Any, float, List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]]],
    eta: Dict[str, np.ndarray],
    n_pipelines: int = 3,
    n_ants: int = 3,
    n_iterations: int = 5,
    seed: int = 42,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.2,
    top_k_pheromone: int = 3,
    use_all_iter_pipelines: bool = False,
    weight_method: str = "rank",
    markov_order: int = 2,
    lambda_smooth: float = 0.7,
) -> Tuple[List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]]:
    rng = np.random.RandomState(seed)

    step_order = list(options.keys())

    pheromones = {step: np.ones(len(vals), dtype=float) for step, vals in options.items()}
    k_conditional_pheromones: Dict[Tuple[str, Tuple[Any, ...]], np.ndarray] = {}

    def get_k_pheromone(step: str, context: Sequence[Any]) -> np.ndarray:
        key = (step, tuple(context))
        if key not in k_conditional_pheromones:
            k_conditional_pheromones[key] = np.ones(len(options[step]), dtype=float)
        return k_conditional_pheromones[key]

    candidate_pipelines: List[Tuple[Dict[str, Any], float]] = []
    eval_cache: Dict[Tuple[Tuple[str, Any], ...], float] = {}

    def sample_config() -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        history: List[Tuple[str, int]] = []
        for step in step_order:
            eta_step = eta[step]
            if len(history) >= markov_order:
                context = tuple(history[-markov_order:])
                k_pher = get_k_pheromone(step, context)
                probs_k = (k_pher ** alpha) * (eta_step ** beta)
                probs_m = (pheromones[step] ** alpha) * (eta_step ** beta)
                probs = lambda_smooth * probs_k + (1 - lambda_smooth) * probs_m
            else:
                probs = (pheromones[step] ** alpha) * (eta_step ** beta)

            if probs.sum() <= 0 or not np.isfinite(probs).all():
                probs = np.ones(len(options[step])) / len(options[step])
            else:
                probs = probs / probs.sum()

            idx = rng.choice(len(options[step]), p=probs)
            cfg[step] = options[step][idx]
            history.append((step, idx))
        return cfg

    for iteration in range(n_iterations):
        sampled: List[Dict[str, Any]] = []
        for _ in range(n_ants):
            cfg = sample_config()
            key = tuple(sorted(cfg.items()))
            if key not in eval_cache:
                sampled.append(cfg)

        if not sampled:
            continue

        best_cfg, best_score, eval_results, unsorted_res = evaluate_fn(sampled)
        if not eval_results:
            logger.info("ACO Iter %s/%s - No valid evaluation", iteration + 1, n_iterations)
            continue

        for cfg, score in eval_results:
            eval_cache[tuple(sorted(cfg.items()))] = score

        for step in pheromones:
            pheromones[step] *= (1 - evaporation)
        for key in k_conditional_pheromones:
            k_conditional_pheromones[key] *= (1 - evaporation)

        cached_results = [(dict(k), sc) for k, sc in eval_cache.items()]
        cached_results.sort(key=lambda x: x[1], reverse=True)

        selected = cached_results if use_all_iter_pipelines else cached_results[: min(top_k_pheromone, len(cached_results))]
        scores = np.array([sc for _, sc in selected])

        if weight_method == "linear" and len(scores) > 1:
            weights = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8) + 1e-3
        elif weight_method == "exponential" and len(scores) > 1:
            scaled = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            weights = np.exp(scaled)
        elif weight_method == "rank":
            selected.sort(key=lambda x: x[1], reverse=True)
            n = len(selected)
            rank_weights = np.arange(n, 0, -1)
            weights = rank_weights / rank_weights.sum()
        elif weight_method == "reciprocal":
            n = len(scores)
            weights = 1 / np.arange(1, n + 1)
        elif weight_method == "power_rank":
            p = 4
            n = len(scores)
            ranks = np.arange(1, n + 1)
            weights = 1 / (ranks ** p)
        elif weight_method == "uniform":
            weights = np.ones_like(scores, dtype=float)
        else:
            weights = np.ones_like(scores, dtype=float)

        for (cfg, _score), weight in zip(selected, weights):
            history: List[Tuple[str, int]] = []
            for step in step_order:
                val_idx = options[step].index(cfg[step])
                pheromones[step][val_idx] += weight
                if len(history) >= markov_order:
                    context = tuple(history[-markov_order:])
                    k_pher = get_k_pheromone(step, context)
                    k_pher[val_idx] += weight
                history.append((step, val_idx))

        candidate_pipelines.extend(unsorted_res)
        logger.info("ACO Iter %s/%s - best: %.4f | k=%s", iteration + 1, n_iterations, best_score, markov_order)

    unsorted_candidate_pipelines = candidate_pipelines.copy()
    candidate_pipelines.sort(key=lambda x: x[1], reverse=True)

    seen: Dict[Tuple[Tuple[str, Any], ...], float] = {}
    final: List[Tuple[Dict[str, Any], float]] = []
    for cfg, sc in candidate_pipelines:
        key = tuple(sorted(cfg.items()))
        if key not in seen or sc > seen[key]:
            seen[key] = sc

    for k, sc in seen.items():
        final.append((dict(k), sc))

    final.sort(key=lambda x: x[1], reverse=True)
    return final[:n_pipelines], unsorted_candidate_pipelines
