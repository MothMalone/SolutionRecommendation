"""k-order Markov conditional ACO search."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)
EPS = 1e-8


def _freeze_value(value: Any) -> Any:
    """Convert nested mutable values into hashable equivalents."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_value(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_value(v) for v in value))
    return value


def _cfg_key(cfg: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((k, _freeze_value(v)) for k, v in cfg.items()))


def compute_sampling_probabilities(
    pheromone: np.ndarray,
    eta_step: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    pheromone_arr = np.asarray(pheromone, dtype=float)
    eta_arr = np.asarray(eta_step, dtype=float)
    probs = (pheromone_arr ** float(alpha)) * (eta_arr ** float(beta))
    if probs.sum() <= 0 or not np.isfinite(probs).all():
        return np.ones_like(pheromone_arr, dtype=float) / max(1, len(pheromone_arr))
    probs = probs / probs.sum()
    return probs


def compute_legacy_mixed_sampling_probabilities(
    marginal_pheromone: np.ndarray,
    eta_step: np.ndarray,
    alpha: float,
    beta: float,
    conditional_pheromone: Optional[np.ndarray] = None,
    lambda_smooth: float = 0.7,
) -> np.ndarray:
    """Notebook-compatible Markov ACO probability calculation.

    The old notebook mixed unnormalized marginal and conditional scores first,
    then normalized once. This is intentionally different from mixing two
    already-normalized distributions.
    """
    marginal = np.asarray(marginal_pheromone, dtype=float)
    eta_arr = np.asarray(eta_step, dtype=float)
    probs_m = (marginal ** float(alpha)) * (eta_arr ** float(beta))

    if conditional_pheromone is None:
        probs = probs_m
    else:
        conditional = np.asarray(conditional_pheromone, dtype=float)
        probs_k = (conditional ** float(alpha)) * (eta_arr ** float(beta))
        lam = float(lambda_smooth)
        probs = lam * probs_k + (1.0 - lam) * probs_m

    if probs.sum() <= 0 or not np.isfinite(probs).all():
        return np.ones_like(marginal, dtype=float) / max(1, len(marginal))
    return probs / probs.sum()


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
    lambda_smooth: float = 0.0,
    early_stop_rounds: int = 0,
    min_improvement: float = 0.0,
    verbose: bool = False,
    return_history: bool = False,
    legacy_notebook_aco: bool = False,
) -> Tuple[List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]]:
    if legacy_notebook_aco:
        # Match the old notebook exactly: it seeded and sampled from NumPy's
        # global RNG, so downstream evaluation code that resets np.random can
        # affect later ACO samples.
        import random

        random.seed(seed)
        np.random.seed(seed)
        rng = None
    else:
        rng = np.random.RandomState(seed)

    step_order = list(options.keys())
    safe_eta: Dict[str, np.ndarray] = {}
    for step in step_order:
        raw_eta = np.asarray(eta[step], dtype=float)
        if raw_eta.shape[0] != len(options[step]):
            raise ValueError(f"eta[{step!r}] size {raw_eta.shape[0]} does not match options size {len(options[step])}")
        arr = raw_eta.copy()
        arr[~np.isfinite(arr)] = EPS
        arr[arr <= 0] = EPS
        safe_eta[step] = arr
    eta = safe_eta
    if verbose:
        print("Phase 3 search: using transferred eta_norm with positive finite entries.")

    pheromones = {step: np.ones(len(vals), dtype=float) for step, vals in options.items()}
    k_conditional_pheromones: Dict[Tuple[str, Tuple[Any, ...]], np.ndarray] = {}

    def get_k_pheromone(step: str, context: Sequence[Any]) -> np.ndarray:
        key = (step, tuple(context))
        if key not in k_conditional_pheromones:
            k_conditional_pheromones[key] = np.ones(len(options[step]), dtype=float)
        return k_conditional_pheromones[key]

    candidate_pipelines: List[Tuple[Dict[str, Any], float]] = []
    eval_cache: Dict[Tuple[Tuple[str, Any], ...], Tuple[Dict[str, Any], float]] = {}
    history: List[Dict[str, Any]] = []
    best_so_far: Optional[float] = None
    no_improve_rounds = 0

    def sample_config() -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        path_history: List[Tuple[str, int]] = []
        for step in step_order:
            eta_step = eta[step]
            if len(path_history) >= markov_order:
                context = tuple(path_history[-markov_order:])
                k_pher = get_k_pheromone(step, context)
                if legacy_notebook_aco:
                    probs = compute_legacy_mixed_sampling_probabilities(
                        marginal_pheromone=pheromones[step],
                        conditional_pheromone=k_pher,
                        eta_step=eta_step,
                        alpha=alpha,
                        beta=beta,
                        lambda_smooth=lambda_smooth,
                    )
                else:
                    probs_k = compute_sampling_probabilities(k_pher, eta_step, alpha=alpha, beta=beta)
                    probs_m = compute_sampling_probabilities(pheromones[step], eta_step, alpha=alpha, beta=beta)
                    probs = lambda_smooth * probs_k + (1 - lambda_smooth) * probs_m
            else:
                if legacy_notebook_aco:
                    probs = compute_legacy_mixed_sampling_probabilities(
                        marginal_pheromone=pheromones[step],
                        eta_step=eta_step,
                        alpha=alpha,
                        beta=beta,
                    )
                else:
                    probs = compute_sampling_probabilities(pheromones[step], eta_step, alpha=alpha, beta=beta)

            if probs.sum() <= 0 or not np.isfinite(probs).all():
                probs = np.ones(len(options[step])) / len(options[step])
            else:
                probs = probs / probs.sum()

            if legacy_notebook_aco:
                idx = np.random.choice(len(options[step]), p=probs)
            else:
                idx = rng.choice(len(options[step]), p=probs)
            cfg[step] = options[step][idx]
            path_history.append((step, idx))
        return cfg

    for iteration in range(n_iterations):
        sampled: List[Dict[str, Any]] = []
        for _ in range(n_ants):
            cfg = sample_config()
            key = _cfg_key(cfg)
            if key not in eval_cache:
                sampled.append(cfg)

        if not sampled:
            carry_best = None
            if eval_cache:
                carry_best = float(max(sc for _cfg, sc in eval_cache.values()))
            history.append({"iteration": iteration + 1, "best_score": carry_best})
            logger.info(
                "ACO Iter %s/%s - No new unique configuration sampled",
                iteration + 1,
                n_iterations,
            )
            if carry_best is not None:
                if best_so_far is None or carry_best > (best_so_far + float(min_improvement)):
                    best_so_far = carry_best
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
            else:
                no_improve_rounds += 1
            if int(early_stop_rounds) > 0 and no_improve_rounds >= int(early_stop_rounds):
                if verbose:
                    print(
                        f"Early stop at iter {iteration+1}: "
                        f"no improvement for {no_improve_rounds} rounds "
                        f"(min_improvement={float(min_improvement):.6f})"
                    )
                else:
                    logger.info(
                        "Early stop at iter %s: no improvement for %s rounds (min_improvement=%.6f)",
                        iteration + 1,
                        no_improve_rounds,
                        float(min_improvement),
                    )
                break
            continue

        best_cfg, best_score, eval_results, unsorted_res = evaluate_fn(sampled)
        if not eval_results:
            history.append({"iteration": iteration + 1, "best_score": None})
            logger.info("ACO Iter %s/%s - No valid evaluation", iteration + 1, n_iterations)
            no_improve_rounds += 1
            if int(early_stop_rounds) > 0 and no_improve_rounds >= int(early_stop_rounds):
                if verbose:
                    print(
                        f"Early stop at iter {iteration+1}: "
                        f"no improvement for {no_improve_rounds} rounds "
                        f"(min_improvement={float(min_improvement):.6f})"
                    )
                else:
                    logger.info(
                        "Early stop at iter %s: no improvement for %s rounds (min_improvement=%.6f)",
                        iteration + 1,
                        no_improve_rounds,
                        float(min_improvement),
                    )
                break
            continue

        for cfg, score in eval_results:
            eval_cache[_cfg_key(cfg)] = (dict(cfg), float(score))

        for step in pheromones:
            pheromones[step] *= (1 - evaporation)
        for key in k_conditional_pheromones:
            k_conditional_pheromones[key] *= (1 - evaporation)

        cached_results = [(cfg, sc) for cfg, sc in eval_cache.values()]
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
            path_context: List[Tuple[str, int]] = []
            for step in step_order:
                val_idx = options[step].index(cfg[step])
                pheromones[step][val_idx] += weight
                if len(path_context) >= markov_order:
                    context = tuple(path_context[-markov_order:])
                    k_pher = get_k_pheromone(step, context)
                    k_pher[val_idx] += weight
                path_context.append((step, val_idx))

        candidate_pipelines.extend(unsorted_res)
        history.append({"iteration": iteration + 1, "best_score": float(best_score)})
        current_best = float(max(sc for _cfg, sc in eval_cache.values()))
        if best_so_far is None or current_best > (best_so_far + float(min_improvement)):
            best_so_far = current_best
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
        if int(early_stop_rounds) > 0 and no_improve_rounds >= int(early_stop_rounds):
            if verbose:
                print(
                    f"Early stop at iter {iteration+1}: "
                    f"no improvement for {no_improve_rounds} rounds "
                    f"(min_improvement={float(min_improvement):.6f})"
                )
            else:
                logger.info(
                    "Early stop at iter %s: no improvement for %s rounds (min_improvement=%.6f)",
                    iteration + 1,
                    no_improve_rounds,
                    float(min_improvement),
                )
            break
        if verbose:
            print(
                f"ACO Iter {iteration+1}/{n_iterations} — "
                f"best: {best_score:.4f} | k={markov_order}"
            )
        else:
            logger.info("ACO Iter %s/%s - best: %.4f | k=%s", iteration + 1, n_iterations, best_score, markov_order)

    unsorted_candidate_pipelines = candidate_pipelines.copy()
    candidate_pipelines.sort(key=lambda x: x[1], reverse=True)

    seen: Dict[Tuple[Tuple[str, Any], ...], Tuple[Dict[str, Any], float]] = {}
    final: List[Tuple[Dict[str, Any], float]] = []
    for cfg, sc in candidate_pipelines:
        key = _cfg_key(cfg)
        if key not in seen or sc > seen[key][1]:
            seen[key] = (dict(cfg), float(sc))

    for cfg, sc in seen.values():
        final.append((cfg, sc))

    final.sort(key=lambda x: x[1], reverse=True)
    if verbose:
        print("\n🏆 Top pipelines (k-order Markov ACO):")
        for i, (cfg, sc) in enumerate(final[:n_pipelines]):
            print(f"  {i+1}. {cfg.get('name', 'Pipeline')} — score: {sc:.4f}")
    if return_history:
        return final[:n_pipelines], unsorted_candidate_pipelines, history
    return final[:n_pipelines], unsorted_candidate_pipelines
