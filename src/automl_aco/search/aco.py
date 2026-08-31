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


def _semantic_cfg_key(
    cfg: Mapping[str, Any],
    option_steps: Sequence[str],
) -> Tuple[Tuple[str, Any], ...]:
    """Key only the operator decisions that define an ACO search point.

    Evaluators are allowed to attach metadata such as ``name`` and ``step_order``
    to a copied config.  Those fields must not turn an already evaluated search
    point into a cache miss.  Step ordering is fixed for one invocation of this
    search function, so it is deliberately outside the within-run cache key.
    """
    return tuple((step, _freeze_value(cfg.get(step))) for step in option_steps)


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


def mix_with_uniform_exploration(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    """Mix a categorical distribution with a uniform exploration policy."""
    probs = np.asarray(probabilities, dtype=float)
    eps = float(epsilon)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if not np.isfinite(eps) or not 0.0 <= eps <= 1.0:
        raise ValueError("epsilon must be finite and within [0, 1]")
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(probs.sum())
    probs = probs / total if total > 0.0 else np.ones(probs.size, dtype=float) / probs.size
    uniform = np.ones(probs.size, dtype=float) / probs.size
    mixed = (1.0 - eps) * probs + eps * uniform
    return mixed / mixed.sum()


def _normalized_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a selection distribution, normalized to [0,1] by log(n).

    1.0 = uniform exploration over a step's operators; ~0.0 = the colony has collapsed onto a
    single operator at that step. Used as the per-step collapse signal.
    """
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    n = len(probs)
    if n <= 1 or p.size == 0:
        return 0.0 if n <= 1 else 1.0
    h = -float(np.sum(p * np.log(p)))
    return h / float(np.log(n))


def _resolve_mmas_bounds(
    evaporation: float,
    tau_min: Optional[float],
    tau_max: Optional[float],
    tau_min_ratio: float,
) -> Tuple[float, float]:
    """Standard MMAS bounds. Auto τ_max = max reward steady state 1/ρ (since the per-iteration
    min-max reward Δ(p) ∈ [0,1], τ → (1-ρ)τ + 1 converges to 1/ρ); τ_min = ratio·τ_max.

    Bounds keep every operator selectable (τ_min > 0 ⇒ non-zero probability) while capping
    runaway exploitation (τ_max), which is MMAS's defining anti-collapse mechanism.
    """
    rho = float(evaporation)
    auto_max = (1.0 / rho) if rho > EPS else 1e6
    t_max = float(tau_max) if tau_max is not None else auto_max
    t_min = float(tau_min) if tau_min is not None else max(EPS, float(tau_min_ratio) * t_max)
    if t_min > t_max:
        t_min, t_max = t_max, t_min
    return t_min, t_max


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


def apply_interaction_prior(
    eta_step: np.ndarray,
    step: str,
    path_history: Sequence[Tuple[str, int]],
    interaction_priors: Optional[Mapping[Tuple[str, int, str], np.ndarray]] = None,
    interaction_prior_strength: float = 0.0,
) -> np.ndarray:
    """Adjust a step heuristic using fixed historical pairwise priors.

    Unlike Markov pheromones, these priors are computed before search from
    retrieved historical pipelines and are not updated from target proxy scores.
    """
    eta_arr = np.asarray(eta_step, dtype=float)
    strength = float(interaction_prior_strength)
    if strength <= 0.0 or not interaction_priors or len(path_history) == 0:
        return eta_arr

    adjusted = eta_arr.copy()
    for prev_step, prev_idx in path_history:
        prior = interaction_priors.get((str(prev_step), int(prev_idx), str(step)))
        if prior is None:
            continue
        prior_arr = np.asarray(prior, dtype=float)
        if prior_arr.shape != adjusted.shape:
            continue
        prior_arr = np.nan_to_num(prior_arr, nan=1.0, posinf=1.0, neginf=1.0)
        prior_arr = np.clip(prior_arr, EPS, None)
        adjusted *= prior_arr ** strength

    adjusted[~np.isfinite(adjusted)] = EPS
    adjusted[adjusted <= 0] = EPS
    return adjusted


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
    interaction_priors: Optional[Mapping[Tuple[str, int, str], np.ndarray]] = None,
    interaction_prior_strength: float = 0.0,
    mmas_bounds: bool = False,
    tau_min: Optional[float] = None,
    tau_max: Optional[float] = None,
    tau_min_ratio: float = 0.05,
    canonical_cache_keys: bool = False,
    deduplicate_iteration: bool = False,
    cache_invalid_configs: bool = False,
    tie_aware_rank_weights: bool = False,
    refill_unique_ants: bool = False,
    max_sampling_attempt_multiplier: int = 100,
    update_policy: str = "global_elite",
    exploration_policy: str = "none",
    exploration_epsilon: float = 0.1,
    exploration_initial_epsilon: float = 0.05,
    exploration_step: float = 0.05,
    exploration_max_epsilon: float = 0.30,
    total_ant_budget: Optional[int] = None,
) -> Tuple[List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]]:
    if total_ant_budget is not None and int(total_ant_budget) < 1:
        raise ValueError("total_ant_budget must be positive when provided")
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

    def config_key(cfg: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        if canonical_cache_keys:
            return _semantic_cfg_key(cfg, step_order)
        return _cfg_key(dict(cfg))
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

    if mmas_bounds:
        bound_min, bound_max = _resolve_mmas_bounds(evaporation, tau_min, tau_max, tau_min_ratio)
        if verbose:
            print(f"[aco:mmas] pheromone bounds [tau_min={bound_min:.4f}, tau_max={bound_max:.4f}]")
    else:
        bound_min, bound_max = None, None

    def _clip_pheromones() -> None:
        if bound_min is None:
            return
        for step in pheromones:
            np.clip(pheromones[step], bound_min, bound_max, out=pheromones[step])
        for key in k_conditional_pheromones:
            np.clip(k_conditional_pheromones[key], bound_min, bound_max, out=k_conditional_pheromones[key])

    def _step_diagnostics() -> Dict[str, Any]:
        ent: Dict[str, float] = {}
        raw_ent: Dict[str, float] = {}
        pher_min = np.inf
        pher_max = -np.inf
        saturated = 0
        total = 0
        for step in step_order:
            probs = compute_sampling_probabilities(pheromones[step], eta[step], alpha, beta)
            raw_ent[step] = _normalized_entropy(probs)
            if effective_epsilon > 0.0:
                probs = mix_with_uniform_exploration(probs, effective_epsilon)
            ent[step] = _normalized_entropy(probs)
            ph = pheromones[step]
            pher_min = min(pher_min, float(ph.min()))
            pher_max = max(pher_max, float(ph.max()))
            if bound_max is not None:
                saturated += int(np.sum(ph >= bound_max - EPS) + np.sum(ph <= bound_min + EPS))
                total += ph.size
        return {
            "step_entropy": ent,
            "raw_step_entropy": raw_ent,
            "mean_entropy": float(np.mean(list(ent.values()))) if ent else None,
            "raw_mean_entropy": float(np.mean(list(raw_ent.values()))) if raw_ent else None,
            "pheromone_min": None if not np.isfinite(pher_min) else pher_min,
            "pheromone_max": None if not np.isfinite(pher_max) else pher_max,
            "pheromone_saturation": (saturated / total) if total else None,
        }

    def get_k_pheromone(step: str, context: Sequence[Any]) -> np.ndarray:
        key = (step, tuple(context))
        if key not in k_conditional_pheromones:
            k_conditional_pheromones[key] = np.ones(len(options[step]), dtype=float)
        return k_conditional_pheromones[key]

    candidate_pipelines: List[Tuple[Dict[str, Any], float]] = []
    eval_cache: Dict[Tuple[Tuple[str, Any], ...], Tuple[Dict[str, Any], float]] = {}
    invalid_cache: set[Tuple[Tuple[str, Any], ...]] = set()
    history: List[Dict[str, Any]] = []
    best_so_far: Optional[float] = None
    no_improve_rounds = 0
    cumulative_draw_count = 0
    cumulative_duplicate_draw_count = 0
    cumulative_cached_draw_count = 0
    cumulative_invalid_cached_draw_count = 0
    cumulative_evaluation_request_count = 0
    conditional_pheromone_enabled = bool(markov_order > 0 and float(lambda_smooth) > 0.0)
    update_policy = str(update_policy).strip().lower()
    exploration_policy = str(exploration_policy).strip().lower()
    if update_policy not in {
        "global_elite",
        "iteration_elite",
        "improvement_only",
        "hybrid_elite",
    }:
        raise ValueError(f"Unsupported ACO update_policy={update_policy!r}")
    if exploration_policy not in {"none", "fixed", "stagnation"}:
        raise ValueError(f"Unsupported ACO exploration_policy={exploration_policy!r}")
    for label, value in {
        "exploration_epsilon": exploration_epsilon,
        "exploration_initial_epsilon": exploration_initial_epsilon,
        "exploration_step": exploration_step,
        "exploration_max_epsilon": exploration_max_epsilon,
    }.items():
        if not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{label} must be finite and within [0, 1]")
    effective_epsilon = 0.0
    search_space_size = int(np.prod([len(options[step]) for step in step_order], dtype=np.int64))
    if cache_invalid_configs and not canonical_cache_keys:
        raise ValueError("cache_invalid_configs requires canonical_cache_keys=True")
    if refill_unique_ants and not (canonical_cache_keys and deduplicate_iteration):
        raise ValueError(
            "refill_unique_ants requires canonical_cache_keys=True and "
            "deduplicate_iteration=True"
        )

    def sample_config() -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        path_history: List[Tuple[str, int]] = []
        for step in step_order:
            eta_step = apply_interaction_prior(
                eta_step=eta[step],
                step=step,
                path_history=path_history,
                interaction_priors=interaction_priors,
                interaction_prior_strength=interaction_prior_strength,
            )
            if conditional_pheromone_enabled and len(path_history) >= markov_order:
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
            if effective_epsilon > 0.0:
                probs = mix_with_uniform_exploration(probs, effective_epsilon)

            if legacy_notebook_aco:
                idx = np.random.choice(len(options[step]), p=probs)
            else:
                idx = rng.choice(len(options[step]), p=probs)
            cfg[step] = options[step][idx]
            path_history.append((step, idx))
        return cfg

    for iteration in range(n_iterations):
        if exploration_policy == "fixed":
            effective_epsilon = float(exploration_epsilon)
        elif exploration_policy == "stagnation":
            effective_epsilon = min(
                float(exploration_max_epsilon),
                float(exploration_initial_epsilon) + no_improve_rounds * float(exploration_step),
            )
        else:
            effective_epsilon = 0.0
        sampled: List[Dict[str, Any]] = []
        sampled_keys: set[Tuple[Tuple[str, Any], ...]] = set()
        draw_count = 0
        duplicate_draw_count = 0
        cached_draw_count = 0
        invalid_cached_draw_count = 0
        target_draws = max(0, int(n_ants))
        if total_ant_budget is not None:
            remaining_draws = max(0, int(total_ant_budget) - int(cumulative_draw_count))
            target_draws = min(target_draws, remaining_draws)
            if remaining_draws <= 0:
                break
        if refill_unique_ants:
            known_keys = set(eval_cache).union(invalid_cache)
            target_draws = min(target_draws, max(0, search_space_size - len(known_keys)))
        max_attempts = target_draws
        if refill_unique_ants:
            max_attempts = max(target_draws, target_draws * max(1, int(max_sampling_attempt_multiplier)))

        while draw_count < max_attempts:
            if refill_unique_ants and len(sampled) >= target_draws:
                break
            if not refill_unique_ants and draw_count >= target_draws:
                break
            cfg = sample_config()
            draw_count += 1
            key = config_key(cfg)
            if key in eval_cache:
                cached_draw_count += 1
                continue
            if cache_invalid_configs and key in invalid_cache:
                invalid_cached_draw_count += 1
                continue
            if deduplicate_iteration and key in sampled_keys:
                duplicate_draw_count += 1
                continue
            sampled.append(cfg)
            sampled_keys.add(key)

        cumulative_draw_count += draw_count
        cumulative_duplicate_draw_count += duplicate_draw_count
        cumulative_cached_draw_count += cached_draw_count
        cumulative_invalid_cached_draw_count += invalid_cached_draw_count
        cumulative_evaluation_request_count += len(sampled)

        if not sampled:
            carry_best = None
            if eval_cache:
                carry_best = float(max(sc for _cfg, sc in eval_cache.values()))
            previous_best = best_so_far
            improved = (
                carry_best is not None
                and (previous_best is None or carry_best > (previous_best + float(min_improvement)))
            )
            history.append(
                {
                    "iteration": iteration + 1,
                    "best_score": carry_best,
                    "global_best_score": carry_best,
                    "iteration_best_score": None,
                    "iteration_mean_score": None,
                    "iteration_min_score": None,
                    "sampled_unique_count": 0,
                    "sampled_distinct_count": 0,
                    "valid_count": 0,
                    "cache_size": len(eval_cache),
                    "invalid_cache_size": len(invalid_cache),
                    "draw_count": int(draw_count),
                    "duplicate_draw_count": int(duplicate_draw_count),
                    "cached_draw_count": int(cached_draw_count),
                    "invalid_cached_draw_count": int(invalid_cached_draw_count),
                    "evaluation_request_count": 0,
                    "cumulative_draw_count": int(cumulative_draw_count),
                    "cumulative_duplicate_draw_count": int(cumulative_duplicate_draw_count),
                    "cumulative_cached_draw_count": int(cumulative_cached_draw_count),
                    "cumulative_invalid_cached_draw_count": int(cumulative_invalid_cached_draw_count),
                    "cumulative_evaluation_request_count": int(cumulative_evaluation_request_count),
                    "conditional_pheromone_enabled": conditional_pheromone_enabled,
                    "update_policy": update_policy,
                    "exploration_policy": exploration_policy,
                    "effective_epsilon": float(effective_epsilon),
                    "pheromone_deposit_count": 0,
                    "global_improved": bool(improved),
                    "global_improvement": (
                        float(carry_best - previous_best)
                        if carry_best is not None and previous_best is not None
                        else (0.0 if carry_best is not None else None)
                    ),
                    "no_improve_rounds": int(no_improve_rounds),
                    "stagnation_count": int(0 if improved else no_improve_rounds + 1),
                    "status": "no_new_unique",
                    **_step_diagnostics(),
                }
            )
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
        valid_keys = {config_key(cfg) for cfg, _score in eval_results}
        if cache_invalid_configs:
            invalid_cache.update(key for key in sampled_keys if key not in valid_keys)
        if not eval_results:
            carry_best = None
            if eval_cache:
                carry_best = float(max(sc for _cfg, sc in eval_cache.values()))
            history.append(
                {
                    "iteration": iteration + 1,
                    "best_score": carry_best,
                    "global_best_score": carry_best,
                    "iteration_best_score": None,
                    "iteration_mean_score": None,
                    "iteration_min_score": None,
                    "sampled_unique_count": len(sampled),
                    "sampled_distinct_count": len(sampled_keys),
                    "valid_count": 0,
                    "cache_size": len(eval_cache),
                    "invalid_cache_size": len(invalid_cache),
                    "draw_count": int(draw_count),
                    "duplicate_draw_count": int(duplicate_draw_count),
                    "cached_draw_count": int(cached_draw_count),
                    "invalid_cached_draw_count": int(invalid_cached_draw_count),
                    "evaluation_request_count": int(len(sampled)),
                    "cumulative_draw_count": int(cumulative_draw_count),
                    "cumulative_duplicate_draw_count": int(cumulative_duplicate_draw_count),
                    "cumulative_cached_draw_count": int(cumulative_cached_draw_count),
                    "cumulative_invalid_cached_draw_count": int(cumulative_invalid_cached_draw_count),
                    "cumulative_evaluation_request_count": int(cumulative_evaluation_request_count),
                    "conditional_pheromone_enabled": conditional_pheromone_enabled,
                    "update_policy": update_policy,
                    "exploration_policy": exploration_policy,
                    "effective_epsilon": float(effective_epsilon),
                    "pheromone_deposit_count": 0,
                    "global_improved": False,
                    "global_improvement": 0.0 if carry_best is not None else None,
                    "no_improve_rounds": int(no_improve_rounds),
                    "stagnation_count": int(no_improve_rounds + 1),
                    "status": "no_valid_evaluation",
                    **_step_diagnostics(),
                }
            )
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
            eval_cache[config_key(cfg)] = (dict(cfg), float(score))

        for step in pheromones:
            pheromones[step] *= (1 - evaporation)
        for key in k_conditional_pheromones:
            k_conditional_pheromones[key] *= (1 - evaporation)

        cached_results = [(cfg, sc) for cfg, sc in eval_cache.values()]
        cached_results.sort(key=lambda x: x[1], reverse=True)
        iteration_results = [(dict(cfg), float(sc)) for cfg, sc in eval_results]
        iteration_results.sort(key=lambda x: x[1], reverse=True)
        current_best = float(cached_results[0][1])
        previous_best = best_so_far
        improved_global = previous_best is None or current_best > (previous_best + float(min_improvement))

        if update_policy == "iteration_elite":
            reinforcement_pool = iteration_results
        elif update_policy == "improvement_only":
            reinforcement_pool = iteration_results if improved_global else []
        elif update_policy == "hybrid_elite":
            # Hybrid update: reserve one of the top-k deposits for the global
            # best and use the remaining slots for the current iteration's
            # elite candidates.  This keeps the deposit budget comparable to
            # the other policies while retaining both long-term memory and a
            # signal from the current iteration.
            hybrid_limit = max(1, int(top_k_pheromone))
            hybrid_candidates = list(cached_results[:1]) + list(iteration_results)
            selected_hybrid: List[Tuple[Dict[str, Any], float]] = []
            seen_hybrid: set[Tuple[Tuple[str, Any], ...]] = set()
            for candidate_cfg, candidate_score in hybrid_candidates:
                candidate_key = config_key(candidate_cfg)
                if candidate_key in seen_hybrid:
                    continue
                selected_hybrid.append((dict(candidate_cfg), float(candidate_score)))
                seen_hybrid.add(candidate_key)
                if len(selected_hybrid) >= hybrid_limit:
                    break
            reinforcement_pool = selected_hybrid
        else:
            reinforcement_pool = cached_results
        selected = (
            reinforcement_pool
            if use_all_iter_pipelines
            else reinforcement_pool[: min(top_k_pheromone, len(reinforcement_pool))]
        )
        scores = np.array([sc for _, sc in selected])

        if len(scores) == 0:
            weights = np.asarray([], dtype=float)
        elif weight_method == "linear" and len(scores) > 1:
            weights = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8) + 1e-3
        elif weight_method == "exponential" and len(scores) > 1:
            scaled = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            weights = np.exp(scaled)
        elif weight_method == "rank":
            selected.sort(key=lambda x: x[1], reverse=True)
            n = len(selected)
            rank_weights = np.arange(n, 0, -1)
            weights = rank_weights / rank_weights.sum()
            if tie_aware_rank_weights and n > 1:
                selected_scores = np.asarray([float(score) for _cfg, score in selected], dtype=float)
                start = 0
                while start < n:
                    end = start + 1
                    while end < n and np.isclose(
                        selected_scores[end], selected_scores[start], atol=EPS, rtol=1e-9
                    ):
                        end += 1
                    weights[start:end] = float(np.mean(weights[start:end]))
                    start = end
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
                if conditional_pheromone_enabled and len(path_context) >= markov_order:
                    context = tuple(path_context[-markov_order:])
                    k_pher = get_k_pheromone(step, context)
                    k_pher[val_idx] += weight
                path_context.append((step, val_idx))

        # MMAS bounds: clip pheromones to [tau_min, tau_max] after evaporation+reinforcement.
        _clip_pheromones()

        candidate_pipelines.extend(unsorted_res)
        iter_scores = np.asarray([float(sc) for _cfg, sc in eval_results], dtype=float)
        history.append(
            {
                "iteration": iteration + 1,
                "best_score": current_best,
                "global_best_score": current_best,
                "iteration_best_score": float(best_score),
                "iteration_mean_score": float(np.mean(iter_scores)) if iter_scores.size else None,
                "iteration_min_score": float(np.min(iter_scores)) if iter_scores.size else None,
                "sampled_unique_count": len(sampled),
                "sampled_distinct_count": len(sampled_keys),
                "valid_count": len(eval_results),
                "cache_size": len(eval_cache),
                "invalid_cache_size": len(invalid_cache),
                "draw_count": int(draw_count),
                "duplicate_draw_count": int(duplicate_draw_count),
                "cached_draw_count": int(cached_draw_count),
                "invalid_cached_draw_count": int(invalid_cached_draw_count),
                "evaluation_request_count": int(len(sampled)),
                "cumulative_draw_count": int(cumulative_draw_count),
                "cumulative_duplicate_draw_count": int(cumulative_duplicate_draw_count),
                "cumulative_cached_draw_count": int(cumulative_cached_draw_count),
                "cumulative_invalid_cached_draw_count": int(cumulative_invalid_cached_draw_count),
                "cumulative_evaluation_request_count": int(cumulative_evaluation_request_count),
                "conditional_pheromone_enabled": conditional_pheromone_enabled,
                "update_policy": update_policy,
                "exploration_policy": exploration_policy,
                "effective_epsilon": float(effective_epsilon),
                "pheromone_deposit_count": int(len(selected)),
                "reinforced_count": int(len(selected)),
                "reinforced_unique_score_count": int(len(np.unique(scores))),
                "reinforcement_scores": [float(value) for value in scores.tolist()],
                "reinforcement_weights": [float(value) for value in weights.tolist()],
                "tie_aware_rank_weights": bool(tie_aware_rank_weights),
                "global_improved": bool(improved_global),
                "global_improvement": (
                    float(current_best - previous_best)
                    if previous_best is not None
                    else 0.0
                ),
                "no_improve_rounds": int(no_improve_rounds),
                "stagnation_count": int(0 if improved_global else no_improve_rounds + 1),
                "status": "ok",
                **_step_diagnostics(),
            }
        )
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
            search_kind = f"k={markov_order}" if conditional_pheromone_enabled else "marginal"
            print(
                f"ACO Iter {iteration+1}/{n_iterations} — "
                f"best: {best_score:.4f} | {search_kind} | "
                f"draws={draw_count} eval_requests={len(sampled)} cache_hits={cached_draw_count} "
                f"duplicates={duplicate_draw_count} invalid_hits={invalid_cached_draw_count} "
                f"update={update_policy} deposits={len(selected)} epsilon={effective_epsilon:.3f}"
            )
        else:
            search_kind = f"k={markov_order}" if conditional_pheromone_enabled else "marginal"
            logger.info(
                "ACO Iter %s/%s - best: %.4f | %s | draws=%s eval_requests=%s cache_hits=%s duplicates=%s invalid_hits=%s update=%s deposits=%s epsilon=%.3f",
                iteration + 1,
                n_iterations,
                best_score,
                search_kind,
                draw_count,
                len(sampled),
                cached_draw_count,
                duplicate_draw_count,
                invalid_cached_draw_count,
                update_policy,
                len(selected),
                effective_epsilon,
            )

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
        label = "k-order Markov ACO" if conditional_pheromone_enabled else "marginal ACO"
        print(f"\n🏆 Top pipelines ({label}):")
        for i, (cfg, sc) in enumerate(final[:n_pipelines]):
            print(f"  {i+1}. {cfg.get('name', 'Pipeline')} — score: {sc:.4f}")
    if return_history:
        return final[:n_pipelines], unsorted_candidate_pipelines, history
    return final[:n_pipelines], unsorted_candidate_pipelines
