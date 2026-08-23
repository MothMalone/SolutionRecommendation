import numpy as np
import pytest

from automl_aco.search.aco import (
    apply_interaction_prior,
    compute_legacy_mixed_sampling_probabilities,
    compute_sampling_probabilities,
    mix_with_uniform_exploration,
    search_pipelines_aco,
)
from automl_aco.search.heuristics import build_pairwise_interaction_priors


def _make_eta(options):
    return {step: np.ones(len(vals), dtype=float) for step, vals in options.items()}


def _dummy_evaluate_factory(options):
    def _evaluate(configs):
        results = []
        for cfg in configs:
            score = 0.0
            for step, vals in options.items():
                score += vals.index(cfg[step]) + 1
            results.append((cfg, score))
        if not results:
            return None, float("nan"), [], []
        unsorted = results.copy()
        results.sort(key=lambda x: x[1], reverse=True)
        best_cfg, best_score = results[0]
        return best_cfg, best_score, results, unsorted

    return _evaluate


def test_aco_returns_k_pipelines():
    options = {"imputation": ["none", "mean"], "scaling": ["none", "standard"]}
    eta = _make_eta(options)
    evaluate_fn = _dummy_evaluate_factory(options)

    final, _unsorted = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate_fn,
        eta=eta,
        n_pipelines=2,
        n_ants=2,
        n_iterations=2,
        seed=42,
    )

    assert len(final) == 2


def test_compute_sampling_probabilities_are_finite_and_normalized():
    probs = compute_sampling_probabilities(
        pheromone=np.array([1.0, 2.0, 3.0], dtype=float),
        eta_step=np.array([0.2, 0.5, 1.0], dtype=float),
        alpha=1.0,
        beta=2.0,
    )
    assert np.isfinite(probs).all()
    assert np.isclose(float(np.sum(probs)), 1.0)
    assert (probs > 0.0).all()


def test_apply_interaction_prior_boosts_supported_pair():
    eta = np.ones(2, dtype=float)
    priors = {("imputation", 0, "scaling"): np.array([0.2, 1.0], dtype=float)}

    adjusted = apply_interaction_prior(
        eta_step=eta,
        step="scaling",
        path_history=[("imputation", 0)],
        interaction_priors=priors,
        interaction_prior_strength=1.0,
    )

    assert adjusted[1] > adjusted[0]


def test_build_pairwise_interaction_priors_preserves_pipeline_context():
    options = {
        "imputation": ["none", "median"],
        "scaling": ["standard", "robust"],
    }
    pipeline_configs = [
        {"name": "p_good", "imputation": "median", "scaling": "robust"},
        {"name": "p_weak", "imputation": "none", "scaling": "standard"},
    ]
    candidates = [
        {"pipeline": "p_good", "candidate_weight": 1.0},
        {"pipeline": "p_weak", "candidate_weight": 0.2},
    ]

    priors = build_pairwise_interaction_priors(
        transfer_candidates=candidates,
        pipeline_configs=pipeline_configs,
        options=options,
        interaction_prior_floor=0.2,
    )

    median_idx = options["imputation"].index("median")
    robust_idx = options["scaling"].index("robust")
    standard_idx = options["scaling"].index("standard")
    prior = priors[("imputation", median_idx, "scaling")]
    assert prior[robust_idx] > prior[standard_idx]


def test_legacy_mixed_sampling_probabilities_match_notebook_raw_mix():
    marginal = np.array([1.0, 20.0, 2.0], dtype=float)
    conditional = np.array([30.0, 1.0, 4.0], dtype=float)
    eta = np.array([1.0, 0.5, 0.2], dtype=float)
    alpha = 1.0
    beta = 2.0
    lam = 0.7

    probs = compute_legacy_mixed_sampling_probabilities(
        marginal_pheromone=marginal,
        conditional_pheromone=conditional,
        eta_step=eta,
        alpha=alpha,
        beta=beta,
        lambda_smooth=lam,
    )

    raw_m = (marginal ** alpha) * (eta ** beta)
    raw_k = (conditional ** alpha) * (eta ** beta)
    expected = lam * raw_k + (1.0 - lam) * raw_m
    expected = expected / expected.sum()

    separately_normalized = (
        lam * compute_sampling_probabilities(conditional, eta, alpha, beta)
        + (1.0 - lam) * compute_sampling_probabilities(marginal, eta, alpha, beta)
    )

    assert np.allclose(probs, expected)
    assert not np.allclose(probs, separately_normalized)


def test_aco_rejects_mismatched_eta_shape():
    options = {"imputation": ["none", "mean"]}
    eta = {"imputation": np.array([1.0], dtype=float)}
    evaluate_fn = _dummy_evaluate_factory(options)

    with pytest.raises(ValueError):
        search_pipelines_aco(
            options=options,
            evaluate_fn=evaluate_fn,
            eta=eta,
            n_pipelines=1,
            n_ants=1,
            n_iterations=1,
            seed=7,
        )


def test_aco_sanitizes_non_finite_eta_and_runs():
    options = {"imputation": ["none", "mean"], "scaling": ["none", "standard"]}
    eta = {
        "imputation": np.array([np.nan, 0.0], dtype=float),
        "scaling": np.array([1.0, np.nan], dtype=float),
    }
    evaluate_fn = _dummy_evaluate_factory(options)

    final, _unsorted = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate_fn,
        eta=eta,
        n_pipelines=1,
        n_ants=2,
        n_iterations=2,
        seed=11,
    )

    assert len(final) == 1


def test_aco_return_history_contains_iteration_records_only():
    options = {
        "imputation": ["none", "mean"],
        "scaling": ["none", "standard"],
        "feature_selection": ["none", "k_best"],
    }
    eta = _make_eta(options)
    evaluate_fn = _dummy_evaluate_factory(options)

    _final, _unsorted, history = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate_fn,
        eta=eta,
        n_pipelines=2,
        n_ants=2,
        n_iterations=4,
        seed=42,
        return_history=True,
    )

    assert len(history) > 1
    assert all(isinstance(row, dict) for row in history)
    assert all("iteration" in row and "best_score" in row for row in history)
    assert all("global_best_score" in row and "iteration_best_score" in row for row in history)
    assert all("sampled_unique_count" in row and "valid_count" in row for row in history)


def test_aco_history_carries_forward_when_no_new_configs_sampled():
    options = {
        "imputation": ["none"],
        "scaling": ["none"],
    }
    eta = _make_eta(options)
    evaluate_fn = _dummy_evaluate_factory(options)

    _final, _unsorted, history = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate_fn,
        eta=eta,
        n_pipelines=1,
        n_ants=3,
        n_iterations=4,
        seed=42,
        return_history=True,
    )

    assert len(history) == 4
    assert [row.get("iteration") for row in history] == [1, 2, 3, 4]
    assert all(row.get("best_score") == history[0].get("best_score") for row in history)


def test_aco_early_stop_stops_history_growth():
    options = {
        "imputation": ["none"],
        "scaling": ["none"],
    }
    eta = _make_eta(options)
    evaluate_fn = _dummy_evaluate_factory(options)

    _final, _unsorted, history = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate_fn,
        eta=eta,
        n_pipelines=1,
        n_ants=3,
        n_iterations=10,
        seed=42,
        early_stop_rounds=2,
        min_improvement=0.0,
        return_history=True,
    )

    assert len(history) == 3
    assert [row.get("iteration") for row in history] == [1, 2, 3]


def test_canonical_cache_matches_evaluator_enriched_configs():
    options = {"imputation": ["none"], "scaling": ["none"]}
    calls = []

    def evaluate(configs):
        calls.extend(configs)
        results = []
        for cfg in configs:
            enriched = dict(cfg)
            enriched["name"] = "generated"
            enriched["step_order"] = list(options)
            results.append((enriched, 1.0))
        return results[0][0], 1.0, results, results.copy()

    _final, _unsorted, history = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate,
        eta=_make_eta(options),
        n_pipelines=1,
        n_ants=10,
        n_iterations=5,
        canonical_cache_keys=True,
        deduplicate_iteration=True,
        return_history=True,
    )

    assert len(calls) == 1
    assert history[-1]["cumulative_evaluation_request_count"] == 1
    assert history[-1]["cumulative_cached_draw_count"] == 40


def test_same_iteration_deduplication_removes_repeated_requests():
    options = {"imputation": ["none"], "scaling": ["none"]}
    requested_batch_sizes = []

    def evaluate(configs):
        requested_batch_sizes.append(len(configs))
        return _dummy_evaluate_factory(options)(configs)

    search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate,
        eta=_make_eta(options),
        n_pipelines=1,
        n_ants=10,
        n_iterations=1,
        canonical_cache_keys=True,
        deduplicate_iteration=True,
    )

    assert requested_batch_sizes == [1]


def test_invalid_config_negative_cache_avoids_repeated_evaluation():
    options = {"imputation": ["invalid"]}
    requested_batch_sizes = []

    def evaluate(configs):
        requested_batch_sizes.append(len(configs))
        return None, float("nan"), [], []

    _final, _unsorted, history = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate,
        eta=_make_eta(options),
        n_pipelines=1,
        n_ants=5,
        n_iterations=4,
        canonical_cache_keys=True,
        deduplicate_iteration=True,
        cache_invalid_configs=True,
        return_history=True,
    )

    assert requested_batch_sizes == [1]
    assert history[-1]["invalid_cache_size"] == 1
    assert history[-1]["cumulative_invalid_cached_draw_count"] == 15


def test_tie_aware_rank_reinforcement_gives_equal_weight_to_ties():
    options = {"imputation": ["a", "b", "c"]}

    def evaluate(configs):
        results = [(dict(cfg), 0.5) for cfg in configs]
        return results[0][0], 0.5, results, results.copy()

    _final, _unsorted, history = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate,
        eta=_make_eta(options),
        n_pipelines=1,
        n_ants=3,
        n_iterations=1,
        top_k_pheromone=3,
        canonical_cache_keys=True,
        deduplicate_iteration=True,
        refill_unique_ants=True,
        tie_aware_rank_weights=True,
        return_history=True,
        seed=2,
    )

    weights = history[0]["reinforcement_weights"]
    assert len(weights) == 3
    assert np.allclose(weights, np.repeat(1.0 / 3.0, 3))


def test_zero_markov_weight_disables_conditional_pheromone_path():
    options = {"imputation": ["none", "mean"], "scaling": ["none", "standard"]}
    _final, _unsorted, history = search_pipelines_aco(
        options=options,
        evaluate_fn=_dummy_evaluate_factory(options),
        eta=_make_eta(options),
        n_pipelines=1,
        n_ants=2,
        n_iterations=1,
        markov_order=2,
        lambda_smooth=0.0,
        return_history=True,
    )

    assert history[0]["conditional_pheromone_enabled"] is False


def test_refill_unique_ants_preserves_unique_evaluation_budget():
    options = {
        "imputation": ["none", "mean", "median"],
        "scaling": ["none", "standard", "robust"],
    }
    requested_batch_sizes = []

    def evaluate(configs):
        requested_batch_sizes.append(len(configs))
        return _dummy_evaluate_factory(options)(configs)

    _final, _unsorted, history = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate,
        eta=_make_eta(options),
        n_pipelines=1,
        n_ants=5,
        n_iterations=1,
        canonical_cache_keys=True,
        deduplicate_iteration=True,
        refill_unique_ants=True,
        return_history=True,
        seed=4,
    )

    assert requested_batch_sizes == [5]
    assert history[0]["evaluation_request_count"] == 5
    assert history[0]["sampled_distinct_count"] == 5


def test_uniform_exploration_mixture_is_normalized_and_floored():
    mixed = mix_with_uniform_exploration(np.array([1.0, 0.0, 0.0]), epsilon=0.3)
    assert np.isclose(mixed.sum(), 1.0)
    assert np.allclose(mixed, np.array([0.8, 0.1, 0.1]))


@pytest.mark.parametrize(
    ("policy", "expected_deposits"),
    [
        ("global_elite", [1, 1, 1]),
        ("iteration_elite", [1, 1, 1]),
        ("improvement_only", [1, 0, 0]),
    ],
)
def test_aco_update_policies_control_reinforcement(policy, expected_deposits):
    options = {"imputation": [f"v{i}" for i in range(10_000)]}

    def evaluate(configs):
        results = [(dict(cfg), 1.0) for cfg in configs]
        return results[0][0], 1.0, results, results.copy()

    _final, _unsorted, history = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate,
        eta=_make_eta(options),
        n_pipelines=1,
        n_ants=1,
        n_iterations=3,
        top_k_pheromone=1,
        update_policy=policy,
        return_history=True,
        seed=8,
    )
    assert [row["pheromone_deposit_count"] for row in history] == expected_deposits


def test_stagnation_exploration_increases_then_resets_after_improvement():
    options = {"imputation": [f"v{i}" for i in range(100)]}
    calls = 0

    def evaluate(configs):
        nonlocal calls
        calls += 1
        score = 2.0 if calls == 3 else 1.0
        results = [(dict(cfg), score) for cfg in configs]
        return results[0][0], score, results, results.copy()

    _final, _unsorted, history = search_pipelines_aco(
        options=options,
        evaluate_fn=evaluate,
        eta=_make_eta(options),
        n_pipelines=1,
        n_ants=1,
        n_iterations=4,
        exploration_policy="stagnation",
        exploration_initial_epsilon=0.05,
        exploration_step=0.05,
        exploration_max_epsilon=0.30,
        return_history=True,
        seed=11,
    )
    assert np.allclose(
        [row["effective_epsilon"] for row in history],
        [0.05, 0.05, 0.10, 0.05],
    )
