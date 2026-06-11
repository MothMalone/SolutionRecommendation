import numpy as np
import pytest

from automl_aco.search.aco import (
    compute_legacy_mixed_sampling_probabilities,
    compute_sampling_probabilities,
    search_pipelines_aco,
)


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
