"""Paper-faithful component tests: Eq 7 flat average, MMAS bounds, ACO diagnostics."""
import numpy as np
import pandas as pd
import pytest

from automl_aco.search.heuristics import (
    aggregate_operator_heuristics_flat,
    compute_global_operator_prior,
    blend_eta_with_prior,
)
from automl_aco.search.aco import (
    _normalized_entropy,
    _resolve_mmas_bounds,
    search_pipelines_aco,
)


# ---- Eq 7: flat (unweighted) average of top-set scores per operator ----

def test_flat_average_is_plain_mean_of_oriented_scores():
    # Two neighbors, each contributing its best pipeline to the top-set.
    # pA uses scaling=standard (score 0.9); pB uses scaling=standard (score 0.5);
    # pC uses scaling=robust (score 0.7). Flat eta(scaling=standard) must be mean(0.9,0.5)=0.7,
    # eta(scaling=robust)=0.7, eta(scaling=minmax)=missing -> falls back to min observed.
    top_l_pipelines = {
        "d1": [{"pipeline": "pA", "oriented_score": 0.9}],
        "d2": [
            {"pipeline": "pB", "oriented_score": 0.5},
            {"pipeline": "pC", "oriented_score": 0.7},
        ],
    }
    configs = [
        {"name": "pA", "scaling": "standard"},
        {"name": "pB", "scaling": "standard"},
        {"name": "pC", "scaling": "robust"},
    ]
    options = {"scaling": ["standard", "robust", "minmax"]}
    raw = aggregate_operator_heuristics_flat(top_l_pipelines, configs, options)
    arr = raw["scaling"]
    assert arr[0] == pytest.approx(0.7)  # mean(0.9, 0.5)  -- NOT weighted by similarity/quality
    assert arr[1] == pytest.approx(0.7)  # single value 0.7
    # missing operator falls back to the step's weakest observed signal (min of {0.7,0.7}=0.7)
    assert arr[2] == pytest.approx(0.7)


def test_flat_average_unweighted_differs_from_quality_weighting():
    # If aggregation were quality-weighted, the high-score pipeline would dominate; flat must not.
    top_l_pipelines = {
        "d1": [{"pipeline": "hi", "oriented_score": 1.0}],
        "d2": [{"pipeline": "lo", "oriented_score": 0.2}],
    }
    configs = [{"name": "hi", "imputation": "mean"}, {"name": "lo", "imputation": "mean"}]
    options = {"imputation": ["mean", "median"]}
    raw = aggregate_operator_heuristics_flat(top_l_pipelines, configs, options)
    assert raw["imputation"][0] == pytest.approx(0.6)  # plain mean(1.0, 0.2)


# ---- MMAS bounds (Eq 9/10 + Min-Max) ----

def test_global_operator_prior_suppresses_harmful_operators():
    # Reference matrix where, vs baseline, pipeline with svd scores lower and with zscore higher.
    perf = pd.DataFrame(
        {
            "d1": {"baseline": 0.80, "use_svd": 0.70, "use_zscore": 0.88},
            "d2": {"baseline": 0.60, "use_svd": 0.50, "use_zscore": 0.66},
        }
    )
    configs = [
        {"name": "baseline", "dimensionality_reduction": "none", "outlier_removal": "none"},
        {"name": "use_svd", "dimensionality_reduction": "svd", "outlier_removal": "none"},
        {"name": "use_zscore", "dimensionality_reduction": "none", "outlier_removal": "zscore"},
    ]
    options = {"dimensionality_reduction": ["none", "svd"], "outlier_removal": ["none", "zscore"]}
    prior = compute_global_operator_prior(perf, configs, options)
    # svd hurts -> floor (0); none for dimred is best -> 1
    assert prior["dimensionality_reduction"][1] < prior["dimensionality_reduction"][0]
    # zscore helps -> higher than none
    assert prior["outlier_removal"][1] > prior["outlier_removal"][0]


def test_blend_eta_with_prior_interpolates_and_floors():
    eta = {"s": np.array([1.0, 1.0, 1.0])}        # neighbor signal: flat
    prior = {"s": np.array([1.0, 0.5, 0.0])}      # global: 3rd op is harmful
    blended = blend_eta_with_prior(eta, prior, weight=0.5, eta_floor=0.05)
    # weight 0.5 -> harmful operator pulled down below the others
    assert blended["s"][2] < blended["s"][0]
    assert blended["s"].min() >= 0.05 - 1e-9


def test_resolve_mmas_bounds_auto():
    tmin, tmax = _resolve_mmas_bounds(0.2, None, None, 0.05)
    assert tmax == pytest.approx(5.0)   # 1/rho
    assert tmin == pytest.approx(0.25)  # 0.05 * tmax


def test_resolve_mmas_bounds_explicit_and_swap():
    assert _resolve_mmas_bounds(0.2, 0.1, 2.0, 0.05) == (0.1, 2.0)
    # swapped inputs are reordered
    assert _resolve_mmas_bounds(0.2, 2.0, 0.1, 0.05) == (0.1, 2.0)


def _toy_problem():
    options = {"a": ["x", "y", "z"], "b": ["p", "q"]}
    eta = {"a": np.array([1.0, 0.5, 0.2]), "b": np.array([0.8, 0.4])}

    def evaluate_fn(cfgs):
        res = []
        for c in cfgs:
            s = (1.0 if c["a"] == "x" else 0.3) + (0.5 if c["b"] == "p" else 0.1)
            res.append((c, s))
        res.sort(key=lambda t: t[1], reverse=True)
        return res[0][0], res[0][1], res, res

    return options, eta, evaluate_fn


def test_mmas_bounds_clip_pheromones():
    options, eta, evaluate_fn = _toy_problem()
    _final, _unsorted, hist = search_pipelines_aco(
        options, evaluate_fn, eta, n_ants=8, n_iterations=10,
        weight_method="linear", mmas_bounds=True, markov_order=2,
        return_history=True, seed=1,
    )
    ok = [h for h in hist if h.get("status") == "ok"]
    assert ok, "expected successful iterations"
    for h in ok:
        # tau_max auto = 1/0.2 = 5.0, tau_min = 0.25
        assert h["pheromone_max"] <= 5.0 + 1e-6
        assert h["pheromone_min"] >= 0.25 - 1e-6


def test_aco_diagnostics_present_and_entropy_bounded():
    options, eta, evaluate_fn = _toy_problem()
    _final, _unsorted, hist = search_pipelines_aco(
        options, evaluate_fn, eta, n_ants=8, n_iterations=5,
        weight_method="linear", mmas_bounds=False, return_history=True, seed=3,
    )
    ok = [h for h in hist if h.get("status") == "ok"]
    for h in ok:
        for key in ("step_entropy", "mean_entropy", "pheromone_min", "pheromone_max", "pheromone_saturation"):
            assert key in h
        assert 0.0 <= h["mean_entropy"] <= 1.0 + 1e-9


def test_normalized_entropy_extremes():
    assert _normalized_entropy(np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(1.0)
    assert _normalized_entropy(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_mmas_bounds_off_by_default_preserves_behavior():
    # Default (no bounds) must not record bound saturation as if bounded.
    options, eta, evaluate_fn = _toy_problem()
    _f, _u, hist = search_pipelines_aco(
        options, evaluate_fn, eta, n_ants=6, n_iterations=4, return_history=True, seed=5,
    )
    ok = [h for h in hist if h.get("status") == "ok"]
    # With bounds off, saturation is reported as 0/None (no tau_max to hit).
    for h in ok:
        assert h["pheromone_saturation"] in (None, 0.0)
