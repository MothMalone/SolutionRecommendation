"""Regression guard: a fixed-seed ACO run on a deterministic synthetic problem must reproduce a
committed best pipeline + score, and the paper flat-average heuristic must reproduce committed eta.

These are deterministic (no torch / no AutoGluon), so any score-moving change to the search loop or
heuristic aggregation will fail here and must be explained. The committed values are computed from
the method itself, not from any of the 24 test datasets.
"""
import json
import os

import numpy as np

from automl_aco.search.aco import search_pipelines_aco
from automl_aco.search.heuristics import aggregate_operator_heuristics_flat
from automl_aco.search.heuristics import normalize_eta_with_floor

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "regression_baseline.json")


def _synthetic_aco_problem():
    options = {
        "imputation": ["none", "mean", "median"],
        "scaling": ["none", "standard", "minmax"],
        "outlier_removal": ["none", "iqr"],
    }
    eta = {
        "imputation": np.array([0.2, 1.0, 0.6]),
        "scaling": np.array([0.3, 1.0, 0.5]),
        "outlier_removal": np.array([1.0, 0.4]),
    }

    def evaluate_fn(cfgs):
        res = []
        for c in cfgs:
            s = 0.0
            s += {"none": 0.10, "mean": 0.40, "median": 0.25}[c["imputation"]]
            s += {"none": 0.05, "standard": 0.30, "minmax": 0.15}[c["scaling"]]
            s += {"none": 0.20, "iqr": 0.05}[c["outlier_removal"]]
            res.append((c, round(s, 6)))
        res.sort(key=lambda t: t[1], reverse=True)
        return res[0][0], res[0][1], res, res

    return options, eta, evaluate_fn


def _run_reference():
    options, eta, evaluate_fn = _synthetic_aco_problem()
    final, _unsorted = search_pipelines_aco(
        options, evaluate_fn, eta,
        n_pipelines=1, n_ants=6, n_iterations=8, seed=42,
        alpha=1.0, beta=2.0, evaporation=0.2, top_k_pheromone=3,
        weight_method="linear", mmas_bounds=True, markov_order=2,
    )
    best_cfg, best_score = final[0]
    return {"best_pipeline": dict(best_cfg), "best_score": float(best_score)}


def test_aco_regression_matches_committed_baseline():
    current = _run_reference()
    if not os.path.exists(BASELINE_PATH):  # first run materializes the baseline
        os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, sort_keys=True)
    with open(BASELINE_PATH, encoding="utf-8") as f:
        committed = json.load(f)
    assert current["best_pipeline"] == committed["best_pipeline"], (
        "ACO chose a different pipeline than the committed baseline; explain the score-moving change."
    )
    assert abs(current["best_score"] - committed["best_score"]) < 1e-9, (
        "ACO best score drifted from the committed baseline; explain the change."
    )


def test_aco_is_deterministic_under_fixed_seed():
    a = _run_reference()
    b = _run_reference()
    assert a == b, "ACO is not deterministic under a fixed seed."


def test_flat_average_heuristic_regression():
    # Deterministic eta for a fixed top-set; guards Eq-7 aggregation + per-step min-max norm.
    top_l_pipelines = {
        "d1": [{"pipeline": "p1", "oriented_score": 0.9}],
        "d2": [{"pipeline": "p2", "oriented_score": 0.6}],
        "d3": [{"pipeline": "p3", "oriented_score": 0.3}],
    }
    configs = [
        {"name": "p1", "scaling": "standard", "imputation": "mean"},
        {"name": "p2", "scaling": "standard", "imputation": "median"},
        {"name": "p3", "scaling": "robust", "imputation": "mean"},
    ]
    options = {"scaling": ["standard", "robust", "minmax"], "imputation": ["mean", "median"]}
    raw = aggregate_operator_heuristics_flat(top_l_pipelines, configs, options)
    # scaling: standard=mean(0.9,0.6)=0.75, robust=0.3, minmax=missing->min(0.75,0.3)=0.3
    np.testing.assert_allclose(raw["scaling"], [0.75, 0.3, 0.3])
    # imputation: mean=mean(0.9,0.3)=0.6, median=0.6
    np.testing.assert_allclose(raw["imputation"], [0.6, 0.6])
    # normalization is deterministic and within [floor, 1]
    norm = normalize_eta_with_floor(raw, eta_floor=0.05)
    assert norm["scaling"].min() >= 0.05 - 1e-9 and norm["scaling"].max() <= 1.0 + 1e-9
