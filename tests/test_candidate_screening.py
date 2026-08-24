"""The middle rung of the selection ladder: cheap real AutoGluon between proxy and gate.

The rung exists because the proxy's #1 is often not the best candidate the search found (measured
Spearman ~0.42 against AutoGluon), while the full CV gate costs (folds+1) fits and cannot absorb
the whole shortlist. These tests pin the three properties that make it safe to turn on:

  * it actually RESCUES a candidate the proxy ranked below #1,
  * it ranks on VALIDATION, never on test,
  * every failure mode degrades to today's behaviour instead of losing the run.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from automl_aco.metalearning import recommender as recommender_module
from automl_aco.metalearning.recommender import MetaPipelineRecommender


def _recommender():
    return MetaPipelineRecommender(
        performance_matrix=pd.DataFrame([[0.8], [0.6]], index=["p1", "p2"], columns=["1"]),
        metafeatures_df=pd.DataFrame([[0.1, 0.2]], index=["1"], columns=["f1", "f2"]),
        pipeline_configs=[{"name": "p1"}, {"name": "p2"}],
        verbose=False,
    )


CANDIDATES: List[Dict[str, Any]] = [{"name": f"c{i}", "imputation": "none"} for i in range(6)]


def test_screening_keeps_the_validation_winner_not_the_proxy_order(monkeypatch):
    """The whole point: c4 was 5th by proxy and must survive because validation likes it."""
    rec = _recommender()
    seen = {}

    def fake_eval(dataset, target_column, candidate_configs, **kw):
        seen.update(kw)
        seen["n_candidates"] = len(candidate_configs)
        # select_on_val=True means the caller returns results ordered by VALIDATION score.
        ordered = [({"name": "c4"}, 0.91), ({"name": "c1"}, 0.88), ({"name": "c0"}, 0.60)]
        return ordered[0][0], 0.91, ordered, ordered

    monkeypatch.setattr(rec, "_evaluate_candidates_with_autogluon", fake_eval)
    kept = rec._screen_candidates_with_autogluon(
        pd.DataFrame({"x": [0, 1], "target": [0, 1]}), "target", CANDIDATES,
        keep=2, time_limit_per_model=30, autogluon_profile="local_rf_xt",
        seed=42, prepare_mode="leakfree",
    )

    assert [c["name"] for c in kept] == ["c4", "c1"]
    assert seen["n_candidates"] == 6, "screening must see the whole shortlist"
    # The two properties that keep this leak-free and cheap.
    assert seen["select_on_val"] is True, "screening must rank on validation, never on test"
    assert seen["cv_select_folds"] == 0, "screening must be ONE fit per candidate, not k+1"
    assert seen["select_default_name"] is None, "the floor belongs to the gate, not this rung"


def test_screening_is_a_noop_when_the_pool_is_not_larger_than_keep(monkeypatch):
    """No shortlist to narrow means no reason to pay for a screening pass."""
    rec = _recommender()

    def boom(*a, **k):
        raise AssertionError("must not evaluate when there is nothing to narrow")

    monkeypatch.setattr(rec, "_evaluate_candidates_with_autogluon", boom)
    kept = rec._screen_candidates_with_autogluon(
        pd.DataFrame({"x": [0, 1], "target": [0, 1]}), "target", CANDIDATES[:2],
        keep=3, time_limit_per_model=30, autogluon_profile="local_rf_xt",
        seed=42, prepare_mode="leakfree",
    )
    assert [c["name"] for c in kept] == ["c0", "c1"]


def test_screening_failure_degrades_to_proxy_order(monkeypatch):
    """A screening crash must cost the RESCUE, not the run."""
    rec = _recommender()

    def fake_eval(*a, **k):
        raise RuntimeError("screening blew up")

    monkeypatch.setattr(rec, "_evaluate_candidates_with_autogluon", fake_eval)
    kept = rec._screen_candidates_with_autogluon(
        pd.DataFrame({"x": [0, 1], "target": [0, 1]}), "target", CANDIDATES,
        keep=2, time_limit_per_model=30, autogluon_profile="local_rf_xt",
        seed=42, prepare_mode="leakfree",
    )
    assert [c["name"] for c in kept] == ["c0", "c1"], "must fall back to the proxy's own order"


def test_screening_propagates_autogluon_unavailable(monkeypatch):
    """--require-autogluon must still be able to fail the run loudly.

    Swallowing this one would silently turn a required-AutoGluon run into a proxy-only run, which
    is exactly the failure that put `autogluon_failed` rows into published results before.
    """
    rec = _recommender()

    def fake_eval(*a, **k):
        raise RuntimeError("AutoGluon not available in environment")

    monkeypatch.setattr(rec, "_evaluate_candidates_with_autogluon", fake_eval)
    monkeypatch.setattr(recommender_module, "_is_autogluon_unavailable_error", lambda exc: True)
    with pytest.raises(RuntimeError, match="not available"):
        rec._screen_candidates_with_autogluon(
            pd.DataFrame({"x": [0, 1], "target": [0, 1]}), "target", CANDIDATES,
            keep=2, time_limit_per_model=30, autogluon_profile="local_rf_xt",
            seed=42, prepare_mode="leakfree",
        )


def test_empty_screening_results_fall_back_to_proxy_order(monkeypatch):
    """Every candidate failing to fit is a real outcome on degenerate frames."""
    rec = _recommender()
    monkeypatch.setattr(rec, "_evaluate_candidates_with_autogluon",
                        lambda *a, **k: (None, float("nan"), [], []))
    kept = rec._screen_candidates_with_autogluon(
        pd.DataFrame({"x": [0, 1], "target": [0, 1]}), "target", CANDIDATES,
        keep=3, time_limit_per_model=30, autogluon_profile="local_rf_xt",
        seed=42, prepare_mode="leakfree",
    )
    assert [c["name"] for c in kept] == ["c0", "c1", "c2"]


def test_screen_pool_is_not_clamped_by_k(monkeypatch):
    """`--k` must not cap the screening shortlist.

    `k` means "top-k similar datasets" for retrieval, but it is ALSO passed as `n_pipelines` and
    truncates the deduped ACO ranking. Screening off that truncated list would silently clamp
    --screen-topk to k (5 by default), so a --screen-topk 20 run would quietly screen 5. This
    pins the full-ranking pool: with k=1, screening must still see all six candidates and the
    validation winner must reach the gate.
    """
    rec = _recommender()
    monkeypatch.setattr(recommender_module, "AUTOGLUON_AVAILABLE", True)

    imps = ["none", "mean", "median", "most_frequent", "constant", "knn"]
    scalers = ["standard", "minmax", "robust", "maxabs", "none", "standard"]
    aco = [({"name": f"c{i}", "imputation": imps[i], "scaling": scalers[i]}, 0.9 - 0.01 * i)
           for i in range(6)]
    monkeypatch.setattr(rec, "_search_pipelines_aco", lambda *a, **k: (aco, aco, []))

    calls = []

    def fake_eval(dataset, target_column, candidate_configs, **kw):
        calls.append((len(candidate_configs), kw.get("cv_select_folds"),
                      [c.get("name") for c in candidate_configs]))
        if not kw.get("cv_select_folds"):        # screening rung
            ordered = [({"name": "c4", "imputation": "constant", "scaling": "none"}, 0.91),
                       ({"name": "c1", "imputation": "mean", "scaling": "minmax"}, 0.88)]
            return ordered[0][0], 0.91, ordered, ordered
        best = candidate_configs[0]
        return best, 0.77, [(best, 0.77)], [(best, 0.77)]

    monkeypatch.setattr(rec, "_evaluate_candidates_with_autogluon", fake_eval)

    rec.recommend(
        new_dataset=pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]}),
        target_column="target", k=1, eval_k=1, use_autogluon=True, use_aco=True,
        metafeatures_func=lambda d: {"f1": 0.1, "f2": 0.2},
        aco_params={"require_autogluon": False, "hybrid_select": True, "cv_select_folds": 3,
                    "screen_topk": 6, "screen_profile": "local_rf_xt", "screen_time_limit": 30},
        final_autogluon_topk=2,
        options={"imputation": imps, "scaling": ["standard", "minmax", "robust", "maxabs", "none"]},
    )

    assert calls[0][0] == 6, f"screening saw {calls[0][0]} candidates, not the full 6 (k clamped it)"
    assert not calls[0][1], "screening rung must not run CV"
    assert "c4" in calls[1][2], f"proxy-rank-5 winner never reached the gate: {calls[1][2]}"
