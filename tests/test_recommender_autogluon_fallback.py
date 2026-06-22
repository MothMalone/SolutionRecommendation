from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from automl_aco.metalearning import recommender as recommender_module
from automl_aco.metalearning.recommender import MetaPipelineRecommender


def test_final_autogluon_unavailable_falls_back_to_simple_models(monkeypatch):
    performance_matrix = pd.DataFrame([[0.8], [0.6]], index=["p1", "p2"], columns=["1"])
    metafeatures = pd.DataFrame([[0.1, 0.2]], index=["1"], columns=["f1", "f2"])
    pipeline_configs: List[Dict[str, Any]] = [
        {"name": "p1", "imputation": "none"},
        {"name": "p2", "imputation": "mean"},
    ]
    recommender = MetaPipelineRecommender(
        performance_matrix=performance_matrix,
        metafeatures_df=metafeatures,
        pipeline_configs=pipeline_configs,
        verbose=False,
    )

    # Force the code path that attempts AutoGluon first.
    monkeypatch.setattr(recommender_module, "AUTOGLUON_AVAILABLE", True)

    def fake_search(*args, **kwargs):
        top = [({"name": "p1", "imputation": "none"}, 0.71)]
        unsorted = [({"name": "p1", "imputation": "none"}, 0.71)]
        history = []
        return top, unsorted, history

    def fake_autogluon(*args, **kwargs):
        raise RuntimeError("AutoGluon not available in environment")

    def fake_simple_models(*args, **kwargs):
        cfg = {"name": "p1", "imputation": "none"}
        score = 0.79
        ranked = [(cfg, score)]
        unsorted = ranked.copy()
        return cfg, score, ranked, unsorted

    monkeypatch.setattr(recommender, "_search_pipelines_aco", fake_search)
    monkeypatch.setattr(recommender, "_evaluate_candidates_with_autogluon", fake_autogluon)
    monkeypatch.setattr(recommender, "_evaluate_candidates_with_simple_models", fake_simple_models)

    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]})

    result = recommender.recommend(
        new_dataset=df,
        target_column="target",
        k=1,
        eval_k=1,
        use_autogluon=True,
        use_aco=True,
        options={"imputation": ["none", "mean"]},
        aco_params={"n_ants": 1, "n_iterations": 1, "require_autogluon": False},
        metafeatures_func=lambda _dataset: {"f1": 0.1, "f2": 0.2},
        final_autogluon_topk=1,
    )

    assert result["final_evaluation"]["method"] == "simple_models_fallback"
    assert float(result["final_evaluation"]["score"]) == 0.79


def test_require_autogluon_fails_fast_when_runtime_unavailable(monkeypatch):
    performance_matrix = pd.DataFrame([[0.8], [0.6]], index=["p1", "p2"], columns=["1"])
    metafeatures = pd.DataFrame([[0.1, 0.2]], index=["1"], columns=["f1", "f2"])
    recommender = MetaPipelineRecommender(
        performance_matrix=performance_matrix,
        metafeatures_df=metafeatures,
        pipeline_configs=[{"name": "p1", "imputation": "none"}],
        verbose=False,
    )

    monkeypatch.setattr(recommender_module, "_autogluon_runtime_error", lambda: "AutoGluon missing")

    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]})

    try:
        recommender.recommend(
            new_dataset=df,
            target_column="target",
            k=1,
            eval_k=1,
            use_autogluon=True,
            use_aco=False,
            options={"imputation": ["none"]},
            aco_params={"require_autogluon": True},
            metafeatures_func=lambda _dataset: {"f1": 0.1, "f2": 0.2},
        )
    except RuntimeError as exc:
        assert "AutoGluon is required for evaluation but is unavailable" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when AutoGluon is required and unavailable")


def test_no_search_recommendation_excludes_query_dataset(monkeypatch):
    performance_matrix = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["self_best", "neighbor_best"],
        columns=["target_ds", "neighbor_ds"],
    )
    metafeatures = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["target_ds", "neighbor_ds"],
        columns=["f1", "f2"],
    )
    pipeline_configs: List[Dict[str, Any]] = [
        {"name": "self_best", "imputation": "mean"},
        {"name": "neighbor_best", "imputation": "median"},
    ]
    recommender = MetaPipelineRecommender(
        performance_matrix=performance_matrix,
        metafeatures_df=metafeatures,
        pipeline_configs=pipeline_configs,
        verbose=False,
    )
    monkeypatch.setattr(
        recommender,
        "_compute_dataset_similarities",
        lambda _mf: [("target_ds", 1.0), ("neighbor_ds", 0.9)],
    )

    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]})
    result = recommender.recommend(
        new_dataset=df,
        target_column="target",
        k=1,
        eval_k=1,
        use_autogluon=False,
        use_aco=False,
        options={"imputation": ["mean", "median"]},
        aco_params={"query_dataset_id": "target_ds"},
        metafeatures_func=lambda _dataset: {"f1": 1.0, "f2": 0.0},
    )

    assert result["similar_datasets"] == ["neighbor_ds"]
    assert result["pipeline_config"]["name"] == "neighbor_best"


def test_aco_final_eval_can_protect_retrieval_incumbent(monkeypatch):
    performance_matrix = pd.DataFrame(
        {
            "target_ds": [0.10, 0.99, 0.20],
            "neighbor_ds": [0.90, 0.20, 0.10],
        },
        index=["neighbor_pipeline", "target_pipeline", "aco_pipeline"],
    )
    metafeatures = pd.DataFrame(
        [[1.0, 0.0], [0.9, 0.1]],
        index=["target_ds", "neighbor_ds"],
        columns=["f1", "f2"],
    )
    pipeline_configs: List[Dict[str, Any]] = [
        {"name": "neighbor_pipeline", "imputation": "none"},
        {"name": "target_pipeline", "imputation": "mean"},
        {"name": "aco_pipeline", "imputation": "median"},
    ]
    recommender = MetaPipelineRecommender(
        performance_matrix=performance_matrix,
        metafeatures_df=metafeatures,
        pipeline_configs=pipeline_configs,
        verbose=False,
    )
    monkeypatch.setattr(recommender_module, "AUTOGLUON_AVAILABLE", True)
    monkeypatch.setattr(
        recommender,
        "_compute_dataset_similarities",
        lambda _mf: [("target_ds", 1.0), ("neighbor_ds", 0.9)],
    )

    def fake_search(*args, **kwargs):
        cfg = {"name": "aco_pipeline", "imputation": "median"}
        return [(cfg, 0.95)], [(cfg, 0.95)], []

    seen_candidates: List[str] = []

    def fake_autogluon(*args, **kwargs):
        candidates = [dict(cfg) for cfg in kwargs.get("candidate_configs", args[2] if len(args) > 2 else [])]
        seen_candidates.extend(str(cfg.get("name")) for cfg in candidates)
        scored = []
        for cfg in candidates:
            score = 0.91 if cfg.get("name") == "neighbor_pipeline" else 0.50
            scored.append((cfg, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[0][0], scored[0][1], scored, scored.copy()

    monkeypatch.setattr(recommender, "_search_pipelines_aco", fake_search)
    monkeypatch.setattr(recommender, "_evaluate_candidates_with_autogluon", fake_autogluon)

    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]})
    result = recommender.recommend(
        new_dataset=df,
        target_column="target",
        k=1,
        eval_k=1,
        use_autogluon=True,
        use_aco=True,
        options={"imputation": ["none", "mean", "median"]},
        aco_params={
            "query_dataset_id": "target_ds",
            "require_autogluon": False,
            "protect_retrieval_incumbent": True,
        },
        metafeatures_func=lambda _dataset: {"f1": 1.0, "f2": 0.0},
        final_autogluon_topk=1,
    )

    assert "neighbor_pipeline" in seen_candidates
    assert "aco_pipeline" in seen_candidates
    assert result["pipeline_config"]["name"] == "neighbor_pipeline"
    assert result["retrieval_incumbent_protection"]["enabled"] is True
