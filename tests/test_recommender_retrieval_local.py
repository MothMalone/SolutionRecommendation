import numpy as np
import pandas as pd

import automl_aco.metalearning.recommender as recommender_module
from automl_aco.metalearning.recommender import MetaPipelineRecommender


def _cfg(name: str, *, imputation: str = "none", scaling: str = "none") -> dict:
    return {
        "name": name,
        "imputation": imputation,
        "scaling": scaling,
        "encoding": "onehot",
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "none",
    }


def test_retrieval_local_protects_nearest_neighbor_incumbent(monkeypatch):
    perf = pd.DataFrame(
        {
            "target": [0.10, 0.99],
            "neighbor": [0.90, 0.20],
        },
        index=["neighbor_pipeline", "target_pipeline"],
    )
    meta = pd.DataFrame(
        {
            "m1": [1.0, 0.9],
            "m2": [0.0, 0.1],
        },
        index=["target", "neighbor"],
    )
    pipeline_configs = [
        _cfg("neighbor_pipeline", imputation="none", scaling="none"),
        _cfg("target_pipeline", imputation="mean", scaling="standard"),
    ]
    options = {
        "imputation": ["none", "mean"],
        "scaling": ["none", "standard"],
        "encoding": ["onehot"],
        "feature_selection": ["none"],
        "outlier_removal": ["none"],
        "dimensionality_reduction": ["none"],
    }
    recommender = MetaPipelineRecommender(perf, meta, pipeline_configs=pipeline_configs)

    def fake_simple(_dataset, _target_column, candidate_configs, proxy_settings=None):
        results = []
        for cfg in candidate_configs:
            score = 0.95 if cfg.get("imputation") == "mean" else 0.50
            results.append((dict(cfg), score))
        sorted_results = sorted(results, key=lambda item: item[1], reverse=True)
        return sorted_results[0][0], sorted_results[0][1], sorted_results, results

    autogluon_candidates = []

    def fake_autogluon(
        _dataset,
        _target_column,
        candidate_configs,
        time_limit_per_model=300,
        autogluon_profile="best_quality",
        **_kwargs,          # the real evaluator keeps gaining optional kwargs
    ):
        autogluon_candidates.extend([dict(cfg) for cfg in candidate_configs])
        results = []
        for cfg in candidate_configs:
            score = 0.97 if cfg.get("imputation") == "mean" else 0.80
            results.append((dict(cfg), score))
        sorted_results = sorted(results, key=lambda item: item[1], reverse=True)
        return sorted_results[0][0], sorted_results[0][1], sorted_results, results

    monkeypatch.setattr(recommender, "_evaluate_candidates_with_simple_models", fake_simple)
    monkeypatch.setattr(recommender, "_evaluate_candidates_with_autogluon", fake_autogluon)
    monkeypatch.setattr(recommender_module, "AUTOGLUON_AVAILABLE", True)

    dataset = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "target": [0, 1, 0, 1],
        }
    )

    result = recommender.recommend(
        new_dataset=dataset,
        target_column="target",
        options=options,
        k=1,
        use_autogluon=True,
        use_aco=True,
        optimizer="retrieval_local",
        sample_budget=4,
        final_autogluon_topk=1,
        metafeatures_func=lambda _df: np.array([1.0, 0.0]),
        aco_params={
            "query_dataset_id": "target",
            "require_autogluon": False,
            "retrieval_local_neighbor_k": 1,
            "retrieval_local_top_l": 1,
            "retrieval_local_radius": 1,
            "seed": 42,
        },
    )

    assert result["optimizer"] == "retrieval_local"
    assert result["retrieval_local_search"]["incumbent_protection"] is True
    assert result["retrieval_local_search"]["protected_candidates"] == ["neighbor_pipeline"]
    assert any(cfg.get("name") == "neighbor_pipeline" for cfg in autogluon_candidates)
    assert any(cfg.get("imputation") == "mean" for cfg in autogluon_candidates)
    assert all(cfg.get("name") != "target_pipeline" for cfg in autogluon_candidates)
    assert result["pipeline_config"]["imputation"] == "mean"
    assert result["final_evaluation"]["method"] == "autogluon"
