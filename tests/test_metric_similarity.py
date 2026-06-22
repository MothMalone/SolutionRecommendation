import numpy as np
import pandas as pd
import pytest

from automl_aco.metalearning.metric import build_similarity_target_matrix
from automl_aco.metalearning.recommender import MetaPipelineRecommender


def test_rank_cosine_target_is_scale_robust():
    perf_profiles = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    sim = build_similarity_target_matrix(
        perf_profiles=perf_profiles,
        similarity_target="rank_cosine",
        score_direction="higher_is_better",
    )
    assert sim.shape == (3, 3)
    assert np.isclose(sim[0, 1], 1.0, atol=1e-6)
    assert sim[0, 1] > sim[0, 2]


def test_score_direction_changes_similarity_target():
    perf_profiles = np.array(
        [
            [0.10, 0.10, 0.30, 0.50],
            [0.30, 0.20, 0.20, 0.40],
            [0.90, 0.70, 0.60, 0.20],
        ],
        dtype=float,
    )
    sim_higher = build_similarity_target_matrix(
        perf_profiles=perf_profiles,
        similarity_target="row_minmax_cosine",
        score_direction="higher_is_better",
    )
    sim_lower = build_similarity_target_matrix(
        perf_profiles=perf_profiles,
        similarity_target="row_minmax_cosine",
        score_direction="lower_is_better",
    )
    assert not np.allclose(sim_higher, sim_lower)


def test_legacy_similarity_target_mode_still_supported():
    perf_profiles = np.array(
        [
            [0.7, 0.8, 0.9],
            [0.4, 0.6, 0.5],
            [0.9, 0.2, 0.1],
        ],
        dtype=float,
    )
    sim = build_similarity_target_matrix(
        perf_profiles=perf_profiles,
        similarity_target="legacy_global_zscore_cosine",
        score_direction="higher_is_better",
    )
    assert np.isfinite(sim).all()
    assert np.allclose(np.diag(sim), np.ones(3))


def test_recommender_trains_metric_on_preprocessed_metafeatures():
    pytest.importorskip("torch")

    perf = pd.DataFrame(
        {
            "1": [0.80, 0.70],
            "2": [0.79, 0.71],
            "3": [0.20, 0.90],
        },
        index=["p1", "p2"],
    )
    meta = pd.DataFrame(
        {
            "rows": [100.0, 110.0, 10_000.0],
            "cols": [20.0, np.nan, 80.0],
        },
        index=["1", "2", "3"],
    )
    recommender = MetaPipelineRecommender(
        perf,
        meta,
        pipeline_configs=[{"name": "p1"}, {"name": "p2"}],
    )

    model = recommender.train_metric(hidden_dim=4, embed_dim=3, epochs=1, seed=7)

    assert model.params["metafeature_preprocessing"] == "preprocessed_input"
    assert model.params["metric_objective"] == "embedding_cosine"
    assert recommender.metric_params["metafeature_preprocessing"] == "preprocessed_input"
