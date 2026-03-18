import numpy as np
import pandas as pd

from automl_aco.metalearning.recommender import MetaPipelineRecommender


def test_encode_pipeline_config():
    perf = pd.DataFrame([[0.8, 0.7]], index=["p1"], columns=[1, 2])
    meta = pd.DataFrame([[0.1, 0.2], [0.2, 0.3]], index=[1, 2], columns=["f1", "f2"])
    recommender = MetaPipelineRecommender(perf, meta, pipeline_configs=[{"name": "p1"}])

    options = {"imputation": ["none", "mean"], "scaling": ["none", "standard"]}
    cfg = {"imputation": "mean", "scaling": "none"}
    encoding = recommender.encode_pipeline_config(cfg, options)

    expected = np.array([0, 1, 1, 0], dtype=float)
    assert np.array_equal(encoding, expected)


def test_recommender_sanitizes_infinite_metafeatures():
    perf = pd.DataFrame([[0.8, 0.7]], index=["p1"], columns=[1, 2])
    meta = pd.DataFrame(
        [[np.inf, 0.2], [0.2, -np.inf]],
        index=[1, 2],
        columns=["f1", "f2"],
    )
    recommender = MetaPipelineRecommender(perf, meta, pipeline_configs=[{"name": "p1"}])
    assert np.isfinite(recommender.metafeatures_imputed).all()
