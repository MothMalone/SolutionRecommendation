import numpy as np
import pandas as pd

from automl_aco.search.heuristics import compute_aco_heuristic


def test_heuristic_eta_shape():
    performance_matrix = pd.DataFrame(
        [[0.8, 0.7], [0.6, 0.9]],
        index=["p1", "p2"],
        columns=[1, 2],
    )
    metafeatures_df = pd.DataFrame(
        [[0.1, 0.2], [0.2, 0.3]],
        index=[1, 2],
        columns=["f1", "f2"],
    )
    pipeline_configs = [
        {"name": "p1", "imputation": "none", "scaling": "none"},
        {"name": "p2", "imputation": "mean", "scaling": "standard"},
    ]
    options = {"imputation": ["none", "mean"], "scaling": ["none", "standard"]}
    new_mf = np.array([0.15, 0.25])

    eta = compute_aco_heuristic(
        performance_matrix=performance_matrix,
        metafeatures_df=metafeatures_df,
        pipeline_configs=pipeline_configs,
        options=options,
        new_metafeatures=new_mf,
        dataset_weighting="equality",
        use_top_pipelines_from_metric=False,
    )

    assert set(eta.keys()) == set(options.keys())
    for step, values in options.items():
        assert eta[step].shape[0] == len(values)


def test_heuristic_similarity_top_k_prefers_nearest_dataset():
    performance_matrix = pd.DataFrame(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        index=["p1", "p2"],
        columns=["d1", "d2", "d3"],
    )
    metafeatures_df = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        index=["d1", "d2", "d3"],
        columns=["f1", "f2"],
    )
    pipeline_configs = [
        {"name": "p1", "imputation": "none"},
        {"name": "p2", "imputation": "mean"},
    ]
    options = {"imputation": ["none", "mean"]}
    new_mf = np.array([1.0, 0.0], dtype=float)

    eta = compute_aco_heuristic(
        performance_matrix=performance_matrix,
        metafeatures_df=metafeatures_df,
        pipeline_configs=pipeline_configs,
        options=options,
        new_metafeatures=new_mf,
        dataset_weighting="similarity",
        top_k=1,
        use_top_pipelines_from_metric=False,
    )

    assert eta["imputation"][0] > eta["imputation"][1]
