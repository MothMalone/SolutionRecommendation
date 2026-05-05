import numpy as np
import pandas as pd

from automl_aco.search.heuristics import (
    aggregate_operator_heuristics,
    build_transfer_candidates,
    compute_aco_heuristic,
    compute_relative_pipeline_quality,
    compute_similarity_weights,
    normalize_eta_with_floor,
    select_top_k_neighbors,
    select_top_l_pipelines_per_neighbor,
)


def test_select_top_k_neighbors_picks_highest_similarity():
    sims = [("d1", 0.2), ("d2", 0.9), ("d3", 0.6), ("d4", 0.1)]
    top = select_top_k_neighbors(sims, top_k=2)
    assert [ds for ds, _ in top] == ["d2", "d3"]


def test_select_top_k_neighbors_excludes_query_dataset():
    sims = [("d1", 0.95), ("d2", 0.9), ("d3", 0.8)]
    top = select_top_k_neighbors(sims, top_k=2, query_dataset_id="d1")
    assert [ds for ds, _ in top] == ["d2", "d3"]


def test_select_top_k_neighbors_excludes_query_dataset_with_id_formats():
    sims = [("248.0", 0.95), ("openml_184", 0.9), ("D_378", 0.8)]
    top = select_top_k_neighbors(sims, top_k=3, query_dataset_id=248)
    assert [ds for ds, _ in top] == ["openml_184", "D_378"]


def test_select_top_l_pipelines_per_neighbor_respects_l():
    performance_matrix = pd.DataFrame(
        [[0.9, 0.2], [0.7, 0.3], [0.1, 0.95]],
        index=["p1", "p2", "p3"],
        columns=["d1", "d2"],
    )
    selected = select_top_l_pipelines_per_neighbor(
        performance_matrix=performance_matrix,
        top_k_neighbors=[("d1", 0.9), ("d2", 0.8)],
        top_l=2,
        score_direction="higher_is_better",
    )
    assert [row["pipeline"] for row in selected["d1"]] == ["p1", "p2"]
    assert [row["pipeline"] for row in selected["d2"]] == ["p3", "p2"]


def test_build_transfer_candidates_preserves_multiset_duplicates():
    performance_matrix = pd.DataFrame(
        [[0.9, 0.9], [0.7, 0.1], [0.2, 0.2]],
        index=["p1", "p2", "p3"],
        columns=["d1", "d2"],
    )
    top_l = {
        "d1": [{"pipeline": "p1", "historical_score": 0.9, "pipeline_rank_within_neighbor": 1}],
        "d2": [{"pipeline": "p1", "historical_score": 0.9, "pipeline_rank_within_neighbor": 1}],
    }
    candidates = build_transfer_candidates(
        performance_matrix=performance_matrix,
        top_k_neighbors=[("d1", 0.8), ("d2", 0.7)],
        top_l_pipelines=top_l,
        similarity_weights={"d1": 0.55, "d2": 0.45},
        score_direction="higher_is_better",
    )
    assert len(candidates) == 2
    assert sum(c["pipeline"] == "p1" for c in candidates) == 2
    assert {c["source_dataset"] for c in candidates} == {"d1", "d2"}


def test_similarity_weighting_softmax_and_uniform_fallback():
    weighted = compute_similarity_weights(
        top_k_neighbors=[("d1", 2.0), ("d2", 1.0)],
        dataset_weighting="similarity",
        similarity_temperature=1.0,
    )
    assert weighted["d1"] > weighted["d2"]
    assert np.isclose(sum(weighted.values()), 1.0)

    fallback = compute_similarity_weights(
        top_k_neighbors=[("d1", 1.0), ("d2", 1.0), ("d3", 1.0)],
        dataset_weighting="similarity",
        similarity_temperature=0.2,
    )
    assert np.isclose(fallback["d1"], fallback["d2"])
    assert np.isclose(fallback["d2"], fallback["d3"])


def test_relative_pipeline_quality_handles_direction_and_equal_rows():
    performance_matrix = pd.DataFrame(
        [[1.0, 0.3, 5.0], [2.0, 0.1, 5.0], [3.0, 0.2, 5.0]],
        index=["p1", "p2", "p3"],
        columns=["score_ds", "loss_ds", "equal_ds"],
    )

    score_quality = compute_relative_pipeline_quality(
        performance_matrix=performance_matrix,
        source_dataset="score_ds",
        score_direction="higher_is_better",
    )
    assert score_quality["p3"] > score_quality["p2"] > score_quality["p1"]
    assert np.isclose(score_quality["p3"], 1.0)
    assert np.isclose(score_quality["p1"], 0.0)

    loss_quality = compute_relative_pipeline_quality(
        performance_matrix=performance_matrix,
        source_dataset="loss_ds",
        score_direction="lower_is_better",
    )
    assert loss_quality["p2"] > loss_quality["p3"] > loss_quality["p1"]
    assert np.isclose(loss_quality["p2"], 1.0)
    assert np.isclose(loss_quality["p1"], 0.0)

    equal_quality = compute_relative_pipeline_quality(
        performance_matrix=performance_matrix,
        source_dataset="equal_ds",
        score_direction="higher_is_better",
    )
    assert np.isfinite(equal_quality.to_numpy()).all()
    assert np.allclose(equal_quality.to_numpy(), np.ones(3))


def test_operator_heuristic_aggregation_uses_transferred_candidates_only():
    options = {"imputation": ["none", "mean", "median"]}
    pipeline_configs = [
        {"name": "p1", "imputation": "none"},
        {"name": "p2", "imputation": "mean"},
        {"name": "p3", "imputation": "none"},
    ]
    transfer_candidates = [
        {"pipeline": "p1", "candidate_weight": 0.5, "relative_pipeline_quality": 1.0},
        {"pipeline": "p2", "candidate_weight": 0.5, "relative_pipeline_quality": 0.2},
    ]

    raw_eta = aggregate_operator_heuristics(
        transfer_candidates=transfer_candidates,
        pipeline_configs=pipeline_configs,
        options=options,
    )
    assert np.isclose(raw_eta["imputation"][0], 1.0)
    assert np.isclose(raw_eta["imputation"][1], 0.2)
    # Missing operator gets the minimum observed value from this step.
    assert np.isclose(raw_eta["imputation"][2], 0.2)


def test_positive_floor_normalization_is_finite_and_nonzero():
    raw_eta = {
        "imputation": np.array([0.2, 0.4, 0.6], dtype=float),
        "scaling": np.array([0.3, 0.3], dtype=float),
    }
    normalized = normalize_eta_with_floor(raw_eta=raw_eta, eta_floor=0.05)
    assert np.isfinite(normalized["imputation"]).all()
    assert np.isfinite(normalized["scaling"]).all()
    assert (normalized["imputation"] >= 0.05).all()
    assert (normalized["scaling"] >= 0.05).all()
    assert np.allclose(normalized["scaling"], np.ones_like(normalized["scaling"]))


def test_compute_aco_heuristic_self_exclusion_changes_neighbor_choice():
    performance_matrix = pd.DataFrame(
        [[0.95, 0.55], [0.20, 0.90], [0.80, 0.10]],
        index=["p1", "p2", "p3"],
        columns=["d1", "d2"],
    )
    metafeatures_df = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["d1", "d2"], columns=["f1", "f2"])
    pipeline_configs = [
        {"name": "p1", "imputation": "none"},
        {"name": "p2", "imputation": "mean"},
        {"name": "p3", "imputation": "none"},
    ]
    options = {"imputation": ["none", "mean"]}

    eta = compute_aco_heuristic(
        performance_matrix=performance_matrix,
        metafeatures_df=metafeatures_df,
        pipeline_configs=pipeline_configs,
        options=options,
        new_metafeatures=np.array([1.0, 0.0], dtype=float),
        dataset_weighting="similarity",
        top_k=1,
        top_l=2,
        dataset_similarity_scores={"d1": 0.99, "d2": 0.98},
        query_dataset_id="d1",
        score_direction="higher_is_better",
        use_top_pipelines_from_metric=False,
    )
    assert eta["imputation"][1] > eta["imputation"][0]


def test_compute_aco_heuristic_uses_top_k_top_l_only_not_full_library():
    performance_matrix = pd.DataFrame(
        [[0.90, 0.10], [0.60, 0.95], [0.20, 0.80]],
        index=["p_none", "p_mean", "p_none_alt"],
        columns=["d1", "d2"],
    )
    metafeatures_df = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["d1", "d2"], columns=["f1", "f2"])
    pipeline_configs = [
        {"name": "p_none", "imputation": "none"},
        {"name": "p_mean", "imputation": "mean"},
        {"name": "p_none_alt", "imputation": "none"},
    ]
    options = {"imputation": ["none", "mean"]}

    eta = compute_aco_heuristic(
        performance_matrix=performance_matrix,
        metafeatures_df=metafeatures_df,
        pipeline_configs=pipeline_configs,
        options=options,
        new_metafeatures=np.array([1.0, 0.0], dtype=float),
        dataset_weighting="similarity",
        top_k=1,
        top_l=2,
        dataset_similarity_scores={"d1": 0.99, "d2": 0.10},
        score_direction="higher_is_better",
        use_top_pipelines_from_metric=False,
    )
    assert eta["imputation"][0] > eta["imputation"][1]
    assert np.isfinite(eta["imputation"]).all()
    assert (eta["imputation"] > 0.0).all()
