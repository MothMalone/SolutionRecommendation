from scripts.rq3_transfer_ablation_common import count_active_operator_assignments, selected_pipeline_operator_stats


def test_operator_count_ignores_identity_and_counts_per_feature_assignments():
    config = {
        "imputation": {"a": "mean", "b": "none"},
        "scaling": "standard",
        "encoding": "onehot",
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "pca",
    }
    assert count_active_operator_assignments(config) == 4


def test_candidate_operator_stats_handles_json_tuple_shape():
    recommendation = {
        "pipeline_config": {"imputation": "mean", "scaling": "none", "encoding": "onehot"},
        "aco_results": [
            [{"imputation": "mean", "scaling": "standard", "encoding": "onehot"}, 0.8],
            [{"imputation": "none", "scaling": "none", "encoding": "onehot"}, 0.7],
        ],
    }
    stats = selected_pipeline_operator_stats(recommendation)
    assert stats["selected_pipeline_active_operator_count"] == 2
    assert stats["selected_candidate_count"] == 2
    assert stats["selected_candidates_active_operator_count_total"] == 4
