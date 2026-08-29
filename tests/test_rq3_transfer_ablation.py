from scripts.rq3_transfer_ablation_common import (
    _build_search_command,
    _variant_name,
    _variant_values,
    build_parser,
    count_active_operator_assignments,
    selected_pipeline_operator_stats,
)


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


def test_num_ants_ablation_defaults_to_requested_sweep():
    args = build_parser("num_ants", "test").parse_args([])
    assert _variant_values(args) == [5, 10, 15, 20]
    assert _variant_name("num_ants", 15) == "A15"


def test_num_ants_ablation_changes_n_ants_and_keeps_k_h_fixed(tmp_path):
    args = build_parser("num_ants", "test").parse_args(
        ["--root", str(tmp_path), "--fixed-k", "5", "--fixed-h", "3", "--n-ants", "10"]
    )
    command = _build_search_command(
        args,
        {
            "performance_matrix": tmp_path / "performance.csv",
            "metafeatures": tmp_path / "features.csv",
            "pipeline_configs": tmp_path / "pipelines.json",
            "data_dir": tmp_path / "data",
        },
        "1066",
        tmp_path / "run",
        15,
    )

    def value_after(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value_after("--n-ants") == "15"
    assert value_after("--heuristic-top-k") == "5"
    assert value_after("--heuristic-top-l") == "3"
