import numpy as np
import pandas as pd

from automl_aco.metalearning.offline_eval import (
    paired_accuracy_summary,
    retrieval_metrics,
    warm_start_regret,
)


def test_retrieval_metrics_reward_correct_ranking():
    target = {"a": 1.0, "b": 0.8, "c": 0.1}
    perfect = retrieval_metrics(target, target, ks=(2,))
    reversed_scores = {"a": 0.1, "b": 0.8, "c": 1.0}
    reversed_result = retrieval_metrics(reversed_scores, target, ks=(2,))
    assert perfect["ndcg_at_2"] > reversed_result["ndcg_at_2"]
    assert perfect["overlap_at_2"] == 1.0


def test_warm_start_regret_uses_neighbor_top_pipelines():
    reference = pd.DataFrame({"neighbor": [0.9, 0.1]}, index=["p1", "p2"])
    query = pd.Series({"p1": 0.7, "p2": 0.95})
    regret = warm_start_regret(query, reference, ["neighbor"], top_l=1)
    assert regret == 0.25


def test_paired_accuracy_summary_is_dataset_macro_average():
    result = paired_accuracy_summary([0.8, 0.9], [0.7, 0.8], bootstrap_samples=100)
    assert np.isclose(result["mean_accuracy_delta"], 0.1)
    assert result["wins"] == 2
