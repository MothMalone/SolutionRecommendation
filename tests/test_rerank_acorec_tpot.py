import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rerank_acorec_tpot.py"
SPEC = importlib.util.spec_from_file_location("rerank_acorec_tpot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_top_unique_candidates_sorts_proxy_scores_and_deduplicates_name():
    recommendation = {
        "pipeline_config": {"imputation": "mean", "name": "selected"},
        "recommended_performance": 0.8,
        "aco_results": [
            [{"imputation": "median", "name": "low"}, 0.7],
            [{"imputation": "mean", "name": "duplicate"}, 0.8],
            [{"imputation": "knn", "name": "high"}, 0.9],
        ],
    }

    candidates = MODULE._top_unique_candidates(recommendation, top_k=5)

    assert [row["pipeline_config"]["imputation"] for row in candidates] == [
        "knn",
        "mean",
        "median",
    ]
    assert [row["proxy_score"] for row in candidates] == [0.9, 0.8, 0.7]


def test_top_unique_candidates_respects_top_k():
    recommendation = {
        "aco_results": [
            [{"imputation": "mean"}, 0.8],
            [{"imputation": "median"}, 0.7],
        ]
    }

    candidates = MODULE._top_unique_candidates(recommendation, top_k=1)

    assert len(candidates) == 1
    assert candidates[0]["pipeline_config"]["imputation"] == "mean"
