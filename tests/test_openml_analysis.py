from __future__ import annotations

import pandas as pd

from automl_aco.data import openml_analysis


def test_select_imputation_candidates_filters_and_sorts():
    df = pd.DataFrame(
        {
            "did": [1, 2, 3, 4],
            "name": ["a", "b", "c", "d"],
            "NumberOfInstances": [100, 20_000, 5000, 7000],
            "NumberOfMissingValues": [10, 1000, 0, 30],
        }
    )
    out = openml_analysis.select_imputation_candidates(df, max_instances=10_000)
    assert list(out["did"]) == [4, 1]


def test_get_cc18_dataset_ids_from_suite(monkeypatch):
    class _Suite:
        data = [10, 20, 30]

    class _Study:
        @staticmethod
        def get_suite(suite_id):
            assert suite_id == 99
            return _Suite()

    class _OpenML:
        study = _Study()

    monkeypatch.setattr(openml_analysis, "_import_openml_api", lambda: _OpenML())
    assert openml_analysis.get_cc18_dataset_ids() == [10, 20, 30]


def test_compute_scale_ratio_and_outlier_ratio():
    X = pd.DataFrame(
        {
            "large": [0, 1000, 2000, 3000, 4000],
            "small": [0.0, 1.0, 2.0, 3.0, 4.0],
            "outlier_col": [1, 1, 1, 1, 100],
        }
    )
    scale_ratio = openml_analysis.compute_scale_ratio(X)
    outlier_ratio = openml_analysis.compute_outlier_ratio(X)
    assert scale_ratio >= 1000.0
    assert 0.0 < outlier_ratio <= 1.0


def test_evaluate_dataset_issues_records_ok_and_failed_rows():
    dataset_map = {
        1: {
            "id": 1,
            "name": "D_1",
            "X": pd.DataFrame({"a": [0, 100, 200], "b": [0.0, 1.0, 2.0]}),
        }
    }

    def _loader(dataset_id, verbose=False):
        if dataset_id not in dataset_map:
            return None
        out = dict(dataset_map[dataset_id])
        out["y"] = pd.Series([0, 1, 1])
        return out

    result = openml_analysis.evaluate_dataset_issues(dataset_ids=[1, 2], loader=_loader, verbose=False)
    assert list(result["ID"]) == [1, 2]
    assert list(result["Load_Status"]) == ["ok", "failed"]
    assert result.loc[result["ID"] == 1, "Scale_Ratio"].iloc[0] >= 1.0

