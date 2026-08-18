import numpy as np
import pandas as pd
import pytest

from automl_aco.preprocessing.autodp import (
    AUTODP_OPTIONS,
    AutoDPResourceLimitError,
    AutoDPPreprocessor,
    autodp_space_size,
    build_autodp_reference_pipelines,
    exclude_holdout_columns,
)


def test_autodp_space_and_reference_pipeline_coverage():
    pipelines = build_autodp_reference_pipelines()
    assert autodp_space_size() == 8400
    assert len(pipelines) == 36
    assert len({config["name"] for config in pipelines}) == 36
    for stage, values in AUTODP_OPTIONS.items():
        assert set(values) == {config[stage] for config in pipelines}


def test_all_reference_pipelines_preserve_test_cardinality_and_schema():
    X_train = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 100.0, 2.0, 2.01, 3.0, 4.0, 5.0, 6.0],
            "category": ["x", "y", None, "z", "y", "y", "x", "z", "x", "y"],
            "correlated": [1, 2, 3, 4, 2, 2, 3, 4, 5, 6],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 1, 1, 0, 1, 0, 1])
    X_test = pd.DataFrame(
        {"numeric": [np.nan, 7.0], "category": ["unseen", None], "correlated": [7, 8]}
    )

    for config in build_autodp_reference_pipelines():
        preprocessor = AutoDPPreprocessor(config, task_type="classification")
        transformed_train, transformed_y = preprocessor.fit_transform(X_train, y_train)
        transformed_test = preprocessor.transform(X_test)
        assert len(transformed_train) == len(transformed_y), config["name"]
        assert len(transformed_test) == len(X_test), config["name"]
        assert transformed_train.columns.tolist() == transformed_test.columns.tolist(), config["name"]


def test_holdout_exclusion_removes_every_present_autodp_column():
    matrix = pd.DataFrame([[0.1, 0.2, 0.3]], columns=["D_36", "D_100", "D_45012"])
    reference, removed = exclude_holdout_columns(matrix)
    assert removed == ["D_36", "D_45012"]
    assert reference.columns.tolist() == ["D_100"]


def test_collinear_resource_guard_runs_before_quadratic_allocation():
    config = {stage: "none" for stage in AUTODP_OPTIONS}
    config["feature_selection"] = "collinear"
    X = pd.DataFrame(np.ones((3, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series([0, 1, 0])
    preprocessor = AutoDPPreprocessor(
        config,
        task_type="classification",
        max_collinear_features=5,
    )

    with pytest.raises(AutoDPResourceLimitError, match="6 numeric features"):
        preprocessor.fit_transform(X, y)
