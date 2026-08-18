import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from automl_aco.preprocessing.autodp import (
    AUTODP_OPTIONS,
    AutoDP36Preprocessor,
    AutoDPPreprocessor,
)
from automl_aco.preprocessing.preprocessor import Preprocessor
from automl_aco.search.evaluation import _make_preprocessor


ROOT = Path(__file__).resolve().parents[1]


def _build_arg_parser():
    spec = importlib.util.spec_from_file_location(
        "run_recommend_for_autodp36_test", ROOT / "scripts/run_recommend.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_arg_parser()


def _empty_autodp36_config():
    return {stage: "none" for stage in AUTODP_OPTIONS}


def test_cli_default_is_unchanged_and_autodp36_is_explicit():
    parser = _build_arg_parser()
    assert parser.parse_args([]).operator_space == "ours"
    assert parser.parse_args(["--operator-space", "autodp36"]).operator_space == "autodp36"


def test_preprocessor_factory_preserves_original_and_routes_autodp36():
    original = _make_preprocessor(
        {
            "imputation": "mean",
            "scaling": "standard",
            "encoding": "onehot",
        }
    )
    autodp36 = _make_preprocessor(_empty_autodp36_config())
    assert isinstance(original, Preprocessor)
    assert isinstance(autodp36, AutoDP36Preprocessor)


def test_autodp36_runtime_adapter_matches_numeric_model_contract():
    X_train = pd.DataFrame(
        {
            "number": [1.0, np.nan, 3.0, 4.0],
            "category": ["a", "b", None, "a"],
        }
    )
    y_train = pd.Series([0, 1, 0, 1])
    X_test = pd.DataFrame(
        {"number": [np.nan, 8.0], "category": ["unseen", None]}
    )

    pre = AutoDP36Preprocessor(_empty_autodp36_config())
    X_train_out, y_train_out = pre.fit_transform(X_train, y_train)
    X_test_out = pre.transform(X_test)

    assert len(X_train_out) == len(y_train_out)
    assert len(X_test_out) == len(X_test)
    assert X_train_out.columns.tolist() == X_test_out.columns.tolist()
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train_out.dtypes)
    assert not X_train_out.sparse.to_dense().isna().to_numpy().any()
    assert not X_test_out.sparse.to_dense().isna().to_numpy().any()


def test_autodp36_assets_are_aligned():
    matrix = pd.read_csv(
        ROOT / "autodp_matrix/merged/training_performance_matrix_autodp36_ready.csv",
        index_col=0,
    )
    configs = json.loads(
        (ROOT / "aco/pipeline_configs_autodp36.json").read_text(encoding="utf-8")
    )
    assert matrix.shape == (36, 818)
    assert matrix.index.tolist() == [config["name"] for config in configs]
    assert matrix.notna().sum(axis=0).min() >= 1


def test_autodp_task_type_can_be_inferred_without_changing_matrix_builder_api():
    classification = AutoDPPreprocessor(_empty_autodp36_config())
    classification.fit_transform(pd.DataFrame({"x": range(6)}), pd.Series([0, 1, 0, 1, 0, 1]))
    assert classification.task_type == "classification"

    regression = AutoDPPreprocessor(_empty_autodp36_config())
    regression.fit_transform(
        pd.DataFrame({"x": range(60)}),
        pd.Series(np.linspace(0.0, 1.0, 60)),
    )
    assert regression.task_type == "regression"
