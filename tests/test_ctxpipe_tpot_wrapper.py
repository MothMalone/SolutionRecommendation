import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_ctxpipe_tpot_test_module",
    ROOT / "scripts" / "evaluate_ctxpipe_tpot.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTPOTClassifier:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fitted_pipeline_ = "fake-classifier"
        FakeTPOTClassifier.last_instance = self

    def fit(self, X, y):
        self.fit_shape = X.shape
        values, counts = np.unique(y, return_counts=True)
        self.majority = values[int(np.argmax(counts))]
        return self

    def predict(self, X):
        self.predict_shape = X.shape
        return np.repeat(self.majority, len(X))


def fake_search_space(group, **kwargs):
    return {"group": group, **kwargs}


def test_ctxpipe_replay_and_tpot_hold_out_outer_test():
    frame = pd.DataFrame(
        {
            "number": np.arange(100, dtype=float),
            "category": ["a", "b", "c", None] * 25,
        }
    )
    frame.loc[3, "number"] = np.nan
    dataset = {
        "X": frame,
        "y": pd.Series(([0] * 55) + ([1] * 45)),
        "task_type": "classification",
    }
    sequence = [
        "ImputerMean",
        "ImputerCatMode",
        "OneHotEncoder",
        "StandardScaler",
        "blank",
        "VarianceThreshold",
    ]
    result, _ = MODULE.evaluate_ctxpipe_sequence(
        dataset,
        sequence,
        estimator_factory=FakeTPOTClassifier,
        search_space_factory=fake_search_space,
        verbose=0,
    )
    instance = FakeTPOTClassifier.last_instance
    assert result["tpot_train_rows"] == 60
    assert result["validation_rows_used_by_ctxpipe_search"] == 20
    assert result["validation_reused_by_tpot"] is False
    assert result["outer_test_rows"] == 20
    assert result["ctxpipe_saw_outer_test"] is False
    assert result["tpot_preprocessing"] is False
    assert instance.fit_shape[0] == 60
    assert instance.predict_shape[0] == 20
    assert instance.kwargs["search_space"]["group"] == "classifiers"


def test_all_native_ctxpipe_operator_names_are_supported():
    supported_sequence_names = {
        "blank",
        "ImputerMean",
        "ImputerMedian",
        "ImputerNumMode",
        "ImputerCatMode",
        "NumericData",
        "LabelEncoder",
        "OneHotEncoder",
        "MinMaxScaler",
        "MaxAbsScaler",
        "RobustScaler",
        "StandardScaler",
        "QuantileTransformer",
        "PowerTransformer",
        "Normalizer",
        "KBinsDiscretizerOrdinal",
        "PolynomialFeatures",
        "InteractionFeatures",
        "PCA_AUTO",
        "IncrementalPCA",
        "KernelPCA",
        "TruncatedSVD",
        "RandomTreesEmbedding",
        "VarianceThreshold",
    }
    assert supported_sequence_names == MODULE.SUPPORTED_NATIVE_OPERATORS


def test_each_native_operator_replay_executes_on_compatible_small_data():
    numeric = pd.DataFrame(
        {
            "a": np.linspace(1.0, 20.0, 20),
            "b": np.linspace(2.0, 40.0, 20),
            "c": np.tile([0.0, 1.0], 10),
            "d": np.arange(20, dtype=float) ** 2,
            "e": np.sin(np.arange(20, dtype=float)),
        }
    )
    numeric_test = numeric.iloc[:5].copy()
    categorical = numeric[["a"]].copy()
    categorical["kind"] = ["x", "y", None, "z"] * 5
    categorical_test = categorical.iloc[:5].copy()
    y = pd.Series(np.tile([0, 1], 10))
    numeric_missing = numeric.copy()
    numeric_missing.loc[1, :] = np.nan

    cases = {
        "ImputerMean": (["ImputerMean"], numeric_missing, numeric_test),
        "ImputerMedian": (["ImputerMedian"], numeric_missing, numeric_test),
        "ImputerNumMode": (["ImputerNumMode"], numeric_missing, numeric_test),
        "ImputerCatMode": (["ImputerCatMode", "LabelEncoder"], categorical, categorical_test),
        "NumericData": (["NumericData"], categorical, categorical_test),
        "LabelEncoder": (["LabelEncoder"], categorical, categorical_test),
        "OneHotEncoder": (["OneHotEncoder"], categorical, categorical_test),
    }
    for name in (
        "MinMaxScaler",
        "MaxAbsScaler",
        "RobustScaler",
        "StandardScaler",
        "QuantileTransformer",
        "PowerTransformer",
        "Normalizer",
        "KBinsDiscretizerOrdinal",
        "PolynomialFeatures",
        "InteractionFeatures",
        "PCA_AUTO",
        "IncrementalPCA",
        "KernelPCA",
        "TruncatedSVD",
        "RandomTreesEmbedding",
        "VarianceThreshold",
        "blank",
    ):
        cases[name] = ([name], numeric, numeric_test)

    for name, (sequence, train, test) in cases.items():
        transformed_train, transformed_test, trace = MODULE.replay_ctxpipe_sequence(
            sequence, train, test, y
        )
        assert len(transformed_train) == len(train), name
        assert len(transformed_test) == len(test), name
        assert transformed_train.shape[1] > 0, name
        assert trace[0]["operator"] == name
