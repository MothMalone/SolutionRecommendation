"""Regression tests for the optional H2O evaluator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from sklearn.metrics import accuracy_score

def _load_evaluator_module():
    source = Path(__file__).resolve().parents[1] / "scripts" / "h2o_evaluator.py"
    spec = importlib.util.spec_from_file_location("h2o_evaluator_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h2o_init_uses_only_current_client_arguments():
    """Protect Kaggle runs against the removed ``silent`` H2O argument."""
    calls = []

    class FakeH2O:
        def init(self, **kwargs):
            calls.append(kwargs)

        def remove_all(self):
            calls.append("remove_all")

    evaluator = _load_evaluator_module()
    evaluator._init_h2o(FakeH2O(), nthreads=1, max_mem_size="6G")

    assert calls == [{"nthreads": 1, "max_mem_size": "6G"}, "remove_all"]


def test_h2o_init_accepts_an_explicit_port_without_changing_default_calls():
    calls = []

    class FakeH2O:
        def init(self, **kwargs):
            calls.append(kwargs)

        def remove_all(self):
            pass

    evaluator = _load_evaluator_module()
    evaluator._init_h2o(FakeH2O(), nthreads=1, max_mem_size="2G", port=54329)

    assert calls == [{"nthreads": 1, "max_mem_size": "2G", "port": 54329}]


def test_h2o_classification_metric_normalizes_numeric_factor_predictions():
    """H2O may emit 0/1 while the uploaded factor target is '0'/'1'."""
    evaluator = _load_evaluator_module()
    actual = evaluator._classification_labels(["0", "1", "0", "1"])
    predicted = evaluator._classification_labels([0, 1, 0, 1])

    assert actual.dtype.kind in {"U", "O"}
    assert predicted.dtype.kind in {"U", "O"}
    assert accuracy_score(actual, predicted) == 1.0


def test_h2o_forces_categorical_predictors_for_every_split():
    evaluator = _load_evaluator_module()
    calls = []

    class Column:
        def __init__(self, frame_name, column_name):
            self.frame_name = frame_name
            self.column_name = column_name

        def asfactor(self):
            calls.append((self.frame_name, self.column_name))
            return self

    class Frame:
        def __init__(self, name):
            self.name = name
            self.names = ["category", "number"]

        def __getitem__(self, column):
            return Column(self.name, column)

        def __setitem__(self, column, value):
            assert column == "category"

    evaluator._force_h2o_categorical_columns(
        (Frame("train"), Frame("validation"), Frame("test")), ["category"]
    )

    assert calls == [
        ("train", "category"),
        ("validation", "category"),
        ("test", "category"),
    ]
