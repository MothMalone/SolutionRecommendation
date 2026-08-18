import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_acorec_tpot_test_module",
    ROOT / "scripts" / "evaluate_acorec_tpot.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTPOTClassifier:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fitted_pipeline_ = "fake-majority-classifier"
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


def test_wrapper_uses_train_60_and_scores_outer_test_20_only():
    dataset = {
        "id": 123,
        "X": pd.DataFrame({"x1": np.arange(100), "x2": np.arange(100) % 7}),
        "y": pd.Series(([0] * 55) + ([1] * 45)),
        "task_type": "classification",
    }
    pipeline = {
        "name": "test_pipeline",
        "imputation": "none",
        "scaling": "standard",
        "encoding": "none",
        "outlier_removal": "none",
        "feature_selection": "none",
        "dimensionality_reduction": "none",
    }

    result, _model = MODULE.evaluate_recommendation(
        dataset,
        pipeline,
        estimator_factory=FakeTPOTClassifier,
        search_space_factory=fake_search_space,
        verbose=0,
    )

    instance = FakeTPOTClassifier.last_instance
    assert result["train_rows_raw"] == 60
    assert result["train_rows_processed"] == 60
    assert result["validation_rows_unused"] == 20
    assert result["test_rows"] == 20
    assert result["test_fraction"] == 0.2
    assert result["primary_metric"] == "accuracy"
    assert instance.fit_shape[0] == 60
    assert instance.predict_shape[0] == 20
    assert instance.kwargs["preprocessing"] is False
    assert instance.kwargs["validation_strategy"] == "none"
    assert instance.kwargs["search_space"]["group"] == "classifiers"


def test_safe_cv_uses_smallest_processed_class():
    y = pd.Series(([0] * 8) + ([1] * 3) + ([2] * 2))
    assert MODULE._safe_cv_folds(y, "classification", 5) == 2
