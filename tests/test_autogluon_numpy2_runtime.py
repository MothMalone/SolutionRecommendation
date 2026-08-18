import sys
import types

import numpy as np

from automl_aco.search import evaluation


def test_autogluon_runtime_import_does_not_reject_numpy2(monkeypatch):
    class FakePredictor:
        pass

    class FakeFeatureGenerator:
        pass

    autogluon = types.ModuleType("autogluon")
    autogluon.__path__ = []
    tabular = types.ModuleType("autogluon.tabular")
    tabular.TabularPredictor = FakePredictor
    features = types.ModuleType("autogluon.features")
    features.__path__ = []
    generators = types.ModuleType("autogluon.features.generators")
    generators.IdentityFeatureGenerator = FakeFeatureGenerator

    monkeypatch.setitem(sys.modules, "autogluon", autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular)
    monkeypatch.setitem(sys.modules, "autogluon.features", features)
    monkeypatch.setitem(sys.modules, "autogluon.features.generators", generators)
    monkeypatch.setattr(np, "__version__", "2.2.5")

    predictor, generator = evaluation._load_autogluon_components()

    assert predictor is FakePredictor
    assert generator is FakeFeatureGenerator
