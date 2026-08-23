import json
from pathlib import Path

import pandas as pd

from automl_aco.data.splits import split_fingerprints, split_train_val_test
from automl_aco.eval_ids import EVAL_ID_SET, holdout_ids
from automl_aco.preprocessing.autodp import AUTODP_60_IDS
from automl_aco.preprocessing.preprocessor import Preprocessor


ROOT = Path(__file__).resolve().parents[1]


def test_meta_dev_manifest_is_frozen_balanced_and_disjoint():
    manifest = json.loads((ROOT / "data/openml/meta_dev18.json").read_text(encoding="utf-8"))
    dataset_ids = [str(value) for value in manifest["dataset_ids"]]
    forbidden = set(EVAL_ID_SET) | {str(value) for value in AUTODP_60_IDS}
    assert manifest["seed"] == 42
    assert len(dataset_ids) == len(set(dataset_ids)) == 18
    assert not (set(dataset_ids) & forbidden)


def test_split_fingerprints_are_deterministic_and_seed_sensitive():
    X = pd.DataFrame({"x": range(100), "category": ["a", "b"] * 50})
    y = pd.Series([0, 1] * 50)
    first = split_fingerprints(split_train_val_test(X, y, seed=42))
    second = split_fingerprints(split_train_val_test(X, y, seed=42))
    other = split_fingerprints(split_train_val_test(X, y, seed=43))
    assert first == second
    assert first != other
    assert set(first) == {"train", "validation", "test"}


def test_preprocessor_supports_boolean_categorical_columns():
    X = pd.DataFrame({"flag": [True, False, None, True], "value": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 0, 1])
    preprocessor = Preprocessor(
        {
            "imputation": "most_frequent",
            "scaling": "none",
            "encoding": "onehot",
            "feature_selection": "none",
            "outlier_removal": "none",
            "dimensionality_reduction": "none",
        }
    )
    transformed, transformed_y = preprocessor.fit_transform(X, y)
    replayed = preprocessor.transform(X)
    assert transformed.shape == replayed.shape
    assert len(transformed_y) == len(X)


def test_query_dataset_is_removed_from_both_loo_reference_views():
    performance = pd.DataFrame({"D_965": [0.5], "D_994": [0.7]}, index=["pipeline"])
    metafeatures = pd.DataFrame({"f": [1.0, 2.0]}, index=[965, 994])
    clean_performance, clean_metafeatures, report = holdout_ids(
        performance, metafeatures, [965]
    )
    assert list(clean_performance.columns) == ["D_994"]
    assert list(clean_metafeatures.index) == [994]
    assert report["held_out_ids"] == ["965"]
