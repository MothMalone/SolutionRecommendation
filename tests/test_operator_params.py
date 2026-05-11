import pandas as pd

from automl_aco.preprocessing.preprocessor import Preprocessor
from automl_aco.utils.operator_spec import base_operator_name, parse_operator_spec


def test_parse_operator_spec():
    base, params = parse_operator_spec("knn@k=7")
    assert base == "knn"
    assert params["k"] == "7"
    assert base_operator_name("pca@n=20") == "pca"
    assert parse_operator_spec("none") == ("none", {})


def test_knn_param_token_controls_neighbors():
    X = pd.DataFrame(
        {
            "a": [1.0, None, 3.0, 4.0, None, 6.0],
            "b": [2.0, 1.0, None, 4.0, 5.0, 6.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])

    cfg = {
        "imputation": "knn@k=3",
        "scaling": "none",
        "encoding": "none",
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "none",
    }
    pre = Preprocessor(cfg)
    X_t, y_t = pre.fit_transform(X, y)

    assert pre.num_imputer is not None
    assert int(pre.num_imputer.n_neighbors) == 3
    assert X_t.isna().sum().sum() == 0
    assert len(y_t) == len(X_t)


def test_per_feature_operator_maps_apply_independently():
    X = pd.DataFrame(
        {
            "num_a": [1.0, None, 3.0, 4.0, 5.0],
            "num_b": [10.0, 20.0, None, 40.0, 50.0],
            "cat": ["x", "y", None, "x", "z"],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0])

    cfg = {
        "imputation": {
            "num_a": "median",
            "num_b": "mean",
            "cat": "most_frequent",
        },
        "scaling": {
            "num_a": "standard",
            "num_b": "minmax",
        },
        "encoding": {"cat": "onehot"},
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "none",
    }

    pre = Preprocessor(cfg)
    X_t, y_t = pre.fit_transform(X, y)
    X_x = pre.transform(X)

    assert pre.num_imputer_map is not None
    assert pre.scaler_map is not None
    assert pre.encoder_map is not None
    assert X_t.isna().sum().sum() == 0
    assert X_x.isna().sum().sum() == 0
    assert len(y_t) == len(X_t)
    assert list(X_t.columns) == list(X_x.columns)
    assert "num_a" in X_t.columns and "num_b" in X_t.columns
    assert any(col.startswith("cat_") for col in X_t.columns)
