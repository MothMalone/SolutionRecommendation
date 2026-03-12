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
