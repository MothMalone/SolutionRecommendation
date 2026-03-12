import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from automl_aco.search.evaluation import evaluate_candidates_simple


def _base_cfg():
    return {
        "name": "cfg",
        "imputation": "none",
        "scaling": "standard",
        "encoding": "none",
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "none",
    }


def _make_dataset():
    X, y = make_classification(
        n_samples=180,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=7,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


def test_proxy_classification_models_run():
    dataset = _make_dataset()
    for model_name in ["logreg", "random_forest", "linear_svm"]:
        best_cfg, best_score, results, _unsorted = evaluate_candidates_simple(
            dataset=dataset,
            target_column="target",
            candidate_configs=[_base_cfg()],
            proxy_settings={
                "classification_model": model_name,
                "split_seeds": [42],
                "logreg_max_iter": 2000,
            },
            verbose=False,
        )
        assert best_cfg is not None
        assert results
        assert np.isfinite(best_score)


def test_missing_data_with_no_imputation_is_fast_rejected():
    dataset = _make_dataset()
    dataset.loc[:10, "f0"] = np.nan
    cfg = _base_cfg()
    cfg["imputation"] = "none"

    best_cfg, best_score, results, _unsorted = evaluate_candidates_simple(
        dataset=dataset,
        target_column="target",
        candidate_configs=[cfg],
        proxy_settings={"classification_model": "logreg", "split_seeds": [42]},
        verbose=False,
    )

    assert best_cfg is None
    assert not results
    assert np.isnan(best_score)
