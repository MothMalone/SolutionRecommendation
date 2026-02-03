"""Imputation helpers."""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer


def build_imputers(
    method: str,
    X_num: Optional[pd.DataFrame],
    X_cat: Optional[pd.DataFrame],
) -> Tuple[Optional[SimpleImputer], Optional[SimpleImputer]]:
    num_imputer = None
    cat_imputer = None

    if X_num is not None and method != "none":
        if method == "knn":
            num_imputer = KNNImputer(n_neighbors=min(5, len(X_num) - 1))
        elif method in ["mean", "median", "most_frequent", "constant"]:
            num_imputer = SimpleImputer(strategy=method)
        else:
            num_imputer = SimpleImputer(strategy="mean")

    if X_cat is not None and method != "none":
        cat_imputer = SimpleImputer(strategy="most_frequent")

    return num_imputer, cat_imputer
