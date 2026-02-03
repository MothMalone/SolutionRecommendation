"""Feature selection helpers."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif


def build_selector(method: str, X: pd.DataFrame, y: pd.Series) -> Optional[object]:
    if method == "variance_threshold":
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X)
        return selector

    k = min(20, X.shape[1])

    if method == "k_best":
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y.values.ravel())
        return selector

    if method == "mutual_info":
        selector = SelectKBest(
            lambda Xv, yv: mutual_info_classif(Xv, yv, discrete_features="auto"),
            k=k,
        )
        selector.fit(X, y.values.ravel())
        return selector

    return None
