"""Dimensionality reduction helpers."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD


def build_reducer(method: str, X: pd.DataFrame) -> Optional[object]:
    if X is None:
        return None
    if method == "none":
        return None
    if X.shape[1] <= 1 or len(X) < 2:
        return None

    n_components = min(10, X.shape[1], len(X) - 1)
    if method == "pca":
        return PCA(n_components=n_components)
    return TruncatedSVD(n_components=n_components)
