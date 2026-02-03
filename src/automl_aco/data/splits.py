"""Leak-free train/val/test split utilities (matches notebook logic)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split data using the exact permutation logic from the notebook."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    np.random.seed(seed)
    N = len(y)
    n_val = int(N * val_ratio)
    n_test = int(N * test_ratio)
    n_train = N - n_test - n_val

    indices = np.random.permutation(N)
    test_indices = indices[:n_test]
    val_indices = indices[n_test : n_test + n_val]
    train_indices = indices[n_test + n_val : n_test + n_val + n_train]

    X_train = X.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    X_val = X.iloc[val_indices].reset_index(drop=True)
    y_val = y.iloc[val_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    return X_train, y_train, X_val, y_val, X_test, y_test
