"""SoluRec-compatible deterministic split utility.

This keeps the same index permutation logic as the SoluRec notebook/module:
split into test, then val, then train using a single seeded permutation.
"""

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
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    np.random.seed(seed)
    n_samples = len(y)
    n_val = int(n_samples * val_ratio)
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_test - n_val

    indices = np.random.permutation(n_samples)
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
