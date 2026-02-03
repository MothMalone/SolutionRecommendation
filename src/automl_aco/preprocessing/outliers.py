"""Outlier removal/cleaning helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor


def compute_outlier_mask(X_num: pd.DataFrame, method: str) -> pd.Series:
    if method == "iqr":
        mask = pd.Series(True, index=X_num.index)
        for col in X_num.columns:
            q1, q3 = X_num[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                mask &= (X_num[col] >= q1 - 1.5 * iqr) & (X_num[col] <= q3 + 1.5 * iqr)
        return mask
    if method == "zscore":
        z = np.abs(zscore(X_num))
        return pd.Series((z < 3).all(axis=1), index=X_num.index)
    if method == "lof":
        lof = LocalOutlierFactor(n_neighbors=20)
        return pd.Series(lof.fit_predict(X_num) == 1, index=X_num.index)
    if method == "isolation_forest":
        iso = IsolationForest(contamination=0.05, random_state=42)
        return pd.Series(iso.fit_predict(X_num) == 1, index=X_num.index)
    return pd.Series(True, index=X_num.index)


def fit_outlier_cleaner(X: pd.DataFrame, method: str) -> Dict[str, Any]:
    X_array = X.values.astype(float)
    params: Dict[str, Any] = {}

    if method.startswith("zscore"):
        nstd = float(method.split("-")[1]) if "-" in method else 3.0
        mean = X_array.mean(axis=0)
        std = X_array.std(axis=0)
        cut = std * nstd
        params["lower"] = (mean - cut).reshape(1, -1)
        params["upper"] = (mean + cut).reshape(1, -1)
        params["mode"] = "cell"
    elif method.startswith("iqr"):
        k = float(method.split("-")[1]) if "-" in method else 1.5
        q25 = np.percentile(X_array, 25, axis=0)
        q75 = np.percentile(X_array, 75, axis=0)
        iqr = q75 - q25
        cut = iqr * k
        params["lower"] = (q25 - cut).reshape(1, -1)
        params["upper"] = (q75 + cut).reshape(1, -1)
        params["mode"] = "cell"
    elif method.startswith("mad"):
        nmad = float(method.split("-")[1]) if "-" in method else 2.5
        median = np.median(X_array, axis=0, keepdims=True)
        mad = np.median(np.abs(X_array - median), axis=0, keepdims=True)
        params["lower"] = median - nmad * mad
        params["upper"] = median + nmad * mad
        params["mode"] = "cell"
    elif method == "lof":
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        lof.fit(X_array)
        params["model"] = lof
        params["mode"] = "row"
    elif method == "isolation_forest":
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(X_array)
        params["model"] = iso
        params["mode"] = "row"
    else:
        raise ValueError(f"Unknown outlier_cleaning method: {method}")

    X_tmp = X_array.copy()
    if params["mode"] == "cell":
        mask = (X_tmp < params["lower"]) | (X_tmp > params["upper"])
        X_tmp[mask] = np.nan
    else:
        row_mask = params["model"].predict(X_tmp) == -1
        X_tmp[row_mask, :] = np.nan

    params["imputer"] = SimpleImputer(strategy="mean")
    params["imputer"].fit(X_tmp)
    return params


def apply_outlier_cleaner(X: pd.DataFrame, cleaner: Dict[str, Any]) -> pd.DataFrame:
    if cleaner is None:
        return X

    X_array = X.values.astype(float)
    if cleaner["mode"] == "cell":
        indicator = (X_array < cleaner["lower"]) | (X_array > cleaner["upper"])
        X_array[indicator] = np.nan
    else:
        model = cleaner["model"]
        row_mask = model.predict(X_array) == -1
        X_array[row_mask, :] = np.nan

    repaired = cleaner["imputer"].transform(X_array)
    return pd.DataFrame(repaired, columns=X.columns, index=X.index)
