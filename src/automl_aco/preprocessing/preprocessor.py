"""Preprocessor implementation (ported from notebook)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore
from copy import deepcopy

try:
    import category_encoders as ce
except Exception:  # pragma: no cover - optional dependency
    ce = None

from ..config import DEFAULT_PREPROCESSOR_ORDER


class Preprocessor:
    """Leak-free preprocessor matching the notebook behavior."""

    def __init__(self, config: Dict[str, str], step_order: Optional[List[str]] = None):
        self.config = config
        self.step_order = step_order or list(DEFAULT_PREPROCESSOR_ORDER)

        self.fitted = False

        # Saved transformers
        self.num_imputer = None
        self.cat_imputer = None
        self.encoder = None
        self.selector = None
        self.scaler = None
        self.reducer = None

        self.selected_columns_ = None
        self.num_cols = None
        self.cat_cols = None
        self.num_columns_ = None
        self.cat_columns_ = None

        self.outlier_cleaner_num = None
        self.outlier_cleaner_cat = None

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")

        X = X.copy()
        X.columns = X.columns.astype(str)

        self.num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        X_num = X[self.num_cols].copy() if self.num_cols else None
        X_cat = X[self.cat_cols].copy() if self.cat_cols else None

        for step in self.step_order:
            if step == "imputation":
                X_num, X_cat = self._fit_imputation(X_num, X_cat)
            elif step == "outlier_removal":
                X_num, X_cat, y = self._fit_outlier_removal(X_num, X_cat, y)
            elif step == "outlier_cleaning":
                X_num, X_cat = self._fit_outlier_cleaning(X_num, X_cat)
            elif step == "encoding":
                X_cat = self._fit_encoding(X_cat)
            elif step == "feature_selection":
                X_num, X_cat = self._fit_feature_selection(X_num, X_cat, y)
            elif step == "scaling":
                X_num = self._fit_scaling(X_num)
            elif step == "dimensionality_reduction":
                X_num = self._fit_dim_reduction(X_num)

        if X_num is not None:
            X_num.columns = X_num.columns.astype(str)
        if X_cat is not None:
            X_cat.columns = X_cat.columns.astype(str)

        X_out = None
        if X_cat is not None and X_num is not None:
            X_out = pd.concat([X_num, X_cat], axis=1)
        elif X_num is not None:
            X_out = X_num
        elif X_cat is not None:
            X_out = X_cat

        self.fitted = True
        return X_out, y

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise AssertionError("You must call fit() before transform()")

        X = X.copy()
        X.columns = X.columns.astype(str)

        X_num = X[self.num_cols].copy() if self.num_cols else None
        X_cat = X[self.cat_cols].copy() if self.cat_cols else None

        for step in self.step_order:
            if step == "imputation":
                X_num, X_cat = self._transform_imputation(X_num, X_cat)
            elif step == "outlier_removal":
                pass
            elif step == "outlier_cleaning":
                X_num, X_cat = self._transform_outlier_cleaning(X_num, X_cat)
            elif step == "encoding":
                X_cat = self._transform_encoding(X_cat)
            elif step == "feature_selection":
                X_num, X_cat = self._transform_feature_selection(X_num, X_cat)
            elif step == "scaling":
                X_num = self._transform_scaling(X_num)
            elif step == "dimensionality_reduction":
                X_num = self._transform_dim_reduction(X_num)

        if X_num is not None:
            X_num.columns = X_num.columns.astype(str)
        if X_cat is not None:
            X_cat.columns = X_cat.columns.astype(str)

        if X_cat is not None and X_num is not None:
            return pd.concat([X_num, X_cat], axis=1).reset_index(drop=True)
        if X_cat is not None:
            return X_cat.reset_index(drop=True)
        return X_num.reset_index(drop=True)

    # -----------------------------
    # 1. Imputation
    # -----------------------------
    @staticmethod
    def _imputer_output_to_df(arr, index, original_columns, imputer):
        """Build DataFrame safely even when sklearn drops all-missing columns."""
        cols = list(original_columns)
        if arr.shape[1] == len(cols):
            return pd.DataFrame(arr, index=index, columns=cols)

        # Some sklearn versions drop columns that are entirely missing at fit-time.
        # Reconstruct surviving column names from imputer.statistics_ when possible.
        recovered_cols = None
        stats = getattr(imputer, "statistics_", None)
        if stats is not None and len(stats) == len(cols):
            keep_mask = ~pd.isna(stats)
            if int(np.sum(keep_mask)) == arr.shape[1]:
                recovered_cols = [c for c, keep in zip(cols, keep_mask) if keep]

        if recovered_cols is None:
            recovered_cols = cols[: arr.shape[1]]

        return pd.DataFrame(arr, index=index, columns=recovered_cols)

    def _fit_imputation(self, X_num, X_cat):
        method = self.config["imputation"]

        if X_num is not None and method != "none":
            if method == "knn":
                self.num_imputer = KNNImputer(n_neighbors=min(5, len(X_num) - 1))
            elif method in ["mean", "median", "most_frequent", "constant"]:
                self.num_imputer = SimpleImputer(strategy=method)
            else:
                self.num_imputer = SimpleImputer(strategy="mean")

            X_num = self._imputer_output_to_df(
                self.num_imputer.fit_transform(X_num),
                index=X_num.index,
                original_columns=X_num.columns,
                imputer=self.num_imputer,
            )

        if X_cat is not None and method != "none":
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            X_cat = self._imputer_output_to_df(
                self.cat_imputer.fit_transform(X_cat),
                index=X_cat.index,
                original_columns=X_cat.columns,
                imputer=self.cat_imputer,
            )

        return X_num, X_cat

    def _transform_imputation(self, X_num, X_cat):
        if X_num is not None and self.num_imputer is not None:
            X_num = self._imputer_output_to_df(
                self.num_imputer.transform(X_num),
                index=X_num.index,
                original_columns=X_num.columns,
                imputer=self.num_imputer,
            )

        if X_cat is not None and self.cat_imputer is not None:
            X_cat = self._imputer_output_to_df(
                self.cat_imputer.transform(X_cat),
                index=X_cat.index,
                original_columns=X_cat.columns,
                imputer=self.cat_imputer,
            )
        return X_num, X_cat

    # -----------------------------
    # 2. Outlier removal
    # -----------------------------
    def _fit_outlier_removal(self, X_num, X_cat, y):
        method = self.config["outlier_removal"]
        if X_num is None or method == "none":
            return X_num, X_cat, y

        if X_num is not None:
            X_num = X_num.reset_index(drop=True)
        if X_cat is not None:
            X_cat = X_cat.reset_index(drop=True)
        if y is not None:
            y = y.reset_index(drop=True)

        if method == "iqr":
            mask = pd.Series(True, index=X_num.index)
            for col in X_num.columns:
                q1, q3 = X_num[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    mask &= (X_num[col] >= q1 - 1.5 * iqr) & (X_num[col] <= q3 + 1.5 * iqr)
        elif method == "zscore":
            z = np.abs(zscore(X_num))
            mask = pd.Series((z < 3).all(axis=1), index=X_num.index)
        elif method == "lof":
            lof = LocalOutlierFactor(n_neighbors=20)
            mask = pd.Series(lof.fit_predict(X_num) == 1, index=X_num.index)
        elif method == "isolation_forest":
            iso = IsolationForest(contamination=0.05, random_state=42)
            mask = pd.Series(iso.fit_predict(X_num) == 1, index=X_num.index)
        else:
            mask = pd.Series(True, index=X_num.index)

        X_num = X_num.loc[mask].reset_index(drop=True)
        if X_cat is not None:
            X_cat = X_cat.loc[mask].reset_index(drop=True)
        if y is not None:
            y = y.loc[mask].reset_index(drop=True)

        return X_num, X_cat, y

    def _fit_outlier_cleaning(self, X_num, X_cat):
        method = self.config.get("outlier_cleaning", "none")

        self.outlier_cleaner_num = None
        self.outlier_cleaner_cat = None

        if method == "none":
            return X_num, X_cat

        is_cat_encoded = (
            X_cat is not None and all(pd.api.types.is_numeric_dtype(X_cat[c]) for c in X_cat.columns)
        )

        def _fit_cleaner(X):
            X_array = X.values.astype(float)
            params = {}

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

            X_tmp = deepcopy(X_array)
            if params["mode"] == "cell":
                mask = (X_tmp < params["lower"]) | (X_tmp > params["upper"])
                X_tmp[mask] = np.nan
            else:
                row_mask = params["model"].predict(X_tmp) == -1
                X_tmp[row_mask, :] = np.nan

            params["imputer"] = SimpleImputer(strategy="mean")
            params["imputer"].fit(X_tmp)
            return params

        if X_num is not None:
            self.outlier_cleaner_num = _fit_cleaner(X_num)
        if is_cat_encoded:
            self.outlier_cleaner_cat = _fit_cleaner(X_cat)
        return X_num, X_cat

    def _transform_outlier_cleaning(self, X_num, X_cat):
        is_cat_encoded = (
            X_cat is not None and all(pd.api.types.is_numeric_dtype(X_cat[c]) for c in X_cat.columns)
        )

        def _apply_cleaner(X, cleaner):
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
            X_repaired = cleaner["imputer"].transform(X_array)
            return pd.DataFrame(X_repaired, columns=X.columns, index=X.index)

        if X_num is not None:
            X_num = _apply_cleaner(X_num, self.outlier_cleaner_num)
        if is_cat_encoded:
            X_cat = _apply_cleaner(X_cat, self.outlier_cleaner_cat)
        return X_num, X_cat

    # -----------------------------
    # 3. Encoding
    # -----------------------------
    def _fit_encoding(self, X_cat):
        if X_cat is None or self.config["encoding"] == "none":
            return X_cat

        method = self.config["encoding"]

        if method == "onehot":
            try:
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:  # pragma: no cover
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            arr = self.encoder.fit_transform(X_cat)
            return pd.DataFrame(
                arr,
                index=X_cat.index,
                columns=self.encoder.get_feature_names_out(X_cat.columns),
            )

        if ce is None:
            raise RuntimeError("category_encoders is required for non-onehot encoding")

        if method == "frequency":
            self.encoder = ce.CountEncoder(normalize=True)
        elif method == "count":
            self.encoder = ce.CountEncoder(normalize=False)
        elif method == "ordinal":
            self.encoder = ce.OrdinalEncoder()
        elif method == "binary":
            self.encoder = ce.BinaryEncoder()
        else:
            self.encoder = ce.OrdinalEncoder()

        return self.encoder.fit_transform(X_cat)

    def _transform_encoding(self, X_cat):
        if X_cat is None or self.encoder is None:
            return X_cat

        arr = self.encoder.transform(X_cat)

        if hasattr(self.encoder, "get_feature_names_out"):
            return pd.DataFrame(arr, index=X_cat.index, columns=self.encoder.get_feature_names_out())

        return pd.DataFrame(arr, index=X_cat.index)

    # -----------------------------
    # 4. Feature selection
    # -----------------------------
    def _fit_feature_selection(self, X_num, X_cat, y):
        fs = self.config["feature_selection"]

        self.selector = None
        self.selected_columns_ = None
        self.num_columns_ = pd.Index([])
        self.cat_columns_ = pd.Index([])

        if fs == "none":
            return X_num, X_cat

        is_cat_encoded = (
            X_cat is not None and all(pd.api.types.is_numeric_dtype(X_cat[c]) for c in X_cat.columns)
        )

        if X_num is None and not is_cat_encoded:
            return X_num, X_cat

        if X_num is not None and is_cat_encoded:
            X_all = pd.concat([X_num, X_cat], axis=1)
            self.num_columns_ = X_num.columns
            self.cat_columns_ = X_cat.columns
        elif X_num is not None:
            X_all = X_num.copy()
            self.num_columns_ = X_num.columns
        else:
            X_all = X_cat.copy()
            self.cat_columns_ = X_cat.columns

        if fs == "variance_threshold":
            self.selector = VarianceThreshold(threshold=0.01)
            self.selector.fit(X_all)
        else:
            k = min(20, X_all.shape[1])
            if fs == "k_best":
                self.selector = SelectKBest(f_classif, k=k)
                self.selector.fit(X_all, y.values.ravel())
            elif fs == "mutual_info":
                self.selector = SelectKBest(
                    lambda Xv, yv: mutual_info_classif(Xv, yv, discrete_features="auto"),
                    k=k,
                )
                self.selector.fit(X_all, y.values.ravel())
            else:
                raise ValueError(f"Unknown feature_selection: {fs}")

        support = self.selector.get_support()
        self.selected_columns_ = X_all.columns[support]
        X_selected = X_all[self.selected_columns_]

        X_num_sel = (
            X_selected[self.selected_columns_.intersection(self.num_columns_)]
            if len(self.num_columns_) > 0
            else None
        )

        X_cat_sel = (
            X_selected[self.selected_columns_.intersection(self.cat_columns_)]
            if is_cat_encoded
            else X_cat
        )

        return X_num_sel, X_cat_sel

    def _transform_feature_selection(self, X_num, X_cat):
        if self.selector is None:
            return X_num, X_cat

        is_cat_encoded = (
            X_cat is not None and all(pd.api.types.is_numeric_dtype(X_cat[c]) for c in X_cat.columns)
        )

        if len(self.num_columns_) > 0 and len(self.cat_columns_) > 0:
            X_all = pd.concat([X_num[self.num_columns_], X_cat[self.cat_columns_]], axis=1)
        elif len(self.num_columns_) > 0:
            X_all = X_num[self.num_columns_]
        else:
            X_all = X_cat[self.cat_columns_]

        arr = self.selector.transform(X_all)
        X_selected = pd.DataFrame(arr, index=X_all.index, columns=self.selected_columns_)

        X_num_sel = (
            X_selected[self.selected_columns_.intersection(self.num_columns_)]
            if len(self.num_columns_) > 0
            else None
        )

        X_cat_sel = (
            X_selected[self.selected_columns_.intersection(self.cat_columns_)]
            if is_cat_encoded
            else X_cat
        )

        return X_num_sel, X_cat_sel

    # -----------------------------
    # 5. Scaling
    # -----------------------------
    def _fit_scaling(self, X):
        method = self.config["scaling"]
        if X is None or method == "none":
            return X

        self.scaler = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabs": MaxAbsScaler(),
        }.get(method)

        if self.scaler:
            return pd.DataFrame(self.scaler.fit_transform(X), index=X.index, columns=X.columns)
        return X

    def _transform_scaling(self, X):
        if X is None or self.scaler is None:
            return X
        return pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)

    # -----------------------------
    # 6. Dimensionality Reduction
    # -----------------------------
    def _fit_dim_reduction(self, X):
        dr = self.config["dimensionality_reduction"]
        if X is None or dr == "none" or X.shape[1] <= 1 or len(X) < 2:
            self.reducer = None
            return X

        n_components = min(10, X.shape[1], len(X) - 1)

        if dr == "pca":
            self.reducer = PCA(n_components=n_components)
        else:
            self.reducer = TruncatedSVD(n_components=n_components)

        arr = self.reducer.fit_transform(X)
        cols = [f"dr_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, index=X.index, columns=cols)

    def _transform_dim_reduction(self, X):
        if X is None or self.reducer is None:
            return X
        arr = self.reducer.transform(X)
        cols = [f"dr_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, index=X.index, columns=cols)
