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
from ..utils.operator_spec import parse_operator_spec


class Preprocessor:

    def __init__(self, config: Dict[str, str], step_order: Optional[List[str]] = None):
        self.config = config
        self.step_order = step_order or list(DEFAULT_PREPROCESSOR_ORDER)

        self.fitted = False

        # Saved transformers
        self.num_imputer = None
        self.cat_imputer = None
        self.num_imputer_map = None
        self.cat_imputer_map = None
        self.encoder = None
        self.encoder_map = None
        self.selector = None
        self.scaler = None
        self.scaler_map = None
        self.reducer = None

        self.selected_columns_ = None
        self.num_cols = None
        self.cat_cols = None
        self.num_columns_ = None
        self.cat_columns_ = None

        self.outlier_cleaner_num = None
        self.outlier_cleaner_cat = None

    @staticmethod
    def _resolve_feature_operator(spec: object, column: str, default: str = "none") -> str:
        if isinstance(spec, dict):
            if column in spec:
                return str(spec[column])
            if "*" in spec:
                return str(spec["*"])
            return default
        if spec is None:
            return default
        return str(spec)

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
        method_raw = self.config["imputation"]
        self.num_imputer = None
        self.cat_imputer = None
        self.num_imputer_map = None
        self.cat_imputer_map = None

        # Per-feature map: {"col_name": "median", "*": "none", ...}
        if isinstance(method_raw, dict):
            if X_num is not None:
                X_num_out = X_num.copy()
                self.num_imputer_map = {}
                for col in X_num.columns:
                    col_method_raw = self._resolve_feature_operator(method_raw, str(col), default="none")
                    col_method, col_params = parse_operator_spec(col_method_raw)
                    if col_method == "none":
                        continue
                    if col_method == "knn":
                        # KNN is multivariate by design; in per-feature mode, fall back to mean.
                        imp = SimpleImputer(strategy="mean")
                    elif col_method in ["mean", "median", "most_frequent", "constant"]:
                        imp = SimpleImputer(strategy=col_method)
                    else:
                        imp = SimpleImputer(strategy="mean")
                    arr = imp.fit_transform(X_num[[col]])
                    X_num_out[col] = arr.ravel()
                    self.num_imputer_map[str(col)] = imp
                X_num = X_num_out

            if X_cat is not None:
                X_cat_out = X_cat.copy()
                self.cat_imputer_map = {}
                for col in X_cat.columns:
                    col_method_raw = self._resolve_feature_operator(method_raw, str(col), default="none")
                    col_method, _col_params = parse_operator_spec(col_method_raw)
                    if col_method == "none":
                        continue
                    imp = SimpleImputer(strategy="most_frequent")
                    arr = imp.fit_transform(X_cat[[col]])
                    X_cat_out[col] = arr.ravel()
                    self.cat_imputer_map[str(col)] = imp
                X_cat = X_cat_out
            return X_num, X_cat

        method, params = parse_operator_spec(method_raw)

        if X_num is not None and method != "none":
            if method == "knn":
                try:
                    k = int(float(params.get("k", 5)))
                except Exception:
                    k = 5
                k = max(1, k)
                self.num_imputer = KNNImputer(n_neighbors=min(k, len(X_num) - 1))
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
        if isinstance(self.config.get("imputation"), dict):
            if X_num is not None and isinstance(self.num_imputer_map, dict):
                X_num_out = X_num.copy()
                for col, imp in self.num_imputer_map.items():
                    if col in X_num_out.columns:
                        arr = imp.transform(X_num_out[[col]])
                        X_num_out[col] = arr.ravel()
                X_num = X_num_out

            if X_cat is not None and isinstance(self.cat_imputer_map, dict):
                X_cat_out = X_cat.copy()
                for col, imp in self.cat_imputer_map.items():
                    if col in X_cat_out.columns:
                        arr = imp.transform(X_cat_out[[col]])
                        X_cat_out[col] = arr.ravel()
                X_cat = X_cat_out
            return X_num, X_cat

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
        method_raw = self.config["outlier_removal"]
        method, params = parse_operator_spec(method_raw)
        if X_num is None or method == "none":
            return X_num, X_cat, y

        if X_num is not None:
            X_num = X_num.reset_index(drop=True)
        if X_cat is not None:
            X_cat = X_cat.reset_index(drop=True)
        if y is not None:
            y = y.reset_index(drop=True)

        if method == "iqr":
            try:
                iqr_k = float(params.get("k", 1.5))
            except Exception:
                iqr_k = 1.5
            mask = pd.Series(True, index=X_num.index)
            for col in X_num.columns:
                q1, q3 = X_num[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    mask &= (X_num[col] >= q1 - iqr_k * iqr) & (X_num[col] <= q3 + iqr_k * iqr)
        elif method == "zscore":
            try:
                z_thr = float(params.get("z", 3.0))
            except Exception:
                z_thr = 3.0
            z = np.abs(zscore(X_num))
            mask = pd.Series((z < z_thr).all(axis=1), index=X_num.index)
        elif method == "lof":
            try:
                lof_n = int(float(params.get("n", 20)))
            except Exception:
                lof_n = 20
            lof_n = max(2, min(lof_n, max(2, len(X_num) - 1)))
            lof = LocalOutlierFactor(n_neighbors=lof_n)
            mask = pd.Series(lof.fit_predict(X_num) == 1, index=X_num.index)
        elif method == "isolation_forest":
            try:
                contamination = float(params.get("c", params.get("contamination", 0.05)))
            except Exception:
                contamination = 0.05
            contamination = float(np.clip(contamination, 1e-4, 0.5))
            iso = IsolationForest(contamination=contamination, random_state=42)
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
        method_raw = self.config["encoding"]
        self.encoder = None
        self.encoder_map = None

        if isinstance(method_raw, dict):
            if X_cat is None:
                return X_cat
            pieces = []
            self.encoder_map = {}
            for col in X_cat.columns:
                col_method_raw = self._resolve_feature_operator(method_raw, str(col), default="none")
                col_method, _col_params = parse_operator_spec(col_method_raw)
                if col_method == "none":
                    pieces.append(X_cat[[col]].copy())
                    self.encoder_map[str(col)] = {"method": "none"}
                    continue

                if col_method == "onehot":
                    try:
                        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                    except TypeError:  # pragma: no cover
                        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
                    arr = enc.fit_transform(X_cat[[col]])
                    names = [str(x) for x in enc.get_feature_names_out([str(col)])]
                    out_df = pd.DataFrame(arr, index=X_cat.index, columns=names)
                    pieces.append(out_df)
                    self.encoder_map[str(col)] = {
                        "method": "onehot",
                        "encoder": enc,
                        "output_columns": names,
                    }
                    continue

                if ce is None:
                    raise RuntimeError("category_encoders is required for non-onehot encoding")
                if col_method == "frequency":
                    enc = ce.CountEncoder(normalize=True)
                elif col_method == "count":
                    enc = ce.CountEncoder(normalize=False)
                elif col_method == "ordinal":
                    enc = ce.OrdinalEncoder()
                elif col_method == "binary":
                    enc = ce.BinaryEncoder()
                else:
                    enc = ce.OrdinalEncoder()

                transformed = enc.fit_transform(X_cat[[col]])
                if isinstance(transformed, pd.DataFrame):
                    out_df = transformed.copy()
                else:
                    out_df = pd.DataFrame(transformed, index=X_cat.index)
                prefixed = [f"{col}__{i}" for i in out_df.columns.astype(str)]
                out_df.columns = prefixed
                pieces.append(out_df)
                self.encoder_map[str(col)] = {
                    "method": col_method,
                    "encoder": enc,
                    "output_columns": prefixed,
                }
            if not pieces:
                return None
            return pd.concat(pieces, axis=1)

        method, _params = parse_operator_spec(method_raw)
        if X_cat is None or method == "none":
            return X_cat

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
        if isinstance(self.config.get("encoding"), dict):
            if X_cat is None:
                return X_cat
            if not isinstance(self.encoder_map, dict):
                return X_cat
            pieces = []
            for col in X_cat.columns:
                state = self.encoder_map.get(str(col), {"method": "none"})
                method = str(state.get("method", "none"))
                if method == "none":
                    pieces.append(X_cat[[col]].copy())
                    continue

                enc = state.get("encoder")
                if enc is None:
                    pieces.append(X_cat[[col]].copy())
                    continue
                arr = enc.transform(X_cat[[col]])
                out_cols = state.get("output_columns")

                if isinstance(arr, pd.DataFrame):
                    out_df = arr.copy()
                    if isinstance(out_cols, list) and len(out_cols) == out_df.shape[1]:
                        out_df.columns = out_cols
                    pieces.append(out_df)
                    continue

                if hasattr(arr, "toarray"):
                    arr = arr.toarray()
                if not isinstance(out_cols, list) or len(out_cols) != arr.shape[1]:
                    out_cols = [f"{col}__{i}" for i in range(arr.shape[1])]
                pieces.append(pd.DataFrame(arr, index=X_cat.index, columns=out_cols))
            if not pieces:
                return None
            return pd.concat(pieces, axis=1)

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
        fs_raw = self.config["feature_selection"]
        fs, params = parse_operator_spec(fs_raw)

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
            try:
                threshold = float(params.get("t", params.get("threshold", 0.01)))
            except Exception:
                threshold = 0.01
            threshold = max(0.0, threshold)
            self.selector = VarianceThreshold(threshold=threshold)
            self.selector.fit(X_all)
        else:
            try:
                k = int(float(params.get("k", 20)))
            except Exception:
                k = 20
            k = max(1, min(k, X_all.shape[1]))
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
        method_raw = self.config["scaling"]
        self.scaler = None
        self.scaler_map = None
        if isinstance(method_raw, dict):
            if X is None:
                return X
            X_out = X.copy()
            self.scaler_map = {}
            for col in X.columns:
                col_method_raw = self._resolve_feature_operator(method_raw, str(col), default="none")
                col_method, _col_params = parse_operator_spec(col_method_raw)
                if col_method == "none":
                    continue
                scaler = {
                    "standard": StandardScaler(),
                    "minmax": MinMaxScaler(),
                    "robust": RobustScaler(),
                    "maxabs": MaxAbsScaler(),
                }.get(col_method)
                if scaler is None:
                    continue
                arr = scaler.fit_transform(X[[col]])
                X_out[col] = arr.ravel()
                self.scaler_map[str(col)] = scaler
            return X_out

        method, _params = parse_operator_spec(method_raw)
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
        if isinstance(self.config.get("scaling"), dict):
            if X is None:
                return X
            if not isinstance(self.scaler_map, dict):
                return X
            X_out = X.copy()
            for col, scaler in self.scaler_map.items():
                if col not in X_out.columns:
                    continue
                arr = scaler.transform(X_out[[col]])
                X_out[col] = arr.ravel()
            return X_out

        if X is None or self.scaler is None:
            return X
        return pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)

    # -----------------------------
    # 6. Dimensionality Reduction
    # -----------------------------
    def _fit_dim_reduction(self, X):
        dr_raw = self.config["dimensionality_reduction"]
        dr, params = parse_operator_spec(dr_raw)
        if X is None or dr == "none" or X.shape[1] <= 1 or len(X) < 2:
            self.reducer = None
            return X

        try:
            requested_n = int(float(params.get("n", 10)))
        except Exception:
            requested_n = 10
        n_components = max(1, min(requested_n, X.shape[1], len(X) - 1))

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
