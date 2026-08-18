"""AutoDP operator space and leakage-safe preprocessing implementation.

The operator names follow Table 1 of the AutoDP paper.  ACORec uses a fixed,
explicit execution order for the reference performance matrix; operator-order
search, if desired, remains a separate ACORec concern.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import zlib

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class AutoDPResourceLimitError(RuntimeError):
    """Raised when an operator would exceed the experiment's safe resource limit."""


AUTODP_OPTIONS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("imputation", ["none", "random", "drop", "knn", "most_frequent", "em", "mice"]),
        ("encoding", ["none", "ordinal", "binary", "frequency", "catboost"]),
        ("normalization", ["none", "zscore", "minmax", "decimal_scale"]),
        ("feature_selection", ["none", "missing_ratio", "wrapper", "collinear", "tree_based"]),
        ("duplicate_removal", ["none", "exact", "approximate"]),
        ("outlier_removal", ["none", "zscore", "iqr", "lof"]),
    ]
)

# Cleaning precedes target-aware encoding; feature selection sees the final
# numeric representation.  This order is held constant for matrix comparability.
DEFAULT_AUTODP_ORDER: Tuple[str, ...] = (
    "imputation",
    "duplicate_removal",
    "outlier_removal",
    "encoding",
    "normalization",
    "feature_selection",
)

AUTODP_CLASSIFICATION_IDS: Tuple[int, ...] = (
    36, 728, 735, 737, 761, 803, 807, 816, 819, 871,
    1021, 1489, 31, 182, 183, 752, 833, 934, 979, 43723,
    32, 310, 1471, 43972, 179, 184, 734, 1053, 1461, 42493,
)

AUTODP_REGRESSION_IDS: Tuple[int, ...] = (
    189, 217, 225, 227, 503, 507, 529, 572, 573, 668,
    4551, 23516, 308, 516, 541, 558, 41021, 41700, 41928, 564,
    23515, 44019, 44031, 44057, 44066, 344, 41539, 41704, 44311, 45012,
)

AUTODP_60_IDS: Tuple[int, ...] = AUTODP_CLASSIFICATION_IDS + AUTODP_REGRESSION_IDS


def autodp_space_size(options: Mapping[str, Sequence[str]] = AUTODP_OPTIONS) -> int:
    """Return the Cartesian size of an AutoDP-style operator space."""
    return int(np.prod([len(values) for values in options.values()]))


def _pipeline(name: str, **overrides: str) -> Dict[str, str]:
    config = {stage: "none" for stage in AUTODP_OPTIONS}
    config.update(overrides)
    return {"name": name, **config}


def build_autodp_reference_pipelines() -> List[Dict[str, str]]:
    """Build 36 deterministic, human-designed reference pipelines.

    The design contains one all-none baseline, one pipeline for each of the 22
    non-none operators, and 13 deliberately diverse multi-stage pipelines.
    """
    pipelines = [_pipeline("autodp_00_baseline")]

    # One-factor-at-a-time coverage makes every operator independently visible
    # in the performance matrix.
    for stage, values in AUTODP_OPTIONS.items():
        for value in values:
            if value != "none":
                pipelines.append(_pipeline(f"autodp_ofat_{stage}_{value}", **{stage: value}))

    combined = [
        _pipeline("autodp_combo_01_basic", imputation="most_frequent", encoding="ordinal", normalization="zscore", feature_selection="missing_ratio", duplicate_removal="exact", outlier_removal="iqr"),
        _pipeline("autodp_combo_02_knn_binary", imputation="knn", encoding="binary", normalization="minmax", feature_selection="collinear", duplicate_removal="exact", outlier_removal="lof"),
        _pipeline("autodp_combo_03_mice_catboost", imputation="mice", encoding="catboost", normalization="zscore", feature_selection="tree_based", duplicate_removal="approximate", outlier_removal="zscore"),
        _pipeline("autodp_combo_04_em_frequency", imputation="em", encoding="frequency", normalization="decimal_scale", feature_selection="wrapper", duplicate_removal="exact", outlier_removal="iqr"),
        _pipeline("autodp_combo_05_random_ordinal", imputation="random", encoding="ordinal", normalization="zscore", feature_selection="collinear", duplicate_removal="approximate", outlier_removal="lof"),
        _pipeline("autodp_combo_06_drop_binary", imputation="drop", encoding="binary", normalization="minmax", feature_selection="tree_based", duplicate_removal="exact", outlier_removal="zscore"),
        _pipeline("autodp_combo_07_mf_frequency", imputation="most_frequent", encoding="frequency", normalization="decimal_scale", feature_selection="missing_ratio", duplicate_removal="approximate", outlier_removal="iqr"),
        _pipeline("autodp_combo_08_knn_catboost", imputation="knn", encoding="catboost", normalization="zscore", feature_selection="wrapper", outlier_removal="lof"),
        _pipeline("autodp_combo_09_mice_ordinal", imputation="mice", encoding="ordinal", normalization="minmax", feature_selection="collinear", duplicate_removal="exact"),
        _pipeline("autodp_combo_10_em_binary", imputation="em", encoding="binary", normalization="decimal_scale", feature_selection="missing_ratio", duplicate_removal="approximate", outlier_removal="zscore"),
        _pipeline("autodp_combo_11_random_frequency", imputation="random", encoding="frequency", normalization="zscore", feature_selection="tree_based", outlier_removal="iqr"),
        _pipeline("autodp_combo_12_drop_catboost", imputation="drop", encoding="catboost", normalization="minmax", feature_selection="wrapper", duplicate_removal="exact", outlier_removal="lof"),
        _pipeline("autodp_combo_13_mf_binary", imputation="most_frequent", encoding="binary", normalization="decimal_scale", feature_selection="collinear", duplicate_removal="approximate", outlier_removal="zscore"),
    ]
    pipelines.extend(combined)
    validate_autodp_reference_pipelines(pipelines)
    return pipelines


def validate_autodp_reference_pipelines(pipelines: Sequence[Mapping[str, str]]) -> None:
    """Fail fast if a reference pipeline is malformed or coverage regresses."""
    if len(pipelines) != 36:
        raise ValueError(f"Expected 36 AutoDP reference pipelines, got {len(pipelines)}")
    names = [config.get("name") for config in pipelines]
    if len(set(names)) != len(names):
        raise ValueError("AutoDP reference pipeline names must be unique")
    for config in pipelines:
        for stage, allowed in AUTODP_OPTIONS.items():
            if config.get(stage) not in allowed:
                raise ValueError(f"Invalid {stage}={config.get(stage)!r} in {config.get('name')}")
    for stage, allowed in AUTODP_OPTIONS.items():
        covered = {config[stage] for config in pipelines}
        missing = set(allowed) - covered
        if missing:
            raise ValueError(f"Reference pipelines do not cover {stage}: {sorted(missing)}")


def exclude_holdout_columns(
    performance_matrix: pd.DataFrame,
    holdout_ids: Iterable[int] = AUTODP_60_IDS,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove all holdout dataset columns before any ACORec training/retrieval."""
    forbidden = {f"D_{int(dataset_id)}" for dataset_id in holdout_ids}
    removed = [str(column) for column in performance_matrix.columns if str(column) in forbidden]
    return performance_matrix.drop(columns=removed), removed


class AutoDPPreprocessor:
    """Train-only fitted implementation of the AutoDP paper operator space.

    ``drop``, duplicate removal, and outlier removal may remove training rows.
    Validation/test rows are never removed, so prediction cardinality is stable.
    For ``drop``, held-out missing values use training-only fallback statistics.

    AutoDP's EM, MICE, approximate duplicate, and wrapper operators do not have
    exact scikit-learn equivalents.  Their deterministic approximations are
    documented in the corresponding methods below.
    """

    def __init__(
        self,
        config: Mapping[str, str],
        *,
        task_type: str,
        step_order: Optional[Sequence[str]] = None,
        random_state: int = 42,
        missing_ratio_threshold: float = 0.40,
        collinear_threshold: float = 0.95,
        max_collinear_features: int = 2_000,
    ) -> None:
        self.config = dict(config)
        self.task_type = str(task_type)
        self.step_order = tuple(step_order or DEFAULT_AUTODP_ORDER)
        self.random_state = int(random_state)
        self.missing_ratio_threshold = float(missing_ratio_threshold)
        self.collinear_threshold = float(collinear_threshold)
        self.max_collinear_features = int(max_collinear_features)
        self.fitted = False

        for stage, allowed in AUTODP_OPTIONS.items():
            value = self.config.get(stage, "none")
            if value not in allowed:
                raise ValueError(f"Unsupported AutoDP operator: {stage}={value!r}")

        self.input_columns_: List[str] = []
        self.output_columns_: List[str] = []
        self.numeric_columns_: List[str] = []
        self.categorical_columns_: List[str] = []
        self.imputation_numeric_columns_: List[str] = []
        self.imputation_categorical_columns_: List[str] = []
        self.num_imputer_ = None
        self.cat_imputer_ = None
        self.fallback_num_imputer_ = None
        self.fallback_cat_imputer_ = None
        self.random_pools_: Dict[str, np.ndarray] = {}
        self.encoder_state_: Dict[str, object] = {}
        self.scaler_ = None
        self.scaler_columns_: List[str] = []
        self.decimal_scales_: Optional[pd.Series] = None
        self.selected_columns_: Optional[List[str]] = None

    @staticmethod
    def _frame(X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        frame.columns = frame.columns.astype(str)
        frame = frame.reset_index(drop=True)
        numeric = frame.select_dtypes(include=[np.number]).columns
        if len(numeric):
            frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
        return frame

    @staticmethod
    def _series(y: Optional[pd.Series]) -> Optional[pd.Series]:
        return None if y is None else pd.Series(y).reset_index(drop=True)

    @staticmethod
    def _safe_strings(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("__MISSING__")

    @staticmethod
    def _imputer_frame(imputer, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.shape[1] == 0:
            return frame.copy()
        values = imputer.transform(frame)
        return pd.DataFrame(values, columns=frame.columns, index=frame.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X_work = self._frame(X)
        y_work = self._series(y)
        if y_work is not None and len(X_work) != len(y_work):
            raise ValueError("X and y must have the same length")
        self.input_columns_ = X_work.columns.tolist()
        self.numeric_columns_ = X_work.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns_ = [c for c in X_work.columns if c not in self.numeric_columns_]

        for step in self.step_order:
            if step == "imputation":
                X_work, y_work = self._fit_imputation(X_work, y_work)
            elif step == "duplicate_removal":
                X_work, y_work = self._fit_duplicate_removal(X_work, y_work)
            elif step == "outlier_removal":
                X_work, y_work = self._fit_outlier_removal(X_work, y_work)
            elif step == "encoding":
                X_work = self._fit_encoding(X_work, y_work)
            elif step == "normalization":
                X_work = self._fit_normalization(X_work)
            elif step == "feature_selection":
                X_work = self._fit_feature_selection(X_work, y_work)

        X_work = X_work.reset_index(drop=True)
        y_work = self._series(y_work)
        self.output_columns_ = X_work.columns.astype(str).tolist()
        self.fitted = True
        return X_work, y_work

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("fit_transform must be called before transform")
        X_work = self._frame(X)
        for column in self.input_columns_:
            if column not in X_work:
                X_work[column] = np.nan
        X_work = X_work[self.input_columns_]

        for step in self.step_order:
            if step == "imputation":
                X_work = self._transform_imputation(X_work)
            elif step in {"duplicate_removal", "outlier_removal"}:
                # Row-removal operators are training-only.
                continue
            elif step == "encoding":
                X_work = self._transform_encoding(X_work)
            elif step == "normalization":
                X_work = self._transform_normalization(X_work)
            elif step == "feature_selection":
                X_work = self._transform_feature_selection(X_work)

        for column in self.output_columns_:
            if column not in X_work:
                X_work[column] = np.nan
        return X_work[self.output_columns_].reset_index(drop=True)

    def _fit_fallback_imputers(self, X: pd.DataFrame) -> None:
        num = X[self.imputation_numeric_columns_].copy()
        cat = X[self.imputation_categorical_columns_].copy()
        if num.shape[1]:
            for column in num.columns[num.isna().all()]:
                num.loc[num.index[0], column] = 0.0
            self.fallback_num_imputer_ = SimpleImputer(strategy="median")
            self.fallback_num_imputer_.fit(num)
        if cat.shape[1]:
            for column in cat.columns[cat.isna().all()]:
                cat.loc[cat.index[0], column] = "__MISSING__"
            self.fallback_cat_imputer_ = SimpleImputer(strategy="most_frequent")
            self.fallback_cat_imputer_.fit(cat)

    def _apply_fallback_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        if self.fallback_num_imputer_ is not None:
            result[self.imputation_numeric_columns_] = self._imputer_frame(
                self.fallback_num_imputer_, result[self.imputation_numeric_columns_]
            )
        if self.fallback_cat_imputer_ is not None:
            result[self.imputation_categorical_columns_] = self._imputer_frame(
                self.fallback_cat_imputer_, result[self.imputation_categorical_columns_]
            )
        return result

    def _random_fill(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        result = X.copy()
        for column in result.columns:
            if fit:
                pool = result[column].dropna().to_numpy()
                if len(pool) == 0:
                    pool = np.array([0.0 if column in self.numeric_columns_ else "__MISSING__"])
                self.random_pools_[column] = pool
            pool = self.random_pools_[column]
            missing = result[column].isna().to_numpy()
            if missing.any():
                seed = self.random_state + zlib.crc32(column.encode("utf-8"))
                rng = np.random.RandomState(seed & 0xFFFFFFFF)
                result.loc[missing, column] = rng.choice(pool, size=int(missing.sum()), replace=True)
        return result

    def _fit_imputation(self, X: pd.DataFrame, y: Optional[pd.Series]):
        method = self.config.get("imputation", "none")
        self.imputation_numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.imputation_categorical_columns_ = [
            column for column in X.columns if column not in self.imputation_numeric_columns_
        ]
        if method == "none":
            return X, y
        self._fit_fallback_imputers(X)
        if method == "drop":
            keep = ~X.isna().any(axis=1)
            if int(keep.sum()) < 2:
                raise ValueError("drop imputation left fewer than two training rows")
            return X.loc[keep].reset_index(drop=True), None if y is None else y.loc[keep].reset_index(drop=True)
        if method == "random":
            return self._random_fill(X, fit=True), y

        result = X.copy()
        num = result[self.imputation_numeric_columns_].copy()
        cat = result[self.imputation_categorical_columns_].copy()
        if num.shape[1]:
            for column in num.columns[num.isna().all()]:
                num.loc[num.index[0], column] = 0.0
            if method == "knn":
                self.num_imputer_ = KNNImputer(n_neighbors=max(1, min(5, len(X) - 1)))
            elif method == "em":
                # Deterministic iterative conditional expectation is the closest
                # stable sklearn analogue to EM imputation.
                self.num_imputer_ = IterativeImputer(max_iter=10, sample_posterior=False, random_state=self.random_state)
            elif method == "mice":
                self.num_imputer_ = IterativeImputer(max_iter=10, sample_posterior=True, random_state=self.random_state)
            else:
                self.num_imputer_ = SimpleImputer(strategy="most_frequent")
            result[self.imputation_numeric_columns_] = pd.DataFrame(
                self.num_imputer_.fit_transform(num), columns=self.imputation_numeric_columns_, index=result.index
            )
        if cat.shape[1]:
            for column in cat.columns[cat.isna().all()]:
                cat.loc[cat.index[0], column] = "__MISSING__"
            self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
            result[self.imputation_categorical_columns_] = pd.DataFrame(
                self.cat_imputer_.fit_transform(cat), columns=self.imputation_categorical_columns_, index=result.index
            )
        return result, y

    def _transform_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        method = self.config.get("imputation", "none")
        if method == "none":
            return X
        if method == "drop":
            return self._apply_fallback_imputation(X)
        if method == "random":
            return self._random_fill(X, fit=False)
        result = X.copy()
        if self.num_imputer_ is not None:
            result[self.imputation_numeric_columns_] = self._imputer_frame(
                self.num_imputer_, result[self.imputation_numeric_columns_]
            )
        if self.cat_imputer_ is not None:
            result[self.imputation_categorical_columns_] = self._imputer_frame(
                self.cat_imputer_, result[self.imputation_categorical_columns_]
            )
        return result

    def _fit_duplicate_removal(self, X: pd.DataFrame, y: Optional[pd.Series]):
        method = self.config.get("duplicate_removal", "none")
        if method == "none" or len(X) < 2:
            return X, y
        if method == "exact":
            duplicate = X.duplicated(keep="first")
        else:
            # Scalable approximation: quantize numeric values to 1% of their
            # training range and normalize string representation before hashing.
            signature = pd.DataFrame(index=X.index)
            numeric = X.select_dtypes(include=[np.number]).columns
            categorical = [c for c in X.columns if c not in numeric]
            for column in numeric:
                values = pd.to_numeric(X[column], errors="coerce")
                low, high = values.min(), values.max()
                scale = high - low
                signature[column] = ((values - low) / scale).round(2) if pd.notna(scale) and scale > 0 else 0.0
                signature[column] = signature[column].fillna(-999.0)
            for column in categorical:
                signature[column] = self._safe_strings(X[column]).str.strip().str.lower()
            duplicate = signature.duplicated(keep="first")
        keep = ~duplicate
        return X.loc[keep].reset_index(drop=True), None if y is None else y.loc[keep].reset_index(drop=True)

    @staticmethod
    def _numeric_detector_frame(X: pd.DataFrame) -> pd.DataFrame:
        numeric = X.select_dtypes(include=[np.number]).copy()
        if numeric.shape[1] == 0:
            return numeric
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        return numeric.fillna(numeric.median()).fillna(0.0)

    def _fit_outlier_removal(self, X: pd.DataFrame, y: Optional[pd.Series]):
        method = self.config.get("outlier_removal", "none")
        detector = self._numeric_detector_frame(X)
        if method == "none" or detector.shape[1] == 0 or len(detector) < 3:
            return X, y
        if method == "zscore":
            std = detector.std(ddof=0).replace(0.0, 1.0)
            keep = ((detector - detector.mean()).abs().div(std) <= 3.0).all(axis=1)
        elif method == "iqr":
            q1, q3 = detector.quantile(0.25), detector.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            varying = iqr > 0
            keep = ((detector.loc[:, varying] >= lower[varying]) & (detector.loc[:, varying] <= upper[varying])).all(axis=1)
        else:
            neighbors = max(2, min(20, len(detector) - 1))
            keep = pd.Series(LocalOutlierFactor(n_neighbors=neighbors).fit_predict(detector) == 1, index=X.index)
        if int(keep.sum()) < max(2, int(0.20 * len(X))):
            # Avoid pathological operators deleting almost the entire training set.
            keep = pd.Series(True, index=X.index)
        return X.loc[keep].reset_index(drop=True), None if y is None else y.loc[keep].reset_index(drop=True)

    def _fit_encoding(self, X: pd.DataFrame, y: Optional[pd.Series]) -> pd.DataFrame:
        method = self.config.get("encoding", "none")
        categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.encoder_state_ = {"method": method, "columns": categorical}
        if method == "none" or not categorical:
            return X
        numeric = X.drop(columns=categorical).reset_index(drop=True)
        encoded = self._encode_categorical(X[categorical].reset_index(drop=True), y, fit=True)
        return pd.concat([numeric, encoded], axis=1)

    def _transform_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        method = str(self.encoder_state_.get("method", "none"))
        categorical = list(self.encoder_state_.get("columns", []))
        if method == "none" or not categorical:
            return X
        numeric = X.drop(columns=categorical).reset_index(drop=True)
        encoded = self._encode_categorical(X[categorical].reset_index(drop=True), None, fit=False)
        return pd.concat([numeric, encoded], axis=1)

    def _encode_categorical(self, X_cat: pd.DataFrame, y: Optional[pd.Series], *, fit: bool) -> pd.DataFrame:
        method = str(self.encoder_state_["method"])
        parts: List[pd.DataFrame] = []
        if fit:
            self.encoder_state_["mappings"] = {}
        mappings = self.encoder_state_["mappings"]

        if method == "catboost":
            if fit and y is None:
                raise ValueError("CatBoost encoding requires y during fit")
            if fit:
                y_num = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce")
                global_mean = float(y_num.mean())
                self.encoder_state_["global_mean"] = global_mean
            else:
                y_num = None
                global_mean = float(self.encoder_state_["global_mean"])
            alpha = 10.0
            for column in X_cat.columns:
                values = self._safe_strings(X_cat[column]).reset_index(drop=True)
                if fit:
                    sums: Dict[str, float] = {}
                    counts: Dict[str, int] = {}
                    ordered = np.empty(len(values), dtype=float)
                    rng = np.random.RandomState(self.random_state + zlib.crc32(column.encode("utf-8")))
                    for index in rng.permutation(len(values)):
                        key = str(values.iloc[index])
                        ordered[index] = (sums.get(key, 0.0) + alpha * global_mean) / (counts.get(key, 0) + alpha)
                        sums[key] = sums.get(key, 0.0) + float(y_num.iloc[index])
                        counts[key] = counts.get(key, 0) + 1
                    mappings[column] = {
                        key: (sums[key] + alpha * global_mean) / (counts[key] + alpha) for key in counts
                    }
                    encoded = ordered
                else:
                    encoded = values.map(mappings[column]).fillna(global_mean).to_numpy(dtype=float)
                parts.append(pd.DataFrame({f"{column}__catboost": encoded}))
            return pd.concat(parts, axis=1)

        for column in X_cat.columns:
            values = self._safe_strings(X_cat[column])
            if fit:
                unique = pd.Index(values.unique())
                if method in {"ordinal", "binary"}:
                    mappings[column] = {str(value): index + 1 for index, value in enumerate(unique)}
                elif method == "frequency":
                    mappings[column] = values.value_counts(normalize=True).to_dict()
            mapping = mappings[column]
            if method == "frequency":
                parts.append(pd.DataFrame({f"{column}__frequency": values.map(mapping).fillna(0.0).astype(float)}))
            else:
                codes = values.map(mapping).fillna(0).astype(int)
                if method == "ordinal":
                    parts.append(pd.DataFrame({f"{column}__ordinal": codes.astype(float)}))
                else:
                    if fit:
                        bits = max(1, int(np.ceil(np.log2(max(mapping.values(), default=0) + 1))))
                        self.encoder_state_.setdefault("bits", {})[column] = bits
                    bits = int(self.encoder_state_["bits"][column])
                    code_values = codes.to_numpy(dtype=np.int64)
                    parts.append(pd.DataFrame({
                        f"{column}__binary_{bit}": np.bitwise_and(
                            np.right_shift(code_values, bit), 1
                        ).astype(float)
                        for bit in range(bits)
                    }))
        return pd.concat(parts, axis=1)

    def _fit_normalization(self, X: pd.DataFrame) -> pd.DataFrame:
        method = self.config.get("normalization", "none")
        self.scaler_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        if method == "none" or not self.scaler_columns_:
            return X
        result = X.copy()
        numeric = result[self.scaler_columns_]
        if method == "decimal_scale":
            max_abs = numeric.abs().max()
            powers = np.where(max_abs.fillna(0.0).to_numpy() > 0, np.floor(np.log10(max_abs.fillna(0.0).to_numpy())) + 1, 0)
            self.decimal_scales_ = pd.Series(np.power(10.0, powers), index=self.scaler_columns_).replace(0.0, 1.0)
            result[self.scaler_columns_] = numeric.divide(self.decimal_scales_, axis=1)
        else:
            self.scaler_ = StandardScaler() if method == "zscore" else MinMaxScaler()
            result[self.scaler_columns_] = pd.DataFrame(
                self.scaler_.fit_transform(numeric), columns=self.scaler_columns_, index=result.index
            )
        return result

    def _transform_normalization(self, X: pd.DataFrame) -> pd.DataFrame:
        method = self.config.get("normalization", "none")
        if method == "none" or not self.scaler_columns_:
            return X
        result = X.copy()
        if method == "decimal_scale":
            result[self.scaler_columns_] = result[self.scaler_columns_].divide(self.decimal_scales_, axis=1)
        else:
            result[self.scaler_columns_] = pd.DataFrame(
                self.scaler_.transform(result[self.scaler_columns_]),
                columns=self.scaler_columns_,
                index=result.index,
            )
        return result

    def _fit_feature_selection(self, X: pd.DataFrame, y: Optional[pd.Series]) -> pd.DataFrame:
        method = self.config.get("feature_selection", "none")
        if method == "none" or X.shape[1] <= 1:
            self.selected_columns_ = X.columns.tolist()
            return X
        if method == "missing_ratio":
            selected = X.columns[X.isna().mean() <= self.missing_ratio_threshold].tolist()
            self.selected_columns_ = selected or [X.isna().mean().idxmin()]
            return X[self.selected_columns_]

        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical = [column for column in X.columns if column not in numeric]
        if len(numeric) <= 1:
            self.selected_columns_ = X.columns.tolist()
            return X
        if method == "collinear":
            if len(numeric) > self.max_collinear_features:
                correlation_gib = (len(numeric) ** 2 * 8) / (1024 ** 3)
                raise AutoDPResourceLimitError(
                    "collinear skipped: "
                    f"{len(numeric):,} numeric features exceed the safe limit "
                    f"of {self.max_collinear_features:,}; one float64 correlation "
                    f"matrix alone would require {correlation_gib:.2f} GiB"
                )
            model_X = self._numeric_detector_frame(X[numeric])
            correlation = model_X.corr().abs()
            upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
            removed = {column for column in upper.columns if (upper[column] > self.collinear_threshold).any()}
            selected_numeric = [column for column in numeric if column not in removed]
        elif y is None:
            selected_numeric = numeric
        elif method == "tree_based":
            model_X = self._numeric_detector_frame(X[numeric])
            if self.task_type == "regression":
                model = ExtraTreesRegressor(n_estimators=64, random_state=self.random_state, n_jobs=-1)
            else:
                model = ExtraTreesClassifier(n_estimators=64, random_state=self.random_state, n_jobs=-1, class_weight="balanced")
            model.fit(model_X, y)
            order = np.argsort(model.feature_importances_)[::-1]
            keep_count = min(20, max(1, int(np.ceil(len(numeric) / 2))))
            selected_numeric = [numeric[index] for index in order[:keep_count]]
        else:
            # RFE is a practical sklearn analogue to AutoDP's wrapper evaluator.
            # Cap to 100 high-variance features to keep 900x36 matrix runs tractable.
            model_X = self._numeric_detector_frame(X[numeric])
            candidate = numeric
            if len(candidate) > 100:
                candidate = model_X.var().sort_values(ascending=False).head(100).index.tolist()
            estimator = Ridge(alpha=1.0) if self.task_type == "regression" else LogisticRegression(
                max_iter=500, solver="liblinear", random_state=self.random_state
            )
            keep_count = min(20, max(1, int(np.ceil(len(candidate) / 2))))
            selector = RFE(estimator=estimator, n_features_to_select=keep_count, step=0.2)
            selector.fit(model_X[candidate], y)
            selected_numeric = [column for column, keep in zip(candidate, selector.support_) if keep]

        self.selected_columns_ = selected_numeric + categorical
        if not self.selected_columns_:
            self.selected_columns_ = [X.columns[0]]
        return X[self.selected_columns_]

    def _transform_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_columns_ is None:
            return X
        for column in self.selected_columns_:
            if column not in X:
                X[column] = np.nan
        return X[self.selected_columns_]


__all__ = [
    "AUTODP_OPTIONS",
    "DEFAULT_AUTODP_ORDER",
    "AUTODP_CLASSIFICATION_IDS",
    "AUTODP_REGRESSION_IDS",
    "AUTODP_60_IDS",
    "AutoDPResourceLimitError",
    "AutoDPPreprocessor",
    "autodp_space_size",
    "build_autodp_reference_pipelines",
    "exclude_holdout_columns",
    "validate_autodp_reference_pipelines",
]
