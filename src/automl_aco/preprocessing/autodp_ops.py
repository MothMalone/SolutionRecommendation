"""AutoDP's operator space, reimplemented under ACORec's fit/transform discipline.

Purpose: make "same operator space" a real claim rather than a family-level approximation. These are
AutoDP's 24 operators (minus 2, see below) re-expressed as ordinary fit-on-train / transform-test
steps, so ACORec can search *their* space and AutoDP can be run over *ours* with the two sides
meaning the same thing by a pipeline.

Codes are UPPERCASE and ACORec's are lowercase, so the two spaces can share every builder without
collision and a config is self-describing.

===============================================================================================
CHARITABLE REIMPLEMENTATION -- the complete deviation table
===============================================================================================
The rule applied throughout: **keep their parameters, fix their defects.** Where their released
code does something that is clearly a bug rather than a design choice, the bug is removed. Every
such change makes their operator stronger, so it biases against our own result -- which is the
safe direction, and it means a win for ACORec cannot be attributed to their implementation errors.

| code | their released behaviour | here | why |
|------|--------------------------|------|-----|
| MEAN   | `fillna(int(col.mean()))` -- fill value truncated to integer | true mean | truncation is a bug; on a [0,1]-scaled column it fills 0 everywhere |
| MEDIAN | same `int()` truncation | true median | as above |
| MF     | most-frequent, all columns | most-frequent | unchanged |
| KNN    | `impyute.fast_knn(X, 4)` | `KNNImputer(n_neighbors=4)` | same method, k kept at 4; impyute has no fit/transform split |
| MICE   | `impyute.mice(X)` | `IterativeImputer(random_state=42)` | IterativeImputer *is* MICE; seeded for reproducibility |
| RAND   | overwrite with `np.random.randn` unseeded | seeded `RandomState(42)` | reproducibility only |
| DROP   | drops NaN rows from train AND test | drops from train only | deleting test rows changes what the score measures |
| OE/BE/FE/CBE | fit on `concat(train, test)`; CBE also on `concat(target, target_test)` | fit on train only | the leak this whole arm exists to remove |
| ZS     | per-split mean/std; constant column set to `1` | train mean/std; constant -> 0 | per-split fitting erases covariate shift; `1` is arbitrary |
| MM/DS  | scaler re-instantiated per split | fitted on train | as above |
| MR/LC  | unsupervised column drops, computed on train | unchanged, computed on train | already correct |
| WR     | `SelectKBest(chi2, k=10)`; negativity guard loses columns via `del` while indexing, and skips the last column | same chi2/k=10, guard rewritten correctly | the guard is a bug, not a design |
| TB     | `ExtraTreesClassifier(n_estimators=10)` unseeded | seeded `random_state=42` | reproducibility only |
| WR/TB  | `select_dtypes(['number'])` deletes every categorical column | categoricals preserved | see NOTE below -- this one is deliberate and must be disclosed |
| ZSB    | modified z-score, threshold 1.6, MAD constant 1.4296 | threshold 1.6 kept, constant 1.4826 | 1.4296 is a typo for the standard constant; the *threshold* is theirs and is kept |
| IQR    | 1.5*IQR; row order scrambled by `set()` | 1.5*IQR, order preserved | ordering is a bug that misaligns labels |
| LOF    | deletes a FIXED `int(threshold*100)` = 30 rows regardless of dataset size; wipes a split under 30 rows | `contamination=0.1`, n_neighbors=4 | fixed count is a bug; their n_neighbors and 0.1 contamination are kept |
| ED     | exact duplicate removal, train and test | train only | as with DROP |
| **EM** | `impyute.em` | **DROPPED** | no fit/transform equivalent exists; hand-rolling a Gaussian EM would be a different operator wearing their label |
| **AD** | approximate string dedup via `py-stringsimjoin` | **DROPPED** | the dependency has no wheel for python >= 3.10 and its setup.py imports pip |

Result: **22 of their 24 operators**. Report that number, and the two omissions, in any table.

NOTE on WR/TB and categoricals. Their `FS_WR_identify_best_subset` / `FS_Tree_based` start with
`select_dtypes(['number'])` and return only those columns, and `transform()` then propagates that
column list to the test split -- so choosing WR or TB **deletes every categorical feature**. That is
behaviour, not a defect, and a faithful port would keep it. It is not kept here, for a structural
reason: ACORec searches step order, so feature_selection may legally run before encoding, where the
same code would silently destroy the categorical block for reasons of ordering rather than merit. We
preserve categoricals instead. This is the one deviation that is *not* a bug fix, and it is
charitable to them (their version discards information). Disclose it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2

try:
    import category_encoders as ce
except Exception:  # pragma: no cover - optional dependency
    ce = None


# --------------------------------------------------------------------------------------------
# The space
# --------------------------------------------------------------------------------------------
IMPUTATION: List[str] = ["MEAN", "MEDIAN", "MF", "KNN", "MICE", "RAND", "DROP"]   # EM dropped
ENCODING: List[str] = ["OE", "BE", "FE", "CBE"]
SCALING: List[str] = ["ZS", "MM", "DS"]
FEATURE_SELECTION: List[str] = ["MR", "WR", "LC", "TB"]
OUTLIER_REMOVAL: List[str] = ["ZSB", "IQR", "LOF"]
DUPLICATE_REMOVAL: List[str] = ["ED"]                                            # AD dropped

DROPPED = {
    "EM": "impyute.em has no fit/transform equivalent; a hand-rolled Gaussian EM would be a "
          "different operator under their label",
    "AD": "py-stringsimjoin has no wheel for python >= 3.10 and its setup.py imports pip",
}

ALL_CODES = set(
    IMPUTATION + ENCODING + SCALING + FEATURE_SELECTION + OUTLIER_REMOVAL + DUPLICATE_REMOVAL
)

#: Operators that drop rows. They run on train only; ``transform`` is a no-op for them.
ROW_DROPPING = {"DROP", "ED"} | set(OUTLIER_REMOVAL)


def is_autodp_op(name: Any) -> bool:
    """True if ``name`` is one of AutoDP's operator codes (uppercase namespace)."""
    return isinstance(name, str) and name in ALL_CODES


# --------------------------------------------------------------------------------------------
# Imputation
# --------------------------------------------------------------------------------------------
class RandomNormalImputer:
    """AutoDP's ``RAND``: replace missing cells with draws from a standard normal.

    Their `NaN_random_replace` overwrites via `DataFrame.update` with an unseeded `randn` frame.
    Seeded here; otherwise identical, including the fact that it ignores the column's own scale.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.n_features_in_: Optional[int] = None

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float).copy()
        rng = np.random.RandomState(self.seed)
        mask = np.isnan(arr)
        if mask.any():
            arr[mask] = rng.randn(int(mask.sum()))
        return arr

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def build_imputers(method: str, X_num, X_cat) -> Tuple[Optional[Any], Optional[Any]]:
    """Numeric/categorical imputer pair for one of AutoDP's imputation codes.

    ``DROP`` returns ``(None, None)``: it deletes rows rather than filling, and is handled by the
    row-dropping path, not here.
    """
    if method == "DROP":
        # NECESSARY DEVIATION. Their DROP deletes incomplete rows from train AND test. Deleting
        # test rows is not available to us -- it changes what the score measures, and there is no
        # leak-free counterpart at inference time (you cannot refuse to predict a row). So: rows
        # are dropped from TRAIN (done in Preprocessor._fit_row_drop_imputation) and the test
        # split is filled from the surviving TRAINING rows' statistics. Without this the test
        # frame keeps NaNs that every downstream operator then trips over.
        return (
            SimpleImputer(strategy="median") if X_num is not None else None,
            SimpleImputer(strategy="most_frequent") if X_cat is not None else None,
        )

    num_imputer = None
    if X_num is not None:
        if method == "MEAN":
            num_imputer = SimpleImputer(strategy="mean")
        elif method == "MEDIAN":
            num_imputer = SimpleImputer(strategy="median")
        elif method == "MF":
            num_imputer = SimpleImputer(strategy="most_frequent")
        elif method == "KNN":
            n = max(1, min(4, len(X_num) - 1))
            num_imputer = KNNImputer(n_neighbors=n)
        elif method == "MICE":
            num_imputer = IterativeImputer(random_state=42, max_iter=10)
        elif method == "RAND":
            num_imputer = RandomNormalImputer(seed=42)
        else:
            raise ValueError(f"Unknown AutoDP imputation code: {method}")

    # Their numeric imputers leave object columns untouched; MF is the only one that covers them.
    # Categoricals still have to be filled or downstream encoders fail, so most_frequent is used --
    # the same choice ACORec makes, keeping the two spaces comparable on this incidental point.
    cat_imputer = SimpleImputer(strategy="most_frequent") if X_cat is not None else None
    return num_imputer, cat_imputer


def missing_row_mask(X_num, X_cat) -> pd.Series:
    """Rows to KEEP for ``DROP`` (their `NaN_drop`), computed across both blocks."""
    frames = [f for f in (X_num, X_cat) if f is not None]
    if not frames:
        return pd.Series(dtype=bool)
    joined = pd.concat(frames, axis=1)
    return joined.notna().all(axis=1)


# --------------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------------
def build_encoder(method: str) -> Optional[Any]:
    """One of AutoDP's four encoders, fitted on train only (they fit on train+test)."""
    if ce is None:
        raise RuntimeError(
            "category_encoders is required for AutoDP's encoders (BE/FE/CBE/OE). "
            "It is also an undeclared runtime dependency of autodatapre itself."
        )
    if method == "OE":
        return ce.OrdinalEncoder(handle_unknown="value", handle_missing="value")
    if method == "BE":
        return ce.BinaryEncoder(handle_unknown="value", handle_missing="value")
    if method == "FE":
        return ce.CountEncoder(normalize=True, handle_unknown=0, handle_missing="count")
    if method == "CBE":
        # Supervised. Fitted on the TRAINING target only -- this is the leak removal.
        return ce.CatBoostEncoder(handle_unknown="value", handle_missing="value")
    raise ValueError(f"Unknown AutoDP encoding code: {method}")


#: Encoders needing ``y`` at fit time.
SUPERVISED_ENCODERS = {"CBE"}


# --------------------------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------------------------
def build_scaler(method: str) -> Optional[Any]:
    if method == "ZS":
        return StandardScaler()
    if method == "MM":
        return MinMaxScaler()
    if method == "DS":
        # Their DS is `quantile_transform(X, n_quantiles=10, random_state=0)`; n_quantiles kept.
        return QuantileTransformer(n_quantiles=10, random_state=0)
    raise ValueError(f"Unknown AutoDP scaling code: {method}")


# --------------------------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------------------------
class _MaskSelector:
    """sklearn-shaped selector over a fixed boolean column mask.

    Exposes ``get_support`` / ``transform`` so it drops straight into ``Preprocessor``'s existing
    feature-selection plumbing.
    """

    def __init__(self, mask: np.ndarray, columns: pd.Index):
        if not mask.any():  # never hand back an empty frame
            mask = np.ones(len(mask), dtype=bool)
        self.mask = mask
        self.columns = columns

    def get_support(self, indices: bool = False):
        return np.where(self.mask)[0] if indices else self.mask

    def transform(self, X):
        arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return arr[:, self.mask]


def _drop_incomplete_rows(X: pd.DataFrame, y):
    """Their supervised selectors call ``df_train.dropna()`` before fitting; mirrored here."""
    keep = X.notna().all(axis=1)
    if bool(keep.all()):
        return X, y
    Xc = X.loc[keep]
    yc = None if y is None else pd.Series(np.asarray(y).ravel())[keep.values]
    return Xc, yc


def build_selector(method: str, X: pd.DataFrame, y: Optional[pd.Series]) -> Optional[Any]:
    """One of AutoDP's four selectors, fitted on the training block only."""
    cols = X.columns
    n = X.shape[1]

    if method == "MR":
        # FS_MR_missing_ratio: drop columns whose missing fraction exceeds 0.2.
        frac = X.isnull().sum() / max(1, X.shape[0])
        return _MaskSelector((frac <= 0.2).values, cols)

    if method == "LC":
        # FS_LC_identify_collinear: drop columns correlated above 0.8 with an earlier column.
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = {c for c in upper.columns if (upper[c] > 0.8).any()}
        return _MaskSelector(np.array([c not in drop for c in cols]), cols)

    if method == "TB":
        # Their FS_Tree_based does `df_train.dropna()` first; mirrored so the operator stays
        # usable when it is ordered before imputation.
        Xc, yc = _drop_incomplete_rows(X, y)
        if Xc.empty:
            return _MaskSelector(np.ones(n, dtype=bool), cols)
        clf = ExtraTreesClassifier(n_estimators=10, random_state=42)
        clf.fit(Xc, np.asarray(yc).ravel())
        importances = clf.feature_importances_
        return _MaskSelector(importances >= importances.mean(), cols)

    if method == "WR":
        # SelectKBest(chi2, k=10). chi2 needs non-negative inputs, so columns containing negatives
        # are excluded first -- their guard for this is buggy (deletes while indexing, skips the
        # last column); rewritten correctly here.
        # Their FS_WR_identify_best_subset does `df_train.dropna()` first; mirrored.
        Xc, yc = _drop_incomplete_rows(X, y)
        if Xc.empty:
            return _MaskSelector(np.ones(n, dtype=bool), cols)
        nonneg = np.array([bool((Xc[c] >= 0).all()) for c in cols])
        if not nonneg.any():
            return _MaskSelector(np.ones(n, dtype=bool), cols)
        X_ok = Xc.loc[:, nonneg]
        k = min(10, X_ok.shape[1])
        sel = SelectKBest(chi2, k=k)
        sel.fit(X_ok, np.asarray(yc).ravel())
        keep_ok = sel.get_support()
        mask = np.zeros(n, dtype=bool)
        mask[np.where(nonneg)[0][keep_ok]] = True
        return _MaskSelector(mask, cols)

    raise ValueError(f"Unknown AutoDP feature_selection code: {method}")


# --------------------------------------------------------------------------------------------
# Outlier removal (train rows only)
# --------------------------------------------------------------------------------------------
def outlier_keep_mask(X_num: pd.DataFrame, method: str) -> pd.Series:
    """Rows to KEEP. Their thresholds are preserved; their defects are not (see the table)."""
    if method == "ZSB":
        # Modified z-score. Threshold 1.6 is theirs and is deliberately kept -- it is aggressive
        # (deletes ~8% of clean gaussian noise) but it is their design point, not an error.
        median = X_num.median(axis=0)
        mad = 1.4826 * (X_num - median).abs().median(axis=0)
        mad = mad.replace(0, np.nan)
        z = (X_num - median).abs() / mad
        return (z.fillna(0) <= 1.6).all(axis=1)

    if method == "IQR":
        q1 = X_num.quantile(0.25)
        q3 = X_num.quantile(0.75)
        iqr = q3 - q1
        keep = pd.Series(True, index=X_num.index)
        for c in X_num.columns:
            if iqr[c] > 0:
                keep &= (X_num[c] >= q1[c] - 1.5 * iqr[c]) & (X_num[c] <= q3[c] + 1.5 * iqr[c])
        return keep

    if method == "LOF":
        # Their LOF_outlier_detection does `dataset.dropna()` first; rows with missing values are
        # therefore not scored. They are kept here rather than deleted, since deleting on the basis
        # of missingness is imputation's job, not the outlier detector's.
        complete = X_num.notna().all(axis=1)
        keep = pd.Series(True, index=X_num.index)
        if complete.sum() < 3:
            return keep
        sub = X_num.loc[complete]
        n = max(2, min(4, len(sub) - 1))  # their n_neighbors=4
        lof = LocalOutlierFactor(n_neighbors=n, contamination=0.1)  # their contamination=0.1
        keep.loc[complete] = lof.fit_predict(sub) == 1
        return keep

    raise ValueError(f"Unknown AutoDP outlier_removal code: {method}")


# --------------------------------------------------------------------------------------------
# Duplicate removal (train rows only)
# --------------------------------------------------------------------------------------------
def duplicate_keep_mask(X_num, X_cat, method: str) -> pd.Series:
    """Rows to KEEP for ``ED`` (exact duplicate removal), keeping the first occurrence."""
    if method != "ED":
        raise ValueError(f"Unknown AutoDP duplicate_removal code: {method}")
    frames = [f for f in (X_num, X_cat) if f is not None]
    if not frames:
        return pd.Series(dtype=bool)
    joined = pd.concat(frames, axis=1)
    return ~joined.duplicated(keep="first")
