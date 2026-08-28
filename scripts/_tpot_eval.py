"""Shared estimator-only TPOT configuration for the downstream-AutoML comparison.

Two evaluators import from here so their TPOT settings CANNOT drift apart:

  * ``scripts/evaluate_acorec_tpot.py`` -- fits a frozen ACORec preprocessing pipeline on the
    fixed 60% train split, then estimator-only TPOT, and scores the untouched 20% test split.
  * ``scripts/evaluate_autodp_tpot.py`` -- reads an AutoDP-prepared frame (stage 2 of
    ``scripts/run_autodatapre.py``) and scores it with the SAME estimator-only TPOT, so the only
    thing that differs between the two numbers is the preprocessing that produced the frame.

"Estimator-only" means TPOT searches over classifiers / regressors only
(``tpot.config.get_search_space("classifiers" | "regressors")``), with ``preprocessing=False`` and
``validation_strategy="none"`` -- it never touches the validation split ACORec used while searching,
and it adds no preprocessing of its own on top of what is under test.

The knobs below are the ones the frozen ACORec TPOT run used
(``notebooks/run-acorec-tpot-kaggle.ipynb``): split seed 42, TPOT ``random_state`` 1 (a separate
knob), ``max_time_mins`` 5, ``max_eval_time_mins`` 1, ``population_size`` 20, ``early_stop`` 5,
``cv`` = ``min(5, smallest training-class count)``. The search space is sized with the
*transformed* ``n_features`` / ``n_samples`` (matching ``evaluate_acorec_tpot`` /
``evaluate_ctxpipe_tpot`` / the DiffPrep TPOT notebook; the standalone No-Preprocessing baseline
notebook sizes it with raw ``n_features`` and ``random_state`` 42 -- a pre-existing inconsistency
on that column alone). Changing a knob here changes both arms at once, which is the point.

Frames that a method's pipeline left with NaN or an object column are run through
``apply_minimal_adapter`` -- the identical train-fitted median/mode-impute + one-hot cleanup the
No-Preprocessing baseline is scored through. Without it TPOT (``preprocessing=False``) cannot
consume the frame at all, and a method that preprocessed nothing (AutoDP frequently) would be
silently dropped rather than scored at the no-preprocessing floor.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------------------------------- knobs (both arms)
TPOT_SPLIT_SEED = 42          # seed-42 0.6/0.2/0.2 split -- the split every number in our tables uses
TPOT_RANDOM_STATE = 1         # TPOT's own random_state; a SEPARATE knob from the split seed
TPOT_MAX_TIME_MINS = 5        # per-dataset TPOT wall budget, in MINUTES
TPOT_MAX_EVAL_TIME_MINS = 1   # per-candidate-pipeline cap, in MINUTES
TPOT_N_JOBS = 2
TPOT_MEMORY_LIMIT = "5GB"
TPOT_POPULATION_SIZE = 20
TPOT_MAX_CV_FOLDS = 5
TPOT_EARLY_STOP = 5


def knob_summary() -> dict:
    """The knob block, verbatim, for the output JSON so a number can be traced to its settings."""
    return {
        "split_seed": TPOT_SPLIT_SEED,
        "tpot_seed": TPOT_RANDOM_STATE,
        "max_time_mins": TPOT_MAX_TIME_MINS,
        "max_eval_time_mins": TPOT_MAX_EVAL_TIME_MINS,
        "n_jobs": TPOT_N_JOBS,
        "memory_limit": TPOT_MEMORY_LIMIT,
        "population_size": TPOT_POPULATION_SIZE,
        "max_cv_folds": TPOT_MAX_CV_FOLDS,
        "early_stop": TPOT_EARLY_STOP,
        "preprocessing": False,
        "validation_strategy": "none",
        "search_space": "classifiers | regressors (estimator-only), sized with transformed n_features/n_samples",
        "compat_adapter": "No-Preprocessing baseline adapter: train-fit median-impute numerics + "
                          "most_frequent-impute + one-hot categoricals; applied only to columns "
                          "the method's pipeline left dirty",
    }


# --------------------------------------------------------------------------- helpers
def normalize_task_type(problem_type: str) -> str:
    """``_detect_problem_type`` returns binary/multiclass/regression; TPOT wants the coarse label."""
    if problem_type in ("binary", "multiclass", "classification"):
        return "classification"
    if problem_type == "regression":
        return "regression"
    raise ValueError(f"Unsupported problem type {problem_type!r}")


def safe_cv_folds(y_train: pd.Series, task_type: str, maximum: int = TPOT_MAX_CV_FOLDS) -> int:
    """Largest CV fold count that every training class can support (regression: bounded by n rows).

    Fed the POST-preprocessing y_train: AutoDP's row deletions can shrink a class below the fold
    count, and ACORec operators can drop rows too.
    """
    maximum = max(2, int(maximum))
    if task_type == "regression":
        folds = min(maximum, len(y_train))
        if folds < 2:
            raise ValueError("TPOT regression CV requires at least two training rows")
        return folds
    counts = pd.Series(y_train).value_counts()
    if len(counts) < 2:
        raise ValueError("TPOT classification requires at least two training classes")
    folds = min(maximum, int(counts.min()))
    if folds < 2:
        raise ValueError("A processed training class has fewer than two rows for TPOT CV")
    return folds


def numeric_matrix(frame: Any, label: str) -> np.ndarray:
    """Coerce an ALREADY-numeric transformed block to a finite float32 matrix, or raise.

    Used only for the array / sparse path (e.g. a method's pipeline whose output is already a
    numeric matrix -- post-PCA, post-SVD). For DataFrame input with residual NaN or object columns,
    callers use ``apply_minimal_adapter`` instead: raising there would hold a method whose pipeline
    left a column dirty to a stricter protocol than the No-Preprocessing baseline, which is scored
    through exactly that adapter.
    """
    if isinstance(frame, pd.DataFrame):
        if any(isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes):
            frame = frame.sparse.to_dense()
        non_numeric = frame.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        if non_numeric:
            raise ValueError(f"{label} matrix still has non-numeric columns: {non_numeric[:10]}")
        values = frame.to_numpy(dtype=np.float32, copy=False)
    elif hasattr(frame, "toarray"):
        values = np.asarray(frame.toarray(), dtype=np.float32)
    else:
        values = np.asarray(frame, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{label} matrix has invalid shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} matrix contains NaN or infinity")
    return values


# ---------------------------------------------------------- the No-Preprocessing compat adapter
# Verbatim from scripts/build_tpot_baselines_notebook.py: the "unavoidable" train-fitted cleanup
# (median-impute numerics, most_frequent-impute + one-hot categoricals) that the No-Preprocessing
# TPOT baseline is scored through. TPOT gets `preprocessing=False`, so a frame with NaN or an
# object column is unscoreable without it. Applying the SAME adapter to every arm keeps a method
# whose pipeline left a column dirty on the same protocol as the baseline it is compared to,
# instead of silently dropping that dataset. It scales / selects / reduces nothing -- the method's
# own preprocessing is still the only thing under test.
def tpot_ready_raw_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Normalise categorical dtypes to str/object (NaN preserved). Learns nothing."""
    frame = frame.copy()
    categorical = list(frame.select_dtypes(exclude=[np.number]).columns)
    for column in categorical:
        missing = frame[column].isna()
        frame[column] = frame[column].astype(str).astype(object)
        frame.loc[missing, column] = np.nan
    return frame, categorical


def build_minimal_adapter(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric = list(x_train.select_dtypes(include=[np.number]).columns)
    categorical = [c for c in x_train.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                         dtype=np.float32)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop",
                             sparse_threshold=0.0, verbose_feature_names_out=False)


def apply_minimal_adapter(
    x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit the compat adapter on train, transform both. Returns (train_arr, test_arr, meta)."""
    tr, _ = tpot_ready_raw_frame(x_train)
    te, _ = tpot_ready_raw_frame(x_test)
    numeric = list(tr.select_dtypes(include=[np.number]).columns)
    categorical = [c for c in tr.columns if c not in numeric]
    nan_numeric = [c for c in numeric if tr[c].isna().any()]
    adapter = build_minimal_adapter(tr)
    train_arr = np.asarray(adapter.fit_transform(tr, y_train), dtype=np.float32)
    test_arr = np.asarray(adapter.transform(te), dtype=np.float32)
    meta = {
        "applied": bool(nan_numeric or categorical),
        "numeric_columns_imputed_median": nan_numeric,
        "categorical_columns_imputed_mode_and_onehot": categorical,
        "n_features_in": int(x_train.shape[1]),
        "n_features_out": int(train_arr.shape[1]),
    }
    return train_arr, test_arr, meta


def to_tpot_matrix(
    x_train: Any, x_test: Any, y_train: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """DataFrame -> compat adapter (median/mode impute + one-hot); array/sparse -> finite check.

    A method's pipeline may hand back a DataFrame (columns, maybe dirty) or an already-numeric
    matrix (post-PCA/SVD). Both arms route through here so their frame handling is identical.
    """
    if isinstance(x_train, pd.DataFrame):
        return apply_minimal_adapter(x_train, x_test, y_train)
    return (numeric_matrix(x_train, "training"), numeric_matrix(x_test, "test"),
            {"applied": False, "note": "already-numeric matrix; adapter not needed"})


def prune_rare_classes(
    x_train: pd.DataFrame, y_train: pd.Series, min_count: int = 2,
) -> Tuple[pd.DataFrame, pd.Series, List[Any]]:
    """Drop training rows whose class appears < min_count times (unusable for stratified CV).

    Triggers only when a method's pipeline deleted rows and pushed a class below the fold floor
    (AutoDP's outlier ops, ACORec's IQR/LOF). Records which classes so it is a footnote, not
    hidden. The test set is untouched, so the score stays comparable.
    """
    counts = pd.Series(y_train).value_counts()
    rare = counts[counts < int(min_count)].index.tolist()
    if not rare:
        return x_train, y_train, []
    keep = ~pd.Series(y_train).isin(rare).to_numpy()
    return (x_train.loc[keep].reset_index(drop=True),
            pd.Series(y_train).loc[keep].reset_index(drop=True), rare)


def default_tpot_components(task_type: str):
    try:
        import tpot as _tpot
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("TPOT 1.1.0 is required for the TPOT evaluator (no `tpot` importable)") from exc
    try:
        from tpot import TPOTClassifier, TPOTRegressor
        from tpot.config import get_search_space
    except Exception as exc:  # pragma: no cover - optional dependency
        # The pre-1.0 `tpot` (0.12.x, still shipped in Kaggle's base image) has neither
        # `tpot.config.get_search_space` nor the `search_space=` API this evaluator is built on.
        raise RuntimeError(
            f"found tpot {getattr(_tpot, '__version__', '?')}, but this evaluator needs the "
            f"1.x API (`tpot.config.get_search_space`). Install TPOT==1.1.0 into an environment "
            f"that shadows the base one -- see requirements-tpot-kaggle.txt."
        ) from exc
    estimator = TPOTClassifier if task_type == "classification" else TPOTRegressor
    return estimator, get_search_space


def build_model(
    task_type: str,
    *,
    n_samples: int,
    n_features: int,
    n_classes: int,
    cv_folds: int,
    tpot_seed: int = TPOT_RANDOM_STATE,
    max_time_mins: int = TPOT_MAX_TIME_MINS,
    max_eval_time_mins: int = TPOT_MAX_EVAL_TIME_MINS,
    n_jobs: int = TPOT_N_JOBS,
    memory_limit: str = TPOT_MEMORY_LIMIT,
    population_size: int = TPOT_POPULATION_SIZE,
    verbose: int = 2,
    estimator_factory: Optional[Callable[..., Any]] = None,
    search_space_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[Any, str]:
    """Return an UNFITTED estimator-only TPOT model plus its search-space group name.

    The single place the TPOT estimator is configured -- both evaluators call this so the settings
    are shared by construction.
    """
    if estimator_factory is None or search_space_factory is None:
        default_estimator, default_search_space = default_tpot_components(task_type)
        estimator_factory = estimator_factory or default_estimator
        search_space_factory = search_space_factory or default_search_space

    group = "classifiers" if task_type == "classification" else "regressors"
    search_space = search_space_factory(
        group,
        n_classes=int(n_classes) if task_type == "classification" else 1,
        n_samples=int(n_samples),
        n_features=int(n_features),
        random_state=int(tpot_seed),
        n_jobs=1,
    )
    primary_metric = "accuracy" if task_type == "classification" else "r2"
    model = estimator_factory(
        search_space=search_space,
        scorers=[primary_metric],
        scorers_weights=[1],
        cv=int(cv_folds),
        preprocessing=False,
        max_time_mins=int(max_time_mins),
        max_eval_time_mins=int(max_eval_time_mins),
        n_jobs=int(n_jobs),
        memory_limit=str(memory_limit),
        validation_strategy="none",
        early_stop=TPOT_EARLY_STOP,
        verbose=int(verbose),
        random_state=int(tpot_seed),
        population_size=int(population_size),
        initial_population_size=int(population_size),
    )
    return model, group


# Back-compat aliases: the frozen ACORec evaluator referenced these private names.
_safe_cv_folds = safe_cv_folds
_numeric_matrix = numeric_matrix
_default_tpot_components = default_tpot_components
