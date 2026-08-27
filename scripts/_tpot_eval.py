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
(``notebooks/run-acorec-tpot-kaggle.ipynb``). Changing one here changes both arms at once, which is
the point.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd


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
        "search_space": "classifiers | regressors (estimator-only)",
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
    """Coerce a transformed frame to a finite float32 matrix, or raise loudly.

    TPOT with ``preprocessing=False`` hands the frame straight to sklearn estimators, so a NaN or a
    leftover object column is a hard failure here -- unlike AutoGluon / H2O, which tolerate both.
    That is deliberate: imputing or encoding at this point would be downstream preprocessing, which
    this protocol forbids. The caller turns the raised error into a ``status: "failed"`` row.
    """
    if isinstance(frame, pd.DataFrame):
        if any(isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes):
            frame = frame.sparse.to_dense()
        non_numeric = frame.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"{label} matrix still has non-numeric columns after residual encoding: "
                f"{non_numeric[:10]}"
            )
        values = frame.to_numpy(dtype=np.float32, copy=False)
    elif hasattr(frame, "toarray"):
        values = np.asarray(frame.toarray(), dtype=np.float32)
    else:
        values = np.asarray(frame, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{label} matrix has invalid shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(
            f"{label} matrix contains NaN or infinity -- the frame under test left values TPOT "
            f"cannot consume, and imputing them here would be downstream preprocessing"
        )
    return values


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
