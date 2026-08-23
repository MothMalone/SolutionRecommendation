#!/usr/bin/env python3
"""Shared AutoGluon split evaluator for the Kaggle solution notebooks."""
from __future__ import annotations

import gc
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)


def _load_components():
    try:
        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import IdentityFeatureGenerator
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("AutoGluon Tabular is required for this evaluator") from exc
    return TabularPredictor, IdentityFeatureGenerator


def _infer_task(y: pd.Series, task_type: Optional[str]) -> tuple[str, str]:
    n_classes = int(y.nunique(dropna=True))
    if task_type == "regression":
        return "regression", "root_mean_squared_error"
    if task_type == "classification":
        return ("binary" if n_classes <= 2 else "multiclass"), "accuracy"
    if y.dtype.kind in "OUSb" or n_classes <= 50:
        return ("binary" if n_classes <= 2 else "multiclass"), "accuracy"
    return "regression", "root_mean_squared_error"


def _frame(X: Any, y: Any, label: str) -> pd.DataFrame:
    frame = pd.DataFrame(X).reset_index(drop=True).copy()
    if label in frame.columns:
        raise ValueError(f"Feature frame already contains label column {label!r}")
    frame[label] = pd.Series(y).reset_index(drop=True).to_numpy()
    return frame


def _is_xgb_compatibility_error(error: Exception) -> bool:
    message = str(error).lower()
    return "xgbclassifier" in message and "n_classes_" in message


def evaluate_autogluon_split(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    X_test: Any,
    y_test: Any,
    *,
    task_type: Optional[str] = None,
    feature_generator: str = "identity",
    time_limit: int = 300,
    presets: str = "best_quality",
    seed: int = 42,
    label: str = "__target__",
    path_root: Optional[str | Path] = None,
    verbosity: int = 0,
) -> dict[str, Any]:
    """Fit AutoGluon on train, use validation for tuning, and score test once.

    ``X_*`` and ``y_*`` must already come from the repository's shared
    ``split_train_val_test`` function. The outer test is never supplied to
    AutoGluon until final prediction.
    """
    if feature_generator not in {"identity", "default"}:
        raise ValueError("feature_generator must be 'identity' or 'default'")
    TabularPredictor, IdentityFeatureGenerator = _load_components()
    random.seed(int(seed))
    np.random.seed(int(seed))
    train_y = pd.Series(y_train).reset_index(drop=True)
    val_y = pd.Series(y_val).reset_index(drop=True)
    test_y = pd.Series(y_test).reset_index(drop=True)
    train_data = _frame(X_train, train_y, label)
    val_data = _frame(X_val, val_y, label)
    test_data = _frame(X_test, test_y, label)
    resolved_problem_type, eval_metric = _infer_task(train_y, task_type)
    is_classification = resolved_problem_type in {"binary", "multiclass"}

    if path_root is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="autogluon_eval_"))
        remove_dir = True
    else:
        root = Path(path_root)
        root.mkdir(parents=True, exist_ok=True)
        temp_dir = root / f"{resolved_problem_type}_{os.getpid()}_{time.time_ns()}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        remove_dir = True

    started = time.perf_counter()
    fit_started = time.perf_counter()
    predictor = None
    predictor_dirs = [temp_dir]
    model_names: list[str] = []
    retry_without_xgb = False
    try:
        predictor = TabularPredictor(
            label=label,
            path=str(temp_dir),
            problem_type=resolved_problem_type,
            eval_metric=eval_metric,
            verbosity=int(verbosity),
        )
        fit_kwargs: dict[str, Any] = {
            "train_data": train_data,
            "tuning_data": val_data,
            "time_limit": int(time_limit),
            "presets": str(presets),
            "raise_on_no_models_fitted": False,
        }
        if feature_generator == "identity":
            fit_kwargs["feature_generator"] = IdentityFeatureGenerator()
        try:
            predictor.fit(**fit_kwargs)
        except Exception as error:
            if not _is_xgb_compatibility_error(error):
                raise
            retry_without_xgb = True
            fit_kwargs["excluded_model_types"] = ["XGB"]
            # A failed fit can leave partially-written learner state behind.
            # Retry in a clean directory instead of reusing that state.
            retry_dir = temp_dir.with_name(temp_dir.name + "_retry_no_xgb")
            shutil.rmtree(retry_dir, ignore_errors=True)
            retry_dir.mkdir(parents=True, exist_ok=False)
            predictor_dirs.append(retry_dir)
            predictor = TabularPredictor(
                label=label,
                path=str(retry_dir),
                problem_type=resolved_problem_type,
                eval_metric=eval_metric,
                verbosity=int(verbosity),
            )
            predictor.fit(**fit_kwargs)
        fit_seconds = float(time.perf_counter() - fit_started)
        try:
            model_names = list(predictor.model_names() or [])
        except Exception:
            model_names = []
        if not model_names:
            raise RuntimeError("AutoGluon fitted no models")

        prediction_started = time.perf_counter()
        val_prediction = predictor.predict(val_data.drop(columns=[label]))
        test_prediction = predictor.predict(test_data.drop(columns=[label]))
        prediction_seconds = float(time.perf_counter() - prediction_started)
        if is_classification:
            val_score = float(accuracy_score(val_y, val_prediction))
            test_score = float(accuracy_score(test_y, test_prediction))
            result = {
                "accuracy": test_score,
                "balanced_accuracy": float(balanced_accuracy_score(test_y, test_prediction)),
                "f1_macro": float(f1_score(test_y, test_prediction, average="macro", zero_division=0)),
                "score": test_score,
                "validation_score": val_score,
            }
        else:
            val_score = float(r2_score(val_y, val_prediction))
            test_score = float(r2_score(test_y, test_prediction))
            result = {
                "r2": test_score,
                "rmse": float(mean_squared_error(test_y, test_prediction) ** 0.5),
                "score": test_score,
                "validation_score": val_score,
            }
        result.update(
            {
                "status": "ok",
                "evaluator": "AutoGluon.TabularPredictor",
                "task_type": "classification" if is_classification else "regression",
                "autogluon_problem_type": resolved_problem_type,
                "primary_metric": "accuracy" if is_classification else "r2",
                "feature_generator": feature_generator,
                "autogluon_presets": str(presets),
                "time_limit": int(time_limit),
                "seed": int(seed),
                "train_rows": int(len(train_y)),
                "validation_rows": int(len(val_y)),
                "test_rows": int(len(test_y)),
                "raw_features": int(pd.DataFrame(X_train).shape[1]),
                "fit_seconds": fit_seconds,
                "prediction_seconds": prediction_seconds,
                "total_seconds": float(time.perf_counter() - started),
                "model_count": int(len(model_names)),
                "model_names": model_names,
                "selected_model": model_names[0] if model_names else None,
                "retry_without_xgb": bool(retry_without_xgb),
                "outer_test_seen_during_training_or_selection": False,
            }
        )
        return result
    finally:
        del predictor
        if remove_dir:
            for predictor_dir in predictor_dirs:
                shutil.rmtree(predictor_dir, ignore_errors=True)
        gc.collect()
