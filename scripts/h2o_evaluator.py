"""Shared H2O AutoML evaluator for the experiment notebooks.

The evaluator receives already-split data. It never uses the outer test frame
to choose a model: models are ranked by validation performance and the chosen
model is scored on test exactly once.
"""
from __future__ import annotations

import gc
import math
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)


def _load_h2o():
    try:
        import h2o
        from h2o.automl import H2OAutoML
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "H2O is required for this evaluator. Install it with `pip install h2o`."
        ) from exc
    return h2o, H2OAutoML


def _shutdown_h2o(h2o: Any) -> None:
    """Release the H2O JVM and frames after one isolated evaluation."""
    try:
        h2o.remove_all()
    finally:
        try:
            h2o.cluster().shutdown(prompt=False)
        except Exception:
            # The cluster may already have exited after a JVM error.
            pass
        gc.collect()


def _init_h2o(h2o: Any, *, nthreads: int, max_mem_size: str) -> None:
    """Start H2O with arguments supported by the pinned 3.46.x client."""
    # H2O 3.46.x removed the legacy ``silent`` keyword from ``h2o.init``.
    # Passing it through raises H2OTypeError before the JVM starts.
    h2o.init(nthreads=int(nthreads), max_mem_size=str(max_mem_size))
    h2o.remove_all()


def _as_frame(value: Any, *, columns: Optional[list[str]] = None) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif hasattr(value, "toarray"):
        frame = pd.DataFrame(value.toarray())
    else:
        frame = pd.DataFrame(np.asarray(value))
    if columns is not None:
        frame.columns = columns
    else:
        frame.columns = [str(column) for column in frame.columns]
    return frame.reset_index(drop=True)


def _prediction_from_frame(h2o_frame: Any) -> np.ndarray:
    prediction = h2o_frame.as_data_frame(use_pandas=True)
    if "predict" in prediction.columns:
        return prediction["predict"].to_numpy()
    return prediction.iloc[:, 0].to_numpy()


def _classification_labels(values: Any) -> np.ndarray:
    """Compare H2O factor predictions and targets using one stable dtype.

    H2O may return a binary factor prediction as numeric ``0``/``1`` even
    when its factor domain originated from the string labels ``"0"``/``"1"``.
    sklearn deliberately rejects mixed label types, so canonicalize both
    arrays only at metric-computation time.
    """
    return pd.Series(values).astype(str).to_numpy()


def _categorical_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return pandas categorical columns whose H2O type must be consistent."""
    return [
        str(column)
        for column in frame.select_dtypes(include=["object", "category", "bool"]).columns
    ]


def _force_h2o_categorical_columns(frames: tuple[Any, ...], columns: list[str]) -> None:
    """Prevent H2O from inferring different feature types per split."""
    for frame in frames:
        for column in columns:
            if column in frame.names:
                frame[column] = frame[column].asfactor()


def _metric_from_model(model: Any, validation_frame: Any, task_type: str) -> float:
    performance = model.model_performance(validation_frame)
    if task_type == "classification":
        # Accuracy is not always the leaderboard's default metric in H2O, so
        # compute it explicitly for every candidate model on validation.
        pred = _classification_labels(_prediction_from_frame(model.predict(validation_frame)))
        actual = _classification_labels(
            validation_frame["__target__"].as_data_frame(use_pandas=True).iloc[:, 0].to_numpy()
        )
        return float(accuracy_score(actual, pred))
    value = performance.r2()
    return float(value) if value is not None and np.isfinite(value) else float("-inf")


def _select_model_by_validation(aml: Any, validation_frame: Any, task_type: str):
    leaderboard = aml.leaderboard.as_data_frame(use_pandas=True)
    if "model_id" not in leaderboard.columns or leaderboard.empty:
        raise RuntimeError("H2O AutoML produced an empty leaderboard")
    best_model = None
    best_score = float("-inf")
    scores = []
    for model_id in leaderboard["model_id"].tolist():
        model = __import__("h2o").get_model(model_id)
        try:
            score = _metric_from_model(model, validation_frame, task_type)
        except Exception as exc:
            scores.append({"model_id": str(model_id), "validation_score": None, "error": str(exc)})
            continue
        scores.append({"model_id": str(model_id), "validation_score": score})
        if score > best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("H2O could not score any leaderboard model on validation")
    return best_model, float(best_score), scores


def evaluate_h2o_frames(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    X_test: Any,
    y_test: Any,
    *,
    task_type: str = "classification",
    h2o_preprocessing: Optional[str] = None,
    max_runtime_secs: int = 300,
    max_runtime_secs_per_model: int = 60,
    max_models: Optional[int] = None,
    nfolds: int = 5,
    seed: int = 42,
    nthreads: int = 1,
    max_mem_size: str = "6G",
    include_algos: Optional[list[str]] = None,
    keep_h2o_alive: bool = False,
) -> Tuple[Dict[str, Any], Any]:
    """Run H2O AutoML and return metrics plus the selected H2O model."""
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task_type: {task_type!r}")
    h2o, H2OAutoML = _load_h2o()
    _init_h2o(h2o, nthreads=nthreads, max_mem_size=max_mem_size)

    train_x = _as_frame(X_train)
    val_x = _as_frame(X_val, columns=list(train_x.columns))
    test_x = _as_frame(X_test, columns=list(train_x.columns))
    train_y = pd.Series(y_train).reset_index(drop=True)
    val_y = pd.Series(y_val).reset_index(drop=True)
    test_y = pd.Series(y_test).reset_index(drop=True)
    if len(train_x) != len(train_y) or len(val_x) != len(val_y) or len(test_x) != len(test_y):
        raise ValueError("H2O evaluator received mismatched X/y lengths")

    train_pd = train_x.copy()
    val_pd = val_x.copy()
    test_pd = test_x.copy()
    if task_type == "classification":
        # H2O factor predictions are returned using the factor labels. Keep
        # the same string representation for validation/test metric scoring.
        train_target = train_y.astype(str)
        val_target = val_y.astype(str)
        test_target = test_y.astype(str)
    else:
        train_target, val_target, test_target = train_y, val_y, test_y
    train_pd["__target__"] = train_target.to_numpy()
    val_pd["__target__"] = val_target.to_numpy()
    test_pd["__target__"] = test_target.to_numpy()
    train_frame = h2o.H2OFrame(train_pd)
    val_frame = h2o.H2OFrame(val_pd)
    test_frame = h2o.H2OFrame(test_pd)
    categorical_predictors = _categorical_feature_columns(train_x)
    _force_h2o_categorical_columns(
        (train_frame, val_frame, test_frame), categorical_predictors
    )
    train_frame["__target__"] = train_frame["__target__"].asnumeric() if task_type == "regression" else train_frame["__target__"].asfactor()
    val_frame["__target__"] = val_frame["__target__"].asnumeric() if task_type == "regression" else val_frame["__target__"].asfactor()
    test_frame["__target__"] = test_frame["__target__"].asnumeric() if task_type == "regression" else test_frame["__target__"].asfactor()
    predictors = [column for column in train_frame.names if column != "__target__"]
    if not predictors:
        raise ValueError("H2O evaluator received zero predictor columns")

    preprocessing = ["target_encoding"] if h2o_preprocessing == "target_encoding" else None
    kwargs: Dict[str, Any] = {
        "max_runtime_secs": int(max_runtime_secs),
        "max_runtime_secs_per_model": int(max_runtime_secs_per_model),
        "nfolds": int(nfolds),
        "seed": int(seed),
        "preprocessing": preprocessing,
        "verbosity": "warn",
    }
    if max_models is not None:
        kwargs["max_models"] = int(max_models)
    if include_algos:
        kwargs["include_algos"] = list(include_algos)

    started = time.perf_counter()
    try:
        aml = H2OAutoML(**kwargs)
        aml.train(
            x=predictors,
            y="__target__",
            training_frame=train_frame,
            leaderboard_frame=val_frame,
        )
        selected_model, validation_score, model_scores = _select_model_by_validation(
            aml, val_frame, task_type
        )
        test_prediction = _prediction_from_frame(selected_model.predict(test_frame))
        fit_seconds = float(time.perf_counter() - started)
    except Exception:
        if not keep_h2o_alive:
            _shutdown_h2o(h2o)
        raise

    result: Dict[str, Any] = {
        "status": "ok",
        "evaluator": "H2OAutoML",
        "task_type": task_type,
        "h2o_preprocessing": h2o_preprocessing or "none",
        "validation_score": float(validation_score),
        "validation_used_for_model_selection": True,
        "outer_test_seen_during_selection": False,
        "model_count": int(len(model_scores)),
        "selected_model_id": str(selected_model.model_id),
        "selected_model_algo": str(getattr(selected_model, "algo", "unknown")),
        "train_rows": int(len(train_y)),
        "validation_rows": int(len(val_y)),
        "test_rows": int(len(test_y)),
        "features": int(len(predictors)),
        "fit_seconds": fit_seconds,
        "max_runtime_secs": int(max_runtime_secs),
        "max_runtime_secs_per_model": int(max_runtime_secs_per_model),
        "nfolds": int(nfolds),
        "seed": int(seed),
        "nthreads": int(nthreads),
        "max_mem_size": str(max_mem_size),
        "model_scores_validation": model_scores,
    }
    if task_type == "classification":
        test_prediction = _classification_labels(test_prediction)
        result.update(
            {
                "accuracy": float(accuracy_score(test_target, test_prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(test_target, test_prediction)),
                "f1_macro": float(f1_score(test_target, test_prediction, average="macro", zero_division=0)),
                "score": float(accuracy_score(test_target, test_prediction)),
            }
        )
    else:
        result.update(
            {
                "r2": float(r2_score(test_y, test_prediction)),
                "rmse": float(math.sqrt(mean_squared_error(test_y, test_prediction))),
                "score": float(r2_score(test_y, test_prediction)),
            }
        )

    if not keep_h2o_alive:
        _shutdown_h2o(h2o)
    return result, selected_model
