"""Candidate evaluation functions (simple models and AutoGluon)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor

from ..data.splits import split_train_val_test
from ..preprocessing.preprocessor import Preprocessor
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _normalize_dataset(dataset: Any, target_column: str) -> pd.DataFrame:
    if isinstance(dataset, dict):
        if "X" in dataset and "y" in dataset:
            df = pd.DataFrame(dataset["X"]).copy()
            df[target_column] = pd.Series(dataset["y"]).copy()
        else:
            df = pd.DataFrame(dataset)
    elif isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
    else:
        raise ValueError("dataset must be DataFrame or dict{'X','y'}")
    return df


def _detect_problem_type(y: pd.Series) -> Tuple[str, str]:
    unique_classes = y.nunique()
    if np.issubdtype(y.dtype, np.number) and unique_classes > 50:
        return "regression", "r2"
    if unique_classes == 2:
        return "binary", "accuracy"
    return "multiclass", "accuracy"


def evaluate_candidates_autogluon(
    dataset: Any,
    target_column: str,
    candidate_configs: List[Dict[str, Any]],
    time_limit_per_model: int = 300,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], float, List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]]:
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore
        from autogluon.features.generators import IdentityFeatureGenerator  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("AutoGluon not available in environment") from exc

    df = _normalize_dataset(dataset, target_column)
    if target_column not in df.columns:
        raise ValueError(f"target_column {target_column} not in dataset")

    y = df[target_column]
    problem_type, eval_metric = _detect_problem_type(y)

    results: List[Tuple[Dict[str, Any], float]] = []

    for cfg in candidate_configs:
        if "name" not in cfg or cfg.get("name") is None:
            cfg["name"] = str(cfg)
        name = cfg["name"]

        try:
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()

            X_train, y_train, _X_val, _y_val, X_test, y_test = split_train_val_test(X, y)

            pre = Preprocessor(cfg)
            result = pre.fit_transform(X_train, y_train)
            if isinstance(result, tuple):
                X_train_proc, y_train_proc = result
            else:
                X_train_proc = result
                y_train_proc = y_train.reset_index(drop=True)

            X_test_proc = pre.transform(X_test)
            y_test_proc = y_test.reset_index(drop=True)

            if X_train_proc.shape[0] == 0:
                if verbose:
                    print(f"    ✗ {name} produced empty TRAIN data after preprocessing")
                else:
                    logger.info("%s produced empty TRAIN data after preprocessing", name)
                continue

            train_df = X_train_proc.copy()
            train_df[target_column] = y_train_proc
            test_df = X_test_proc.copy()

            if len(y_test_proc) != len(X_test_proc):
                if verbose:
                    print(f"    ✗ {name} - y_test length mismatch after preprocessing")
                else:
                    logger.info("%s - y_test length mismatch after preprocessing", name)
                continue

            import os, tempfile, uuid
            temp_dir = os.path.join(tempfile.gettempdir(), f"autogluon_{uuid.uuid4().hex}")

            try:
                predictor = TabularPredictor(
                    label=target_column,
                    path=temp_dir,
                    problem_type=problem_type,
                    eval_metric=eval_metric,
                    verbosity=2 if verbose else 0,
                )
                predictor.fit(
                    train_data=train_df,
                    time_limit=time_limit_per_model,
                    presets="best_quality",
                    feature_generator=IdentityFeatureGenerator(),
                    raise_on_no_models_fitted=False,
                )
                try:
                    model_names = predictor.model_names()
                    if len(model_names) == 0:
                        if verbose:
                            print(f"    ✗ {name} - AutoGluon fitted no models")
                        else:
                            logger.info("%s - AutoGluon fitted no models", name)
                        continue
                except Exception:
                    pass
                preds = predictor.predict(test_df)

                if problem_type == "regression":
                    score = r2_score(y_test_proc, preds)
                else:
                    score = accuracy_score(y_test_proc, preds)
            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

            results.append((cfg, float(score)))
            if verbose:
                print(f"    ✓ {name} -> {score:.4f}")
        except Exception as exc:
            if verbose:
                print(f"    ✗ Error evaluating cfg {name}: {exc}")
            else:
                logger.exception("Error evaluating cfg %s: %s", name, exc)
            continue

    if not results:
        if verbose:
            print("No candidate produced valid evaluation results")
        else:
            logger.info("No candidate produced valid evaluation results")
        results.append((candidate_configs[0], 0.0))

    unsorted_res = results.copy()
    results.sort(key=lambda x: x[1], reverse=True)
    best_cfg, best_score = results[0]
    return best_cfg, best_score, results, unsorted_res


def evaluate_candidates_simple(
    dataset: Any,
    target_column: str,
    candidate_configs: List[Dict[str, Any]],
    verbose: bool = False,
) -> Tuple[Optional[Dict[str, Any]], float, List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]]:
    df = _normalize_dataset(dataset, target_column)
    if target_column not in df.columns:
        raise ValueError(f"target_column {target_column} not found")

    y_all = df[target_column]
    problem_type, _eval_metric = _detect_problem_type(y_all)

    if problem_type != "regression":
        _, counts = np.unique(y_all, return_counts=True)
        if counts.min() < 3:
            return None, np.nan, [], []

    results: List[Tuple[Dict[str, Any], float]] = []

    for cfg in candidate_configs:
        if "name" not in cfg or cfg.get("name") is None:
            cfg["name"] = str(cfg)
        name = cfg["name"]

        try:
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()

            X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y)

            pre = Preprocessor(cfg)
            result = pre.fit_transform(X_train, y_train)
            if isinstance(result, tuple):
                X_train_p, y_train_p = result
            else:
                X_train_p = result
                y_train_p = y_train.reset_index(drop=True)

            X_val_p = pre.transform(X_val)
            X_test_p = pre.transform(X_test)

            y_val_p = y_val.reset_index(drop=True)
            y_test_p = y_test.reset_index(drop=True)

            if X_train_p.shape[0] == 0:
                if verbose:
                    print(f"    ✗ {name} produced empty TRAIN data")
                else:
                    logger.info("%s produced empty TRAIN data", name)
                continue
            if X_val_p.shape[0] == 0:
                if verbose:
                    print(f"    ✗ {name} produced empty VAL data")
                else:
                    logger.info("%s produced empty VAL data", name)
                continue
            if X_test_p.shape[0] == 0:
                if verbose:
                    print(f"    ✗ {name} produced empty TEST data")
                else:
                    logger.info("%s produced empty TEST data", name)
                continue
            if len(X_train_p) != len(y_train_p):
                if verbose:
                    print(f"    ✗ {name} - TRAIN X/y length mismatch")
                else:
                    logger.info("%s - TRAIN X/y length mismatch", name)
                continue
            if len(X_val_p) != len(y_val_p):
                if verbose:
                    print(f"    ✗ {name} - VAL X/y length mismatch")
                else:
                    logger.info("%s - VAL X/y length mismatch", name)
                continue

            if problem_type == "regression":
                models = [
                    LinearRegression(),
                    RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
                ]
                scores = []
                for model in models:
                    try:
                        model.fit(X_train_p, y_train_p)
                        pred = model.predict(X_val_p)
                        scores.append(r2_score(y_val_p, pred))
                    except Exception as exc:
                        logger.debug("Model %s failed: %s", type(model).__name__, exc)
                if not scores:
                    continue
                score = float(np.mean(scores))
            else:
                logreg_grid = {
                    "C": [0.01, 0.1, 1.0],
                    "solver": ["lbfgs"],
                    "class_weight": [None, "balanced"],
                }
                scores = []
                for C in logreg_grid["C"]:
                    for solver in logreg_grid["solver"]:
                        for cw in logreg_grid["class_weight"]:
                            try:
                                clf = LogisticRegression(
                                    C=C,
                                    solver=solver,
                                    penalty="l2",
                                    multi_class="auto",
                                    class_weight=cw,
                                    max_iter=1000,
                                    n_jobs=-1,
                                    random_state=42,
                                )
                                clf.fit(X_train_p, y_train_p)
                                pred = clf.predict(X_val_p)
                                scores.append(accuracy_score(y_val_p, pred))
                            except Exception:
                                pass
                if not scores:
                    continue
                score = float(max(scores))

            results.append((cfg, score))
            if verbose:
                print(f"    ✓ {name} -> {score:.4f}")
        except Exception as exc:
            if verbose:
                print(f"    ✗ Error evaluating cfg {name}: {exc}")
            else:
                logger.exception("Error evaluating cfg %s: %s", name, exc)
            continue

    if not results:
        if verbose:
            print("❌ No candidate produced valid evaluation results")
        return None, np.nan, [], []

    unsorted_res = results.copy()
    results.sort(key=lambda x: x[1], reverse=True)
    best_cfg, best_score = results[0]
    return best_cfg, best_score, results, unsorted_res
