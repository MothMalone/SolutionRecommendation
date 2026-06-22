"""Candidate evaluation functions (simple models and AutoGluon)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from ..data.splits import split_train_val_test
from ..preprocessing.preprocessor import Preprocessor
from ..utils.operator_spec import base_operator_name
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _normalize_proxy_settings(proxy_settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(proxy_settings or {})
    seeds = cfg.get("split_seeds", [42])
    if isinstance(seeds, int):
        seeds = [int(seeds)]
    elif isinstance(seeds, (tuple, list)):
        seeds = [int(s) for s in seeds]
    else:
        seeds = [42]
    if not seeds:
        seeds = [42]
    cfg["split_seeds"] = seeds

    cfg.setdefault("active_step_penalty", 0.0)
    cfg.setdefault("row_drop_penalty", 0.0)
    cfg.setdefault("imputation_low_missing_penalty", 0.0)
    cfg.setdefault("low_missing_threshold", 0.0)
    cfg.setdefault("outlier_removal_penalty", 0.0)
    cfg.setdefault("dimred_small_feature_penalty", 0.0)
    cfg.setdefault("dimred_small_feature_threshold", 0)
    cfg.setdefault("verbose_components", False)
    cfg.setdefault("classification_model", "logreg")
    cfg.setdefault("regression_model", "ensemble")
    cfg.setdefault("logreg_max_iter", 3000)
    return cfg


def _count_active_steps(cfg: Dict[str, Any]) -> int:
    steps = [
        "imputation",
        "scaling",
        "outlier_removal",
        "feature_selection",
        "dimensionality_reduction",
    ]
    return sum(1 for step in steps if not _step_is_none(cfg, step))


def _step_is_none(cfg: Dict[str, Any], step: str) -> bool:
    val = cfg.get(step)
    if isinstance(val, dict):
        if len(val) == 0:
            return True
        for op in val.values():
            if base_operator_name(op) != "none":
                return False
        return True
    if val is None:
        return True
    return base_operator_name(val) == "none"


def _proxy_penalty(
    cfg: Dict[str, Any],
    *,
    n_rows_before: int,
    n_rows_after: int,
    n_features_before: int,
    missing_ratio: float,
    proxy_cfg: Dict[str, Any],
) -> float:
    penalty = 0.0
    penalty += float(proxy_cfg["active_step_penalty"]) * float(_count_active_steps(cfg))

    row_drop = 0.0
    if n_rows_before > 0 and n_rows_after >= 0:
        row_drop = max(0.0, (float(n_rows_before) - float(n_rows_after)) / float(n_rows_before))
    penalty += float(proxy_cfg["row_drop_penalty"]) * row_drop

    if (
        missing_ratio <= float(proxy_cfg["low_missing_threshold"])
        and not _step_is_none(cfg, "imputation")
    ):
        penalty += float(proxy_cfg["imputation_low_missing_penalty"])

    if not _step_is_none(cfg, "outlier_removal"):
        penalty += float(proxy_cfg["outlier_removal_penalty"])

    if (
        not _step_is_none(cfg, "dimensionality_reduction")
        and n_features_before <= int(proxy_cfg["dimred_small_feature_threshold"])
    ):
        penalty += float(proxy_cfg["dimred_small_feature_penalty"])

    return float(max(0.0, penalty))


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


def _make_preprocessor(cfg: Dict[str, Any]) -> Preprocessor:
    step_order = cfg.get("step_order")
    pre_cfg = {k: v for k, v in cfg.items() if k != "step_order"}
    if isinstance(step_order, list) and len(step_order) > 0:
        return Preprocessor(pre_cfg, step_order=step_order)
    return Preprocessor(pre_cfg)


def _should_retry_without_xgb(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "xgbclassifier" in msg and "n_classes_" in msg


def _fit_predict_with_autogluon(
    *,
    TabularPredictor,
    IdentityFeatureGenerator,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    eval_metric: str,
    time_limit_per_model: int,
    verbosity: int,
    autogluon_profile: str = "best_quality",
    excluded_model_types: Optional[Sequence[str]] = None,
    select_df: Optional[pd.DataFrame] = None,
):
    import os
    import shutil
    import tempfile
    import uuid

    temp_dir = os.path.join(tempfile.gettempdir(), f"autogluon_{uuid.uuid4().hex}")
    try:
        predictor = TabularPredictor(
            label=target_column,
            path=temp_dir,
            problem_type=problem_type,
            eval_metric=eval_metric,
            verbosity=verbosity,
        )
        profile = str(autogluon_profile or "best_quality").strip().lower()
        fit_kwargs = dict(
            train_data=train_df,
            time_limit=time_limit_per_model,
            feature_generator=IdentityFeatureGenerator(),
            raise_on_no_models_fitted=False,
        )
        if profile == "best_quality":
            fit_kwargs["presets"] = "best_quality"
        elif profile == "medium_quality":
            fit_kwargs["presets"] = "medium_quality_faster_train"
        elif profile == "local_rf_xt":
            # Local debugging profile: avoids model families that are fragile
            # on lightweight macOS/dev installs while still using AutoGluon.
            if problem_type == "regression":
                fit_kwargs["hyperparameters"] = {
                    "RF": {"criterion": "squared_error"},
                    "XT": {"criterion": "squared_error"},
                }
            else:
                fit_kwargs["hyperparameters"] = {
                    "RF": {"criterion": "gini"},
                    "XT": {"criterion": "gini"},
                }
        else:
            raise ValueError(
                "autogluon_profile must be one of: best_quality, medium_quality, local_rf_xt"
            )
        if excluded_model_types:
            fit_kwargs["excluded_model_types"] = list(excluded_model_types)

        predictor.fit(**fit_kwargs)
        try:
            model_names = predictor.model_names()
            if len(model_names) == 0:
                return None, None, "no_models_fitted"
        except Exception:
            pass
        preds = predictor.predict(test_df)
        select_preds = predictor.predict(select_df) if select_df is not None else None
        return preds, select_preds, None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def evaluate_candidates_autogluon(
    dataset: Any,
    target_column: str,
    candidate_configs: List[Dict[str, Any]],
    time_limit_per_model: int = 300,
    autogluon_profile: str = "best_quality",
    verbose: bool = False,
    select_on_val: bool = False,
) -> Tuple[Optional[Dict[str, Any]], float, List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]]:
    # When select_on_val is True, every candidate is trained on the train split and scored on the
    # VALIDATION split for SELECTION; the winner's TEST score is reported. This is the leak-free way
    # to choose among candidates (e.g. ACO pipeline vs no-preprocessing) without selecting on test,
    # and it is the mechanism that protects against over-preprocessing.
    try:
        import numpy as _np
        ver = _np.__version__.split(".")
        if len(ver) >= 1 and int(ver[0]) >= 2:
            raise RuntimeError("AutoGluon requires NumPy < 2.0; please install numpy<2")
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
    val_scores: Dict[str, Optional[float]] = {}

    for cfg in candidate_configs:
        if "name" not in cfg or cfg.get("name") is None:
            cfg["name"] = str(cfg)
        name = cfg["name"]

        try:
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()

            X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y)

            pre = _make_preprocessor(cfg)
            result = pre.fit_transform(X_train, y_train)
            if isinstance(result, tuple):
                X_train_proc, y_train_proc = result
            else:
                X_train_proc = result
                y_train_proc = y_train.reset_index(drop=True)

            X_test_proc = pre.transform(X_test)
            y_test_proc = y_test.reset_index(drop=True)

            val_df = None
            y_val_proc = None
            if select_on_val:
                X_val_proc = pre.transform(X_val)
                y_val_proc = y_val.reset_index(drop=True)
                if len(y_val_proc) == len(X_val_proc) and X_val_proc.shape[0] > 0:
                    val_df = X_val_proc.copy()

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

            try:
                preds, select_preds, ag_issue = _fit_predict_with_autogluon(
                    TabularPredictor=TabularPredictor,
                    IdentityFeatureGenerator=IdentityFeatureGenerator,
                    train_df=train_df,
                    test_df=test_df,
                    target_column=target_column,
                    problem_type=problem_type,
                    eval_metric=eval_metric,
                    time_limit_per_model=time_limit_per_model,
                    verbosity=2 if verbose else 0,
                    autogluon_profile=autogluon_profile,
                    select_df=val_df,
                )

                if ag_issue == "no_models_fitted":
                    if verbose:
                        print(f"    ✗ {name} - AutoGluon fitted no models")
                    else:
                        logger.info("%s - AutoGluon fitted no models", name)
                    continue

                score = (r2_score if problem_type == "regression" else accuracy_score)(y_test_proc, preds)
            except Exception as exc:
                if _should_retry_without_xgb(exc):
                    if verbose:
                        print(f"    ! {name} hit XGBoost compatibility issue, retrying without XGB models")
                    preds, select_preds, ag_issue = _fit_predict_with_autogluon(
                        TabularPredictor=TabularPredictor,
                        IdentityFeatureGenerator=IdentityFeatureGenerator,
                        train_df=train_df,
                        test_df=test_df,
                        target_column=target_column,
                        problem_type=problem_type,
                        eval_metric=eval_metric,
                        time_limit_per_model=time_limit_per_model,
                        verbosity=2 if verbose else 0,
                        autogluon_profile=autogluon_profile,
                        excluded_model_types=["XGB"],
                        select_df=val_df,
                    )
                    if ag_issue == "no_models_fitted":
                        if verbose:
                            print(f"    ✗ {name} - AutoGluon fitted no models (retry without XGB)")
                        else:
                            logger.info("%s - AutoGluon fitted no models (retry without XGB)", name)
                        continue
                    score = (r2_score if problem_type == "regression" else accuracy_score)(y_test_proc, preds)
                else:
                    raise

            # Selection score on the held-out validation split (leak-free), when requested.
            val_score = None
            if select_on_val and val_df is not None and select_preds is not None and y_val_proc is not None:
                try:
                    val_score = float((r2_score if problem_type == "regression" else accuracy_score)(y_val_proc, select_preds))
                except Exception:
                    val_score = None

            results.append((cfg, float(score)))
            val_scores[name] = val_score
            if verbose:
                vtxt = f" (val={val_score:.4f})" if val_score is not None else ""
                print(f"    ✓ {name} -> {score:.4f}{vtxt}")
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
        return None, np.nan, [], []

    unsorted_res = results.copy()
    if select_on_val and any(v is not None for v in val_scores.values()):
        # Pick the candidate with the best VALIDATION score (leak-free selection); report its TEST
        # score. Missing val scores fall back to the test score so a candidate is always chosen.
        def _sel_key(item):
            cfg, test_score = item
            v = val_scores.get(cfg.get("name"))
            return (v if v is not None else -np.inf, test_score)
        results_sorted = sorted(results, key=_sel_key, reverse=True)
        best_cfg, best_score = results_sorted[0]
        if verbose:
            print(f"    [hybrid-select] chose {best_cfg.get('name')} by val; reporting its test={best_score:.4f}")
        return best_cfg, best_score, results_sorted, unsorted_res
    results.sort(key=lambda x: x[1], reverse=True)
    best_cfg, best_score = results[0]
    return best_cfg, best_score, results, unsorted_res


def evaluate_candidates_simple(
    dataset: Any,
    target_column: str,
    candidate_configs: List[Dict[str, Any]],
    proxy_settings: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[Optional[Dict[str, Any]], float, List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]]:
    df = _normalize_dataset(dataset, target_column)
    if target_column not in df.columns:
        raise ValueError(f"target_column {target_column} not found")

    proxy_cfg = _normalize_proxy_settings(proxy_settings)
    split_seeds = proxy_cfg["split_seeds"]
    missing_ratio = float(df.drop(columns=[target_column]).isna().to_numpy().mean()) if len(df.columns) > 1 else 0.0
    n_features_before = int(df.drop(columns=[target_column]).shape[1]) if len(df.columns) > 1 else 0
    has_missing = bool(missing_ratio > 0.0)

    y_all = df[target_column]
    problem_type, _eval_metric = _detect_problem_type(y_all)

    if problem_type != "regression":
        _, counts = np.unique(y_all, return_counts=True)
        if counts.min() < 3:
            return None, np.nan, [], []

    results: List[Tuple[Dict[str, Any], float]] = []

    def _score_regression_proxy(
        X_train_p: pd.DataFrame,
        y_train_p: pd.Series,
        X_val_p: pd.DataFrame,
        y_val_p: pd.Series,
        *,
        model_name: str,
    ) -> Optional[float]:
        name = str(model_name).lower().strip()
        models: List[Any] = []
        if name == "linear":
            models = [LinearRegression()]
        elif name == "random_forest":
            models = [RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)]
        else:  # "ensemble" default
            models = [
                LinearRegression(),
                RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
            ]

        scores: List[float] = []
        for model in models:
            try:
                model.fit(X_train_p, y_train_p)
                pred = model.predict(X_val_p)
                scores.append(float(r2_score(y_val_p, pred)))
            except Exception as exc:
                logger.debug("Regression proxy model %s failed: %s", type(model).__name__, exc)
        if not scores:
            return None
        if name == "ensemble":
            return float(np.mean(scores))
        return float(scores[0])

    def _score_classification_proxy(
        X_train_p: pd.DataFrame,
        y_train_p: pd.Series,
        X_val_p: pd.DataFrame,
        y_val_p: pd.Series,
        *,
        model_name: str,
        logreg_max_iter: int,
    ) -> Optional[float]:
        name = str(model_name).lower().strip()
        if name == "logreg":
            best = -np.inf
            for C in (0.01, 0.1, 1.0):
                for class_weight in (None, "balanced"):
                    try:
                        clf = LogisticRegression(
                            C=C,
                            solver="lbfgs",
                            class_weight=class_weight,
                            max_iter=max(200, int(logreg_max_iter)),
                            random_state=42,
                        )
                        clf.fit(X_train_p, y_train_p)
                        pred = clf.predict(X_val_p)
                        best = max(best, float(accuracy_score(y_val_p, pred)))
                    except Exception:
                        continue
            if np.isfinite(best):
                return float(best)
            return None

        if name == "linear_svm":
            clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
        elif name == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
        elif name == "extra_trees":
            clf = ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif name == "knn":
            clf = KNeighborsClassifier(n_neighbors=7, weights="distance")
        elif name == "hist_gbdt":
            clf = HistGradientBoostingClassifier(
                max_depth=None,
                learning_rate=0.1,
                max_iter=200,
                random_state=42,
            )
        else:
            raise ValueError(f"Unsupported classification proxy model: {model_name}")

        try:
            clf.fit(X_train_p, y_train_p)
            pred = clf.predict(X_val_p)
            return float(accuracy_score(y_val_p, pred))
        except Exception as exc:
            logger.debug("Classification proxy model %s failed: %s", type(clf).__name__, exc)
            return None

    for cfg in candidate_configs:
        if "name" not in cfg or cfg.get("name") is None:
            cfg["name"] = str(cfg)
        name = cfg["name"]

        # Fast reject: this search space does not include NaN-tolerant downstream estimators.
        # If data has missing values and no imputation is selected, the config is invalid by design.
        if has_missing and _step_is_none(cfg, "imputation"):
            if verbose:
                print(f"    ✗ {name} skipped: missing values require non-'none' imputation")
            continue

        try:
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()

            seed_scores: List[float] = []
            seed_raw_scores: List[float] = []
            seed_penalties: List[float] = []

            for split_seed in split_seeds:
                X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, seed=int(split_seed))

                pre = _make_preprocessor(cfg)
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
                    seed_scores = []
                    break
                if X_val_p.shape[0] == 0:
                    if verbose:
                        print(f"    ✗ {name} produced empty VAL data")
                    else:
                        logger.info("%s produced empty VAL data", name)
                    seed_scores = []
                    break
                if X_test_p.shape[0] == 0:
                    if verbose:
                        print(f"    ✗ {name} produced empty TEST data")
                    else:
                        logger.info("%s produced empty TEST data", name)
                    seed_scores = []
                    break
                if len(X_train_p) != len(y_train_p):
                    if verbose:
                        print(f"    ✗ {name} - TRAIN X/y length mismatch")
                    else:
                        logger.info("%s - TRAIN X/y length mismatch", name)
                    seed_scores = []
                    break
                if len(X_val_p) != len(y_val_p):
                    if verbose:
                        print(f"    ✗ {name} - VAL X/y length mismatch")
                    else:
                        logger.info("%s - VAL X/y length mismatch", name)
                    seed_scores = []
                    break

                if problem_type == "regression":
                    raw_score = _score_regression_proxy(
                        X_train_p,
                        y_train_p,
                        X_val_p,
                        y_val_p,
                        model_name=str(proxy_cfg.get("regression_model", "ensemble")),
                    )
                    if raw_score is None:
                        seed_scores = []
                        break
                else:
                    raw_score = _score_classification_proxy(
                        X_train_p,
                        y_train_p,
                        X_val_p,
                        y_val_p,
                        model_name=str(proxy_cfg.get("classification_model", "logreg")),
                        logreg_max_iter=int(proxy_cfg.get("logreg_max_iter", 3000)),
                    )
                    if raw_score is None:
                        seed_scores = []
                        break

                penalty = _proxy_penalty(
                    cfg=cfg,
                    n_rows_before=len(X_train),
                    n_rows_after=len(X_train_p),
                    n_features_before=n_features_before,
                    missing_ratio=missing_ratio,
                    proxy_cfg=proxy_cfg,
                )
                adjusted_score = float(raw_score - penalty)
                seed_raw_scores.append(raw_score)
                seed_penalties.append(penalty)
                seed_scores.append(adjusted_score)

            if not seed_scores:
                continue

            score = float(np.mean(seed_scores))
            results.append((cfg, score))
            if verbose:
                if proxy_cfg.get("verbose_components", False):
                    print(
                        f"    ✓ {name} -> {score:.4f} "
                        f"(raw={np.mean(seed_raw_scores):.4f}, penalty={np.mean(seed_penalties):.4f}, "
                        f"seeds={len(seed_scores)})"
                    )
                else:
                    print(f"    ✓ {name} -> {score:.4f}")
        except Exception as exc:
            if verbose:
                print(f"    ✗ Error evaluating cfg {name}: {exc}")
            else:
                logger.warning("Error evaluating cfg %s: %s", name, exc)
            continue

    if not results:
        if verbose and len(candidate_configs) > 1:
            print("❌ No candidate produced valid evaluation results")
        return None, np.nan, [], []

    unsorted_res = results.copy()
    results.sort(key=lambda x: x[1], reverse=True)
    best_cfg, best_score = results[0]
    return best_cfg, best_score, results, unsorted_res
