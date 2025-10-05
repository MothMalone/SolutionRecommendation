#!/usr/bin/env python3
"""Utility script to run the SolutionRecommendation warm-start recommenders on
smaller, custom pipeline configurations and evaluate the top suggestions with
AutoGluon.

The workflow implemented here mirrors the original repository but is streamlined
for the user's reduced pipeline space (12 preprocessing-only pipelines) and
custom meta-data files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import (SelectKBest, VarianceThreshold,
                                       f_classif, mutual_info_classif)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MaxAbsScaler, MinMaxScaler,
                                   OneHotEncoder, RobustScaler, StandardScaler)
from sklearn.utils import shuffle
from scipy.stats import zscore

try:  # AutoGluon can be heavy; provide a clear message if it's missing.
    from autogluon.tabular import TabularDataset, TabularPredictor
except ImportError as _autogluon_exc:  # pragma: no cover - handled at runtime
    TabularPredictor = None  # type: ignore
    _AUTOGLUON_IMPORT_ERROR = _autogluon_exc
else:
    _AUTOGLUON_IMPORT_ERROR = None

import gplvm
import kernels
import my_recommends as mr
import my_tool as mt

# --------------------------------------------------------------------------------------
# Configuration constants ----------------------------------------------------------------
# --------------------------------------------------------------------------------------

AG_ARGS_FIT = {
    "ag.max_memory_usage_ratio": 0.3,
    "num_gpus": 0,
    "num_cpus": min(10, os.cpu_count() if os.cpu_count() else 4),
}

STABLE_MODELS = [
    "GBM",
    "CAT",
    "XGB",
    "RF",
    "XT",
    "KNN",
    "LR",
    "NN_TORCH",
    "FASTAI",
    "NN_MXNET",
    "TABPFN",
    "DUMMY",
    "NB",
]

PIPELINE_CONFIGS: List[Dict[str, str]] = [
    {
        "name": "baseline",
        "imputation": "none",
        "scaling": "none",
        "encoding": "none",
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "none",
    },
    {
        "name": "simple_preprocess",
        "imputation": "mean",
        "scaling": "standard",
        "encoding": "onehot",
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "none",
    },
    {
        "name": "robust_preprocess",
        "imputation": "median",
        "scaling": "robust",
        "encoding": "onehot",
        "feature_selection": "none",
        "outlier_removal": "iqr",
        "dimensionality_reduction": "none",
    },
    {
        "name": "feature_selection",
        "imputation": "median",
        "scaling": "standard",
        "encoding": "onehot",
        "feature_selection": "k_best",
        "outlier_removal": "none",
        "dimensionality_reduction": "none",
    },
    {
        "name": "dimension_reduction",
        "imputation": "mean",
        "scaling": "standard",
        "encoding": "onehot",
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "pca",
    },
    {
        "name": "conservative",
        "imputation": "median",
        "scaling": "minmax",
        "encoding": "onehot",
        "feature_selection": "variance_threshold",
        "outlier_removal": "none",
        "dimensionality_reduction": "none",
    },
    {
        "name": "aggressive",
        "imputation": "mean",
        "scaling": "standard",
        "encoding": "onehot",
        "feature_selection": "k_best",
        "outlier_removal": "iqr",
        "dimensionality_reduction": "pca",
    },
    {
        "name": "knn_impute_pca",
        "imputation": "knn",
        "scaling": "standard",
        "encoding": "onehot",
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "pca",
    },
    {
        "name": "mutual_info_zscore",
        "imputation": "median",
        "scaling": "robust",
        "encoding": "onehot",
        "feature_selection": "mutual_info",
        "outlier_removal": "zscore",
        "dimensionality_reduction": "none",
    },
    {
        "name": "constant_maxabs_iforest",
        "imputation": "constant",
        "scaling": "maxabs",
        "encoding": "onehot",
        "feature_selection": "variance_threshold",
        "outlier_removal": "isolation_forest",
        "dimensionality_reduction": "none",
    },
    {
        "name": "mean_minmax_lof_svd",
        "imputation": "mean",
        "scaling": "minmax",
        "encoding": "onehot",
        "feature_selection": "k_best",
        "outlier_removal": "lof",
        "dimensionality_reduction": "svd",
    },
    {
        "name": "mostfreq_standard_iqr",
        "imputation": "most_frequent",
        "scaling": "standard",
        "encoding": "onehot",
        "feature_selection": "none",
        "outlier_removal": "iqr",
        "dimensionality_reduction": "none",
    },
]

DEFAULT_TRAIN_DATASET_IDS: Sequence[int] = (
    22, 23, 24, 26, 28, 29, 30, 31, 32, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 53, 54, 55,
    56, 57, 59, 60, 61, 62, 163, 164, 171, 181, 182, 185, 186,
    187, 188, 275, 276, 277, 278, 285, 300, 301, 307, 308,
    310, 311, 312, 313, 316, 327, 328, 329, 333, 334, 335, 336,
    337, 338, 339, 340, 342, 343, 346, 372, 375,
)

DEFAULT_TEST_DATASET_IDS: Sequence[int] = (
    1503, 23517, 1551, 1552, 183, 255, 545, 546, 475, 481,
    516, 3, 6, 8, 10, 12, 14, 9, 11, 5,
)

# --------------------------------------------------------------------------------------
# Utility helpers ----------------------------------------------------------------------
# --------------------------------------------------------------------------------------


def _ensure_autogluon_available() -> None:
    if TabularPredictor is None:
        raise RuntimeError(
            "autogluon.tabular is required to train the evaluation models."
            " Install AutoGluon before running this script."
        ) from _AUTOGLUON_IMPORT_ERROR


def parse_dataset_id(column_name: str) -> int:
    """Extract the numeric OpenML dataset id from a column label like "D22"."""
    cleaned = ''.join(ch for ch in str(column_name) if ch.isdigit())
    if not cleaned:
        raise ValueError(f"Unable to parse dataset id from column '{column_name}'.")
    return int(cleaned)


def load_meta_performance(
    performance_csv: Path,
    dataset_feats_csv: Path,
    pipeline_configs: Sequence[Dict[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Return (performance_df, dataset_feats_df, ordered_pipeline_names).

    The first column (index) of ``performance_csv`` is assumed to contain pipeline
    names (case-insensitive). Dataset columns are expected to be labelled with IDs
    such as ``D22``.
    """

    perf_df = pd.read_csv(performance_csv, index_col=0)
    perf_df.index = perf_df.index.str.strip().str.lower()
    dataset_ids = [parse_dataset_id(col) for col in perf_df.columns]
    perf_df.columns = dataset_ids

    cfg_names = [cfg["name"].strip().lower() for cfg in pipeline_configs]
    missing = set(cfg_names) - set(perf_df.index)
    if missing:
        raise ValueError(
            "The performance table is missing pipeline rows for: " + ', '.join(sorted(missing))
        )

    ordered_perf = perf_df.loc[cfg_names]

    dataset_feats = pd.read_csv(dataset_feats_csv)
    dataset_feats.columns = [col.strip() for col in dataset_feats.columns]
    dataset_feats.iloc[:, 0] = dataset_feats.iloc[:, 0].apply(parse_dataset_id)

    return ordered_perf, dataset_feats, cfg_names


def split_meta_matrices(
    performance_df: pd.DataFrame,
    dataset_feats_df: pd.DataFrame,
    train_ids: Sequence[int],
    test_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Transform pandas inputs into the numpy matrices consumed by recommenders."""

    dataset_feats_df = dataset_feats_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _select(ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        feat_rows = dataset_feats_df[dataset_feats_df.iloc[:, 0].isin(ids)]
        if feat_rows.empty:
            raise ValueError(f"No dataset features found for ids: {ids}")
        perf_cols = [col for col in performance_df.columns if col in ids]
        if len(perf_cols) != len(ids):
            missing = set(ids) - set(perf_cols)
            raise ValueError(f"Missing performance columns for datasets: {missing}")
        return performance_df[perf_cols].to_numpy(dtype=np.float64), feat_rows.iloc[:, 1:].to_numpy(dtype=np.float64)

    Ytrain, Ftrain = _select(train_ids)
    Ytest, Ftest = _select(test_ids)

    scaler = StandardScaler()
    scaler.fit(Ftrain)

    Ftrain_norm = scaler.transform(Ftrain)
    Ftest_norm = scaler.transform(Ftest)

    return Ytrain, Ytest, Ftrain_norm, Ftest_norm, Ftrain, Ftest, scaler


def build_pipeline_feature_matrix(pipeline_configs: Sequence[Dict[str, str]]) -> np.ndarray:
    """Create a simple one-hot encoding for the discrete pipeline hyper-parameters."""

    categories: Dict[str, List[str]] = {}
    keys = [
        "imputation",
        "scaling",
        "encoding",
        "feature_selection",
        "outlier_removal",
        "dimensionality_reduction",
    ]
    for key in keys:
        categories[key] = sorted({cfg[key] for cfg in pipeline_configs})

    features: List[List[int]] = []
    for cfg in pipeline_configs:
        row: List[int] = []
        for key in keys:
            row.extend(1 if cfg[key] == option else 0 for option in categories[key])
        features.append(row)
    return np.asarray(features, dtype=np.float32)


def make_preprocessor(config: Dict[str, str]) -> "Preprocessor":
    return Preprocessor(config)


class Preprocessor:
    """Stateful preprocessing pipeline copied from the original implementation."""

    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.column_transformer: Optional[ColumnTransformer] = None
        self.selection_model: Optional[object] = None
        self.reduction_model: Optional[object] = None
        self.outlier_models: Dict[str, object] = {}
        self.fitted_columns: Optional[pd.Index] = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            X_processed = X.copy()

            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

            imputers = {
                "mean": SimpleImputer(strategy="mean"),
                "median": SimpleImputer(strategy="median"),
                "knn": KNNImputer(n_neighbors=min(5, max(len(X) - 1, 1))),
                "most_frequent": SimpleImputer(strategy="most_frequent"),
                "constant": SimpleImputer(strategy="constant", fill_value=0),
            }
            scalers = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
                "maxabs": MaxAbsScaler(),
            }

            numeric_steps: List[Tuple[str, object]] = []
            if self.config.get("imputation") in imputers:
                numeric_steps.append(("imputer", imputers[self.config["imputation"]]))
            if self.config.get("scaling") in scalers:
                numeric_steps.append(("scaler", scalers[self.config["scaling"]]))
            numeric_pipeline: object = Pipeline(steps=numeric_steps) if numeric_steps else "passthrough"

            categorical_pipeline: object = "passthrough"
            if self.config.get("encoding") == "onehot" and categorical_features:
                categorical_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
                    ]
                )

            transformers: List[Tuple[str, object, List[str]]] = []
            if numeric_features:
                transformers.append(("num", numeric_pipeline, numeric_features))
            if categorical_features:
                transformers.append(("cat", categorical_pipeline, categorical_features))

            if not transformers:
                raise ValueError("No features to transform")

            self.column_transformer = ColumnTransformer(transformers, remainder="drop")
            self.column_transformer.fit(X_processed)
            X_processed = pd.DataFrame(
                self.column_transformer.transform(X_processed),
                columns=self.column_transformer.get_feature_names_out(),
                index=X.index,
            )

            if X_processed.shape[1] == 0:
                raise ValueError("No features remained after initial transformation")

            if self.config["feature_selection"] in ["k_best", "mutual_info"]:
                k = min(20, X_processed.shape[1])
                if k > 0 and len(X_processed) > k:
                    selector_func = f_classif if self.config["feature_selection"] == "k_best" else mutual_info_classif
                    selector = SelectKBest(selector_func, k=k)
                    selector.fit(X_processed, y)
                    self.selection_model = selector
                    X_processed = pd.DataFrame(
                        selector.transform(X_processed),
                        columns=getattr(selector, "get_feature_names_out", lambda: None)(),
                        index=X_processed.index,
                    )
            elif self.config["feature_selection"] == "variance_threshold":
                selector = VarianceThreshold(threshold=0.0)
                selector.fit(X_processed)
                transformed = selector.transform(X_processed)
                if transformed.shape[1] > 0:
                    self.selection_model = selector
                    X_processed = pd.DataFrame(transformed, index=X_processed.index)

            if self.config["dimensionality_reduction"] in ["pca", "svd"]:
                n_components = min(10, X_processed.shape[1], max(len(X_processed) - 1, 1))
                if n_components > 0:
                    reducer = (
                        PCA(n_components=n_components)
                        if self.config["dimensionality_reduction"] == "pca"
                        else TruncatedSVD(n_components=n_components)
                    )
                    self.reduction_model = reducer.fit(X_processed)

            if self.config["outlier_removal"] in ["isolation_forest", "lof"] and len(X_processed) > 10:
                model = (
                    IsolationForest(random_state=42, contamination=0.05)
                    if self.config["outlier_removal"] == "isolation_forest"
                    else LocalOutlierFactor()
                )
                self.outlier_models["model"] = model.fit(X_processed)

            self.fitted_columns = X_processed.columns
            self.fitted = True
        except Exception as exc:  # pragma: no cover - logging fallback
            logging.warning("Preprocessor fit failed: %s", exc)
            self.fitted = False

    def transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.fitted or self.column_transformer is None:
            logging.warning("Preprocessor not fitted; returning original data.")
            return X.reset_index(drop=True), y.reset_index(drop=True)

        try:
            X_processed = pd.DataFrame(
                self.column_transformer.transform(X),
                columns=self.column_transformer.get_feature_names_out(),
                index=X.index,
            )
            y_processed = y.copy()

            if self.selection_model is not None:
                X_processed = pd.DataFrame(
                    self.selection_model.transform(X_processed),
                    columns=getattr(self.selection_model, "get_feature_names_out", lambda: None)(),
                    index=X_processed.index,
                )

            if self.reduction_model is not None:
                X_processed = pd.DataFrame(
                    self.reduction_model.transform(X_processed),
                    index=X_processed.index,
                )

            if self.config["outlier_removal"] != "none" and len(X_processed) > 20:
                original_size = len(X_processed)
                if self.config["outlier_removal"] == "iqr":
                    mask = pd.Series(True, index=X_processed.index)
                    for col in X_processed.columns:
                        q1, q3 = X_processed[col].quantile(0.25), X_processed[col].quantile(0.75)
                        iqr = q3 - q1
                        if iqr > 0:
                            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                            mask &= X_processed[col].between(lower, upper)
                    X_processed = X_processed[mask]
                elif self.config["outlier_removal"] == "zscore":
                    try:
                        z_scores = np.abs(zscore(X_processed, nan_policy="omit"))
                        mask = (z_scores < 3).all(axis=1)
                        X_processed = X_processed.loc[mask]
                    except Exception:  # pragma: no cover - defensive
                        pass
                elif self.config["outlier_removal"] in ["isolation_forest", "lof"] and "model" in self.outlier_models:
                    try:
                        predictions = self.outlier_models["model"].fit_predict(X_processed)
                        X_processed = X_processed[predictions == 1]
                    except Exception:  # pragma: no cover - defensive
                        pass

                if len(X_processed) < original_size * 0.1:
                    logging.warning("Outlier removal would drop >90%% of rows; reverting.")
                    X_processed = pd.DataFrame(
                        self.column_transformer.transform(X),
                        columns=self.column_transformer.get_feature_names_out(),
                        index=X.index,
                    )
                    if self.selection_model is not None:
                        X_processed = pd.DataFrame(
                            self.selection_model.transform(X_processed),
                            index=X_processed.index,
                        )
                    if self.reduction_model is not None:
                        X_processed = pd.DataFrame(
                            self.reduction_model.transform(X_processed),
                            index=X_processed.index,
                        )
                else:
                    y_processed = y_processed.loc[X_processed.index]

            return X_processed.reset_index(drop=True), y_processed.reset_index(drop=True)
        except Exception as exc:  # pragma: no cover - logging fallback
            logging.warning("Preprocessor transform failed: %s", exc)
            return X.reset_index(drop=True), y.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Recommendation & evaluation utilities ------------------------------------------------
# --------------------------------------------------------------------------------------


def train_recommenders(
    Ytrain: np.ndarray,
    Ftrain_norm: np.ndarray,
    FPipeline: np.ndarray,
    warm_starter_type: str = "knn",
) -> mr.AverageRankRecommender:
    warm_starters = {
        "average": mr.AverageRankRecommender(),
        "knn": mr.KnnRecommender({"n_neighbors": min(5, max(len(Ftrain_norm) - 1, 1))}),
        "l1": mr.L1Recommender(),
        "rf": mr.RFRecommender({"n_estimators": 100, "random_state": 42}),
    }

    if warm_starter_type not in warm_starters:
        raise ValueError(f"Unsupported warm starter '{warm_starter_type}'. Options: {sorted(warm_starters)}")

    starter = warm_starters[warm_starter_type]
    starter.train(Ytrain, Ftrain_norm)
    if hasattr(starter, "FPipeline"):
        starter.FPipeline = FPipeline
    return starter


def recommend_pipelines(
    starter: mr.AverageRankRecommender,
    dataset_feature_vector: np.ndarray,
    pipeline_names: Sequence[str],
    top_k: int,
) -> List[str]:
    ranking = starter.recommend(dataset_feature_vector)
    if ranking.ndim > 1:
        ranking = ranking[0]
    indices = ranking[:top_k]
    return [pipeline_names[idx] for idx in indices]


def load_openml_dataset(dataset_id: int) -> Optional[Dict[str, object]]:
    """Wrapper around the repo helper with a custom log path."""
    try:
        from sklearn.datasets import fetch_openml
    except Exception as exc:  # pragma: no cover - defensive
        logging.error("scikit-learn missing openml support: %s", exc)
        return None

    try:
        dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
        X, y = dataset.data, dataset.target

        if isinstance(X, pd.DataFrame):
            categorical_cols = X.select_dtypes(["category"]).columns
            if len(categorical_cols) > 0:
                X = X.copy()
                X.loc[:, categorical_cols] = X.loc[:, categorical_cols].astype(object)

        if y.dtype == "object" or getattr(y.dtype, "name", "") == "category":
            y = pd.Series(LabelEncoder().fit_transform(y), name=y.name)

        valid_indices = y.dropna().index
        X = X.loc[valid_indices].reset_index(drop=True)
        y = y.loc[valid_indices].reset_index(drop=True)

        if len(X) > 5000:
            X, y = shuffle(X, y, n_samples=5000, random_state=42)
            X, y = X.reset_index(drop=True), y.reset_index(drop=True)

        if len(X) < 20:
            logging.error("Dataset %s too small: %s samples", dataset_id, len(X))
            return None

        logging.info("Loaded OpenML dataset %s with shape %s and %s classes", dataset_id, X.shape, y.nunique())
        return {"id": dataset_id, "name": f"D_{dataset_id}", "X": X, "y": y.astype(int)}
    except Exception as exc:
        logging.error("Failed to fetch dataset %s: %s", dataset_id, exc)
        with open("failed_datasets.log", "a", encoding="utf-8") as handle:
            handle.write(f"Dataset {dataset_id}: {exc}\n")
        return None


def evaluate_pipeline_with_autogluon(
    dataset: Dict[str, object],
    pipeline_cfg: Dict[str, str],
    problem_time_limit: int,
    seed: int,
) -> Tuple[float, float]:
    _ensure_autogluon_available()

    X: pd.DataFrame = dataset["X"]  # type: ignore[assignment]
    y: pd.Series = dataset["y"]  # type: ignore[assignment]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if y.nunique() > 1 else None
    )

    preprocessor = make_preprocessor(pipeline_cfg)
    preprocessor.fit(X_train, y_train)
    X_train_proc, y_train_proc = preprocessor.transform(X_train, y_train)
    X_test_proc, y_test_proc = preprocessor.transform(X_test, y_test)

    train_df = X_train_proc.copy()
    train_df["target"] = y_train_proc.values

    test_df = X_test_proc.copy()
    test_df["target"] = y_test_proc.values

    problem_type = "binary" if y_train_proc.nunique() <= 2 else "multiclass"

    with tempfile.TemporaryDirectory(prefix="autogluon_") as tmp_dir:
        predictor = TabularPredictor(
            label="target",
            path=tmp_dir,
            problem_type=problem_type,
            eval_metric="accuracy",
            verbosity=0,
        )
        predictor.fit(
            TabularDataset(train_df),
            time_limit=problem_time_limit,
            presets="medium_quality",
            included_model_types=STABLE_MODELS,
            hyperparameter_tune_kwargs=None,
            feature_generator=None,
            ag_args_fit=AG_ARGS_FIT,
            raise_on_no_models_fitted=False,
        )

        preds = predictor.predict(TabularDataset(test_df.drop(columns=["target"])))
        acc = accuracy_score(y_test_proc, preds)
        leaderboard = predictor.leaderboard(silent=True)
        best_score = leaderboard.loc[leaderboard["score_val"].idxmax(), "score_val"]

    return float(acc), float(best_score)


# --------------------------------------------------------------------------------------
# Optional GPLVM offline evaluation ----------------------------------------------------
# --------------------------------------------------------------------------------------


def offline_evaluate_gplvm(
    Ytrain: np.ndarray,
    Ytest: np.ndarray,
    Ftrain_norm: np.ndarray,
    Ftest_norm: np.ndarray,
    pipeline_names: Sequence[str],
    test_dataset_ids: Sequence[int],
    starter: mr.AverageRankRecommender,
    bo_n_init: int,
    bo_n_iters: int,
) -> Dict[str, object]:
    """Full GPLVM offline evaluation replicating the original paper routine.

    The procedure follows the published configuration: a 20-dimensional latent
    GPLVM trained with SGD over mini-batches and subsequently evaluated via the
    Bayesian optimisation loop implemented in :mod:`my_tool`. The evaluation is
    purely offline and relies on the ground-truth accuracies ``Ytest``. The
    returned dictionary now also reports which pipeline produced the best score
    for each dataset alongside aggregate counts to help interpret the BO
    trajectory inside the constrained 12-pipeline space.
    """

    logging.info("Training GPLVM model with paper configuration for offline evaluation ...")

    Ytrain32 = Ytrain.astype(np.float32)
    Ytest32 = Ytest.astype(np.float32)

    latent_dim = 20
    batch_size = 50
    n_epochs = 300
    learning_rate = 1e-7
    N_max = 1000

    maxiter = max(1, int(np.ceil(Ytrain.shape[1] / batch_size * n_epochs)))

    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_init = PCA(latent_dim).fit_transform(imp.fit(Ytrain32).transform(Ytrain32))

    kernel = kernels.Add(kernels.RBF(latent_dim, lengthscale=None), kernels.White(latent_dim))
    model = gplvm.GPLVM(latent_dim, X_init, Ytrain32, kernel, N_max=N_max, D_max=batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.train()
    start_time = time.time()
    for iteration in range(1, maxiter + 1):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % 50 == 0:
            logging.debug(
                "GPLVM iteration %s/%s | neg log-lik per dim %.6f",
                iteration,
                maxiter,
                float(loss.item() / max(1, model.D)),
            )

    train_time = time.time() - start_time
    model.eval()

    logging.info("GPLVM training finished in %.2fs (%s iterations)", train_time, maxiter)
    logging.info("Evaluating BO trajectories on %s test datasets ...", Ytest.shape[1])

    curves: List[np.ndarray] = []
    eval_times: List[float] = []
    best_pipeline_per_dataset: Dict[int, str] = {}
    pipeline_counter: Counter[str] = Counter()

    with torch.no_grad():
        for idx in range(Ytest32.shape[1]):
            y_column = Ytest32[:, idx]
            if np.all(np.isnan(y_column)):
                continue

            start_eval = time.time()
            y_curve, ix_evaled = mt.bo_search(
                model,
                bo_n_init=bo_n_init,
                bo_n_iters=bo_n_iters,
                Ytrain=Ytrain32,
                Ftrain=Ftrain_norm,
                ftest=Ftest_norm[idx],
                ytest=y_column,
                warm_start="custom",
                warm_starter=starter,
            )
            eval_times.append(time.time() - start_eval)
            curves.append(y_curve)

            best_ix = None
            best_val = float("-inf")
            for step_ix in ix_evaled:
                if step_ix < 0:
                    continue
                candidate = float(y_column[step_ix])
                if candidate > best_val:
                    best_val = candidate
                    best_ix = step_ix

            if best_ix is not None:
                dataset_id = int(test_dataset_ids[idx]) if idx < len(test_dataset_ids) else idx
                pipeline_name = str(pipeline_names[best_ix]) if best_ix < len(pipeline_names) else str(best_ix)
                best_pipeline_per_dataset[dataset_id] = pipeline_name
                pipeline_counter[pipeline_name] += 1

    if not curves:
        logging.warning("No valid BO trajectories produced during GPLVM evaluation.")
        return {
            "bo_curve": [],
            "final_best": 0.0,
            "train_time": float(train_time),
            "avg_eval_time": 0.0,
            "best_pipeline_per_dataset": {},
            "best_pipeline_counts": {},
        }

    stacked = np.vstack(curves)
    avg_curve = np.nanmean(stacked, axis=0)

    return {
        "bo_curve": avg_curve.tolist(),
        "final_best": float(avg_curve[-1]) if avg_curve.size > 0 else 0.0,
        "train_time": float(train_time),
        "avg_eval_time": float(np.mean(eval_times)) if eval_times else 0.0,
        "best_pipeline_per_dataset": {int(k): v for k, v in best_pipeline_per_dataset.items()},
        "best_pipeline_counts": dict(pipeline_counter),
    }


# --------------------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--meta-performance",
        type=Path,
        default=Path("/drive1/nammt/SoluRec/Data/preprocessed_performance.csv"),
        help="CSV file containing pipeline (rows) x dataset (columns) accuracy table.",
    )
    parser.add_argument(
        "--dataset-features",
        type=Path,
        default=Path("/drive1/nammt/SoluRec/Data/dataset_feats.csv"),
        help="CSV file with OpenML dataset meta-features (first column must be dataset ids).",
    )
    parser.add_argument(
        "--train-ids",
        type=str,
        default="",
        help="Comma-separated list of dataset ids for training the recommender. Defaults to the published split.",
    )
    parser.add_argument(
        "--test-ids",
        type=str,
        default="",
        help="Comma-separated list of dataset ids to evaluate. Defaults to the published split.",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=100,
        help="Maximum number of OpenML datasets to fetch and evaluate (to keep runtime manageable).",
    )
    parser.add_argument(
        "--warm-starter",
        type=str,
        default="knn",
        help="Warm starter to use (average, knn, l1, rf).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top pipelines to evaluate per dataset."
        " The script stops early if a pipeline fit fails.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="AutoGluon time limit per dataset (seconds).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for preprocessing splits and recommenders.",
    )
    parser.add_argument(
        "--offline-gplvm",
        action="store_true",
        help=(
            "Train the full GPLVM model from the paper and replay the offline BO search"
            " on the meta-data. This mirrors the original evaluation and can take a long"
            " time."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("run_recommender_results.json"),
        help="Where to store the JSON results summary.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    args = parser.parse_args(argv)
    return args


def maybe_parse_id_list(raw: str, default: Sequence[int]) -> List[int]:
    if not raw:
        return list(default)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI entry
    args = parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    mt.setup_seed(args.random_seed)
    np.random.seed(args.random_seed)

    train_ids = maybe_parse_id_list(args.train_ids, DEFAULT_TRAIN_DATASET_IDS)
    test_ids = maybe_parse_id_list(args.test_ids, DEFAULT_TEST_DATASET_IDS)

    logging.info("Loading meta-performance and feature tables ...")
    performance_df, dataset_feats_df, pipeline_names = load_meta_performance(
        args.meta_performance,
        args.dataset_features,
        PIPELINE_CONFIGS,
    )
    pipeline_display_names = [cfg["name"] for cfg in PIPELINE_CONFIGS]

    logging.info("Preparing matrices for recommenders ...")
    Ytrain, Ytest, Ftrain_norm, Ftest_norm, Ftrain_raw, Ftest_raw, scaler = split_meta_matrices(
        performance_df, dataset_feats_df, train_ids, test_ids
    )

    FPipeline = build_pipeline_feature_matrix(PIPELINE_CONFIGS)

    logging.info("Training warm-starter '%s' ...", args.warm_starter)
    starter = train_recommenders(Ytrain, Ftrain_norm, FPipeline, args.warm_starter)

    offline_metrics: Optional[Dict[str, float]] = None
    if args.offline_gplvm:
        offline_metrics = offline_evaluate_gplvm(
            Ytrain,
            Ytest,
            Ftrain_norm,
            Ftest_norm,
            pipeline_display_names,
            test_ids,
            starter,
            bo_n_init=5,
            bo_n_iters=20,
        )
        logging.info("Offline GPLVM metrics: %s", offline_metrics)

    evaluation_ids = test_ids[: args.max_datasets]
    logging.info("Evaluating top-%s pipelines on %s OpenML datasets ...", args.top_k, len(evaluation_ids))

    results: List[Dict[str, object]] = []

    for dataset_id in evaluation_ids:
        dataset = load_openml_dataset(dataset_id)
        if dataset is None:
            continue

        feat_vector = scaler.transform(dataset_feats_df[dataset_feats_df.iloc[:, 0] == dataset_id].iloc[:, 1:])[0]
        recommended_pipelines = recommend_pipelines(starter, feat_vector, pipeline_names, args.top_k)

        for pipeline_name in recommended_pipelines:
            cfg = next(cfg for cfg in PIPELINE_CONFIGS if cfg["name"].lower() == pipeline_name)
            logging.info("Dataset %s -> evaluating pipeline '%s'", dataset_id, pipeline_name)
            try:
                accuracy, best_score = evaluate_pipeline_with_autogluon(
                    dataset,
                    cfg,
                    args.time_limit,
                    args.random_seed,
                )
            except Exception as exc:
                logging.error("Pipeline '%s' failed on dataset %s: %s", pipeline_name, dataset_id, exc)
                break

            results.append(
                {
                    "dataset_id": dataset_id,
                    "pipeline": pipeline_name,
                    "accuracy": accuracy,
                    "best_model_score": best_score,
                    "time_limit": args.time_limit,
                }
            )

    results_df = pd.DataFrame(results)
    pipeline_lookup = {cfg["name"].lower(): cfg["name"] for cfg in PIPELINE_CONFIGS}
    summary_files: Dict[str, str] = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset_summary_records: List[Dict[str, object]] = []
    pipeline_summary_records: List[Dict[str, object]] = []
    offline_curve_records: List[Dict[str, object]] = []

    if not results_df.empty:
        results_df["pipeline_display"] = results_df["pipeline"].map(
            lambda name: pipeline_lookup.get(str(name).lower(), str(name))
        )

        dataset_summary_df = (
            results_df.groupby("dataset_id")
            .agg(
                best_accuracy=("accuracy", "max"),
                mean_accuracy=("accuracy", "mean"),
                pipelines_evaluated=("pipeline", "nunique"),
                evaluations=("pipeline", "count"),
            )
            .reset_index()
        )
        best_pipeline_idx = results_df.groupby("dataset_id")["accuracy"].idxmax()
        dataset_summary_df = dataset_summary_df.merge(
            results_df.loc[best_pipeline_idx, ["dataset_id", "pipeline_display"]],
            on="dataset_id",
            how="left",
        )
        dataset_summary_df = dataset_summary_df.rename(columns={"pipeline_display": "best_pipeline"})

        pipeline_summary_df = (
            results_df.groupby("pipeline_display")
            .agg(
                datasets_tested=("dataset_id", "nunique"),
                evaluations=("dataset_id", "count"),
                mean_accuracy=("accuracy", "mean"),
                median_accuracy=("accuracy", "median"),
            )
            .reset_index()
            .rename(columns={"pipeline_display": "pipeline"})
            .sort_values(by="mean_accuracy", ascending=False)
            .reset_index(drop=True)
        )

        results_csv = args.output.with_name(f"{args.output.stem}_results.csv")
        per_dataset_csv = args.output.with_name(f"{args.output.stem}_per_dataset.csv")
        per_pipeline_csv = args.output.with_name(f"{args.output.stem}_per_pipeline.csv")

        results_df.to_csv(results_csv, index=False)
        dataset_summary_df.to_csv(per_dataset_csv, index=False)
        pipeline_summary_df.to_csv(per_pipeline_csv, index=False)

        summary_files.update(
            {
                "results_csv": str(results_csv),
                "per_dataset_csv": str(per_dataset_csv),
                "per_pipeline_csv": str(per_pipeline_csv),
            }
        )

        dataset_summary_records = dataset_summary_df.to_dict(orient="records")
        pipeline_summary_records = pipeline_summary_df.to_dict(orient="records")

        logging.info(
            "Average accuracy across %s evaluations: %.4f",
            len(results_df),
            results_df["accuracy"].mean(),
        )
        logging.info(
            "Top dataset summary rows:\n%s",
            dataset_summary_df.head(min(5, len(dataset_summary_df))).to_string(index=False),
        )
        logging.info(
            "Top pipeline summary rows:\n%s",
            pipeline_summary_df.head(min(5, len(pipeline_summary_df))).to_string(index=False),
        )
    else:
        logging.warning("No successful AutoGluon evaluations were recorded.")

    if offline_metrics:
        canonical_counts = {
            pipeline_lookup.get(str(name).lower(), str(name)): count
            for name, count in offline_metrics.get("best_pipeline_counts", {}).items()
        }
        offline_metrics["best_pipeline_counts_pretty"] = canonical_counts
        offline_metrics["best_pipeline_per_dataset_pretty"] = {
            str(dataset_id): pipeline_lookup.get(str(name).lower(), str(name))
            for dataset_id, name in offline_metrics.get("best_pipeline_per_dataset", {}).items()
        }

        if offline_metrics.get("bo_curve"):
            offline_curve_df = pd.DataFrame(
                {
                    "iteration": list(range(1, len(offline_metrics["bo_curve"]) + 1)),
                    "avg_best_accuracy": offline_metrics["bo_curve"],
                }
            )
            offline_curve_csv = args.output.with_name(f"{args.output.stem}_offline_bo.csv")
            offline_curve_df.to_csv(offline_curve_csv, index=False)
            offline_curve_records = offline_curve_df.to_dict(orient="records")
            summary_files["offline_bo_csv"] = str(offline_curve_csv)

    summary = {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "warm_starter": args.warm_starter,
        "top_k": args.top_k,
        "time_limit": args.time_limit,
        "offline_gplvm": offline_metrics,
        "results": results,
        "dataset_summary": dataset_summary_records,
        "pipeline_summary": pipeline_summary_records,
        "offline_bo_curve": offline_curve_records,
        "artifacts": summary_files,
    }

    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Saved results to %s", args.output)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    raise SystemExit(main())
