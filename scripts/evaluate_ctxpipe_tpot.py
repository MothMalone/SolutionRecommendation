#!/usr/bin/env python3
"""Replay a frozen native CtxPipe sequence and evaluate it with model-only TPOT.

CtxPipe pipeline search must be run separately on the outer train+validation 80%.
This evaluator reloads the original dataset, fits the frozen preprocessing
sequence on the fixed 60% train split, leaves the 20% validation split out of
TPOT, and reports accuracy exactly once on the untouched outer test 20%.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA, KernelPCA, PCA, TruncatedSVD
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automl_aco.data.loaders import load_gitlab_openml_dataset  # noqa: E402
from automl_aco.data.splits import split_train_val_test  # noqa: E402
from automl_aco.eval_ids import EVAL_IDS  # noqa: E402


BLANK_NAMES = {"blank", "none", "", "primitive"}
SUPPORTED_NATIVE_OPERATORS = {
    "blank",
    "ImputerMean",
    "ImputerMedian",
    "ImputerNumMode",
    "ImputerCatMode",
    "NumericData",
    "LabelEncoder",
    "OneHotEncoder",
    "MinMaxScaler",
    "MaxAbsScaler",
    "RobustScaler",
    "StandardScaler",
    "QuantileTransformer",
    "PowerTransformer",
    "Normalizer",
    "KBinsDiscretizerOrdinal",
    "PolynomialFeatures",
    "InteractionFeatures",
    "PCA_AUTO",
    "IncrementalPCA",
    "KernelPCA",
    "TruncatedSVD",
    "RandomTreesEmbedding",
    "VarianceThreshold",
}


def _safe_cv_folds(y_train: pd.Series, maximum: int) -> int:
    counts = pd.Series(y_train).value_counts()
    if len(counts) < 2:
        raise ValueError("TPOT classification requires at least two training classes")
    folds = min(max(2, int(maximum)), int(counts.min()))
    if folds < 2:
        raise ValueError("A training class has fewer than two rows for TPOT CV")
    return folds


def _numeric_columns(frame: pd.DataFrame) -> list:
    return frame.select_dtypes(include=["number", "bool"]).columns.tolist()


def _categorical_columns(frame: pd.DataFrame) -> list:
    numeric = set(_numeric_columns(frame))
    return [column for column in frame.columns if column not in numeric]


def _replace_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence,
    train_values: Any,
    test_values: Any,
    *,
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    remaining = [column for column in train.columns if column not in set(columns)]
    train_remaining = train[remaining].reset_index(drop=True)
    test_remaining = test[remaining].reset_index(drop=True)
    occupied = {str(column) for column in remaining}
    names = []
    for index in range(np.asarray(train_values).shape[1]):
        candidate = f"{prefix}_{index}"
        suffix = 1
        while candidate in occupied:
            candidate = f"{prefix}_{index}_{suffix}"
            suffix += 1
        occupied.add(candidate)
        names.append(candidate)
    train_new = pd.DataFrame(train_values, columns=names).reset_index(drop=True)
    test_new = pd.DataFrame(test_values, columns=names).reset_index(drop=True)
    return (
        pd.concat([train_remaining, train_new], axis=1),
        pd.concat([test_remaining, test_new], axis=1),
    )


def _guard_shape(frame: pd.DataFrame, label: str, maximum_cells: int) -> None:
    cells = int(frame.shape[0]) * int(frame.shape[1])
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"CtxPipe produced empty {label} data: {frame.shape}")
    if cells > int(maximum_cells):
        raise MemoryError(
            f"CtxPipe {label} matrix {frame.shape} has {cells:,} cells, above "
            f"the configured safety limit {int(maximum_cells):,}"
        )


def replay_ctxpipe_sequence(
    sequence: Sequence[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    *,
    maximum_cells: int = 200_000_000,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Fit native-space operators on train and transform test without leakage."""
    train = pd.DataFrame(X_train).copy().reset_index(drop=True)
    test = pd.DataFrame(X_test).copy().reset_index(drop=True)
    train.columns = [str(column) for column in train.columns]
    test.columns = list(train.columns)
    trace: list[dict] = []

    for raw_name in sequence:
        name = str(raw_name).strip()
        if name.lower() not in BLANK_NAMES and name not in SUPPORTED_NATIVE_OPERATORS:
            raise ValueError(f"Unsupported native CtxPipe operator: {name!r}")
        before = [int(train.shape[0]), int(train.shape[1])]
        if name.lower() in BLANK_NAMES:
            trace.append({"operator": name or "blank", "before": before, "after": before})
            continue

        if name in {"ImputerMean", "ImputerMedian", "ImputerNumMode"}:
            columns = _numeric_columns(train)
            if columns:
                strategy = {
                    "ImputerMean": "mean",
                    "ImputerMedian": "median",
                    "ImputerNumMode": "most_frequent",
                }[name]
                transformer = SimpleImputer(strategy=strategy, keep_empty_features=True)
                train_values = transformer.fit_transform(train[columns])
                test_values = transformer.transform(test[columns])
                train, test = _replace_columns(
                    train, test, columns, train_values, test_values, prefix="num"
                )
        elif name == "ImputerCatMode":
            columns = _categorical_columns(train)
            if columns:
                transformer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
                train_values = transformer.fit_transform(train[columns].astype(object))
                test_values = transformer.transform(test[columns].astype(object))
                train, test = _replace_columns(
                    train, test, columns, train_values, test_values, prefix="cat"
                )
        elif name == "NumericData":
            columns = _numeric_columns(train)
            train = train[columns].reset_index(drop=True)
            test = test[columns].reset_index(drop=True)
        elif name in {"LabelEncoder", "OneHotEncoder"}:
            columns = _categorical_columns(train)
            if columns:
                train_cat = train[columns].astype("string").fillna("__CTXPIPE_MISSING__")
                test_cat = test[columns].astype("string").fillna("__CTXPIPE_MISSING__")
                if name == "LabelEncoder":
                    transformer = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32
                    )
                    prefix = "ordinal"
                else:
                    estimated_features = sum(
                        int(train_cat[column].nunique(dropna=False)) for column in columns
                    )
                    remaining_features = int(train.shape[1] - len(columns))
                    if int(len(train)) * (estimated_features + remaining_features) > int(
                        maximum_cells
                    ):
                        raise MemoryError(
                            "OneHotEncoder would exceed the configured matrix-cell limit"
                        )
                    transformer = OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False, dtype=np.float32
                    )
                    prefix = "onehot"
                train_values = transformer.fit_transform(train_cat)
                test_values = transformer.transform(test_cat)
                train, test = _replace_columns(
                    train, test, columns, train_values, test_values, prefix=prefix
                )
        elif name in {
            "MinMaxScaler",
            "MaxAbsScaler",
            "RobustScaler",
            "StandardScaler",
            "QuantileTransformer",
            "PowerTransformer",
            "Normalizer",
            "KBinsDiscretizerOrdinal",
        }:
            transformers = {
                "MinMaxScaler": MinMaxScaler,
                "MaxAbsScaler": MaxAbsScaler,
                "RobustScaler": RobustScaler,
                "StandardScaler": StandardScaler,
                "QuantileTransformer": lambda: QuantileTransformer(random_state=0),
                "PowerTransformer": PowerTransformer,
                "Normalizer": Normalizer,
                "KBinsDiscretizerOrdinal": lambda: KBinsDiscretizer(
                    encode="ordinal", random_state=0
                ),
            }
            transformer = transformers[name]()
            train_values = transformer.fit_transform(train)
            test_values = transformer.transform(test)
            columns = [f"feature_{index}" for index in range(train_values.shape[1])]
            train = pd.DataFrame(train_values, columns=columns)
            test = pd.DataFrame(test_values, columns=columns)
        elif name in {"PolynomialFeatures", "InteractionFeatures"}:
            interaction_only = name == "InteractionFeatures"
            output_features = (
                train.shape[1]
                + (
                    train.shape[1]
                    * (train.shape[1] + (-1 if interaction_only else 1))
                )
                // 2
            )
            if int(train.shape[0]) * int(output_features) > int(maximum_cells):
                raise MemoryError(
                    f"{name} would create about {output_features:,} features on "
                    f"{len(train):,} training rows"
                )
            transformer = PolynomialFeatures(
                interaction_only=interaction_only, include_bias=False
            )
            train_values = transformer.fit_transform(train)
            test_values = transformer.transform(test)
            columns = [f"poly_{index}" for index in range(train_values.shape[1])]
            train = pd.DataFrame(train_values, columns=columns)
            test = pd.DataFrame(test_values, columns=columns)
        elif name in {
            "PCA_AUTO",
            "IncrementalPCA",
            "KernelPCA",
            "TruncatedSVD",
            "RandomTreesEmbedding",
        }:
            if name == "KernelPCA" and int(len(train)) ** 2 > int(maximum_cells):
                raise MemoryError(
                    f"KernelPCA would allocate a kernel with {int(len(train)) ** 2:,} cells"
                )
            if name == "RandomTreesEmbedding":
                # sklearn defaults to 100 depth-5 trees, hence at most 3,200 leaves.
                estimated_features = 100 * (2**5)
                if int(len(train)) * estimated_features > int(maximum_cells):
                    raise MemoryError(
                        "RandomTreesEmbedding would exceed the configured matrix-cell limit"
                    )
            transformers = {
                "PCA_AUTO": lambda: PCA(svd_solver="auto", random_state=0),
                "IncrementalPCA": IncrementalPCA,
                "KernelPCA": lambda: KernelPCA(n_components=2),
                "TruncatedSVD": lambda: TruncatedSVD(n_components=2, random_state=0),
                "RandomTreesEmbedding": lambda: RandomTreesEmbedding(random_state=0),
            }
            transformer = transformers[name]()
            train_values = transformer.fit_transform(train)
            test_values = transformer.transform(test)
            if hasattr(train_values, "toarray"):
                train_values = train_values.toarray()
                test_values = test_values.toarray()
            columns = [f"engine_{index}" for index in range(train_values.shape[1])]
            train = pd.DataFrame(train_values, columns=columns)
            test = pd.DataFrame(test_values, columns=columns)
        elif name == "VarianceThreshold":
            transformer = VarianceThreshold()
            train_values = transformer.fit_transform(train)
            test_values = transformer.transform(test)
            columns = [f"selected_{index}" for index in range(train_values.shape[1])]
            train = pd.DataFrame(train_values, columns=columns)
            test = pd.DataFrame(test_values, columns=columns)
        train = train.replace([np.inf, -np.inf], np.nan)
        test = test.replace([np.inf, -np.inf], np.nan)
        _guard_shape(train, "training", maximum_cells)
        _guard_shape(test, "test", maximum_cells)
        trace.append(
            {
                "operator": name,
                "before": before,
                "after": [int(train.shape[0]), int(train.shape[1])],
            }
        )

    non_numeric = train.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"CtxPipe sequence left categorical columns: {non_numeric[:10]}")
    train = train.astype(np.float32)
    test = test.astype(np.float32)
    if not np.isfinite(train.to_numpy()).all() or not np.isfinite(test.to_numpy()).all():
        raise ValueError("CtxPipe replay produced NaN or infinity")
    return train, test, trace


def _default_tpot_components():
    try:
        from tpot import TPOTClassifier
        from tpot.config import get_search_space
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("TPOT 1.1.0 is required") from exc
    return TPOTClassifier, get_search_space


def evaluate_ctxpipe_sequence(
    dataset: Dict[str, Any],
    sequence: Sequence[str],
    *,
    split_seed: int = 42,
    tpot_seed: int = 1,
    max_time_mins: int = 5,
    max_eval_time_mins: int = 1,
    n_jobs: int = 2,
    memory_limit: str = "5GB",
    population_size: int = 20,
    max_cv_folds: int = 5,
    maximum_cells: int = 200_000_000,
    verbose: int = 2,
    estimator_factory: Optional[Callable[..., Any]] = None,
    search_space_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[Dict[str, Any], Any]:
    if str(dataset.get("task_type", "classification")) != "classification":
        raise ValueError("Native CtxPipe TPOT experiment currently supports classification only")
    X = pd.DataFrame(dataset["X"]).copy()
    y = pd.Series(dataset["y"]).copy()
    X_train, y_train, X_val, _y_val, X_test, y_test = split_train_val_test(
        X, y, seed=int(split_seed)
    )
    train_processed, test_processed, trace = replay_ctxpipe_sequence(
        sequence,
        X_train,
        X_test,
        y_train,
        maximum_cells=int(maximum_cells),
    )
    cv_folds = _safe_cv_folds(y_train, max_cv_folds)
    if estimator_factory is None or search_space_factory is None:
        default_estimator, default_search_space = _default_tpot_components()
        estimator_factory = estimator_factory or default_estimator
        search_space_factory = search_space_factory or default_search_space
    search_space = search_space_factory(
        "classifiers",
        n_classes=int(y_train.nunique()),
        n_samples=int(len(train_processed)),
        n_features=int(train_processed.shape[1]),
        random_state=int(tpot_seed),
        n_jobs=1,
    )
    model = estimator_factory(
        search_space=search_space,
        scorers=["accuracy"],
        scorers_weights=[1],
        cv=int(cv_folds),
        preprocessing=False,
        max_time_mins=int(max_time_mins),
        max_eval_time_mins=int(max_eval_time_mins),
        n_jobs=int(n_jobs),
        memory_limit=str(memory_limit),
        validation_strategy="none",
        early_stop=5,
        verbose=int(verbose),
        random_state=int(tpot_seed),
        population_size=int(population_size),
        initial_population_size=int(population_size),
    )
    started = time.perf_counter()
    model.fit(train_processed.to_numpy(), y_train.to_numpy())
    prediction = model.predict(test_processed.to_numpy())
    fit_seconds = float(time.perf_counter() - started)
    result = {
        "status": "ok",
        "method": "ctxpipe_tpot",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "score": float(accuracy_score(y_test, prediction)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "f1_macro": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "split_seed": int(split_seed),
        "tpot_seed": int(tpot_seed),
        "raw_rows": int(len(X)),
        "tpot_train_rows": int(len(X_train)),
        "validation_rows_used_by_ctxpipe_search": int(len(X_val)),
        "validation_reused_by_tpot": False,
        "outer_test_rows": int(len(X_test)),
        "outer_test_fraction": 0.2,
        "ctxpipe_saw_outer_test": False,
        "raw_features": int(X.shape[1]),
        "transformed_features": int(train_processed.shape[1]),
        "ctxpipe_sequence": [str(value) for value in sequence],
        "ctxpipe_replay": "leakfree_fit_train_only",
        "operator_trace": trace,
        "tpot_space": "classifiers",
        "tpot_preprocessing": False,
        "cv_folds": int(cv_folds),
        "max_time_mins": int(max_time_mins),
        "max_eval_time_mins": int(max_eval_time_mins),
        "n_jobs": int(n_jobs),
        "memory_limit": str(memory_limit),
        "population_size": int(population_size),
        "fit_seconds": fit_seconds,
        "selected_estimator": str(
            getattr(model, "fitted_pipeline_", getattr(model, "fitted_pipeline", ""))
        ),
    }
    return result, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctxpipe-result-json", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--tpot-seed", type=int, default=1)
    parser.add_argument("--max-time-mins", type=int, default=5)
    parser.add_argument("--max-eval-time-mins", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--memory-limit", default="5GB")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--max-cv-folds", type=int, default=5)
    parser.add_argument("--maximum-cells", type=int, default=200_000_000)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_json.exists() and not args.force:
        existing = json.loads(args.output_json.read_text(encoding="utf-8"))
        if existing.get("status") == "ok":
            print("SKIP successful result:", args.output_json)
            return 0
    ctxpipe_result = json.loads(args.ctxpipe_result_json.read_text(encoding="utf-8"))
    sequence = ctxpipe_result.get("sequence")
    if not isinstance(sequence, list) or len(sequence) != 6:
        raise ValueError("CtxPipe result must contain a six-operator sequence")
    dataset = load_gitlab_openml_dataset(
        args.dataset_id,
        cache_dir=str(args.data_dir),
        test_dataset_ids=[int(value) for value in EVAL_IDS],
        verbose=True,
        max_samples_if_test=int(args.max_samples),
    )
    if dataset is None:
        raise RuntimeError(f"Could not load dataset {args.dataset_id}")
    model = None
    try:
        result, model = evaluate_ctxpipe_sequence(
            dataset,
            sequence,
            split_seed=args.split_seed,
            tpot_seed=args.tpot_seed,
            max_time_mins=args.max_time_mins,
            max_eval_time_mins=args.max_eval_time_mins,
            n_jobs=args.n_jobs,
            memory_limit=args.memory_limit,
            population_size=args.population_size,
            max_cv_folds=args.max_cv_folds,
            maximum_cells=args.maximum_cells,
            verbose=args.verbose,
        )
        result.update(
            {
                "dataset_id": int(args.dataset_id),
                "dataset_name": dataset.get("name", f"D_{args.dataset_id}"),
                "dataset_backend": dataset.get("download_backend"),
                "max_samples": int(args.max_samples),
                "ctxpipe_checkpoint": ctxpipe_result.get("checkpoint"),
                "ctxpipe_native_reward": ctxpipe_result.get("native_reward"),
                "ctxpipe_official_commit": ctxpipe_result.get("official_commit"),
                "ctxpipe_search_rows": ctxpipe_result.get("search_rows"),
                "ctxpipe_internal_train_rows": ctxpipe_result.get(
                    "native_internal_train_rows"
                ),
                "ctxpipe_internal_reward_rows": ctxpipe_result.get(
                    "native_internal_reward_rows"
                ),
            }
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "method": "ctxpipe_tpot",
            "dataset_id": int(args.dataset_id),
            "ctxpipe_sequence": sequence,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        raise
    finally:
        del model
        gc.collect()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"CtxPipe + TPOT outer-test accuracy: {result['accuracy']:.6f}")
    print("Saved:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
