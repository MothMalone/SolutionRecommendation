"""Three-way hybrid validation for retrieval-only vs ACO search.

This script tests a practical question that the standard recommendation run
does not isolate: after search/no-search pick candidate pipelines on a search
development split, can a hybrid selector choose the better family on a separate
validation split and still generalize to a final holdout split?

The default reported hybrid score is selected on the selector split and scored
on the final split. This avoids using the final test split to choose the
pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automl_aco.config import DEFAULT_PIPELINE_OPTIONS, NOTEBOOK_LEGACY_PIPELINE_OPTIONS
from automl_aco.data.loaders import load_kaggle_dataset, load_openml_dataset
from automl_aco.data.metafeatures import extract_enhanced_metafeatures
from automl_aco.metalearning.recommender import MetaPipelineRecommender
from automl_aco.preprocessing.preprocessor import Preprocessor
from automl_aco.search.evaluation import _detect_problem_type
from automl_aco.utils.operator_spec import base_operator_name


PIPELINE_STEPS = [
    "imputation",
    "scaling",
    "encoding",
    "feature_selection",
    "outlier_removal",
    "dimensionality_reduction",
]


def _normalize_id(val: object) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        f = float(val)
        if np.isfinite(f) and abs(f - round(f)) <= 1e-9:
            return str(int(round(f)))
        return str(val).strip()

    text = str(val).strip()
    float_like = re.fullmatch(r"([0-9]+)\.0+", text)
    if float_like:
        return float_like.group(1)

    prefixed = re.fullmatch(r"(?i)(?:d|dataset|openml)[_\-: ]*([0-9]+)", text)
    if prefixed:
        return prefixed.group(1)
    return text


def _parse_dataset_ids(raw: Optional[List[str]]) -> List[Any]:
    out: List[Any] = []
    for token in raw or []:
        for piece in str(token).split(","):
            text = piece.strip()
            if text:
                out.append(int(text) if text.isdigit() else text)
    return out


def _maybe_set_meta_index(meta_df: pd.DataFrame, perf_df: pd.DataFrame, explicit_col: Optional[str]) -> pd.DataFrame:
    perf_norm = {_normalize_id(c) for c in perf_df.columns}

    def overlap(values: Iterable[object]) -> int:
        return len({_normalize_id(v) for v in values} & perf_norm)

    if explicit_col:
        if explicit_col not in meta_df.columns:
            raise ValueError(f"--metafeatures-id-column={explicit_col!r} not found")
        return meta_df.set_index(explicit_col)

    best_col: Optional[str] = None
    best_overlap = overlap(meta_df.index)
    prioritized = ["dataset_id", "did", "openml_id", "id", "Dataset", "dataset", "Unnamed: 0"]
    candidate_cols = [c for c in prioritized if c in meta_df.columns] + [c for c in meta_df.columns if c not in prioritized]
    for col in candidate_cols:
        cur = overlap(meta_df[col])
        if cur > best_overlap:
            best_overlap = cur
            best_col = str(col)

    if best_col is not None and best_overlap > 0:
        return meta_df.set_index(best_col)
    return meta_df


def _lookup_metafeatures(dataset: Mapping[str, Any], meta_df: pd.DataFrame) -> Dict[str, Any]:
    dataset_id = dataset.get("id") if isinstance(dataset, Mapping) else None
    if dataset_id is None:
        return extract_enhanced_metafeatures(dict(dataset), meta_features_df=meta_df)

    dataset_norm = _normalize_id(dataset_id)
    for idx in meta_df.index:
        if _normalize_id(idx) == dataset_norm:
            return meta_df.loc[idx].to_dict()
    return extract_enhanced_metafeatures(dict(dataset), meta_features_df=meta_df)


def _infer_pipeline_config_from_name(pipeline_name: str) -> Dict[str, Any]:
    token = str(pipeline_name).strip().lower()
    cfg: Dict[str, Any] = {
        "name": str(pipeline_name),
        "imputation": "none",
        "scaling": "none",
        "encoding": "onehot",
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "none",
    }

    if "knn" in token:
        cfg["imputation"] = "knn"
    elif "mostfreq" in token or "most_frequent" in token:
        cfg["imputation"] = "most_frequent"
    elif "constant" in token:
        cfg["imputation"] = "constant"
    elif "median" in token:
        cfg["imputation"] = "median"
    elif "mean" in token:
        cfg["imputation"] = "mean"

    if "no_scale" in token:
        cfg["scaling"] = "none"
    elif "robust" in token:
        cfg["scaling"] = "robust"
    elif "minmax" in token:
        cfg["scaling"] = "minmax"
    elif "maxabs" in token:
        cfg["scaling"] = "maxabs"
    elif "standard" in token:
        cfg["scaling"] = "standard"
    elif "uniform" in token or "quantile" in token:
        cfg["scaling"] = "minmax"

    if "mutualinfo" in token or "mutual_info" in token:
        cfg["feature_selection"] = "mutual_info"
    elif "kbest" in token or "k_best" in token:
        cfg["feature_selection"] = "k_best"
    elif "variance" in token:
        cfg["feature_selection"] = "variance_threshold"

    if "iforest" in token or "isolation" in token:
        cfg["outlier_removal"] = "isolation_forest"
    elif "zscore" in token:
        cfg["outlier_removal"] = "zscore"
    elif "iqr" in token:
        cfg["outlier_removal"] = "iqr"
    elif "lof" in token:
        cfg["outlier_removal"] = "lof"

    if "pca" in token:
        cfg["dimensionality_reduction"] = "pca"
    elif "svd" in token:
        cfg["dimensionality_reduction"] = "svd"
    return cfg


def _load_pipeline_configs(path: Path, perf: pd.DataFrame, allow_inferred: bool) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        configs = json.load(f)

    existing = {str(cfg.get("name")) for cfg in configs if isinstance(cfg, dict) and "name" in cfg}
    missing = [str(name) for name in perf.index if str(name) not in existing]
    if missing:
        if not allow_inferred:
            sample = ", ".join(missing[:10])
            raise ValueError(
                "Performance matrix contains rows absent from pipeline configs: "
                f"{sample}. Use matching files or pass --allow-inferred-pipeline-configs."
            )
        configs = list(configs) + [_infer_pipeline_config_from_name(name) for name in missing]
    return configs


def _load_dataset_as_frame(
    dataset_id: Any,
    *,
    dataset_source: str,
    openml_local_folder: Path,
    kaggle_data_folder: Path,
    kaggle_target_column: str,
    verbose: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if dataset_source == "openml":
        dataset = load_openml_dataset(dataset_id, verbose=verbose, local_data_folder=str(openml_local_folder))
    else:
        dataset = load_kaggle_dataset(
            dataset_id,
            data_folder=str(kaggle_data_folder),
            target_column=kaggle_target_column,
            verbose=verbose,
        )
    if dataset is None or "X" not in dataset or "y" not in dataset:
        raise RuntimeError(f"Could not load dataset {dataset_id}")

    df = dataset["X"].copy()
    df["target"] = dataset["y"]
    return dataset, df


def _config_signature(cfg: Mapping[str, Any], options: Mapping[str, Sequence[str]]) -> str:
    return "|".join(f"{step}={base_operator_name(cfg.get(step, 'none'))}" for step in options.keys())


def _display_config(cfg: Optional[Mapping[str, Any]], options: Mapping[str, Sequence[str]]) -> str:
    if not cfg:
        return ""
    return "; ".join(f"{step}={cfg.get(step, 'none')}" for step in options.keys())


def _build_options(args: argparse.Namespace) -> Dict[str, List[str]]:
    base = NOTEBOOK_LEGACY_PIPELINE_OPTIONS if args.notebook_legacy_options else DEFAULT_PIPELINE_OPTIONS
    return {step: list(vals) for step, vals in base.items()}


def _adapt_options_to_dataset(options: Dict[str, List[str]], X: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[str]]:
    notes: List[str] = []
    out = {k: list(v) for k, v in options.items()}
    has_missing = bool(X.isna().to_numpy().any())
    if has_missing and "imputation" in out:
        before = len(out["imputation"])
        out["imputation"] = [v for v in out["imputation"] if base_operator_name(v) != "none"]
        if not out["imputation"]:
            out["imputation"] = ["mean"]
        if len(out["imputation"]) != before:
            notes.append("removed imputation=none because development data has missing values")
    return out, notes


def _build_proxy_settings(args: argparse.Namespace) -> Dict[str, Any]:
    if args.proxy_profile == "robust":
        settings: Dict[str, Any] = {
            "split_seeds": [42, 52, 62],
            "active_step_penalty": 0.003,
            "row_drop_penalty": 0.10,
            "imputation_low_missing_penalty": 0.010,
            "low_missing_threshold": 0.001,
            "outlier_removal_penalty": 0.007,
            "dimred_small_feature_penalty": 0.008,
            "dimred_small_feature_threshold": 120,
            "verbose_components": bool(args.verbose),
        }
    else:
        settings = {
            "split_seeds": [42],
            "active_step_penalty": 0.0,
            "row_drop_penalty": 0.0,
            "imputation_low_missing_penalty": 0.0,
            "low_missing_threshold": 0.0,
            "outlier_removal_penalty": 0.0,
            "dimred_small_feature_penalty": 0.0,
            "dimred_small_feature_threshold": 0,
            "verbose_components": bool(args.verbose),
        }

    settings["classification_model"] = str(args.proxy_clf_model)
    settings["regression_model"] = str(args.proxy_reg_model)
    settings["logreg_max_iter"] = int(args.proxy_logreg_max_iter)

    if args.proxy_seeds:
        parsed: List[int] = []
        for token in str(args.proxy_seeds).split(","):
            token = token.strip()
            if token:
                parsed.append(int(token))
        if parsed:
            settings["split_seeds"] = parsed
    return settings


def _build_aco_params(args: argparse.Namespace, dataset_id: Any) -> Dict[str, Any]:
    return {
        "n_ants": int(args.n_ants),
        "n_iterations": int(args.n_iterations),
        "seed": int(args.seed),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "evaporation": float(args.evaporation),
        "top_k_pheromone": int(args.top_k_pheromone),
        "weight_method": str(args.aco_weight_method),
        "markov_order": int(args.aco_markov_order),
        "lambda_smooth": float(args.aco_lambda_smooth),
        "early_stop_rounds": int(args.aco_early_stop_rounds),
        "min_improvement": float(args.aco_min_improvement),
        "dataset_weighting": str(args.dataset_weighting),
        "heuristic_top_k": int(args.heuristic_top_k),
        "heuristic_top_l": int(args.heuristic_top_l),
        "heuristic_transfer_method": str(args.heuristic_transfer_method),
        "heuristic_similarity_temperature": float(args.similarity_temperature),
        "heuristic_eta_floor": float(args.eta_floor),
        "score_direction": "higher_is_better",
        "query_dataset_id": dataset_id,
        "require_autogluon": bool(args.require_autogluon),
        "legacy_notebook_aco": bool(args.legacy_notebook_aco),
        "per_feature_independent_search": bool(args.per_feature_independent_search),
        "per_feature_steps": str(args.per_feature_steps),
        "per_feature_early_stop_rounds": int(args.per_feature_early_stop_rounds),
        "per_feature_min_improvement": float(args.per_feature_min_improvement),
        "per_feature_feature_patience": int(args.per_feature_feature_patience),
        "per_feature_feature_min_improvement": float(args.per_feature_feature_min_improvement),
        "per_feature_max_features": int(args.per_feature_max_features),
    }


def _stratify_if_safe(df: pd.DataFrame, target_column: str, test_size: float) -> Optional[pd.Series]:
    y = df[target_column]
    if not (np.issubdtype(y.dtype, np.number) and y.nunique() > 50):
        counts = y.value_counts(dropna=False)
        min_count = int(counts.min()) if not counts.empty else 0
        n_test = int(round(len(df) * float(test_size)))
        if min_count >= 2 and n_test >= y.nunique():
            return y
    return None


def _three_way_split(
    df: pd.DataFrame,
    target_column: str,
    *,
    selector_size: float,
    final_test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if selector_size <= 0 or final_test_size <= 0 or selector_size + final_test_size >= 0.8:
        raise ValueError("--selector-size and --final-test-size must be positive and leave enough search data")

    final_stratify = _stratify_if_safe(df, target_column, final_test_size)
    dev_selector_df, final_df = train_test_split(
        df,
        test_size=float(final_test_size),
        random_state=int(seed),
        shuffle=True,
        stratify=final_stratify,
    )
    selector_fraction_of_remaining = float(selector_size) / max(1e-12, 1.0 - float(final_test_size))
    selector_stratify = _stratify_if_safe(dev_selector_df, target_column, selector_fraction_of_remaining)
    search_df, selector_df = train_test_split(
        dev_selector_df,
        test_size=selector_fraction_of_remaining,
        random_state=int(seed) + 1,
        shuffle=True,
        stratify=selector_stratify,
    )
    return (
        search_df.reset_index(drop=True),
        selector_df.reset_index(drop=True),
        final_df.reset_index(drop=True),
    )


def _make_preprocessor(cfg: Mapping[str, Any]) -> Preprocessor:
    step_order = cfg.get("step_order")
    pre_cfg = {k: v for k, v in dict(cfg).items() if k != "step_order"}
    if isinstance(step_order, list) and step_order:
        return Preprocessor(pre_cfg, step_order=step_order)
    return Preprocessor(pre_cfg)


def _should_retry_without_xgb(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "xgbclassifier" in msg and "n_classes_" in msg


def _fit_predict_autogluon(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    eval_metric: str,
    time_limit: int,
    verbose: bool,
    excluded_model_types: Optional[Sequence[str]] = None,
) -> Tuple[Any, Optional[str]]:
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore
        from autogluon.features.generators import IdentityFeatureGenerator  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("AutoGluon not available in environment") from exc

    temp_dir = os.path.join(tempfile.gettempdir(), f"autogluon_outer_{uuid.uuid4().hex}")
    try:
        predictor = TabularPredictor(
            label=target_column,
            path=temp_dir,
            problem_type=problem_type,
            eval_metric=eval_metric,
            verbosity=2 if verbose else 0,
        )
        fit_kwargs = dict(
            train_data=train_df,
            time_limit=int(time_limit),
            presets="best_quality",
            feature_generator=IdentityFeatureGenerator(),
            raise_on_no_models_fitted=False,
        )
        if excluded_model_types:
            fit_kwargs["excluded_model_types"] = list(excluded_model_types)
        predictor.fit(**fit_kwargs)
        try:
            if len(predictor.model_names()) == 0:
                return None, "no_models_fitted"
        except Exception:
            pass
        return predictor.predict(test_df), None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _evaluate_candidate_outer_holdout(
    *,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_column: str,
    cfg: Mapping[str, Any],
    time_limit: int,
    verbose: bool,
) -> Tuple[float, str]:
    X_train = train_df.drop(columns=[target_column]).copy()
    y_train = train_df[target_column].reset_index(drop=True)
    X_holdout = holdout_df.drop(columns=[target_column]).copy()
    y_holdout = holdout_df[target_column].reset_index(drop=True)

    problem_type, eval_metric = _detect_problem_type(y_train)
    pre = _make_preprocessor(cfg)
    transformed = pre.fit_transform(X_train, y_train)
    if isinstance(transformed, tuple):
        X_train_proc, y_train_proc = transformed
    else:
        X_train_proc = transformed
        y_train_proc = y_train
    X_holdout_proc = pre.transform(X_holdout)

    if X_train_proc is None or X_train_proc.shape[0] == 0:
        raise RuntimeError("candidate produced empty outer TRAIN data")
    if X_holdout_proc is None or X_holdout_proc.shape[0] == 0:
        raise RuntimeError("candidate produced empty outer HOLDOUT data")
    if len(y_train_proc) != len(X_train_proc):
        raise RuntimeError("outer train X/y length mismatch after preprocessing")
    if len(y_holdout) != len(X_holdout_proc):
        raise RuntimeError("evaluation X/y length mismatch after preprocessing")

    ag_train = X_train_proc.copy()
    ag_train[target_column] = y_train_proc.reset_index(drop=True)
    ag_test = X_holdout_proc.copy()
    try:
        preds, issue = _fit_predict_autogluon(
            train_df=ag_train,
            test_df=ag_test,
            target_column=target_column,
            problem_type=problem_type,
            eval_metric=eval_metric,
            time_limit=int(time_limit),
            verbose=verbose,
        )
    except Exception as exc:
        if not _should_retry_without_xgb(exc):
            raise
        preds, issue = _fit_predict_autogluon(
            train_df=ag_train,
            test_df=ag_test,
            target_column=target_column,
            problem_type=problem_type,
            eval_metric=eval_metric,
            time_limit=int(time_limit),
            verbose=verbose,
            excluded_model_types=["XGB"],
        )
    if issue == "no_models_fitted":
        raise RuntimeError("AutoGluon fitted no models")

    if problem_type == "regression":
        return float(r2_score(y_holdout, preds)), "r2"
    return float(accuracy_score(y_holdout, preds)), "accuracy"


def _append_candidate(
    candidates: Dict[str, Tuple[str, Dict[str, Any]]],
    *,
    source: str,
    cfg: Any,
    options: Mapping[str, Sequence[str]],
) -> None:
    if not isinstance(cfg, Mapping):
        return
    cfg_dict = dict(cfg)
    sig = _config_signature(cfg_dict, options)
    if sig not in candidates:
        candidates[sig] = (source, cfg_dict)


def _collect_candidates(
    *,
    no_search_rec: Mapping[str, Any],
    search_rec: Mapping[str, Any],
    options: Mapping[str, Sequence[str]],
    include_inner_topk: bool,
    inner_topk: int,
) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    candidates: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    _append_candidate(candidates, source="no_search_selected", cfg=no_search_rec.get("pipeline_config"), options=options)
    _append_candidate(candidates, source="search_selected", cfg=search_rec.get("pipeline_config"), options=options)

    if not include_inner_topk:
        return candidates

    for item in list(no_search_rec.get("top_candidates_evaluated") or [])[: max(0, int(inner_topk))]:
        if isinstance(item, (list, tuple)) and item:
            _append_candidate(candidates, source="no_search_inner_topk", cfg=item[0], options=options)

    for item in list(search_rec.get("aco_results") or [])[: max(0, int(inner_topk))]:
        if isinstance(item, (list, tuple)) and item:
            _append_candidate(candidates, source="search_inner_topk", cfg=item[0], options=options)
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid outer-holdout validation for retrieval and ACO search")
    parser.add_argument("--root", default=os.environ.get("ROOT", str(ROOT)))
    parser.add_argument("--performance-matrix", required=True)
    parser.add_argument("--metafeatures", required=True)
    parser.add_argument("--metafeatures-id-column", default=None)
    parser.add_argument("--pipeline-configs", required=True)
    parser.add_argument("--allow-inferred-pipeline-configs", action="store_true")
    parser.add_argument("--dataset-source", choices=["openml", "kaggle"], default="openml")
    parser.add_argument("--openml-local-folder", default=None)
    parser.add_argument("--kaggle-data-folder", default=None)
    parser.add_argument("--kaggle-target-column", default="target")
    parser.add_argument("--dataset-ids", nargs="+", required=True)
    parser.add_argument("--selector-size", type=float, default=0.2)
    parser.add_argument("--final-test-size", type=float, default=0.2)
    parser.add_argument("--outer-seed", type=int, default=2026)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--require-autogluon", action="store_true")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--eval-k", type=int, default=3)
    parser.add_argument("--heuristic-top-k", type=int, default=1)
    parser.add_argument("--heuristic-top-l", type=int, default=1)
    parser.add_argument("--dataset-weighting", choices=["similarity", "equality"], default="similarity")
    parser.add_argument("--heuristic-transfer-method", choices=["weighted_topk_topl", "legacy_weighted_average"], default="weighted_topk_topl")
    parser.add_argument("--similarity-temperature", type=float, default=1.0)
    parser.add_argument("--eta-floor", type=float, default=0.05)
    parser.add_argument("--n-ants", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--evaporation", type=float, default=0.2)
    parser.add_argument("--top-k-pheromone", type=int, default=3)
    parser.add_argument("--aco-weight-method", default="rank")
    parser.add_argument("--aco-markov-order", type=int, default=2)
    parser.add_argument("--aco-lambda-smooth", type=float, default=0.0)
    parser.add_argument("--aco-early-stop-rounds", type=int, default=0)
    parser.add_argument("--aco-min-improvement", type=float, default=0.0)
    parser.add_argument("--per-feature-independent-search", action="store_true")
    parser.add_argument("--per-feature-steps", default="imputation,scaling,encoding")
    parser.add_argument("--per-feature-early-stop-rounds", type=int, default=2)
    parser.add_argument("--per-feature-min-improvement", type=float, default=0.001)
    parser.add_argument("--per-feature-feature-patience", type=int, default=0)
    parser.add_argument("--per-feature-feature-min-improvement", type=float, default=0.0)
    parser.add_argument("--per-feature-max-features", type=int, default=0)
    parser.add_argument("--legacy-notebook-aco", action="store_true")
    parser.add_argument("--notebook-legacy-options", action="store_true")
    parser.add_argument("--notebook-legacy-mode", action="store_true")
    parser.add_argument("--metric-path", default=None)
    parser.add_argument("--train-metric-inline", action="store_true")
    parser.add_argument("--no-train-metric-inline", action="store_true")
    parser.add_argument("--metric-hidden-dim", type=int, default=64)
    parser.add_argument("--metric-embed-dim", type=int, default=64)
    parser.add_argument("--metric-epochs", type=int, default=100)
    parser.add_argument("--metric-lr", type=float, default=1e-3)
    parser.add_argument(
        "--metric-similarity-target",
        choices=["rank_cosine", "row_zscore_cosine", "row_minmax_cosine", "legacy_global_zscore_cosine"],
        default="rank_cosine",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-autogluon-topk", type=int, default=3)
    parser.add_argument("--proxy-profile", choices=["default", "robust"], default="default")
    parser.add_argument("--proxy-seeds", default=None)
    parser.add_argument("--proxy-clf-model", choices=["logreg", "linear_svm", "random_forest", "extra_trees", "knn", "hist_gbdt"], default="logreg")
    parser.add_argument("--proxy-reg-model", choices=["ensemble", "linear", "random_forest"], default="ensemble")
    parser.add_argument("--proxy-logreg-max-iter", type=int, default=3000)
    parser.add_argument("--include-inner-topk-candidates", action="store_true")
    parser.add_argument("--inner-topk-candidates", type=int, default=3)
    parser.add_argument("--evaluate-all-candidates-on-final", action="store_true")
    parser.add_argument("--output-dir", default="/kaggle/working/rq3_hybrid_outer_holdout")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.notebook_legacy_mode:
        args.notebook_legacy_options = True
        args.heuristic_transfer_method = "legacy_weighted_average"
        args.dataset_weighting = "equality"
        args.legacy_notebook_aco = True
        args.aco_lambda_smooth = 0.7
        args.proxy_logreg_max_iter = 1000
        args.metric_similarity_target = "legacy_global_zscore_cosine"
        args.metric_hidden_dim = 32
        args.metric_embed_dim = 32
        if not args.metric_path:
            args.train_metric_inline = True
    if args.no_train_metric_inline:
        args.train_metric_inline = False

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    openml_local = Path(args.openml_local_folder or root / "test_data_local")
    kaggle_data = Path(args.kaggle_data_folder or root / "test_data_local")

    perf = pd.read_csv(args.performance_matrix, index_col=0)
    raw_meta = pd.read_csv(args.metafeatures)
    meta = _maybe_set_meta_index(raw_meta, perf, args.metafeatures_id_column)
    configs = _load_pipeline_configs(Path(args.pipeline_configs), perf, bool(args.allow_inferred_pipeline_configs))

    recommender = MetaPipelineRecommender(perf, meta, configs, verbose=bool(args.verbose))
    if args.metric_path:
        recommender.load_metric(args.metric_path)
    elif args.train_metric_inline:
        if args.verbose:
            print(
                "Training Siamese metric inline: "
                f"hidden_dim={args.metric_hidden_dim}, embed_dim={args.metric_embed_dim}, "
                f"epochs={args.metric_epochs}, target={args.metric_similarity_target}"
            )
        recommender.train_metric(
            method="regression",
            hidden_dim=int(args.metric_hidden_dim),
            embed_dim=int(args.metric_embed_dim),
            epochs=int(args.metric_epochs),
            lr=float(args.metric_lr),
            seed=int(args.seed),
            similarity_target=str(args.metric_similarity_target),
            score_direction="higher_is_better",
        )

    proxy_settings = _build_proxy_settings(args)
    dataset_ids = _parse_dataset_ids(args.dataset_ids)
    rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    for run_idx, dataset_id in enumerate(dataset_ids, start=1):
        start = time.perf_counter()
        if args.verbose:
            print(f"\n=== Hybrid three-way validation dataset {dataset_id} ({run_idx}/{len(dataset_ids)}) ===")
        try:
            dataset, df = _load_dataset_as_frame(
                dataset_id,
                dataset_source=str(args.dataset_source),
                openml_local_folder=openml_local,
                kaggle_data_folder=kaggle_data,
                kaggle_target_column=str(args.kaggle_target_column),
                verbose=bool(args.verbose),
            )
            search_df, selector_df, final_df = _three_way_split(
                df,
                "target",
                selector_size=float(args.selector_size),
                final_test_size=float(args.final_test_size),
                seed=int(args.outer_seed),
            )
            options, option_notes = _adapt_options_to_dataset(_build_options(args), search_df.drop(columns=["target"]))
            for note in option_notes:
                if args.verbose:
                    print(f"  Auto option guard: {note}")

            def mf_func(_df: pd.DataFrame, _dataset: Mapping[str, Any] = dataset) -> Dict[str, Any]:
                return _lookup_metafeatures(_dataset, meta)

            aco_params = _build_aco_params(args, dataset_id)
            no_search_rec = recommender.recommend(
                new_dataset=search_df,
                target_column="target",
                options=options,
                k=max(1, int(args.k)),
                eval_k=max(1, int(args.eval_k)),
                use_autogluon=True,
                use_aco=False,
                aco_params=aco_params,
                time_limit_per_model=int(args.time_limit),
                metafeatures_func=mf_func,
                proxy_settings=proxy_settings,
                final_autogluon_topk=max(1, int(args.final_autogluon_topk)),
            )
            search_rec = recommender.recommend(
                new_dataset=search_df,
                target_column="target",
                options=options,
                k=max(1, int(args.k)),
                eval_k=max(1, int(args.eval_k)),
                use_autogluon=True,
                use_aco=True,
                optimizer="aco",
                aco_params=aco_params,
                time_limit_per_model=int(args.time_limit),
                metafeatures_func=mf_func,
                proxy_settings=proxy_settings,
                final_autogluon_topk=max(1, int(args.final_autogluon_topk)),
            )

            candidates = _collect_candidates(
                no_search_rec=no_search_rec,
                search_rec=search_rec,
                options=options,
                include_inner_topk=bool(args.include_inner_topk_candidates),
                inner_topk=int(args.inner_topk_candidates),
            )
            if not candidates:
                raise RuntimeError("No hybrid candidates collected")

            selector_scored: List[Tuple[str, str, Dict[str, Any], float, str, str, str]] = []
            for sig, (source, cfg) in candidates.items():
                try:
                    score, metric = _evaluate_candidate_outer_holdout(
                        train_df=search_df,
                        holdout_df=selector_df,
                        target_column="target",
                        cfg=cfg,
                        time_limit=int(args.time_limit),
                        verbose=bool(args.verbose),
                    )
                    status = "ok"
                    error = ""
                except Exception as exc:
                    score = np.nan
                    metric = ""
                    status = "failed"
                    error = str(exc)
                selector_scored.append((sig, source, cfg, score, metric, status, error))
                if args.verbose:
                    print(f"  selector {source}: {score if np.isfinite(score) else np.nan:.4f} | {sig}")

            no_sig = _config_signature(no_search_rec.get("pipeline_config") or {}, options)
            search_sig = _config_signature(search_rec.get("pipeline_config") or {}, options)
            no_selector = next((float(sc) for sig, _src, _cfg, sc, _m, st, _err in selector_scored if sig == no_sig and st == "ok"), np.nan)
            search_selector = next((float(sc) for sig, _src, _cfg, sc, _m, st, _err in selector_scored if sig == search_sig and st == "ok"), np.nan)

            valid_selector = [item for item in selector_scored if item[5] == "ok" and np.isfinite(item[3])]
            if not valid_selector:
                raise RuntimeError("No candidate produced valid selector score")
            best_sig, best_source, best_cfg, best_selector_score, best_selector_metric, _status, _error = max(
                valid_selector,
                key=lambda item: float(item[3]),
            )

            final_train_df = pd.concat([search_df, selector_df], ignore_index=True)
            final_eval_sigs = {no_sig, search_sig, best_sig}
            if args.evaluate_all_candidates_on_final:
                final_eval_sigs = {sig for sig, *_rest in selector_scored}

            final_scores: Dict[str, Tuple[float, str, str, str]] = {}
            for sig, source, cfg, _selector_score, _selector_metric, _selector_status, _selector_error in selector_scored:
                if sig not in final_eval_sigs:
                    continue
                try:
                    final_score, final_metric = _evaluate_candidate_outer_holdout(
                        train_df=final_train_df,
                        holdout_df=final_df,
                        target_column="target",
                        cfg=cfg,
                        time_limit=int(args.time_limit),
                        verbose=bool(args.verbose),
                    )
                    final_scores[sig] = (final_score, final_metric, "ok", "")
                except Exception as exc:
                    final_scores[sig] = (np.nan, "", "failed", str(exc))
                if args.verbose:
                    final_value = final_scores[sig][0]
                    print(f"  final {source}: {final_value if np.isfinite(final_value) else np.nan:.4f} | {sig}")

            for sig, source, cfg, selector_score, selector_metric, selector_status, selector_error in selector_scored:
                final_score, final_metric, final_status, final_error = final_scores.get(sig, (np.nan, "", "", ""))
                candidate_rows.append(
                    {
                        "dataset_id": str(dataset_id),
                        "candidate_source": source,
                        "signature": sig,
                        "selector_score": selector_score,
                        "selector_metric": selector_metric,
                        "selector_status": selector_status,
                        "selector_error": selector_error,
                        "final_score": final_score,
                        "final_metric": final_metric,
                        "final_status": final_status,
                        "final_error": final_error,
                        "selected_by_hybrid": bool(sig == best_sig),
                        "pipeline_config": _display_config(cfg, options),
                    }
                )

            no_final = final_scores.get(no_sig, (np.nan, "", "", ""))[0]
            search_final = final_scores.get(search_sig, (np.nan, "", "", ""))[0]
            hybrid_final, hybrid_final_metric, hybrid_final_status, hybrid_final_error = final_scores.get(
                best_sig,
                (np.nan, "", "", ""),
            )
            if not np.isfinite(hybrid_final):
                raise RuntimeError(f"Hybrid selected candidate failed final evaluation: {hybrid_final_error}")

            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "status": "ok",
                    "error": "",
                    "outer_seed": int(args.outer_seed),
                    "selector_size": float(args.selector_size),
                    "final_test_size": float(args.final_test_size),
                    "search_rows": int(len(search_df)),
                    "selector_rows": int(len(selector_df)),
                    "final_rows": int(len(final_df)),
                    "candidate_count": int(len(candidates)),
                    "no_search_inner_score": no_search_rec.get("final_performance", np.nan),
                    "no_search_inner_method": (no_search_rec.get("final_evaluation") or {}).get("method", ""),
                    "no_search_selector_score": no_selector,
                    "no_search_final_score": no_final,
                    "no_search_pipeline": _display_config(no_search_rec.get("pipeline_config") or {}, options),
                    "search_proxy_score": search_rec.get("recommended_performance", np.nan),
                    "search_inner_score": search_rec.get("final_performance", np.nan),
                    "search_inner_method": (search_rec.get("final_evaluation") or {}).get("method", ""),
                    "search_selector_score": search_selector,
                    "search_final_score": search_final,
                    "search_pipeline": _display_config(search_rec.get("pipeline_config") or {}, options),
                    "hybrid_selector_score": float(best_selector_score),
                    "hybrid_selector_metric": best_selector_metric,
                    "hybrid_final_score": float(hybrid_final),
                    "hybrid_source": best_source,
                    "hybrid_final_metric": hybrid_final_metric,
                    "hybrid_final_status": hybrid_final_status,
                    "hybrid_pipeline": _display_config(best_cfg, options),
                    "hybrid_minus_no_search_final": (
                        float(hybrid_final - no_final) if np.isfinite(no_final) else np.nan
                    ),
                    "hybrid_minus_search_final": (
                        float(hybrid_final - search_final) if np.isfinite(search_final) else np.nan
                    ),
                    "search_minus_no_search_selector": (
                        float(search_selector - no_selector) if np.isfinite(search_selector) and np.isfinite(no_selector) else np.nan
                    ),
                    "search_minus_no_search_final": (
                        float(search_final - no_final) if np.isfinite(search_final) and np.isfinite(no_final) else np.nan
                    ),
                    "search_same_as_no_search": bool(search_sig == no_sig),
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
            )
            if args.verbose:
                print(f"Dataset {dataset_id} failed: {exc}")

    summary_df = pd.DataFrame(rows)
    candidate_df = pd.DataFrame(candidate_rows)
    summary_df.to_csv(out_dir / "hybrid_outer_holdout_summary.csv", index=False)
    candidate_df.to_csv(out_dir / "hybrid_outer_holdout_candidates.csv", index=False)

    ok = summary_df[summary_df.get("status", "") == "ok"].copy() if not summary_df.empty else pd.DataFrame()
    aggregate: Dict[str, Any] = {
        "n_datasets": int(ok["dataset_id"].nunique()) if not ok.empty else 0,
    }
    if not ok.empty:
        for col in [
            "no_search_selector_score",
            "search_selector_score",
            "no_search_final_score",
            "search_final_score",
            "hybrid_selector_score",
            "hybrid_final_score",
            "hybrid_minus_no_search_final",
            "hybrid_minus_search_final",
            "search_minus_no_search_selector",
            "search_minus_no_search_final",
        ]:
            aggregate[f"mean_{col}"] = float(pd.to_numeric(ok[col], errors="coerce").mean())
        aggregate["hybrid_beats_no_search_rate"] = float(
            (pd.to_numeric(ok["hybrid_minus_no_search_final"], errors="coerce") > 0).mean()
        )
        aggregate["hybrid_beats_search_rate"] = float(
            (pd.to_numeric(ok["hybrid_minus_search_final"], errors="coerce") > 0).mean()
        )
        aggregate["search_beats_no_search_rate"] = float(
            (pd.to_numeric(ok["search_minus_no_search_final"], errors="coerce") > 0).mean()
        )

    with (out_dir / "hybrid_outer_holdout_aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(pd.DataFrame([aggregate]).to_string(index=False))
    print(f"Saved hybrid outer-holdout outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
