"""Audit whether operator decomposition loses useful full-pipeline signal.

This diagnostic isolates three stages of the current approach:

1. Controlled operator transfer: evaluate each operator alone against a
   baseline pipeline on the target, then compare target-side operator lift with
   historical neighbor-side operator lift.
2. Eta predictive power: test whether the operators ranked highest by eta are
   actually the best target-side operators under the controlled evaluation.
3. Contextual operator contribution: remove one active operator at a time from
   the retrieved full pipeline and measure whether that operator helped inside
   its original pipeline context.
4. ACO search dynamics: summarize whether ACO proxy scores actually improve
   over iterations after receiving eta.

The full-pipeline vs eta-top comparison is still emitted as a secondary sanity
check, but the main isolation is per-operator. This avoids the trivial case
where topK=1/topL=1 makes the retrieved full pipeline and eta-top recomposition
nearly identical.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automl_aco.config import DEFAULT_PIPELINE_OPTIONS, NOTEBOOK_LEGACY_PIPELINE_OPTIONS
from automl_aco.data.loaders import load_kaggle_dataset, load_openml_dataset
from automl_aco.data.metafeatures import extract_enhanced_metafeatures
from automl_aco.metalearning.recommender import MetaPipelineRecommender
from automl_aco.search.evaluation import evaluate_candidates_autogluon, evaluate_candidates_simple
from automl_aco.search.heuristics import (
    aggregate_operator_heuristics,
    build_transfer_candidates,
    compute_dataset_similarities,
    compute_similarity_weights,
    normalize_eta_with_floor,
    select_top_k_neighbors,
    select_top_l_pipelines_per_neighbor,
)
from automl_aco.utils.operator_spec import base_operator_name


PIPELINE_STEPS = [
    "imputation",
    "scaling",
    "encoding",
    "feature_selection",
    "outlier_removal",
    "dimensionality_reduction",
]
RISKY_OPERATORS = {"iqr", "zscore", "lof", "isolation_forest", "pca", "svd"}


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
            if not text:
                continue
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


def _lookup_metafeatures(dataset: Dict[str, Any], meta_df: pd.DataFrame) -> Dict[str, Any]:
    dataset_id = dataset.get("id") if isinstance(dataset, dict) else None
    if dataset_id is None:
        return extract_enhanced_metafeatures(dataset, meta_features_df=meta_df)

    target_norm = _normalize_id(dataset_id)
    for idx in meta_df.index:
        if _normalize_id(idx) == target_norm:
            return meta_df.loc[idx].to_dict()
    return extract_enhanced_metafeatures(dataset, meta_features_df=meta_df)


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


def _load_pipeline_configs(path: Path, perf: pd.DataFrame, allow_inferred: bool) -> Tuple[List[Dict[str, Any]], List[str]]:
    with path.open("r", encoding="utf-8") as f:
        configs = json.load(f)

    existing = {str(cfg.get("name")) for cfg in configs if isinstance(cfg, dict) and cfg.get("name") is not None}
    missing = [str(name) for name in perf.index if str(name) not in existing]
    if missing:
        if not allow_inferred:
            sample = ", ".join(missing[:10])
            raise ValueError(
                "Performance matrix contains pipeline rows missing from pipeline_configs: "
                f"{sample}. Pass --allow-inferred-pipeline-configs for legacy/debug records."
            )
        configs = list(configs) + [_infer_pipeline_config_from_name(name) for name in missing]
    return configs, missing


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


def _coerce_to_options(cfg: Mapping[str, Any], options: Mapping[str, Sequence[str]], *, name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": name}
    for step, choices_raw in options.items():
        choices = [str(v) for v in choices_raw]
        raw = str(cfg.get(step, "none"))
        if raw in choices:
            out[step] = raw
            continue
        raw_base = base_operator_name(raw)
        matched = next((choice for choice in choices if base_operator_name(choice) == raw_base), None)
        if matched is not None:
            out[step] = matched
            continue
        out[step] = "none" if "none" in choices else choices[0]
    return out


def _eta_top_config(eta: Mapping[str, np.ndarray], options: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"name": "eta_top_decomposed"}
    for step, choices_raw in options.items():
        choices = [str(v) for v in choices_raw]
        vals = np.asarray(eta.get(step, np.ones(len(choices))), dtype=float)
        if vals.size != len(choices):
            vals = np.ones(len(choices), dtype=float)
        vals[~np.isfinite(vals)] = -np.inf
        cfg[step] = choices[int(np.argmax(vals))]
    return cfg


def _baseline_config(options: Mapping[str, Sequence[str]], *, name: str = "audit_baseline") -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"name": name}
    for step, choices_raw in options.items():
        choices = [str(v) for v in choices_raw]
        if step == "encoding" and "onehot" in choices:
            cfg[step] = "onehot"
        elif "none" in choices:
            cfg[step] = "none"
        else:
            cfg[step] = choices[0]
    return cfg


def _operator_only_config(
    step: str,
    operator: str,
    options: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    cfg = _baseline_config(options, name=f"operator_only:{step}={operator}")
    cfg[step] = str(operator)
    return cfg


def _minus_operator_config(
    cfg: Mapping[str, Any],
    step: str,
    options: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    out = dict(cfg)
    baseline = _baseline_config(options)
    out[step] = baseline.get(step, "none")
    out["name"] = f"{cfg.get('name', 'pipeline')}|minus:{step}"
    return out


def _score_sign(value: Any, min_abs: float = 1e-12) -> float:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        return np.nan
    val = float(value)
    if abs(val) <= min_abs:
        return 0.0
    return 1.0 if val > 0 else -1.0


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _spearman(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    data = pd.DataFrame({"x": xs, "y": ys})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.shape[0] < 2:
        return np.nan
    if data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan
    return float(data["x"].rank(method="average").corr(data["y"].rank(method="average")))


def _neighbor_operator_scores(
    *,
    perf: pd.DataFrame,
    cfg_by_name: Mapping[str, Mapping[str, Any]],
    neighbor_id: Any,
    step: str,
    operator: str,
) -> Dict[str, Any]:
    neighbor_col = neighbor_id if neighbor_id in perf.columns else None
    if neighbor_col is None:
        neighbor_norm = _normalize_id(neighbor_id)
        for col in perf.columns:
            if _normalize_id(col) == neighbor_norm:
                neighbor_col = col
                break

    if neighbor_col is None:
        return {
            "neighbor_baseline_score": np.nan,
            "neighbor_operator_best_score": np.nan,
            "neighbor_operator_mean_score": np.nan,
            "neighbor_operator_best_lift": np.nan,
            "neighbor_operator_mean_lift": np.nan,
            "neighbor_operator_best_pipeline": "",
            "neighbor_operator_support_count": 0,
        }

    scores = pd.to_numeric(perf[neighbor_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    baseline_score = float(scores.get("baseline", np.nan)) if "baseline" in scores.index else np.nan
    target_base = base_operator_name(operator)
    matched: List[Tuple[str, float]] = []
    for pipeline_name, cfg in cfg_by_name.items():
        if pipeline_name not in scores.index:
            continue
        score = scores.get(pipeline_name, np.nan)
        if pd.isna(score) or not np.isfinite(float(score)):
            continue
        if base_operator_name(cfg.get(step, "none")) == target_base:
            matched.append((pipeline_name, float(score)))

    if not matched:
        return {
            "neighbor_baseline_score": baseline_score,
            "neighbor_operator_best_score": np.nan,
            "neighbor_operator_mean_score": np.nan,
            "neighbor_operator_best_lift": np.nan,
            "neighbor_operator_mean_lift": np.nan,
            "neighbor_operator_best_pipeline": "",
            "neighbor_operator_support_count": 0,
        }

    best_name, best_score = max(matched, key=lambda item: item[1])
    mean_score = float(np.mean([score for _name, score in matched]))
    return {
        "neighbor_baseline_score": baseline_score,
        "neighbor_operator_best_score": best_score,
        "neighbor_operator_mean_score": mean_score,
        "neighbor_operator_best_lift": best_score - baseline_score if np.isfinite(baseline_score) else np.nan,
        "neighbor_operator_mean_lift": mean_score - baseline_score if np.isfinite(baseline_score) else np.nan,
        "neighbor_operator_best_pipeline": best_name,
        "neighbor_operator_support_count": int(len(matched)),
    }


def _nanmean(values: Sequence[Any]) -> float:
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return np.nan
    return float(vals.mean())


def _nanrate(values: Sequence[Any]) -> float:
    vals = pd.Series(list(values)).dropna()
    if vals.empty:
        return np.nan
    return float(vals.astype(bool).mean())


def _sign_agreement_rate(left: Sequence[Any], right: Sequence[Any]) -> float:
    signs = pd.DataFrame({"left": [_score_sign(v) for v in left], "right": [_score_sign(v) for v in right]})
    signs = signs.replace([np.inf, -np.inf], np.nan).dropna()
    if signs.empty:
        return np.nan
    return float((signs["left"] == signs["right"]).mean())


def _score_candidates(
    df: pd.DataFrame,
    candidates: Sequence[Dict[str, Any]],
    *,
    backend: str,
    time_limit: int,
    require_autogluon: bool,
    proxy_settings: Mapping[str, Any],
    verbose: bool,
) -> Tuple[Dict[str, float], str]:
    if not candidates:
        return {}, "none"

    if backend == "autogluon":
        try:
            _best_cfg, _best_score, sorted_results, _unsorted = evaluate_candidates_autogluon(
                dataset=df,
                target_column="target",
                candidate_configs=[dict(cfg) for cfg in candidates],
                time_limit_per_model=int(time_limit),
                verbose=verbose,
            )
            return {
                str(cfg.get("__audit_signature__", cfg.get("name", idx))): float(score)
                for idx, (cfg, score) in enumerate(sorted_results)
            }, "autogluon"
        except Exception:
            if require_autogluon:
                raise

    _best_cfg, _best_score, sorted_results, _unsorted = evaluate_candidates_simple(
        dataset=df,
        target_column="target",
        candidate_configs=[dict(cfg) for cfg in candidates],
        proxy_settings=dict(proxy_settings),
        verbose=verbose,
    )
    return {
        str(cfg.get("__audit_signature__", cfg.get("name", idx))): float(score)
        for idx, (cfg, score) in enumerate(sorted_results)
    }, "simple"


def _prepare_eval_candidates(
    configs: Sequence[Mapping[str, Any]],
    options: Mapping[str, Sequence[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_sig: Dict[str, Dict[str, Any]] = {}
    for cfg in configs:
        sig = _config_signature(cfg, options)
        if sig in by_sig:
            continue
        clean = {k: v for k, v in dict(cfg).items() if k in set(options.keys()) | {"name", "step_order"}}
        clean["__audit_signature__"] = sig
        by_sig[sig] = clean
    return list(by_sig.values()), by_sig


def _eta_step_stats(
    *,
    dataset_id: Any,
    eta: Mapping[str, np.ndarray],
    raw_eta: Mapping[str, np.ndarray],
    options: Mapping[str, Sequence[str]],
    retrieved_top_cfg: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    margins: List[float] = []
    entropies: List[float] = []
    matches: List[bool] = []
    flat_steps = 0

    for step, choices_raw in options.items():
        choices = [str(v) for v in choices_raw]
        vals = np.asarray(eta.get(step, np.ones(len(choices))), dtype=float)
        raw_vals = np.asarray(raw_eta.get(step, np.ones(len(choices))), dtype=float)
        if vals.size != len(choices):
            vals = np.ones(len(choices), dtype=float)
        order = np.argsort(-vals)
        top_idx = int(order[0])
        second = float(vals[int(order[1])]) if len(order) > 1 else float(vals[top_idx])
        margin = float(vals[top_idx] - second)
        margins.append(margin)
        if margin <= 1e-8:
            flat_steps += 1

        positive = np.clip(vals.astype(float), 0.0, None)
        if positive.sum() > 0 and len(positive) > 1:
            probs = positive / positive.sum()
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs)))
        else:
            entropy = 0.0
        entropies.append(entropy)

        retrieved_operator = str(retrieved_top_cfg.get(step, "none"))
        top_operator = choices[top_idx]
        top_matches = base_operator_name(top_operator) == base_operator_name(retrieved_operator)
        matches.append(top_matches)

        for rank, idx in enumerate(order, start=1):
            op = choices[int(idx)]
            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "step": step,
                    "operator": op,
                    "operator_base": base_operator_name(op),
                    "eta_norm": float(vals[int(idx)]),
                    "eta_raw": float(raw_vals[int(idx)]) if int(idx) < raw_vals.size else np.nan,
                    "eta_rank": int(rank),
                    "is_eta_top": bool(rank == 1),
                    "retrieved_top_operator": retrieved_operator,
                    "matches_retrieved_top_operator": bool(
                        base_operator_name(op) == base_operator_name(retrieved_operator)
                    ),
                    "top1_margin_for_step": margin,
                    "entropy_norm_for_step": entropy,
                }
            )

    return rows, {
        "eta_mean_top1_margin": float(np.mean(margins)) if margins else np.nan,
        "eta_min_top1_margin": float(np.min(margins)) if margins else np.nan,
        "eta_mean_entropy_norm": float(np.mean(entropies)) if entropies else np.nan,
        "eta_flat_step_count": int(flat_steps),
        "eta_retrieved_operator_match_rate": float(np.mean(matches)) if matches else np.nan,
        "eta_retrieved_operator_mismatch_count": int(sum(not m for m in matches)),
    }


def _risky_count(cfg: Mapping[str, Any], options: Mapping[str, Sequence[str]]) -> int:
    count = 0
    for step in options.keys():
        if base_operator_name(cfg.get(step, "none")) in RISKY_OPERATORS:
            count += 1
    return count


def _history_summary(history: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [row for row in history if isinstance(row, Mapping)]

    def finite_values(key: str) -> List[float]:
        vals: List[float] = []
        for row in rows:
            val = row.get(key)
            if isinstance(val, (int, float)) and np.isfinite(float(val)):
                vals.append(float(val))
        return vals

    global_vals = finite_values("global_best_score") or finite_values("best_score")
    iter_vals = finite_values("iteration_best_score")
    valid_counts = finite_values("valid_count")
    cache_sizes = finite_values("cache_size")
    improved = [bool(row.get("global_improved")) for row in rows]

    if not global_vals:
        return {
            "aco_proxy_initial_global_best": np.nan,
            "aco_proxy_final_global_best": np.nan,
            "aco_proxy_gain": np.nan,
            "aco_proxy_iteration_best_max": np.nan,
            "aco_history_iterations": int(len(rows)),
            "aco_proxy_improving_iterations": int(sum(improved)),
            "aco_proxy_unique_evaluated": np.nan,
            "aco_proxy_valid_evaluations": np.nan,
        }

    return {
        "aco_proxy_initial_global_best": float(global_vals[0]),
        "aco_proxy_final_global_best": float(global_vals[-1]),
        "aco_proxy_gain": float(global_vals[-1] - global_vals[0]),
        "aco_proxy_iteration_best_max": float(max(iter_vals)) if iter_vals else np.nan,
        "aco_history_iterations": int(len(rows)),
        "aco_proxy_improving_iterations": int(sum(improved)),
        "aco_proxy_unique_evaluated": int(max(cache_sizes)) if cache_sizes else np.nan,
        "aco_proxy_valid_evaluations": int(sum(valid_counts)) if valid_counts else np.nan,
    }


def _build_proxy_settings(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [42]
    if args.proxy_seeds:
        parsed = []
        for token in str(args.proxy_seeds).split(","):
            token = token.strip()
            if token:
                parsed.append(int(token))
        if parsed:
            seeds = parsed
    return {
        "split_seeds": seeds,
        "active_step_penalty": 0.0,
        "row_drop_penalty": 0.0,
        "imputation_low_missing_penalty": 0.0,
        "low_missing_threshold": 0.0,
        "outlier_removal_penalty": 0.0,
        "dimred_small_feature_penalty": 0.0,
        "dimred_small_feature_threshold": 0,
        "verbose_components": bool(args.verbose),
        "classification_model": str(args.proxy_clf_model),
        "regression_model": str(args.proxy_reg_model),
        "logreg_max_iter": int(args.proxy_logreg_max_iter),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit full-pipeline vs decomposed-operator transfer")
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
    parser.add_argument("--backend", choices=["simple", "autogluon"], default="simple")
    parser.add_argument(
        "--operator-effect-backend",
        choices=["same", "simple", "autogluon"],
        default="simple",
        help=(
            "Backend for controlled/contextual per-operator tests. Use 'same' to match --backend. "
            "The default simple backend keeps the audit affordable because this evaluates many candidates."
        ),
    )
    parser.add_argument("--time-limit", type=int, default=240)
    parser.add_argument("--require-autogluon", action="store_true")
    parser.add_argument("--k", type=int, default=1)
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
    parser.add_argument("--legacy-notebook-aco", action="store_true")
    parser.add_argument("--notebook-legacy-options", action="store_true")
    parser.add_argument("--notebook-legacy-mode", action="store_true")
    parser.add_argument("--metric-path", default=None)
    parser.add_argument("--train-metric-inline", action="store_true")
    parser.add_argument(
        "--no-train-metric-inline",
        action="store_true",
        help=(
            "Disable inline Siamese metric training even when --notebook-legacy-mode would enable it. "
            "Without --metric-path, retrieval falls back to raw metafeature cosine."
        ),
    )
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
    parser.add_argument("--proxy-seeds", default=None)
    parser.add_argument("--proxy-clf-model", choices=["logreg", "linear_svm", "random_forest", "extra_trees", "knn", "hist_gbdt"], default="logreg")
    parser.add_argument("--proxy-reg-model", choices=["ensemble", "linear", "random_forest"], default="ensemble")
    parser.add_argument("--proxy-logreg-max-iter", type=int, default=3000)
    parser.add_argument("--output-dir", default="/kaggle/working/rq3_operator_decomposition_audit")
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
    configs, inferred_names = _load_pipeline_configs(
        Path(args.pipeline_configs),
        perf,
        allow_inferred=bool(args.allow_inferred_pipeline_configs),
    )
    cfg_by_name = {str(cfg.get("name")): cfg for cfg in configs if isinstance(cfg, dict) and cfg.get("name") is not None}
    options = {
        step: list(vals)
        for step, vals in (
            NOTEBOOK_LEGACY_PIPELINE_OPTIONS if args.notebook_legacy_options else DEFAULT_PIPELINE_OPTIONS
        ).items()
    }
    proxy_settings = _build_proxy_settings(args)

    recommender = MetaPipelineRecommender(perf, meta, configs, verbose=args.verbose)
    similarity_source = "raw_metafeature_cosine"
    if args.metric_path:
        print(f"Similarity source: loaded Siamese metric from {args.metric_path}")
        recommender.load_metric(args.metric_path)
        similarity_source = "loaded_siamese_metric"
    elif args.train_metric_inline:
        print(
            "Similarity source: training Siamese metric inline "
            f"(hidden_dim={int(args.metric_hidden_dim)}, "
            f"embed_dim={int(args.metric_embed_dim)}, "
            f"epochs={int(args.metric_epochs)}, "
            f"seed={int(args.seed)}, "
            f"target={str(args.metric_similarity_target)})"
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
        similarity_source = "inline_siamese_metric"
    else:
        print("Similarity source: raw metafeature cosine (no Siamese metric loaded/trained)")

    run_metadata = {
        "similarity_source": similarity_source,
        "metric_path": args.metric_path,
        "train_metric_inline": bool(args.train_metric_inline),
        "metric_similarity_target": str(args.metric_similarity_target),
        "metric_hidden_dim": int(args.metric_hidden_dim),
        "metric_embed_dim": int(args.metric_embed_dim),
        "metric_epochs": int(args.metric_epochs),
        "seed": int(args.seed),
        "performance_matrix": str(args.performance_matrix),
        "metafeatures": str(args.metafeatures),
        "pipeline_configs": str(args.pipeline_configs),
        "aligned_dataset_count": int(len(recommender.metafeatures_df.index)),
    }
    with (out_dir / "similarity_run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, default=str)

    summary_rows: List[Dict[str, Any]] = []
    eta_rows: List[Dict[str, Any]] = []
    retrieval_rows: List[Dict[str, Any]] = []
    neighbor_rank_rows: List[Dict[str, Any]] = []
    aco_history_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    controlled_operator_rows: List[Dict[str, Any]] = []
    contextual_operator_rows: List[Dict[str, Any]] = []
    eta_target_step_rows: List[Dict[str, Any]] = []

    dataset_ids = _parse_dataset_ids(args.dataset_ids)
    for dataset_id in dataset_ids:
        start = time.perf_counter()
        try:
            dataset, df = _load_dataset_as_frame(
                dataset_id,
                dataset_source=str(args.dataset_source),
                openml_local_folder=openml_local,
                kaggle_data_folder=kaggle_data,
                kaggle_target_column=str(args.kaggle_target_column),
                verbose=bool(args.verbose),
            )

            target_mf = _lookup_metafeatures(dataset, meta)
            target_mf_df = pd.DataFrame([target_mf]).reindex(columns=recommender.metafeatures_df.columns, fill_value=0)
            target_mf_scaled = recommender.scaler.transform(recommender.imputer.transform(target_mf_df)).ravel()
            metric_sims = recommender._compute_dataset_similarities(target_mf_scaled)
            dataset_similarity_scores = dict(metric_sims)
            dataset_similarities = compute_dataset_similarities(
                metafeatures_df=recommender.metafeatures_df,
                new_metafeatures=target_mf_scaled,
                metafeatures_scaled=recommender.metafeatures_scaled,
                dataset_similarity_scores=dataset_similarity_scores,
            )
            top_neighbors = select_top_k_neighbors(
                dataset_similarities=dataset_similarities,
                top_k=max(1, int(args.heuristic_top_k)),
                query_dataset_id=dataset_id,
            )
            ranking_neighbors = select_top_k_neighbors(
                dataset_similarities=dataset_similarities,
                top_k=max(10, int(args.heuristic_top_k)),
                query_dataset_id=dataset_id,
            )
            for rank, (neighbor_id, sim) in enumerate(ranking_neighbors, start=1):
                neighbor_rank_rows.append(
                    {
                        "dataset_id": str(dataset_id),
                        "rank": int(rank),
                        "neighbor_id": str(neighbor_id),
                        "similarity": float(sim),
                        "selected_for_transfer": bool(
                            any(_normalize_id(neighbor_id) == _normalize_id(sel_id) for sel_id, _ in top_neighbors)
                        ),
                        "similarity_source": similarity_source,
                    }
                )
            similarity_weights = compute_similarity_weights(
                top_k_neighbors=top_neighbors,
                dataset_weighting=str(args.dataset_weighting),
                similarity_temperature=float(args.similarity_temperature),
            )
            top_l_pipelines = select_top_l_pipelines_per_neighbor(
                performance_matrix=recommender.performance_matrix_imputed,
                top_k_neighbors=top_neighbors,
                top_l=max(1, int(args.heuristic_top_l)),
                score_direction="higher_is_better",
            )
            transfer_candidates = build_transfer_candidates(
                performance_matrix=recommender.performance_matrix_imputed,
                top_k_neighbors=top_neighbors,
                top_l_pipelines=top_l_pipelines,
                similarity_weights=similarity_weights,
                score_direction="higher_is_better",
            )
            raw_eta = aggregate_operator_heuristics(
                transfer_candidates=transfer_candidates,
                pipeline_configs=configs,
                options=options,
            )
            eta = recommender._compute_aco_heuristic(
                target_mf_scaled,
                options,
                dataset_weighting=str(args.dataset_weighting),
                top_k=max(1, int(args.heuristic_top_k)),
                use_top_pipelines_from_metric=True,
                recommend_kwargs={
                    "new_dataset": df,
                    "target_column": "target",
                    "options": options,
                    "k": 5,
                    "eval_k": 3,
                    "use_aco": False,
                    "time_limit_per_model": int(args.time_limit),
                    "use_autogluon": False,
                    "metafeatures_func": lambda _df, _dataset=dataset: _lookup_metafeatures(_dataset, meta),
                },
                top_l=max(1, int(args.heuristic_top_l)),
                similarity_temperature=float(args.similarity_temperature),
                eta_floor=float(args.eta_floor),
                heuristic_transfer_method=str(args.heuristic_transfer_method),
                score_direction="higher_is_better",
                query_dataset_id=dataset_id,
            )
            eta_raw_source = "weighted_topk_topl_raw"
            if str(args.heuristic_transfer_method).strip().lower() == "legacy_weighted_average":
                # Legacy transfer returns only eta_norm, so keep eta_raw aligned
                # with the actual ACO handoff instead of reporting unrelated raw
                # values from the modern weighted_topk_topl path.
                eta_raw_source = "legacy_eta_norm_no_raw_available"
                raw_eta = {
                    step: np.asarray(eta.get(step, np.ones(len(vals), dtype=float)), dtype=float)
                    for step, vals in options.items()
                }

            retrieved_cfgs: List[Dict[str, Any]] = []
            for neighbor_id, pipes in top_l_pipelines.items():
                for pipe in pipes:
                    pname = str(pipe.get("pipeline"))
                    raw_cfg = cfg_by_name.get(pname)
                    if raw_cfg is None:
                        continue
                    cfg = _coerce_to_options(raw_cfg, options, name=pname)
                    retrieved_cfgs.append(cfg)
                    retrieval_rows.append(
                        {
                            "dataset_id": str(dataset_id),
                            "neighbor_id": str(neighbor_id),
                            "neighbor_similarity": float(dict(top_neighbors).get(neighbor_id, np.nan)),
                            "neighbor_weight": float(similarity_weights.get(neighbor_id, np.nan)),
                            "pipeline_name": pname,
                            "pipeline_rank": int(pipe.get("pipeline_rank_within_neighbor", 0)),
                            "historical_score": float(pipe.get("historical_score", np.nan)),
                            "pipeline_config": _display_config(cfg, options),
                        }
                    )

            if not retrieved_cfgs:
                raise RuntimeError("No retrieved full-pipeline configs found.")

            retrieved_top_cfg = retrieved_cfgs[0]
            eta_top_cfg = _eta_top_config(eta, options)
            eta_detail, eta_stats = _eta_step_stats(
                dataset_id=dataset_id,
                eta=eta,
                raw_eta=raw_eta,
                options=options,
                retrieved_top_cfg=retrieved_top_cfg,
            )
            eta_rows.extend(eta_detail)

            eval_configs, by_sig = _prepare_eval_candidates([*retrieved_cfgs, eta_top_cfg], options)
            scores_by_sig, eval_method = _score_candidates(
                df,
                eval_configs,
                backend=str(args.backend),
                time_limit=int(args.time_limit),
                require_autogluon=bool(args.require_autogluon),
                proxy_settings=proxy_settings,
                verbose=bool(args.verbose),
            )
            retrieved_scores = []
            for cfg in retrieved_cfgs:
                sig = _config_signature(cfg, options)
                score = scores_by_sig.get(sig, np.nan)
                retrieved_scores.append((cfg, sig, score))
                candidate_rows.append(
                    {
                        "dataset_id": str(dataset_id),
                        "candidate_type": "retrieved_full_pipeline",
                        "candidate_name": str(cfg.get("name", "")),
                        "signature": sig,
                        "score": score,
                        "eval_method": eval_method,
                        "pipeline_config": _display_config(cfg, options),
                    }
                )
            eta_sig = _config_signature(eta_top_cfg, options)
            eta_top_score = scores_by_sig.get(eta_sig, np.nan)
            candidate_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "candidate_type": "eta_top_decomposed",
                    "candidate_name": str(eta_top_cfg.get("name", "")),
                    "signature": eta_sig,
                    "score": eta_top_score,
                    "eval_method": eval_method,
                    "pipeline_config": _display_config(eta_top_cfg, options),
                }
            )

            effect_backend = str(args.backend) if str(args.operator_effect_backend) == "same" else str(args.operator_effect_backend)
            baseline_cfg = _baseline_config(options)
            baseline_sig = _config_signature(baseline_cfg, options)

            eta_norm_by_step_op: Dict[Tuple[str, str], float] = {}
            eta_raw_by_step_op: Dict[Tuple[str, str], float] = {}
            eta_rank_by_step_op: Dict[Tuple[str, str], int] = {}
            eta_top_by_step: Dict[str, str] = {}
            eta_top2_by_step: Dict[str, List[str]] = {}
            for step, choices_raw in options.items():
                choices = [str(v) for v in choices_raw]
                vals = np.asarray(eta.get(step, np.ones(len(choices))), dtype=float)
                raw_vals = np.asarray(raw_eta.get(step, np.ones(len(choices))), dtype=float)
                if vals.size != len(choices):
                    vals = np.ones(len(choices), dtype=float)
                if raw_vals.size != len(choices):
                    raw_vals = np.ones(len(choices), dtype=float)
                vals = vals.copy()
                vals[~np.isfinite(vals)] = -np.inf
                order = np.argsort(-vals)
                eta_top_by_step[step] = choices[int(order[0])]
                eta_top2_by_step[step] = [choices[int(idx)] for idx in order[: min(2, len(order))]]
                for rank, idx in enumerate(order, start=1):
                    op = choices[int(idx)]
                    eta_norm_by_step_op[(step, op)] = float(vals[int(idx)])
                    eta_raw_by_step_op[(step, op)] = float(raw_vals[int(idx)])
                    eta_rank_by_step_op[(step, op)] = int(rank)

            operator_only_configs: List[Dict[str, Any]] = [baseline_cfg]
            operator_only_signature: Dict[Tuple[str, str], str] = {}
            for step, choices_raw in options.items():
                for operator in [str(v) for v in choices_raw]:
                    cfg = _operator_only_config(step, operator, options)
                    sig = _config_signature(cfg, options)
                    operator_only_signature[(step, operator)] = sig
                    operator_only_configs.append(cfg)

            operator_eval_configs, _operator_by_sig = _prepare_eval_candidates(operator_only_configs, options)
            operator_scores_by_sig, operator_eval_method = _score_candidates(
                df,
                operator_eval_configs,
                backend=effect_backend,
                time_limit=int(args.time_limit),
                require_autogluon=bool(args.require_autogluon and effect_backend == "autogluon"),
                proxy_settings=proxy_settings,
                verbose=bool(args.verbose),
            )
            target_baseline_score = operator_scores_by_sig.get(baseline_sig, np.nan)

            target_score_by_step_op: Dict[Tuple[str, str], float] = {}
            target_lift_by_step_op: Dict[Tuple[str, str], float] = {}
            dataset_controlled_rows: List[Dict[str, Any]] = []
            top_neighbor_raw_id = top_neighbors[0][0] if top_neighbors else None
            top_neighbor_norm = _normalize_id(top_neighbor_raw_id) if top_neighbor_raw_id is not None else ""

            for step, choices_raw in options.items():
                choices = [str(v) for v in choices_raw]
                for operator in choices:
                    sig = operator_only_signature[(step, operator)]
                    target_score = operator_scores_by_sig.get(sig, np.nan)
                    target_lift = (
                        float(target_score - target_baseline_score)
                        if np.isfinite(target_score) and np.isfinite(target_baseline_score)
                        else np.nan
                    )
                    target_score_by_step_op[(step, operator)] = target_score
                    target_lift_by_step_op[(step, operator)] = target_lift

                    for neighbor_id, neighbor_sim in top_neighbors:
                        neighbor_stats = _neighbor_operator_scores(
                            perf=recommender.performance_matrix_imputed,
                            cfg_by_name=cfg_by_name,
                            neighbor_id=neighbor_id,
                            step=step,
                            operator=operator,
                        )
                        eta_rank = eta_rank_by_step_op.get((step, operator), 0)
                        row = {
                            "dataset_id": str(dataset_id),
                            "neighbor_id": str(neighbor_id),
                            "neighbor_similarity": float(neighbor_sim),
                            "neighbor_weight": float(similarity_weights.get(neighbor_id, np.nan)),
                            "is_top_neighbor": bool(_normalize_id(neighbor_id) == top_neighbor_norm),
                            "step": step,
                            "operator": operator,
                            "operator_base": base_operator_name(operator),
                            "operator_only_signature": sig,
                            "operator_effect_backend": effect_backend,
                            "operator_effect_eval_method": operator_eval_method,
                            "target_baseline_score": target_baseline_score,
                            "target_operator_only_score": target_score,
                            "target_operator_lift_vs_baseline": target_lift,
                            "target_supports_operator": bool(np.isfinite(target_lift) and target_lift > 0),
                            "target_hurts_operator": bool(np.isfinite(target_lift) and target_lift < 0),
                            "eta_norm": eta_norm_by_step_op.get((step, operator), np.nan),
                            "eta_raw": eta_raw_by_step_op.get((step, operator), np.nan),
                            "eta_rank": int(eta_rank),
                            "is_eta_top": bool(eta_rank == 1),
                            "is_eta_top2": bool(eta_rank in {1, 2}),
                            **neighbor_stats,
                        }
                        row["neighbor_best_target_sign_agreement"] = (
                            bool(
                                _score_sign(row["neighbor_operator_best_lift"])
                                == _score_sign(row["target_operator_lift_vs_baseline"])
                            )
                            if _is_finite_number(row["neighbor_operator_best_lift"])
                            and _is_finite_number(row["target_operator_lift_vs_baseline"])
                            else np.nan
                        )
                        row["neighbor_mean_target_sign_agreement"] = (
                            bool(
                                _score_sign(row["neighbor_operator_mean_lift"])
                                == _score_sign(row["target_operator_lift_vs_baseline"])
                            )
                            if _is_finite_number(row["neighbor_operator_mean_lift"])
                            and _is_finite_number(row["target_operator_lift_vs_baseline"])
                            else np.nan
                        )
                        controlled_operator_rows.append(row)
                        dataset_controlled_rows.append(row)

            dataset_eta_step_rows: List[Dict[str, Any]] = []
            first_neighbor_controlled = [
                row for row in dataset_controlled_rows if bool(row.get("is_top_neighbor"))
            ]
            for step, choices_raw in options.items():
                choices = [str(v) for v in choices_raw]
                step_lifts = {
                    op: target_lift_by_step_op.get((step, op), np.nan)
                    for op in choices
                }
                finite_step_lifts = {
                    op: lift
                    for op, lift in step_lifts.items()
                    if isinstance(lift, (int, float)) and np.isfinite(float(lift))
                }
                if finite_step_lifts:
                    target_best_operator = max(finite_step_lifts, key=lambda op: float(finite_step_lifts[op]))
                    target_best_lift = float(finite_step_lifts[target_best_operator])
                else:
                    target_best_operator = ""
                    target_best_lift = np.nan
                eta_top_operator = eta_top_by_step.get(step, "")
                eta_top2 = eta_top2_by_step.get(step, [])
                eta_top_lift = target_lift_by_step_op.get((step, eta_top_operator), np.nan)
                step_eta_vals = [eta_norm_by_step_op.get((step, op), np.nan) for op in choices]
                step_target_lifts = [target_lift_by_step_op.get((step, op), np.nan) for op in choices]
                step_neighbor_best_lifts = []
                step_neighbor_mean_lifts = []
                for op in choices:
                    row = next(
                        (
                            item
                            for item in first_neighbor_controlled
                            if item.get("step") == step and item.get("operator") == op
                        ),
                        None,
                    )
                    step_neighbor_best_lifts.append(
                        row.get("neighbor_operator_best_lift", np.nan) if row else np.nan
                    )
                    step_neighbor_mean_lifts.append(
                        row.get("neighbor_operator_mean_lift", np.nan) if row else np.nan
                    )

                step_row = {
                    "dataset_id": str(dataset_id),
                    "step": step,
                    "operator_effect_backend": effect_backend,
                    "operator_effect_eval_method": operator_eval_method,
                    "target_baseline_score": target_baseline_score,
                    "eta_top_operator": eta_top_operator,
                    "eta_top_operator_target_lift": eta_top_lift,
                    "eta_top_operator_supports_target": bool(np.isfinite(eta_top_lift) and eta_top_lift > 0),
                    "target_best_operator": target_best_operator,
                    "target_best_operator_lift": target_best_lift,
                    "eta_top_matches_target_best": bool(
                        target_best_operator != ""
                        and base_operator_name(eta_top_operator) == base_operator_name(target_best_operator)
                    ),
                    "eta_top2_contains_target_best": bool(
                        target_best_operator != ""
                        and any(base_operator_name(op) == base_operator_name(target_best_operator) for op in eta_top2)
                    ),
                    "target_best_eta_rank": eta_rank_by_step_op.get((step, target_best_operator), np.nan)
                    if target_best_operator
                    else np.nan,
                    "eta_target_lift_spearman": _spearman(step_eta_vals, step_target_lifts),
                    "neighbor_best_target_lift_spearman": _spearman(step_neighbor_best_lifts, step_target_lifts),
                    "neighbor_mean_target_lift_spearman": _spearman(step_neighbor_mean_lifts, step_target_lifts),
                    "eta_top1_margin_for_step": next(
                        (
                            item["top1_margin_for_step"]
                            for item in eta_detail
                            if item.get("step") == step and item.get("is_eta_top")
                        ),
                        np.nan,
                    ),
                    "eta_entropy_norm_for_step": next(
                        (
                            item["entropy_norm_for_step"]
                            for item in eta_detail
                            if item.get("step") == step and item.get("is_eta_top")
                        ),
                        np.nan,
                    ),
                }
                eta_target_step_rows.append(step_row)
                dataset_eta_step_rows.append(step_row)

            active_context_steps: List[Tuple[str, str, Dict[str, Any]]] = []
            for step in options.keys():
                operator = str(retrieved_top_cfg.get(step, "none"))
                baseline_operator = str(baseline_cfg.get(step, "none"))
                if base_operator_name(operator) == base_operator_name(baseline_operator):
                    continue
                active_context_steps.append((step, operator, _minus_operator_config(retrieved_top_cfg, step, options)))

            contextual_configs = [retrieved_top_cfg] + [minus_cfg for _step, _op, minus_cfg in active_context_steps]
            contextual_eval_configs, _contextual_by_sig = _prepare_eval_candidates(contextual_configs, options)
            contextual_scores_by_sig, contextual_eval_method = _score_candidates(
                df,
                contextual_eval_configs,
                backend=effect_backend,
                time_limit=int(args.time_limit),
                require_autogluon=bool(args.require_autogluon and effect_backend == "autogluon"),
                proxy_settings=proxy_settings,
                verbose=bool(args.verbose),
            )
            retrieved_top_sig = _config_signature(retrieved_top_cfg, options)
            contextual_full_score = contextual_scores_by_sig.get(retrieved_top_sig, np.nan)
            dataset_contextual_rows: List[Dict[str, Any]] = []
            for step, operator, minus_cfg in active_context_steps:
                minus_sig = _config_signature(minus_cfg, options)
                minus_score = contextual_scores_by_sig.get(minus_sig, np.nan)
                contextual_lift = (
                    float(contextual_full_score - minus_score)
                    if np.isfinite(contextual_full_score) and np.isfinite(minus_score)
                    else np.nan
                )
                row = {
                    "dataset_id": str(dataset_id),
                    "neighbor_id": str(top_neighbor_raw_id) if top_neighbor_raw_id is not None else "",
                    "retrieved_pipeline_name": str(retrieved_top_cfg.get("name", "")),
                    "step": step,
                    "operator": operator,
                    "operator_base": base_operator_name(operator),
                    "operator_effect_backend": effect_backend,
                    "operator_effect_eval_method": contextual_eval_method,
                    "retrieved_full_signature": retrieved_top_sig,
                    "minus_operator_signature": minus_sig,
                    "retrieved_full_score": contextual_full_score,
                    "minus_operator_score": minus_score,
                    "target_contextual_lift_full_minus_without_operator": contextual_lift,
                    "target_context_supports_operator": bool(np.isfinite(contextual_lift) and contextual_lift > 0),
                    "target_context_hurts_operator": bool(np.isfinite(contextual_lift) and contextual_lift < 0),
                    "eta_norm": eta_norm_by_step_op.get((step, operator), np.nan),
                    "eta_rank": eta_rank_by_step_op.get((step, operator), np.nan),
                    "controlled_target_lift_vs_baseline": target_lift_by_step_op.get((step, operator), np.nan),
                    "minus_pipeline_config": _display_config(minus_cfg, options),
                    "retrieved_pipeline_config": _display_config(retrieved_top_cfg, options),
                }
                contextual_operator_rows.append(row)
                dataset_contextual_rows.append(row)

            controlled_df = pd.DataFrame(first_neighbor_controlled)
            contextual_df = pd.DataFrame(dataset_contextual_rows)
            eta_step_df = pd.DataFrame(dataset_eta_step_rows)
            controlled_summary = {
                "operator_effect_backend": effect_backend,
                "operator_effect_eval_method": operator_eval_method,
                "controlled_baseline_score": target_baseline_score,
                "controlled_operator_test_count": int(len(controlled_df)),
                "controlled_target_support_rate": (
                    float((pd.to_numeric(controlled_df["target_operator_lift_vs_baseline"], errors="coerce") > 0).mean())
                    if not controlled_df.empty
                    else np.nan
                ),
                "controlled_target_negative_rate": (
                    float((pd.to_numeric(controlled_df["target_operator_lift_vs_baseline"], errors="coerce") < 0).mean())
                    if not controlled_df.empty
                    else np.nan
                ),
                "mean_controlled_target_lift": (
                    _nanmean(controlled_df["target_operator_lift_vs_baseline"])
                    if not controlled_df.empty
                    else np.nan
                ),
                "eta_target_controlled_spearman": (
                    _spearman(controlled_df["eta_norm"], controlled_df["target_operator_lift_vs_baseline"])
                    if not controlled_df.empty
                    else np.nan
                ),
                "neighbor_best_target_controlled_spearman": (
                    _spearman(controlled_df["neighbor_operator_best_lift"], controlled_df["target_operator_lift_vs_baseline"])
                    if not controlled_df.empty
                    else np.nan
                ),
                "neighbor_mean_target_controlled_spearman": (
                    _spearman(controlled_df["neighbor_operator_mean_lift"], controlled_df["target_operator_lift_vs_baseline"])
                    if not controlled_df.empty
                    else np.nan
                ),
                "neighbor_best_target_sign_agreement_rate": (
                    _sign_agreement_rate(
                        controlled_df["neighbor_operator_best_lift"],
                        controlled_df["target_operator_lift_vs_baseline"],
                    )
                    if not controlled_df.empty
                    else np.nan
                ),
                "neighbor_mean_target_sign_agreement_rate": (
                    _sign_agreement_rate(
                        controlled_df["neighbor_operator_mean_lift"],
                        controlled_df["target_operator_lift_vs_baseline"],
                    )
                    if not controlled_df.empty
                    else np.nan
                ),
                "eta_top_operator_hit_rate": (
                    float(
                        (
                            pd.to_numeric(
                                controlled_df.loc[controlled_df["is_eta_top"], "target_operator_lift_vs_baseline"],
                                errors="coerce",
                            )
                            > 0
                        ).mean()
                    )
                    if not controlled_df.empty and bool(controlled_df["is_eta_top"].any())
                    else np.nan
                ),
                "eta_top_operator_negative_rate": (
                    float(
                        (
                            pd.to_numeric(
                                controlled_df.loc[controlled_df["is_eta_top"], "target_operator_lift_vs_baseline"],
                                errors="coerce",
                            )
                            < 0
                        ).mean()
                    )
                    if not controlled_df.empty and bool(controlled_df["is_eta_top"].any())
                    else np.nan
                ),
                "eta_target_top1_match_rate": (
                    _nanrate(eta_step_df["eta_top_matches_target_best"]) if not eta_step_df.empty else np.nan
                ),
                "eta_target_top2_match_rate": (
                    _nanrate(eta_step_df["eta_top2_contains_target_best"]) if not eta_step_df.empty else np.nan
                ),
                "mean_eta_target_step_spearman": (
                    _nanmean(eta_step_df["eta_target_lift_spearman"]) if not eta_step_df.empty else np.nan
                ),
                "mean_neighbor_best_target_step_spearman": (
                    _nanmean(eta_step_df["neighbor_best_target_lift_spearman"])
                    if not eta_step_df.empty
                    else np.nan
                ),
                "contextual_operator_test_count": int(len(contextual_df)),
                "contextual_target_support_rate": (
                    float(
                        (
                            pd.to_numeric(
                                contextual_df["target_contextual_lift_full_minus_without_operator"],
                                errors="coerce",
                            )
                            > 0
                        ).mean()
                    )
                    if not contextual_df.empty
                    else np.nan
                ),
                "contextual_target_negative_rate": (
                    float(
                        (
                            pd.to_numeric(
                                contextual_df["target_contextual_lift_full_minus_without_operator"],
                                errors="coerce",
                            )
                            < 0
                        ).mean()
                    )
                    if not contextual_df.empty
                    else np.nan
                ),
                "mean_contextual_target_lift": (
                    _nanmean(contextual_df["target_contextual_lift_full_minus_without_operator"])
                    if not contextual_df.empty
                    else np.nan
                ),
            }

            finite_retrieved = [(cfg, sig, sc) for cfg, sig, sc in retrieved_scores if np.isfinite(sc)]
            if finite_retrieved:
                best_retrieved_cfg, best_retrieved_sig, best_retrieved_score = max(
                    finite_retrieved,
                    key=lambda item: float(item[2]),
                )
            else:
                best_retrieved_cfg = retrieved_top_cfg
                best_retrieved_sig = _config_signature(retrieved_top_cfg, options)
                best_retrieved_score = np.nan

            recommendation = recommender.recommend(
                new_dataset=df,
                target_column="target",
                k=max(1, int(args.k)),
                eval_k=max(1, int(args.heuristic_top_l)),
                use_autogluon=str(args.backend) == "autogluon",
                time_limit_per_model=int(args.time_limit),
                metafeatures_func=lambda _df, _dataset=dataset: _lookup_metafeatures(_dataset, meta),
                use_aco=True,
                options=options,
                optimizer="aco",
                final_autogluon_topk=max(1, int(args.final_autogluon_topk)),
                proxy_settings=proxy_settings,
                aco_params={
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
                },
            )

            aco_history = recommendation.get("aco_history") or []
            for row in aco_history:
                if isinstance(row, Mapping):
                    out = dict(row)
                    out["dataset_id"] = str(dataset_id)
                    aco_history_rows.append(out)
            hist_stats = _history_summary(aco_history)
            aco_cfg = recommendation.get("pipeline_config") or {}
            aco_sig = _config_signature(aco_cfg, options) if isinstance(aco_cfg, Mapping) else ""
            aco_final_score = recommendation.get("final_performance", np.nan)
            aco_proxy_score = recommendation.get("recommended_performance", np.nan)
            candidate_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "candidate_type": "aco_selected",
                    "candidate_name": str(aco_cfg.get("name", "aco_selected")) if isinstance(aco_cfg, Mapping) else "aco_selected",
                    "signature": aco_sig,
                    "score": aco_final_score,
                    "eval_method": (recommendation.get("final_evaluation") or {}).get("method", ""),
                    "pipeline_config": _display_config(aco_cfg, options) if isinstance(aco_cfg, Mapping) else "",
                }
            )

            top_neighbor_id = str(top_neighbors[0][0]) if top_neighbors else ""
            top_neighbor_sim = float(top_neighbors[0][1]) if top_neighbors else np.nan
            summary_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "status": "ok",
                    "error": "",
                    "top_neighbor": top_neighbor_id,
                    "top_neighbor_similarity": top_neighbor_sim,
                    "similarity_source": similarity_source,
                    "metric_similarity_target": str(args.metric_similarity_target),
                    "metric_hidden_dim": int(args.metric_hidden_dim),
                    "metric_embed_dim": int(args.metric_embed_dim),
                    "metric_epochs": int(args.metric_epochs),
                    "decomposition_method": str(args.heuristic_transfer_method),
                    "aco_heuristic_transfer_method": str(args.heuristic_transfer_method),
                    "eta_raw_source": eta_raw_source,
                    "heuristic_top_k": int(args.heuristic_top_k),
                    "heuristic_top_l": int(args.heuristic_top_l),
                    "backend": str(args.backend),
                    "retrieved_top_pipeline": str(retrieved_top_cfg.get("name", "")),
                    "retrieved_top_pipeline_config": _display_config(retrieved_top_cfg, options),
                    "best_retrieved_pipeline": str(best_retrieved_cfg.get("name", "")),
                    "best_retrieved_pipeline_config": _display_config(best_retrieved_cfg, options),
                    "best_retrieved_score": best_retrieved_score,
                    "eta_top_pipeline_config": _display_config(eta_top_cfg, options),
                    "eta_top_score": eta_top_score,
                    "eta_top_minus_best_retrieved": (
                        float(eta_top_score - best_retrieved_score)
                        if np.isfinite(eta_top_score) and np.isfinite(best_retrieved_score)
                        else np.nan
                    ),
                    "aco_pipeline_config": _display_config(aco_cfg, options) if isinstance(aco_cfg, Mapping) else "",
                    "aco_proxy_score": aco_proxy_score,
                    "aco_final_score": aco_final_score,
                    "aco_final_method": (recommendation.get("final_evaluation") or {}).get("method", ""),
                    "aco_final_minus_best_retrieved": (
                        float(aco_final_score - best_retrieved_score)
                        if np.isfinite(aco_final_score) and np.isfinite(best_retrieved_score)
                        else np.nan
                    ),
                    "aco_final_minus_eta_top": (
                        float(aco_final_score - eta_top_score)
                        if np.isfinite(aco_final_score) and np.isfinite(eta_top_score)
                        else np.nan
                    ),
                    "eta_top_matches_best_retrieved_signature": bool(eta_sig == best_retrieved_sig),
                    "aco_matches_best_retrieved_signature": bool(aco_sig == best_retrieved_sig),
                    "aco_matches_eta_top_signature": bool(aco_sig == eta_sig),
                    "risky_operator_count_retrieved_top": int(_risky_count(retrieved_top_cfg, options)),
                    "risky_operator_count_eta_top": int(_risky_count(eta_top_cfg, options)),
                    "risky_operator_count_aco": int(_risky_count(aco_cfg, options)) if isinstance(aco_cfg, Mapping) else np.nan,
                    "elapsed_seconds": float(time.perf_counter() - start),
                    "inferred_pipeline_config_count": int(len(inferred_names)),
                    **eta_stats,
                    **controlled_summary,
                    **hist_stats,
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
            )
            if args.verbose:
                print(f"Dataset {dataset_id} failed: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    eta_df = pd.DataFrame(eta_rows)
    retrieval_df = pd.DataFrame(retrieval_rows)
    neighbor_rank_df = pd.DataFrame(neighbor_rank_rows)
    history_df = pd.DataFrame(aco_history_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    controlled_operator_df = pd.DataFrame(controlled_operator_rows)
    contextual_operator_df = pd.DataFrame(contextual_operator_rows)
    eta_target_step_df = pd.DataFrame(eta_target_step_rows)

    summary_df.to_csv(out_dir / "decomposition_audit_summary.csv", index=False)
    eta_df.to_csv(out_dir / "operator_eta_detail.csv", index=False)
    retrieval_df.to_csv(out_dir / "retrieved_pipeline_detail.csv", index=False)
    neighbor_rank_df.to_csv(out_dir / "neighbor_ranking.csv", index=False)
    history_df.to_csv(out_dir / "aco_proxy_history.csv", index=False)
    candidate_df.to_csv(out_dir / "candidate_score_detail.csv", index=False)
    controlled_operator_df.to_csv(out_dir / "controlled_operator_effect_rows.csv", index=False)
    contextual_operator_df.to_csv(out_dir / "contextual_operator_effect_rows.csv", index=False)
    eta_target_step_df.to_csv(out_dir / "eta_target_step_summary.csv", index=False)

    ok = summary_df[summary_df.get("status", "") == "ok"].copy() if not summary_df.empty else pd.DataFrame()
    if not ok.empty:
        def col_mean(col: str) -> float:
            if col not in ok.columns:
                return np.nan
            return float(pd.to_numeric(ok[col], errors="coerce").mean())

        aggregate = {
            "n_datasets": int(ok["dataset_id"].nunique()),
            "mean_best_retrieved_score": col_mean("best_retrieved_score"),
            "mean_eta_top_score": col_mean("eta_top_score"),
            "mean_aco_final_score": col_mean("aco_final_score"),
            "mean_eta_top_minus_best_retrieved": col_mean("eta_top_minus_best_retrieved"),
            "mean_aco_final_minus_best_retrieved": col_mean("aco_final_minus_best_retrieved"),
            "mean_aco_proxy_gain": col_mean("aco_proxy_gain"),
            "eta_top_wins_vs_retrieved_rate": float((pd.to_numeric(ok["eta_top_minus_best_retrieved"], errors="coerce") > 0).mean()),
            "aco_wins_vs_retrieved_rate": float((pd.to_numeric(ok["aco_final_minus_best_retrieved"], errors="coerce") > 0).mean()),
            "mean_eta_target_controlled_spearman": col_mean("eta_target_controlled_spearman"),
            "mean_neighbor_best_target_controlled_spearman": col_mean("neighbor_best_target_controlled_spearman"),
            "mean_neighbor_best_target_sign_agreement_rate": col_mean("neighbor_best_target_sign_agreement_rate"),
            "mean_eta_top_operator_hit_rate": col_mean("eta_top_operator_hit_rate"),
            "mean_eta_target_top1_match_rate": col_mean("eta_target_top1_match_rate"),
            "mean_contextual_target_support_rate": col_mean("contextual_target_support_rate"),
            "mean_contextual_target_negative_rate": col_mean("contextual_target_negative_rate"),
        }
        with (out_dir / "decomposition_audit_aggregate.json").open("w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, default=str)

        tex_cols = [
            "dataset_id",
            "best_retrieved_score",
            "eta_top_score",
            "aco_final_score",
            "eta_top_minus_best_retrieved",
            "aco_final_minus_best_retrieved",
            "aco_proxy_gain",
            "eta_target_controlled_spearman",
            "neighbor_best_target_controlled_spearman",
            "neighbor_best_target_sign_agreement_rate",
            "eta_top_operator_hit_rate",
            "eta_target_top1_match_rate",
            "contextual_target_support_rate",
            "eta_mean_top1_margin",
            "eta_mean_entropy_norm",
        ]
        available = [col for col in tex_cols if col in ok.columns]
        ok[available].to_latex(out_dir / "decomposition_audit_summary.tex", index=False, float_format="%.4f")
        print(pd.DataFrame([aggregate]).to_string(index=False))
    else:
        print("No successful audit rows.")

    print(f"Saved audit outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
