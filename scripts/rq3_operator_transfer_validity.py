"""RQ3 diagnostic: does retrieved operator evidence transfer to the target?

The script compares each active operator in retrieved neighbor pipelines against
the same pipeline with that operator removed. This tests whether the marginal
operator signal given to ACO is actually predictive on the target dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automl_aco.config import DEFAULT_PIPELINE_OPTIONS
from automl_aco.data.loaders import load_kaggle_dataset, load_openml_dataset
from automl_aco.data.metafeatures import extract_enhanced_metafeatures
from automl_aco.metalearning.recommender import MetaPipelineRecommender
from automl_aco.search.evaluation import evaluate_candidates_autogluon, evaluate_candidates_simple
from automl_aco.utils.operator_spec import base_operator_name


class RecordSpec(NamedTuple):
    name: str
    perf_path: Path
    allow_inferred_configs: bool


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


def _maybe_set_meta_index(meta_df: pd.DataFrame, perf_df: pd.DataFrame, explicit_col: Optional[str]) -> pd.DataFrame:
    perf_norm = {_normalize_id(c) for c in perf_df.columns}

    def overlap(series: Iterable[object]) -> int:
        return len({_normalize_id(v) for v in series} & perf_norm)

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

    dataset_norm = _normalize_id(dataset_id)
    for idx in meta_df.index:
        if _normalize_id(idx) == dataset_norm:
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


def _load_pipeline_configs(
    path: Path,
    perf: pd.DataFrame,
    allow_inferred: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    with path.open("r", encoding="utf-8") as f:
        configs = json.load(f)

    existing = {str(cfg.get("name")) for cfg in configs if isinstance(cfg, dict) and "name" in cfg}
    missing = [str(name) for name in perf.index if str(name) not in existing]
    if missing:
        if not allow_inferred:
            sample = ", ".join(missing[:10])
            raise ValueError(
                "Performance matrix contains pipeline rows that are not present in "
                f"{path}: {sample}. Use the matching data/openml matrix, or pass "
                "--allow-inferred-pipeline-configs only for legacy/debug runs."
            )
        configs = list(configs) + [_infer_pipeline_config_from_name(name) for name in missing]
    return configs, missing


def _resolve_record_spec(args: argparse.Namespace, root: Path) -> RecordSpec:
    if args.performance_matrix:
        return RecordSpec(
            name="custom",
            perf_path=Path(args.performance_matrix),
            allow_inferred_configs=bool(args.allow_inferred_pipeline_configs),
        )
    if args.record_space == "aco":
        return RecordSpec(
            name="aco",
            perf_path=root / "aco" / "training_performance_matrix_autogluon.csv",
            allow_inferred_configs=True,
        )
    return RecordSpec(
        name="openml",
        perf_path=root / "data" / "openml" / "training_performance_matrix_autogluon.csv",
        allow_inferred_configs=bool(args.allow_inferred_pipeline_configs),
    )


def _parse_dataset_ids(raw: Optional[List[str]]) -> List[Any]:
    if not raw:
        return []
    out: List[Any] = []
    for token in raw:
        for piece in str(token).split(","):
            text = piece.strip()
            if not text:
                continue
            out.append(int(text) if text.isdigit() else text)
    return out


def _config_signature(cfg: Mapping[str, Any]) -> str:
    payload = {step: cfg.get(step, "none") for step in DEFAULT_PIPELINE_OPTIONS}
    if "step_order" in cfg:
        payload["step_order"] = list(cfg["step_order"])
    return json.dumps(payload, sort_keys=True, default=str)


def _short_sig(signature: str) -> str:
    return hashlib.md5(signature.encode("utf-8")).hexdigest()[:10]


def _display_config(cfg: Mapping[str, Any]) -> str:
    parts = [f"{step}={cfg.get(step, 'none')}" for step in DEFAULT_PIPELINE_OPTIONS]
    return "{" + ", ".join(parts) + "}"


def _minus_operator_config(cfg: Mapping[str, Any], step: str) -> Dict[str, Any]:
    out = {k: v for k, v in cfg.items() if k != "name"}
    out[step] = "none"
    out["name"] = f"{cfg.get('name', 'pipeline')}__minus_{step}"
    return out


def _baseline_config() -> Dict[str, Any]:
    return {
        "name": "controlled_baseline",
        "imputation": "none",
        "scaling": "none",
        "encoding": "onehot",
        "feature_selection": "none",
        "outlier_removal": "none",
        "dimensionality_reduction": "none",
    }


def _operator_only_config(step: str, operator: str) -> Dict[str, Any]:
    cfg = _baseline_config()
    cfg[step] = operator
    cfg["name"] = f"operator_only__{step}__{operator}"
    return cfg


def _eta_top_config(eta: Mapping[str, np.ndarray], options: Mapping[str, List[str]]) -> Dict[str, Any]:
    cfg = _baseline_config()
    cfg["name"] = "eta_top_composed"
    for step, operators in options.items():
        if step == "encoding":
            cfg[step] = "onehot"
            continue
        vals = np.asarray(eta.get(step, np.ones(len(operators))), dtype=float)
        if len(operators) == 0:
            continue
        idx = int(np.nanargmax(vals)) if np.isfinite(vals).any() else 0
        cfg[step] = str(operators[idx])
    return cfg


def _active_operator_rows(cfg: Mapping[str, Any], steps: Iterable[str]) -> List[Tuple[str, str]]:
    active: List[Tuple[str, str]] = []
    for step in steps:
        if step not in DEFAULT_PIPELINE_OPTIONS:
            continue
        op = str(cfg.get(step, "none"))
        if base_operator_name(op) != "none":
            active.append((step, op))
    return active


def _safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    df = pd.DataFrame({"x": list(x), "y": list(y)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 2 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan
    return float(df["x"].corr(df["y"], method="spearman"))


def _rank_desc(values: Mapping[str, float]) -> Dict[str, int]:
    items = sorted(
        [(str(k), float(v)) for k, v in values.items()],
        key=lambda item: (np.nan_to_num(item[1], nan=-np.inf), item[0]),
        reverse=True,
    )
    return {key: rank for rank, (key, _val) in enumerate(items, start=1)}


def _load_aco_scores(args: argparse.Namespace) -> Dict[str, float]:
    if not args.aco_summary_csv:
        return {}
    path = Path(args.aco_summary_csv)
    df = pd.read_csv(path)
    if args.aco_dataset_column not in df.columns:
        raise ValueError(f"ACO dataset column {args.aco_dataset_column!r} not found in {path}")
    if args.aco_score_column not in df.columns:
        raise ValueError(f"ACO score column {args.aco_score_column!r} not found in {path}")
    out: Dict[str, float] = {}
    for row in df.itertuples(index=False):
        dataset_id = getattr(row, args.aco_dataset_column)
        score = getattr(row, args.aco_score_column)
        if pd.notna(score):
            out[_normalize_id(dataset_id)] = float(score)
    return out


def _load_dataset_as_frame(
    dataset_id: Any,
    *,
    dataset_source: str,
    openml_local: Path,
    kaggle_data: Path,
    kaggle_target_column: str,
    verbose: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if dataset_source == "openml":
        ds = load_openml_dataset(dataset_id, verbose=verbose, local_data_folder=str(openml_local))
    else:
        ds = load_kaggle_dataset(
            dataset_id,
            data_folder=str(kaggle_data),
            target_column=kaggle_target_column,
            verbose=verbose,
        )
    if ds is None or "X" not in ds or "y" not in ds:
        raise RuntimeError(f"Could not load dataset {dataset_id}")
    data = ds["X"].copy()
    data["target"] = ds["y"]
    return ds, data


def _evaluate_config_signatures(
    dataset_df: pd.DataFrame,
    configs_by_sig: Mapping[str, Dict[str, Any]],
    *,
    backend: str,
    time_limit: int,
    require_autogluon: bool,
    verbose: bool,
) -> Tuple[Dict[str, float], str]:
    candidates: List[Dict[str, Any]] = []
    sig_by_name: Dict[str, str] = {}
    for sig, cfg in configs_by_sig.items():
        cand = dict(cfg)
        name = f"cand_{_short_sig(sig)}"
        cand["name"] = name
        candidates.append(cand)
        sig_by_name[name] = sig

    if not candidates:
        return {}, backend

    method = backend
    try:
        if backend == "autogluon":
            _best, _score, _sorted_results, unsorted_results = evaluate_candidates_autogluon(
                dataset_df,
                "target",
                candidates,
                time_limit_per_model=int(time_limit),
                verbose=verbose,
            )
        else:
            _best, _score, _sorted_results, unsorted_results = evaluate_candidates_simple(
                dataset_df,
                "target",
                candidates,
                verbose=verbose,
            )
            method = "simple"
    except Exception as exc:
        if backend == "autogluon" and not require_autogluon:
            _best, _score, _sorted_results, unsorted_results = evaluate_candidates_simple(
                dataset_df,
                "target",
                candidates,
                verbose=verbose,
            )
            method = f"simple_fallback_after_autogluon_error:{exc}"
        else:
            raise

    scores: Dict[str, float] = {}
    for cfg, score in unsorted_results:
        sig = sig_by_name.get(str(cfg.get("name")))
        if sig is not None:
            scores[sig] = float(score)
    return scores, method


def _best_historical_match(
    signature: str,
    *,
    signature_to_names: Mapping[str, List[str]],
    perf: pd.DataFrame,
    neighbor_id: str,
) -> Tuple[Optional[str], float]:
    names = signature_to_names.get(signature, [])
    best_name: Optional[str] = None
    best_score = np.nan
    for name in names:
        if name not in perf.index or neighbor_id not in perf.columns:
            continue
        score = perf.at[name, neighbor_id]
        if pd.isna(score):
            continue
        if best_name is None or float(score) > float(best_score):
            best_name = name
            best_score = float(score)
    return best_name, float(best_score) if best_name is not None else np.nan


def _sign(value: float, threshold: float) -> float:
    if not np.isfinite(value):
        return np.nan
    if value > threshold:
        return 1.0
    if value < -threshold:
        return -1.0
    return 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose whether retrieved operators transfer to target datasets")
    parser.add_argument("--root", default=os.environ.get("ROOT", str(Path(__file__).resolve().parents[1])))
    parser.add_argument("--record-space", choices=["openml", "aco"], default="openml")
    parser.add_argument("--performance-matrix", default=None)
    parser.add_argument("--metafeatures", default=None)
    parser.add_argument("--metafeatures-id-column", default=None)
    parser.add_argument("--pipeline-configs", default=None)
    parser.add_argument("--allow-inferred-pipeline-configs", action="store_true")
    parser.add_argument("--dataset-source", choices=["openml", "kaggle"], default="openml")
    parser.add_argument("--openml-local-folder", default=None)
    parser.add_argument("--kaggle-data-folder", default=None)
    parser.add_argument("--kaggle-target-column", default="target")
    parser.add_argument("--dataset-ids", nargs="+", required=True)
    parser.add_argument("--neighbor-k", type=int, default=1)
    parser.add_argument("--top-l", type=int, default=3)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["imputation", "scaling", "feature_selection", "outlier_removal", "dimensionality_reduction"],
        help="Pipeline steps to ablate. Encoding is omitted by default because onehot is usually required for categorical data.",
    )
    parser.add_argument("--backend", choices=["autogluon", "simple"], default="autogluon")
    parser.add_argument("--time-limit", type=int, default=240)
    parser.add_argument("--require-autogluon", action="store_true")
    parser.add_argument("--eta-floor", type=float, default=0.05)
    parser.add_argument("--similarity-temperature", type=float, default=1.0)
    parser.add_argument(
        "--evaluate-neighbor-raw",
        action="store_true",
        help="Also evaluate controlled operator-only and full-vs-minus candidates on raw neighbor datasets.",
    )
    parser.add_argument("--aco-summary-csv", default=None, help="Optional CSV containing ACO final scores to join")
    parser.add_argument("--aco-dataset-column", default="dataset_id")
    parser.add_argument("--aco-score-column", default="final_score")
    parser.add_argument("--min-lift", type=float, default=0.0)
    parser.add_argument("--output-dir", default="/kaggle/working/rq3_operator_transfer_validity")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    record_spec = _resolve_record_spec(args, root)
    meta_path = Path(args.metafeatures or root / "data" / "openml" / "dataset_feats.csv")
    cfg_path = Path(args.pipeline_configs or root / "aco" / "pipeline_configs.json")
    openml_local = Path(args.openml_local_folder or root / "test_data_local")
    kaggle_data = Path(args.kaggle_data_folder or root / "test_data_local")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    perf = pd.read_csv(record_spec.perf_path, index_col=0)
    raw_meta = pd.read_csv(meta_path)
    meta = _maybe_set_meta_index(raw_meta, perf, args.metafeatures_id_column)
    configs, inferred_names = _load_pipeline_configs(
        cfg_path,
        perf,
        allow_inferred=bool(record_spec.allow_inferred_configs),
    )
    cfg_by_name = {str(cfg.get("name")): cfg for cfg in configs if isinstance(cfg, dict) and cfg.get("name") is not None}
    signature_to_names: Dict[str, List[str]] = {}
    for cfg in configs:
        signature_to_names.setdefault(_config_signature(cfg), []).append(str(cfg.get("name")))

    options = {step: list(values) for step, values in DEFAULT_PIPELINE_OPTIONS.items()}
    aco_scores = _load_aco_scores(args)
    rows: List[Dict[str, Any]] = []
    operator_only_rows: List[Dict[str, Any]] = []
    eta_operator_rows: List[Dict[str, Any]] = []
    eta_step_rows: List[Dict[str, Any]] = []
    compression_rows: List[Dict[str, Any]] = []
    eval_cache: Dict[Tuple[str, str, str], float] = {}
    method_cache: Dict[Tuple[str, str], str] = {}
    dataset_ids = _parse_dataset_ids(args.dataset_ids)

    for dataset_id in dataset_ids:
        target_start = time.perf_counter()
        target_norm = _normalize_id(dataset_id)
        try:
            target_ds, target_df = _load_dataset_as_frame(
                dataset_id,
                dataset_source=args.dataset_source,
                openml_local=openml_local,
                kaggle_data=kaggle_data,
                kaggle_target_column=args.kaggle_target_column,
                verbose=args.verbose,
            )

            ref_columns = [col for col in perf.columns if _normalize_id(col) != target_norm]
            ref_index = [idx for idx in meta.index if _normalize_id(idx) != target_norm]
            ref_perf = perf.loc[:, ref_columns]
            ref_meta = meta.loc[ref_index]
            recommender = MetaPipelineRecommender(ref_perf, ref_meta, configs, verbose=args.verbose)

            target_mf = _lookup_metafeatures(target_ds, meta)
            target_mf_df = pd.DataFrame([target_mf]).reindex(columns=recommender.metafeatures_df.columns, fill_value=0)
            target_mf_imputed = recommender.imputer.transform(target_mf_df)
            target_mf_scaled = recommender.scaler.transform(target_mf_imputed).ravel()
            sims = sorted(recommender._compute_dataset_similarities(target_mf_scaled), key=lambda x: x[1], reverse=True)
            neighbors = [(str(ds), float(sim)) for ds, sim in sims[: max(1, int(args.neighbor_k))]]

            eta = recommender._compute_aco_heuristic(
                target_mf_scaled,
                options,
                dataset_weighting="similarity",
                top_k=max(1, int(args.neighbor_k)),
                top_l=max(1, int(args.top_l)),
                similarity_temperature=float(args.similarity_temperature),
                eta_floor=float(args.eta_floor),
                heuristic_transfer_method="weighted_topk_topl",
                score_direction="higher_is_better",
                query_dataset_id=dataset_id,
            )

            baseline_cfg = _baseline_config()
            baseline_sig = _config_signature(baseline_cfg)
            eta_top_cfg = _eta_top_config(eta, options)
            eta_top_sig = _config_signature(eta_top_cfg)

            target_candidates: Dict[str, Dict[str, Any]] = {
                baseline_sig: baseline_cfg,
                eta_top_sig: eta_top_cfg,
            }
            operator_only_sigs: Dict[Tuple[str, str], str] = {}
            for step in args.steps:
                if step not in options:
                    continue
                for operator in options[step]:
                    op_cfg = _operator_only_config(step, str(operator))
                    op_sig = _config_signature(op_cfg)
                    operator_only_sigs[(step, str(operator))] = op_sig
                    target_candidates.setdefault(op_sig, op_cfg)

            neighbor_candidates: Dict[str, Dict[str, Dict[str, Any]]] = {}

            for neighbor_rank, (neighbor_id, neighbor_sim) in enumerate(neighbors, start=1):
                if neighbor_id not in recommender.performance_matrix.columns:
                    continue
                neighbor_candidates.setdefault(neighbor_id, {}).setdefault(baseline_sig, baseline_cfg)
                neighbor_candidates.setdefault(neighbor_id, {}).setdefault(eta_top_sig, eta_top_cfg)
                for (step, operator), op_sig in operator_only_sigs.items():
                    neighbor_candidates.setdefault(neighbor_id, {}).setdefault(op_sig, target_candidates[op_sig])

                neighbor_baseline_score = (
                    float(recommender.performance_matrix.at["baseline", neighbor_id])
                    if "baseline" in recommender.performance_matrix.index
                    else np.nan
                )
                for (step, operator), op_sig in operator_only_sigs.items():
                    best_hist_name = ""
                    best_hist_score = np.nan
                    target_base = base_operator_name(operator)
                    for candidate_name, candidate_cfg in cfg_by_name.items():
                        if base_operator_name(candidate_cfg.get(step, "none")) != target_base:
                            continue
                        if candidate_name not in recommender.performance_matrix.index:
                            continue
                        score = recommender.performance_matrix.at[candidate_name, neighbor_id]
                        if pd.isna(score):
                            continue
                        if not np.isfinite(best_hist_score) or float(score) > best_hist_score:
                            best_hist_score = float(score)
                            best_hist_name = candidate_name
                    neighbor_operator_lift = (
                        best_hist_score - neighbor_baseline_score
                        if np.isfinite(best_hist_score) and np.isfinite(neighbor_baseline_score)
                        else np.nan
                    )
                    operator_only_rows.append(
                        {
                            "record_space": record_spec.name,
                            "dataset_id": str(dataset_id),
                            "neighbor_id": neighbor_id,
                            "neighbor_rank": int(neighbor_rank),
                            "neighbor_similarity": float(neighbor_sim),
                            "step": step,
                            "operator": operator,
                            "operator_only_signature": op_sig,
                            "operator_only_pipeline": _display_config(target_candidates[op_sig]),
                            "neighbor_reference_pipeline": "baseline",
                            "neighbor_reference_score": neighbor_baseline_score,
                            "neighbor_best_pipeline_with_operator": best_hist_name,
                            "neighbor_operator_score": best_hist_score,
                            "neighbor_operator_lift": neighbor_operator_lift,
                            "neighbor_raw_reference_score": np.nan,
                            "neighbor_raw_operator_score": np.nan,
                            "neighbor_raw_operator_lift": np.nan,
                            "target_reference_score": np.nan,
                            "target_operator_score": np.nan,
                            "target_operator_lift": np.nan,
                            "hist_target_sign_agreement": np.nan,
                            "raw_target_sign_agreement": np.nan,
                            "target_eval_method": "",
                            "neighbor_raw_eval_method": "",
                            "status": "pending",
                            "error": "",
                        }
                    )

                neighbor_scores = recommender.performance_matrix[neighbor_id].dropna().sort_values(ascending=False)
                for pipeline_rank, (pipeline_name, pipeline_hist_score) in enumerate(
                    neighbor_scores.head(max(1, int(args.top_l))).items(),
                    start=1,
                ):
                    full_cfg = cfg_by_name.get(str(pipeline_name))
                    if full_cfg is None:
                        continue
                    full_sig = _config_signature(full_cfg)
                    target_candidates.setdefault(full_sig, dict(full_cfg))
                    neighbor_candidates.setdefault(neighbor_id, {}).setdefault(full_sig, dict(full_cfg))

                    for step, operator in _active_operator_rows(full_cfg, args.steps):
                        minus_cfg = _minus_operator_config(full_cfg, step)
                        minus_sig = _config_signature(minus_cfg)
                        if minus_sig == full_sig:
                            continue
                        target_candidates.setdefault(minus_sig, minus_cfg)
                        neighbor_candidates.setdefault(neighbor_id, {}).setdefault(minus_sig, minus_cfg)

                        hist_minus_name, hist_minus_score = _best_historical_match(
                            minus_sig,
                            signature_to_names=signature_to_names,
                            perf=recommender.performance_matrix,
                            neighbor_id=neighbor_id,
                        )
                        neighbor_hist_lift = (
                            float(pipeline_hist_score) - float(hist_minus_score)
                            if np.isfinite(hist_minus_score)
                            else np.nan
                        )

                        rows.append(
                            {
                                "record_space": record_spec.name,
                                "performance_matrix": str(record_spec.perf_path),
                                "inferred_pipeline_config_count": int(len(inferred_names)),
                                "dataset_id": str(dataset_id),
                                "neighbor_id": neighbor_id,
                                "neighbor_rank": int(neighbor_rank),
                                "neighbor_similarity": float(neighbor_sim),
                                "pipeline_name": str(pipeline_name),
                                "pipeline_rank_in_neighbor": int(pipeline_rank),
                                "neighbor_pipeline_hist_score": float(pipeline_hist_score),
                                "step": step,
                                "operator": operator,
                                "full_signature": full_sig,
                                "minus_signature": minus_sig,
                                "full_pipeline": _display_config(full_cfg),
                                "minus_operator_pipeline": _display_config(minus_cfg),
                                "neighbor_hist_minus_pipeline": hist_minus_name or "",
                                "neighbor_hist_minus_score": hist_minus_score,
                                "neighbor_hist_lift": neighbor_hist_lift,
                                "target_full_score": np.nan,
                                "target_minus_score": np.nan,
                                "target_lift": np.nan,
                                "target_supports_operator": np.nan,
                                "neighbor_raw_full_score": np.nan,
                                "neighbor_raw_minus_score": np.nan,
                                "neighbor_raw_lift": np.nan,
                                "hist_target_sign_agreement": np.nan,
                                "raw_target_sign_agreement": np.nan,
                                "target_eval_method": "",
                                "neighbor_raw_eval_method": "",
                                "elapsed_seconds_target": np.nan,
                                "status": "pending",
                                "error": "",
                            }
                        )

            target_scores, target_method = _evaluate_config_signatures(
                target_df,
                target_candidates,
                backend=args.backend,
                time_limit=int(args.time_limit),
                require_autogluon=bool(args.require_autogluon),
                verbose=args.verbose,
            )
            for sig, score in target_scores.items():
                eval_cache[(str(dataset_id), "target", sig)] = score
            method_cache[(str(dataset_id), "target")] = target_method

            if args.evaluate_neighbor_raw:
                for neighbor_id, candidates in neighbor_candidates.items():
                    try:
                        neighbor_ds, neighbor_df = _load_dataset_as_frame(
                            neighbor_id,
                            dataset_source=args.dataset_source,
                            openml_local=openml_local,
                            kaggle_data=kaggle_data,
                            kaggle_target_column=args.kaggle_target_column,
                            verbose=args.verbose,
                        )
                        _ = neighbor_ds
                        neighbor_scores_raw, neighbor_method = _evaluate_config_signatures(
                            neighbor_df,
                            candidates,
                            backend=args.backend,
                            time_limit=int(args.time_limit),
                            require_autogluon=bool(args.require_autogluon),
                            verbose=args.verbose,
                        )
                        for sig, score in neighbor_scores_raw.items():
                            eval_cache[(neighbor_id, "neighbor_raw", sig)] = score
                        method_cache[(neighbor_id, "neighbor_raw")] = neighbor_method
                    except Exception as exc:
                        method_cache[(neighbor_id, "neighbor_raw")] = f"failed:{exc}"

            for row in rows:
                if row["dataset_id"] != str(dataset_id):
                    continue
                full_sig = str(row["full_signature"])
                minus_sig = str(row["minus_signature"])
                target_full = eval_cache.get((str(dataset_id), "target", full_sig), np.nan)
                target_minus = eval_cache.get((str(dataset_id), "target", minus_sig), np.nan)
                target_lift = target_full - target_minus if np.isfinite(target_full) and np.isfinite(target_minus) else np.nan

                neighbor_id = str(row["neighbor_id"])
                raw_full = eval_cache.get((neighbor_id, "neighbor_raw", full_sig), np.nan)
                raw_minus = eval_cache.get((neighbor_id, "neighbor_raw", minus_sig), np.nan)
                raw_lift = raw_full - raw_minus if np.isfinite(raw_full) and np.isfinite(raw_minus) else np.nan

                hist_sign = _sign(float(row["neighbor_hist_lift"]), float(args.min_lift))
                target_sign = _sign(float(target_lift), float(args.min_lift))
                raw_sign = _sign(float(raw_lift), float(args.min_lift))

                row.update(
                    {
                        "target_full_score": target_full,
                        "target_minus_score": target_minus,
                        "target_lift": target_lift,
                        "target_supports_operator": bool(target_lift > float(args.min_lift))
                        if np.isfinite(target_lift)
                        else np.nan,
                        "neighbor_raw_full_score": raw_full,
                        "neighbor_raw_minus_score": raw_minus,
                        "neighbor_raw_lift": raw_lift,
                        "hist_target_sign_agreement": bool(hist_sign == target_sign)
                        if np.isfinite(hist_sign) and np.isfinite(target_sign)
                        else np.nan,
                        "raw_target_sign_agreement": bool(raw_sign == target_sign)
                        if np.isfinite(raw_sign) and np.isfinite(target_sign)
                        else np.nan,
                        "target_eval_method": method_cache.get((str(dataset_id), "target"), ""),
                        "neighbor_raw_eval_method": method_cache.get((neighbor_id, "neighbor_raw"), ""),
                        "elapsed_seconds_target": time.perf_counter() - target_start,
                        "status": "ok" if np.isfinite(target_lift) else "target_eval_missing",
                    }
                )

            for op_row in operator_only_rows:
                if op_row["dataset_id"] != str(dataset_id):
                    continue
                neighbor_id = str(op_row["neighbor_id"])
                op_sig = str(op_row["operator_only_signature"])
                target_reference = eval_cache.get((str(dataset_id), "target", baseline_sig), np.nan)
                target_operator = eval_cache.get((str(dataset_id), "target", op_sig), np.nan)
                target_lift = (
                    target_operator - target_reference
                    if np.isfinite(target_operator) and np.isfinite(target_reference)
                    else np.nan
                )
                raw_reference = eval_cache.get((neighbor_id, "neighbor_raw", baseline_sig), np.nan)
                raw_operator = eval_cache.get((neighbor_id, "neighbor_raw", op_sig), np.nan)
                raw_lift = (
                    raw_operator - raw_reference
                    if np.isfinite(raw_operator) and np.isfinite(raw_reference)
                    else np.nan
                )

                hist_sign = _sign(float(op_row["neighbor_operator_lift"]), float(args.min_lift))
                raw_sign = _sign(float(raw_lift), float(args.min_lift))
                target_sign = _sign(float(target_lift), float(args.min_lift))

                op_row.update(
                    {
                        "neighbor_raw_reference_score": raw_reference,
                        "neighbor_raw_operator_score": raw_operator,
                        "neighbor_raw_operator_lift": raw_lift,
                        "target_reference_score": target_reference,
                        "target_operator_score": target_operator,
                        "target_operator_lift": target_lift,
                        "hist_target_sign_agreement": bool(hist_sign == target_sign)
                        if np.isfinite(hist_sign) and np.isfinite(target_sign)
                        else np.nan,
                        "raw_target_sign_agreement": bool(raw_sign == target_sign)
                        if np.isfinite(raw_sign) and np.isfinite(target_sign)
                        else np.nan,
                        "target_eval_method": method_cache.get((str(dataset_id), "target"), ""),
                        "neighbor_raw_eval_method": method_cache.get((neighbor_id, "neighbor_raw"), ""),
                        "status": "ok" if np.isfinite(target_lift) else "target_eval_missing",
                    }
                )

            target_lift_by_step_operator: Dict[Tuple[str, str], float] = {}
            hist_lift_by_step_operator: Dict[Tuple[str, str], List[float]] = {}
            for op_row in operator_only_rows:
                if op_row["dataset_id"] != str(dataset_id):
                    continue
                key = (str(op_row["step"]), str(op_row["operator"]))
                if np.isfinite(float(op_row["target_operator_lift"])):
                    target_lift_by_step_operator[key] = float(op_row["target_operator_lift"])
                if np.isfinite(float(op_row["neighbor_operator_lift"])):
                    hist_lift_by_step_operator.setdefault(key, []).append(float(op_row["neighbor_operator_lift"]))

            for step in args.steps:
                if step not in options:
                    continue
                operators = [str(op) for op in options[step]]
                eta_values = np.asarray(eta.get(step, np.ones(len(operators))), dtype=float)
                eta_by_operator = {op: float(eta_values[idx]) for idx, op in enumerate(operators)}
                target_by_operator = {
                    op: float(target_lift_by_step_operator.get((step, op), np.nan))
                    for op in operators
                }
                hist_by_operator = {
                    op: float(np.mean(hist_lift_by_step_operator.get((step, op), [np.nan])))
                    for op in operators
                }
                eta_rank = _rank_desc(eta_by_operator)
                target_rank = _rank_desc(target_by_operator)
                hist_rank = _rank_desc(hist_by_operator)
                eta_top_operator = min(eta_rank, key=eta_rank.get) if eta_rank else ""
                best_target_operator = min(target_rank, key=target_rank.get) if target_rank else ""
                top2_target = {op for op, rank in target_rank.items() if rank <= 2}
                corr_eta_target = _safe_spearman(
                    [eta_by_operator[op] for op in operators],
                    [target_by_operator[op] for op in operators],
                )
                corr_hist_target = _safe_spearman(
                    [hist_by_operator[op] for op in operators],
                    [target_by_operator[op] for op in operators],
                )

                for op in operators:
                    eta_operator_rows.append(
                        {
                            "record_space": record_spec.name,
                            "dataset_id": str(dataset_id),
                            "step": step,
                            "operator": op,
                            "eta_score": eta_by_operator.get(op, np.nan),
                            "eta_rank": eta_rank.get(op, np.nan),
                            "target_operator_lift": target_by_operator.get(op, np.nan),
                            "target_rank": target_rank.get(op, np.nan),
                            "neighbor_operator_lift_mean": hist_by_operator.get(op, np.nan),
                            "neighbor_hist_rank": hist_rank.get(op, np.nan),
                            "is_eta_top_operator": bool(op == eta_top_operator),
                            "is_best_target_operator": bool(op == best_target_operator),
                            "eta_top_hits_target_top1": bool(eta_top_operator == best_target_operator)
                            if op == eta_top_operator
                            else np.nan,
                            "eta_top_hits_target_top2": bool(eta_top_operator in top2_target)
                            if op == eta_top_operator
                            else np.nan,
                        }
                    )

                eta_sorted = sorted(eta_by_operator.values(), reverse=True)
                eta_margin = eta_sorted[0] - eta_sorted[1] if len(eta_sorted) >= 2 else np.nan
                eta_step_rows.append(
                    {
                        "record_space": record_spec.name,
                        "dataset_id": str(dataset_id),
                        "step": step,
                        "eta_top_operator": eta_top_operator,
                        "best_target_operator": best_target_operator,
                        "eta_top_target_lift": target_by_operator.get(eta_top_operator, np.nan),
                        "best_target_lift": target_by_operator.get(best_target_operator, np.nan),
                        "eta_top1_match": bool(eta_top_operator == best_target_operator),
                        "eta_top2_match": bool(eta_top_operator in top2_target),
                        "eta_target_spearman": corr_eta_target,
                        "neighbor_hist_target_spearman": corr_hist_target,
                        "eta_margin_top1_top2": eta_margin,
                    }
                )

            retrieved_full_scores: List[Tuple[str, float]] = []
            seen_full: set[str] = set()
            for row in rows:
                if row["dataset_id"] != str(dataset_id):
                    continue
                full_sig = str(row["full_signature"])
                if full_sig in seen_full:
                    continue
                seen_full.add(full_sig)
                score = eval_cache.get((str(dataset_id), "target", full_sig), np.nan)
                if np.isfinite(score):
                    retrieved_full_scores.append((str(row["pipeline_name"]), float(score)))
            if retrieved_full_scores:
                best_retrieved_pipeline, best_retrieved_score = max(retrieved_full_scores, key=lambda item: item[1])
            else:
                best_retrieved_pipeline, best_retrieved_score = "", np.nan

            eta_top_score = eval_cache.get((str(dataset_id), "target", eta_top_sig), np.nan)
            baseline_score = eval_cache.get((str(dataset_id), "target", baseline_sig), np.nan)
            aco_score = aco_scores.get(target_norm, np.nan)
            compression_rows.append(
                {
                    "record_space": record_spec.name,
                    "dataset_id": str(dataset_id),
                    "best_retrieved_full_pipeline": best_retrieved_pipeline,
                    "best_retrieved_full_score": best_retrieved_score,
                    "controlled_baseline_score": baseline_score,
                    "eta_top_composed_pipeline": _display_config(eta_top_cfg),
                    "eta_top_composed_score": eta_top_score,
                    "aco_score": aco_score,
                    "retrieved_minus_eta_top": best_retrieved_score - eta_top_score
                    if np.isfinite(best_retrieved_score) and np.isfinite(eta_top_score)
                    else np.nan,
                    "aco_minus_retrieved": aco_score - best_retrieved_score
                    if np.isfinite(aco_score) and np.isfinite(best_retrieved_score)
                    else np.nan,
                    "aco_minus_eta_top": aco_score - eta_top_score
                    if np.isfinite(aco_score) and np.isfinite(eta_top_score)
                    else np.nan,
                    "target_eval_method": method_cache.get((str(dataset_id), "target"), ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "record_space": record_spec.name,
                    "performance_matrix": str(record_spec.perf_path),
                    "inferred_pipeline_config_count": int(len(inferred_names)),
                    "dataset_id": str(dataset_id),
                    "neighbor_id": "",
                    "neighbor_rank": np.nan,
                    "neighbor_similarity": np.nan,
                    "pipeline_name": "",
                    "pipeline_rank_in_neighbor": np.nan,
                    "neighbor_pipeline_hist_score": np.nan,
                    "step": "",
                    "operator": "",
                    "full_signature": "",
                    "minus_signature": "",
                    "full_pipeline": "",
                    "minus_operator_pipeline": "",
                    "neighbor_hist_minus_pipeline": "",
                    "neighbor_hist_minus_score": np.nan,
                    "neighbor_hist_lift": np.nan,
                    "target_full_score": np.nan,
                    "target_minus_score": np.nan,
                    "target_lift": np.nan,
                    "target_supports_operator": np.nan,
                    "neighbor_raw_full_score": np.nan,
                    "neighbor_raw_minus_score": np.nan,
                    "neighbor_raw_lift": np.nan,
                    "hist_target_sign_agreement": np.nan,
                    "raw_target_sign_agreement": np.nan,
                    "target_eval_method": "",
                    "neighbor_raw_eval_method": "",
                    "elapsed_seconds_target": time.perf_counter() - target_start,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    rows_df = pd.DataFrame(rows)
    operator_only_df = pd.DataFrame(operator_only_rows)
    eta_operator_df = pd.DataFrame(eta_operator_rows)
    eta_step_df = pd.DataFrame(eta_step_rows)
    compression_df = pd.DataFrame(compression_rows)

    rows_path = out_dir / "contextual_operator_transfer_rows.csv"
    operator_only_path = out_dir / "controlled_operator_only_rows.csv"
    eta_operator_path = out_dir / "eta_predictive_operator_rows.csv"
    eta_step_path = out_dir / "eta_predictive_step_summary.csv"
    compression_path = out_dir / "full_pipeline_vs_compressed_eta.csv"
    rows_df.to_csv(rows_path, index=False)
    operator_only_df.to_csv(operator_only_path, index=False)
    eta_operator_df.to_csv(eta_operator_path, index=False)
    eta_step_df.to_csv(eta_step_path, index=False)
    compression_df.to_csv(compression_path, index=False)

    ok = rows_df[rows_df["status"].isin(["ok", "target_eval_missing"])].copy()
    supported = ok[ok["status"] == "ok"].copy()
    if not supported.empty:
        contextual_overall = pd.DataFrame(
            [
                {
                    "record_space": record_spec.name,
                    "n_target_datasets": supported["dataset_id"].nunique(),
                    "n_contextual_operator_tests": len(supported),
                    "contextual_target_support_rate": supported["target_supports_operator"].astype(float).mean(),
                    "contextual_target_negative_rate": (supported["target_lift"] < -float(args.min_lift)).astype(float).mean(),
                    "mean_contextual_target_lift": supported["target_lift"].mean(),
                    "median_contextual_target_lift": supported["target_lift"].median(),
                    "hist_lift_available_rate": supported["neighbor_hist_lift"].notna().mean(),
                    "hist_target_sign_agreement_rate": supported["hist_target_sign_agreement"].dropna().astype(float).mean()
                    if supported["hist_target_sign_agreement"].notna().any()
                    else np.nan,
                    "raw_target_sign_agreement_rate": supported["raw_target_sign_agreement"].dropna().astype(float).mean()
                    if supported["raw_target_sign_agreement"].notna().any()
                    else np.nan,
                }
            ]
        )
        contextual_by_operator = (
            supported.groupby(["step", "operator"], as_index=False)
            .agg(
                n_tests=("target_lift", "size"),
                n_datasets=("dataset_id", "nunique"),
                target_support_rate=("target_supports_operator", lambda x: x.astype(float).mean()),
                mean_target_lift=("target_lift", "mean"),
                median_target_lift=("target_lift", "median"),
                mean_neighbor_hist_lift=("neighbor_hist_lift", "mean"),
                hist_lift_available_rate=("neighbor_hist_lift", lambda x: x.notna().mean()),
                hist_target_sign_agreement_rate=(
                    "hist_target_sign_agreement",
                    lambda x: x.dropna().astype(float).mean() if x.notna().any() else np.nan,
                ),
            )
            .sort_values(["target_support_rate", "mean_target_lift"], ascending=False)
        )
    else:
        contextual_overall = pd.DataFrame()
        contextual_by_operator = pd.DataFrame()

    op_ok = operator_only_df[operator_only_df.get("status", pd.Series(dtype=str)) == "ok"].copy()
    if not op_ok.empty:
        controlled_overall = pd.DataFrame(
            [
                {
                    "record_space": record_spec.name,
                    "n_target_datasets": op_ok["dataset_id"].nunique(),
                    "n_controlled_operator_tests": len(op_ok),
                    "controlled_target_support_rate": (op_ok["target_operator_lift"] > float(args.min_lift)).astype(float).mean(),
                    "controlled_target_negative_rate": (op_ok["target_operator_lift"] < -float(args.min_lift)).astype(float).mean(),
                    "mean_controlled_target_lift": op_ok["target_operator_lift"].mean(),
                    "median_controlled_target_lift": op_ok["target_operator_lift"].median(),
                    "hist_target_sign_agreement_rate": op_ok["hist_target_sign_agreement"].dropna().astype(float).mean()
                    if op_ok["hist_target_sign_agreement"].notna().any()
                    else np.nan,
                    "hist_target_spearman_all_operator_rows": _safe_spearman(
                        op_ok["neighbor_operator_lift"], op_ok["target_operator_lift"]
                    ),
                    "raw_target_sign_agreement_rate": op_ok["raw_target_sign_agreement"].dropna().astype(float).mean()
                    if op_ok["raw_target_sign_agreement"].notna().any()
                    else np.nan,
                    "raw_target_spearman_all_operator_rows": _safe_spearman(
                        op_ok["neighbor_raw_operator_lift"], op_ok["target_operator_lift"]
                    ),
                }
            ]
        )
        controlled_by_operator = (
            op_ok.groupby(["step", "operator"], as_index=False)
            .agg(
                n_tests=("target_operator_lift", "size"),
                n_datasets=("dataset_id", "nunique"),
                target_support_rate=("target_operator_lift", lambda x: (x > float(args.min_lift)).astype(float).mean()),
                mean_target_operator_lift=("target_operator_lift", "mean"),
                median_target_operator_lift=("target_operator_lift", "median"),
                mean_neighbor_operator_lift=("neighbor_operator_lift", "mean"),
                hist_target_sign_agreement_rate=(
                    "hist_target_sign_agreement",
                    lambda x: x.dropna().astype(float).mean() if x.notna().any() else np.nan,
                ),
            )
            .sort_values(["target_support_rate", "mean_target_operator_lift"], ascending=False)
        )
    else:
        controlled_overall = pd.DataFrame()
        controlled_by_operator = pd.DataFrame()

    if not eta_step_df.empty:
        eta_overall = pd.DataFrame(
            [
                {
                    "record_space": record_spec.name,
                    "n_target_datasets": eta_step_df["dataset_id"].nunique(),
                    "n_step_tests": len(eta_step_df),
                    "eta_top1_match_rate": eta_step_df["eta_top1_match"].astype(float).mean(),
                    "eta_top2_match_rate": eta_step_df["eta_top2_match"].astype(float).mean(),
                    "mean_eta_target_spearman": eta_step_df["eta_target_spearman"].mean(),
                    "median_eta_target_spearman": eta_step_df["eta_target_spearman"].median(),
                    "mean_neighbor_hist_target_spearman": eta_step_df["neighbor_hist_target_spearman"].mean(),
                    "mean_eta_margin_top1_top2": eta_step_df["eta_margin_top1_top2"].mean(),
                }
            ]
        )
    else:
        eta_overall = pd.DataFrame()

    if not compression_df.empty:
        compression_overall = pd.DataFrame(
            [
                {
                    "record_space": record_spec.name,
                    "n_target_datasets": compression_df["dataset_id"].nunique(),
                    "mean_best_retrieved_full_score": compression_df["best_retrieved_full_score"].mean(),
                    "mean_eta_top_composed_score": compression_df["eta_top_composed_score"].mean(),
                    "mean_controlled_baseline_score": compression_df["controlled_baseline_score"].mean(),
                    "mean_aco_score": compression_df["aco_score"].mean(),
                    "mean_retrieved_minus_eta_top": compression_df["retrieved_minus_eta_top"].mean(),
                    "mean_aco_minus_retrieved": compression_df["aco_minus_retrieved"].mean(),
                    "mean_aco_minus_eta_top": compression_df["aco_minus_eta_top"].mean(),
                }
            ]
        )
    else:
        compression_overall = pd.DataFrame()

    contextual_overall_path = out_dir / "contextual_operator_transfer_overall.csv"
    contextual_by_operator_path = out_dir / "contextual_operator_transfer_by_operator.csv"
    controlled_overall_path = out_dir / "controlled_operator_transfer_overall.csv"
    controlled_by_operator_path = out_dir / "controlled_operator_transfer_by_operator.csv"
    eta_overall_path = out_dir / "eta_predictive_overall.csv"
    compression_overall_path = out_dir / "full_pipeline_vs_compressed_eta_overall.csv"

    contextual_overall.to_csv(contextual_overall_path, index=False)
    contextual_by_operator.to_csv(contextual_by_operator_path, index=False)
    controlled_overall.to_csv(controlled_overall_path, index=False)
    controlled_by_operator.to_csv(controlled_by_operator_path, index=False)
    eta_overall.to_csv(eta_overall_path, index=False)
    compression_overall.to_csv(compression_overall_path, index=False)

    if not controlled_overall.empty:
        print("\n=== CONTROLLED OPERATOR TRANSFER OVERALL ===")
        print(controlled_overall.to_string(index=False))
    if not eta_overall.empty:
        print("\n=== ETA PREDICTIVE POWER OVERALL ===")
        print(eta_overall.to_string(index=False))
    if not compression_overall.empty:
        print("\n=== FULL PIPELINE VS COMPRESSED ETA OVERALL ===")
        print(compression_overall.to_string(index=False))
    if not contextual_overall.empty:
        print("\n=== CONTEXTUAL OPERATOR TRANSFER OVERALL ===")
        print(contextual_overall.to_string(index=False))

    print("\nSaved:")
    for path in [
        rows_path,
        operator_only_path,
        eta_operator_path,
        eta_step_path,
        compression_path,
        contextual_overall_path,
        contextual_by_operator_path,
        controlled_overall_path,
        controlled_by_operator_path,
        eta_overall_path,
        compression_overall_path,
    ]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
