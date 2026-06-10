"""RQ3 retrieval-only baseline.

This tests whether nearest-neighbor pipeline retrieval is already enough,
without Phase-2 eta aggregation or Phase-3 optimizer search.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def _load_pipeline_configs(path: Path, perf: pd.DataFrame) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        configs = json.load(f)

    existing = {str(cfg.get("name")) for cfg in configs if isinstance(cfg, dict) and "name" in cfg}
    missing = [str(name) for name in perf.index if str(name) not in existing]
    if missing:
        configs = list(configs) + [_infer_pipeline_config_from_name(name) for name in missing]
    return configs


def _format_pipeline(cfg: Optional[Dict[str, Any]]) -> str:
    if not cfg:
        return ""
    parts = []
    for step in DEFAULT_PIPELINE_OPTIONS:
        if step in cfg:
            parts.append(f"{step}={cfg[step]}")
    return "{" + ", ".join(parts) + "}"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval-only nearest-neighbor baselines")
    parser.add_argument("--root", default=os.environ.get("ROOT", str(Path(__file__).resolve().parents[1])))
    parser.add_argument("--performance-matrix", default=None)
    parser.add_argument("--metafeatures", default=None)
    parser.add_argument("--metafeatures-id-column", default=None)
    parser.add_argument("--pipeline-configs", default=None)
    parser.add_argument("--dataset-source", choices=["openml", "kaggle"], default="openml")
    parser.add_argument("--openml-local-folder", default=None)
    parser.add_argument("--kaggle-data-folder", default=None)
    parser.add_argument("--kaggle-target-column", default="target")
    parser.add_argument("--dataset-ids", nargs="+", required=True)
    parser.add_argument("--eval-l-values", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--retrieval-k-values", nargs="+", type=int, default=[1])
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--require-autogluon", action="store_true")
    parser.add_argument("--output-dir", default="/kaggle/working/retrieval_only_baseline")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    perf_path = Path(args.performance_matrix or root / "aco" / "training_performance_matrix_autogluon.csv")
    meta_path = Path(args.metafeatures or root / "data" / "openml" / "dataset_feats.csv")
    cfg_path = Path(args.pipeline_configs or root / "aco" / "pipeline_configs.json")
    openml_local = Path(args.openml_local_folder or root / "test_data_local")
    kaggle_data = Path(args.kaggle_data_folder or root / "test_data_local")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    perf = pd.read_csv(perf_path, index_col=0)
    meta = _maybe_set_meta_index(pd.read_csv(meta_path), perf, args.metafeatures_id_column)
    configs = _load_pipeline_configs(cfg_path, perf)
    recommender = MetaPipelineRecommender(perf, meta, configs, verbose=args.verbose)

    dataset_ids = _parse_dataset_ids(args.dataset_ids)
    rows: List[Dict[str, Any]] = []

    for dataset_id in dataset_ids:
        start = time.perf_counter()
        try:
            if args.dataset_source == "openml":
                ds = load_openml_dataset(dataset_id, verbose=args.verbose, local_data_folder=str(openml_local))
            else:
                ds = load_kaggle_dataset(
                    dataset_id,
                    data_folder=str(kaggle_data),
                    target_column=args.kaggle_target_column,
                    verbose=args.verbose,
                )
            if ds is None or "X" not in ds or "y" not in ds:
                raise RuntimeError(f"Could not load dataset {dataset_id}")

            data = ds["X"].copy()
            data["target"] = ds["y"]

            def mf_func(_df: pd.DataFrame, _dataset: Dict[str, Any] = ds) -> np.ndarray:
                return extract_enhanced_metafeatures(_dataset, meta_features_df=meta)

            for retrieval_k in args.retrieval_k_values:
                for eval_l in args.eval_l_values:
                    rec = recommender.recommend(
                        new_dataset=data,
                        target_column="target",
                        k=max(1, int(retrieval_k)),
                        eval_k=max(1, int(eval_l)),
                        use_autogluon=True,
                        time_limit_per_model=int(args.time_limit),
                        metafeatures_func=mf_func,
                        use_aco=False,
                        aco_params={"require_autogluon": bool(args.require_autogluon)},
                        final_autogluon_topk=1,
                    )

                    sims = rec.get("similarity_scores", {}) or {}
                    similar = list(rec.get("similar_datasets", []) or [])
                    top_neighbor = str(similar[0]) if similar else ""
                    retrieved = rec.get("top_candidates_evaluated") or rec.get("top_candidates") or []
                    retrieved_names = [str(item[0].get("name", "")) if isinstance(item[0], dict) else str(item[0]) for item in retrieved]

                    rows.append(
                        {
                            "dataset_id": str(dataset_id),
                            "mode": f"retrieval_k{int(retrieval_k)}_direct_top{int(eval_l)}",
                            "retrieval_k": int(retrieval_k),
                            "eval_l": int(eval_l),
                            "top_neighbor": top_neighbor,
                            "top_neighbor_similarity": float(sims.get(similar[0], np.nan)) if similar else np.nan,
                            "retrieved_pipeline_names": "|".join(retrieved_names),
                            "selected_pipeline": _format_pipeline(rec.get("pipeline_config")),
                            "score": rec.get("expected_performance"),
                            "evaluation_method": rec.get("evaluation_method"),
                            "elapsed_seconds_total_dataset_so_far": time.perf_counter() - start,
                            "status": "ok",
                            "error": "",
                        }
                    )
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "mode": "failed",
                    "retrieval_k": np.nan,
                    "eval_l": np.nan,
                    "top_neighbor": "",
                    "top_neighbor_similarity": np.nan,
                    "retrieved_pipeline_names": "",
                    "selected_pipeline": "",
                    "score": np.nan,
                    "evaluation_method": "",
                    "elapsed_seconds_total_dataset_so_far": time.perf_counter() - start,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "retrieval_only_results.csv", index=False)
    with (out_dir / "retrieval_only_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    ok = df[df["status"] == "ok"].copy()
    if not ok.empty:
        summary = (
            ok.groupby("mode", as_index=False)
            .agg(n_datasets=("dataset_id", "nunique"), mean_score=("score", "mean"), median_score=("score", "median"))
            .sort_values("mean_score", ascending=False)
        )
        summary.to_csv(out_dir / "retrieval_only_summary.csv", index=False)
        print(summary.to_string(index=False))
    else:
        print("No successful retrieval-only rows.")
    print(f"Saved results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
