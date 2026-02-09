"""Run pipeline recommendation from CLI."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from automl_aco.config import (
    DEFAULT_PIPELINE_OPTIONS,
    KAGGLE_METAFEATURES_PATH,
    KAGGLE_PIPELINES_PATH,
    KAGGLE_REPO_ROOT,
    KAGGLE_TRAIN_PERF_PATHS,
    KAGGLE_DATA_FOLDER,
    LOCAL_METAFEATURES_PATH,
    LOCAL_PIPELINES_PATH,
    LOCAL_PIPELINES_PATH_ALT,
    LOCAL_TRAIN_PERF_PATH,
)
from automl_aco.data.loaders import (
    load_openml_dataset,
    load_kaggle_dataset,
    load_dummy_dataset,
    load_csv_dataset,
)
from automl_aco.data.metafeatures import extract_enhanced_metafeatures
from automl_aco.metalearning.recommender import MetaPipelineRecommender
from automl_aco.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline recommendation")
    parser.add_argument("--performance-matrix", required=False, help="Path to performance matrix CSV")
    parser.add_argument("--metafeatures", required=False, help="Path to metafeatures CSV")
    parser.add_argument("--pipeline-configs", required=False, help="Path to pipeline configs JSON")
    parser.add_argument(
        "--dataset-source",
        choices=["openml", "kaggle", "csv", "dummy"],
        default=None,
        help="Dataset source (openml|kaggle|csv|dummy). If omitted, inferred.",
    )
    parser.add_argument("--dataset-csv", required=False, help="Path to dataset CSV (csv source only)")
    parser.add_argument("--target-column", default="target", help="Target column name")
    parser.add_argument("--dataset-id", required=False, help="Dataset id for metafeature lookup")
    parser.add_argument("--kaggle-data-folder", default=KAGGLE_DATA_FOLDER, help="Kaggle data folder for csv by id")
    parser.add_argument("--kaggle-target-column", default="target", help="Target column for kaggle CSVs")
    parser.add_argument("--kaggle-root", default=KAGGLE_REPO_ROOT, help="Kaggle repo root path")
    parser.add_argument("--use-aco", action="store_true", help="Enable ACO search")
    parser.add_argument("--k", type=int, default=5, help="Top-k similar datasets")
    parser.add_argument("--eval-k", type=int, default=3, help="Number of top pipelines to evaluate")
    parser.add_argument("--n-ants", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--metric-path", required=False, help="Optional metric path to load")
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Use Kaggle default paths for performance matrix, metafeatures, and pipelines",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Output directory for saved recommendation and plots (default: /kaggle/working or ./outputs)",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show sklearn warnings during evaluation",
    )
    parser.add_argument("--verbose", action="store_true", help="Match notebook-style progress output")
    return parser


def main() -> None:
    configure_logging()
    args = build_arg_parser().parse_args()

    if not args.show_warnings:
        try:
            from sklearn.exceptions import ConvergenceWarning

            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        except Exception:
            pass

    use_kaggle = args.kaggle or os.path.isdir("/kaggle/working")

    def pick_existing(label: str, candidates):
        for path in candidates:
            if path and os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find {label}. Tried: {candidates}")

    if args.performance_matrix:
        performance_matrix_path = args.performance_matrix
    else:
        if use_kaggle:
            repo_perf = os.path.join(args.kaggle_root, "aco", "training_performance_matrix_autogluon.csv")
            performance_matrix_path = pick_existing(
                "performance matrix",
                [repo_perf] + KAGGLE_TRAIN_PERF_PATHS,
            )
        else:
            performance_matrix_path = pick_existing(
                "performance matrix",
                [LOCAL_TRAIN_PERF_PATH],
            )

    if args.metafeatures:
        metafeatures_path = args.metafeatures
    else:
        if use_kaggle:
            repo_meta = os.path.join(args.kaggle_root, "aco", "dataset_feats.csv")
            repo_meta_alt = os.path.join(args.kaggle_root, "data", "openml", "dataset_feats.csv")
            metafeatures_path = pick_existing("metafeatures", [repo_meta, repo_meta_alt, KAGGLE_METAFEATURES_PATH])
        else:
            metafeatures_path = pick_existing(
                "metafeatures",
                ["dataset_feats.csv", LOCAL_METAFEATURES_PATH, "data/openml/dataset_feats.csv", "Data/openml/dataset_feats.csv"],
            )

    if args.pipeline_configs:
        pipeline_configs_path = args.pipeline_configs
    else:
        if use_kaggle:
            repo_pipelines = os.path.join(args.kaggle_root, "aco", "pipeline_configs.json")
            repo_pipelines_alt = os.path.join(args.kaggle_root, "Data", "openml", "pipelines.json")
            pipeline_configs_path = pick_existing(
                "pipeline configs",
                [repo_pipelines, repo_pipelines_alt, KAGGLE_PIPELINES_PATH],
            )
        else:
            pipeline_configs_path = pick_existing(
                "pipeline configs",
                [LOCAL_PIPELINES_PATH, LOCAL_PIPELINES_PATH_ALT],
            )

    perf = pd.read_csv(performance_matrix_path, index_col=0)
    meta = pd.read_csv(metafeatures_path, index_col=0)
    if args.verbose:
        print(f"Loaded performance matrix: {performance_matrix_path}")
        print(f"Loaded metafeatures: {metafeatures_path}")
        print(f"Loaded pipeline configs: {pipeline_configs_path}")

    def _normalize_id(val: object) -> str:
        s = str(val).strip()
        if s.startswith("D_"):
            s = s[2:]
        if s.startswith("Dataset_"):
            s = s.split("_", 1)[1]
        return s

    def _maybe_set_meta_index(meta_df: pd.DataFrame, perf_df: pd.DataFrame) -> pd.DataFrame:
        perf_norm = {_normalize_id(c) for c in perf_df.columns}
        candidate_cols = ["dataset_id", "did", "id", "Unnamed: 0"]
        best_col = None
        best_overlap = 0
        for col in candidate_cols:
            if col in meta_df.columns:
                vals = meta_df[col].astype(str).map(_normalize_id)
                overlap = len(set(vals) & perf_norm)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_col = col
        if best_col is not None and best_overlap > 0:
            if args.verbose:
                print(f"Using metafeatures id column: {best_col} (overlap={best_overlap})")
            meta_df = meta_df.set_index(best_col)
        return meta_df

    meta = _maybe_set_meta_index(meta, perf)

    if pipeline_configs_path:
        with open(pipeline_configs_path, "r", encoding="utf-8") as f:
            pipeline_configs = json.load(f)
    else:
        pipeline_configs = [
            {
                "name": "baseline",
                "imputation": "none",
                "scaling": "none",
                "encoding": "onehot",
                "feature_selection": "none",
                "outlier_removal": "none",
                "dimensionality_reduction": "none",
            }
        ]

    dataset_source = args.dataset_source
    if dataset_source is None:
        if args.dataset_csv:
            dataset_source = "csv"
        elif use_kaggle:
            dataset_source = "kaggle"
        else:
            dataset_source = "openml"

    dataset_id = None
    if args.dataset_id is not None and str(args.dataset_id).isdigit():
        dataset_id = int(args.dataset_id)
    elif args.dataset_id is not None:
        dataset_id = args.dataset_id

    if dataset_source == "openml":
        if dataset_id is None:
            raise ValueError("--dataset-id is required for openml source")
        dataset = load_openml_dataset(dataset_id, verbose=args.verbose)
    elif dataset_source == "kaggle":
        if dataset_id is None:
            raise ValueError("--dataset-id is required for kaggle source")
        dataset = load_kaggle_dataset(
            dataset_id,
            data_folder=args.kaggle_data_folder,
            target_column=args.kaggle_target_column,
            verbose=args.verbose,
        )
    elif dataset_source == "csv":
        if not args.dataset_csv:
            raise ValueError("--dataset-csv is required for csv source")
        dataset = load_csv_dataset(
            args.dataset_csv,
            target_column=args.target_column,
            dataset_id=dataset_id,
            verbose=args.verbose,
        )
    elif dataset_source == "dummy":
        if dataset_id is None:
            raise ValueError("--dataset-id is required for dummy source")
        dataset = load_dummy_dataset(dataset_id, verbose=args.verbose)
    else:
        raise ValueError(f"Unknown dataset source: {dataset_source}")

    if dataset is None or "X" not in dataset or "y" not in dataset:
        raise ValueError("Dataset loading failed or did not return X/y")

    X = dataset["X"]
    y = dataset["y"]
    test_dataset_df = X.copy()
    test_dataset_df["target"] = y

    recommender = MetaPipelineRecommender(perf, meta, pipeline_configs, verbose=args.verbose)
    if args.metric_path:
        recommender.load_metric(args.metric_path)

    def _mf_func(_df):
        return extract_enhanced_metafeatures(dataset, meta_features_df=meta)

    recommendation = recommender.recommend(
        new_dataset=test_dataset_df,
        target_column="target",
        options=DEFAULT_PIPELINE_OPTIONS,
        k=args.k,
        eval_k=args.eval_k,
        use_autogluon=True,
        use_aco=args.use_aco,
        aco_params={"n_ants": args.n_ants, "n_iterations": args.n_iterations},
        time_limit_per_model=args.time_limit,
        metafeatures_func=_mf_func,
    )

    def _get_output_dir() -> str:
        if args.output_dir:
            out_dir = args.output_dir
        elif os.path.isdir("/kaggle/working"):
            out_dir = "/kaggle/working"
        else:
            out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _format_pipeline(cfg: dict) -> str:
        if not cfg:
            return "None"
        parts = []
        for step in DEFAULT_PIPELINE_OPTIONS.keys():
            if step in cfg:
                parts.append(f"{step}={cfg[step]}")
        return "{" + ", ".join(parts) + "}"

    def _build_history(history_raw, aco_results, n_ants: int, n_iterations: int):
        if history_raw and isinstance(history_raw, list) and isinstance(history_raw[0], dict):
            if "iteration" in history_raw[0]:
                return history_raw
        flat = None
        if history_raw and isinstance(history_raw, list):
            if isinstance(history_raw[0], (list, tuple)) and len(history_raw[0]) >= 2:
                flat = history_raw
        if flat is None and aco_results and isinstance(aco_results, list):
            if isinstance(aco_results[0], (list, tuple)) and len(aco_results[0]) >= 2:
                flat = aco_results
        if flat is None:
            return []
        n_ants = max(int(n_ants), 1)
        history = []
        for i in range(0, len(flat), n_ants):
            chunk = flat[i : i + n_ants]
            scores = []
            for _cfg, sc in chunk:
                if isinstance(sc, (int, float)):
                    scores.append(float(sc))
            best = max(scores) if scores else None
            history.append({"iteration": len(history) + 1, "best_score": best})
            if n_iterations and len(history) >= int(n_iterations):
                break
        return history

    def _save_history_plot(history, output_dir: str) -> Optional[str]:
        if not history:
            return None
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None
        iters = [h["iteration"] for h in history if h.get("best_score") is not None]
        scores = [h["best_score"] for h in history if h.get("best_score") is not None]
        if not iters:
            return None
        plt.figure(figsize=(6, 4))
        plt.plot(iters, scores, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Best Score")
        plt.title("ACO Best Score per Iteration")
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(output_dir, "aco_progress.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path

    output_dir = _get_output_dir()
    aco_results = recommendation.get("aco_results") or []
    history = _build_history(recommendation.get("aco_history"), aco_results, args.n_ants, args.n_iterations)
    if history and (not recommendation.get("aco_history") or not isinstance(recommendation.get("aco_history"), list)):
        recommendation["aco_history"] = history

    rec_path = os.path.join(output_dir, "recommendation.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2, default=str)
    history_path = None
    if history:
        history_path = os.path.join(output_dir, "aco_history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False)

    plot_path = _save_history_plot(history, output_dir)

    pipeline_cfg = recommendation.get("pipeline_config") or {}
    if "recommended_performance" in recommendation:
        proxy_score = recommendation.get("recommended_performance")
    else:
        proxy_score = recommendation.get("expected_performance")
    final_eval = recommendation.get("final_evaluation", {})
    final_score = recommendation.get("final_performance", final_eval.get("score"))

    print("\nFinal recommendation")
    print(f"  Pipeline: {_format_pipeline(pipeline_cfg)}")
    if proxy_score is not None:
        print(f"  Proxy score: {float(proxy_score):.4f}")
    if final_score is not None and final_eval:
        print(f"  Final eval ({final_eval.get('method', 'unknown')}): {float(final_score):.4f}")
    print(f"  Saved recommendation: {rec_path}")
    if history_path:
        print(f"  Saved ACO history: {history_path}")
    if plot_path:
        print(f"  Saved ACO plot: {plot_path}")
    elif history:
        print("  ACO plot skipped (matplotlib not available)")

    logger.info("Saved recommendation to %s", rec_path)


if __name__ == "__main__":
    main()
