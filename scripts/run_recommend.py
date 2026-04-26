"""Run pipeline recommendation from CLI."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Any, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import numpy as np

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
from automl_aco.utils.operator_spec import base_operator_name
from automl_aco.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline recommendation")
    parser.add_argument("--performance-matrix", required=False, help="Path to performance matrix CSV")
    parser.add_argument("--metafeatures", required=False, help="Path to metafeatures CSV")
    parser.add_argument("--pipeline-configs", required=False, help="Path to pipeline configs JSON")
    parser.add_argument(
        "--pipeline-override",
        required=False,
        help=(
            "Comma-separated step=choice list to force the pipeline. "
            "Example: imputation=none,encoding=none,scaling=standard,outlier_removal=iqr,"
            "feature_selection=k_best,dimensionality_reduction=pca"
        ),
    )
    parser.add_argument(
        "--dataset-source",
        choices=["openml", "kaggle", "csv", "dummy"],
        default=None,
        help="Dataset source (openml|kaggle|csv|dummy). If omitted, inferred.",
    )
    parser.add_argument("--dataset-csv", required=False, help="Path to dataset CSV (csv source only)")
    parser.add_argument("--target-column", default="target", help="Target column name")
    parser.add_argument("--dataset-id", required=False, help="Dataset id for metafeature lookup")
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        required=False,
        help="Dataset ids for batch runs (comma-separated and/or space-separated)",
    )
    parser.add_argument(
        "--openml-local-folder",
        required=False,
        default=None,
        help=(
            "Optional local folder for OpenML fallback files (e.g., 1520.csv or 1520.csv.zip). "
            "Used only when --dataset-source openml and API fetch fails."
        ),
    )
    parser.add_argument("--kaggle-data-folder", default=KAGGLE_DATA_FOLDER, help="Kaggle data folder for csv by id")
    parser.add_argument("--kaggle-target-column", default="target", help="Target column for kaggle CSVs")
    parser.add_argument("--kaggle-root", default=KAGGLE_REPO_ROOT, help="Kaggle repo root path")
    parser.add_argument("--use-aco", action="store_true", help="Enable ACO search")
    parser.add_argument("--k", type=int, default=5, help="Top-k similar datasets")
    parser.add_argument(
        "--heuristic-top-k",
        type=int,
        default=None,
        help="Top-k similar datasets used to build Phase-2 ACO heuristic (default: use --k)",
    )
    parser.add_argument(
        "--dataset-weighting",
        choices=["equality", "similarity"],
        default="similarity",
        help="How to weight top-K neighbors when transferring Phase-2 heuristic",
    )
    parser.add_argument(
        "--heuristic-top-l",
        type=int,
        default=3,
        help="Top-L historical pipelines selected per neighbor for Phase-2 transfer",
    )
    parser.add_argument(
        "--heuristic-transfer-method",
        choices=["weighted_topk_topl", "legacy_weighted_average"],
        default="weighted_topk_topl",
        help="Phase-2 heuristic transfer algorithm",
    )
    parser.add_argument(
        "--heuristic-similarity-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for similarity weighting over top-K neighbors",
    )
    parser.add_argument(
        "--heuristic-eta-floor",
        type=float,
        default=0.05,
        help="Positive floor for per-step eta normalization",
    )
    parser.add_argument(
        "--score-direction",
        choices=["higher_is_better", "lower_is_better"],
        default="higher_is_better",
        help="Direction of performance-matrix values used in transfer.",
    )
    parser.add_argument("--eval-k", type=int, default=3, help="Number of top pipelines to evaluate")
    parser.add_argument("--n-ants", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0, help="ACO alpha: pheromone importance")
    parser.add_argument("--beta", type=float, default=2.0, help="ACO beta: heuristic importance")
    parser.add_argument("--evaporation", type=float, default=0.2, help="ACO pheromone evaporation rate")
    parser.add_argument(
        "--optimizer",
        choices=["aco", "dqn", "random", "ga", "sa", "greedy", "mcts", "beam", "tpe", "exhaustive"],
        default="aco",
        help="Search optimizer. ACO uses n-ants*n-iterations; DQN/others use sample-budget.",
    )
    parser.add_argument("--sample-budget", type=int, default=100, help="Config evaluation budget for non-ACO optimizers")
    parser.add_argument(
        "--dqn-epochs",
        type=int,
        default=1,
        help="Legacy alias for DQN updates-per-episode (optimizer=dqn)",
    )
    parser.add_argument("--dqn-batch-size", type=int, default=64, help="Replay batch size for DQN updates (optimizer=dqn)")
    parser.add_argument("--dqn-lr", type=float, default=3e-4, help="Offline DQN learning rate (optimizer=dqn)")
    parser.add_argument("--dqn-gamma", type=float, default=0.95, help="Offline DQN discount factor (optimizer=dqn)")
    parser.add_argument("--dqn-target-update", type=int, default=5, help="Target-net sync interval in epochs")
    parser.add_argument("--dqn-loss-fn", choices=["huber", "mse"], default="huber", help="DQN TD loss")
    parser.add_argument("--dqn-huber-delta", type=float, default=1.0, help="Huber delta if dqn-loss-fn=huber")
    parser.add_argument("--dqn-grad-clip-norm", type=float, default=5.0, help="Gradient clipping norm for DQN")
    parser.add_argument("--dqn-reward-clip", type=float, default=1.0, help="Reward clip value for DQN targets")
    parser.add_argument("--dqn-target-q-clip", type=float, default=5.0, help="Clamp TD target Q to [-clip, clip]")
    parser.add_argument(
        "--dqn-use-double-dqn",
        dest="dqn_use_double_dqn",
        action="store_true",
        help="Use Double-DQN target action selection",
    )
    parser.add_argument(
        "--no-dqn-use-double-dqn",
        dest="dqn_use_double_dqn",
        action="store_false",
        help="Disable Double-DQN target action selection",
    )
    parser.set_defaults(dqn_use_double_dqn=True)
    parser.add_argument(
        "--dqn-updates-per-episode",
        type=int,
        default=1,
        help="Number of replay updates after each newly evaluated pipeline (optimizer=dqn)",
    )
    parser.add_argument(
        "--dqn-replay-warmup",
        type=int,
        default=16,
        help="Number of evaluated pipelines before starting replay updates (optimizer=dqn)",
    )
    parser.add_argument(
        "--dqn-order-policy",
        choices=["fixed", "ctxpipe"],
        default="ctxpipe",
        help="Order policy mode for DQN. 'ctxpipe' learns logical pipeline order like CtxPipe.",
    )
    parser.add_argument(
        "--dqn-num-logic-orders",
        type=int,
        default=6,
        help="Maximum logical pipeline orders considered in DQN ctxpipe mode",
    )
    parser.add_argument(
        "--dqn-order-updates-per-episode",
        type=int,
        default=1,
        help="Replay updates for logical-order policy after each evaluated pipeline",
    )
    parser.add_argument(
        "--dqn-order-replay-warmup",
        type=int,
        default=16,
        help="Replay warmup size before training logical-order policy",
    )
    parser.add_argument(
        "--dqn-order-epsilon-start",
        type=float,
        default=0.35,
        help="Start epsilon for logical-order exploration",
    )
    parser.add_argument(
        "--dqn-order-epsilon-end",
        type=float,
        default=0.05,
        help="End epsilon for logical-order exploration",
    )
    parser.add_argument("--dqn-epsilon-start", type=float, default=0.35, help="Start epsilon for DQN sampling")
    parser.add_argument("--dqn-epsilon-end", type=float, default=0.05, help="End epsilon for DQN sampling")
    parser.add_argument(
        "--dqn-warmstart-weight",
        type=float,
        default=0.5,
        help="Weight for warm-start priors in DQN action scores",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ACO and ordering search")
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument(
        "--ordering-quick-time-limit",
        type=int,
        default=30,
        help="Quick AutoGluon time limit (seconds) per ordering iteration",
    )
    parser.add_argument("--search-ordering", action="store_true", help="Search over valid step-type orders")
    parser.add_argument("--num-orders", type=int, default=10, help="Number of candidate orders to evaluate")
    parser.add_argument(
        "--order-strategy",
        choices=["fixed", "random", "heuristic", "scored", "all"],
        default="fixed",
        help="Order proposal strategy",
    )
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
    parser.add_argument(
        "--proxy-profile",
        choices=["default", "robust"],
        default="default",
        help=(
            "Proxy scoring profile. robust uses multi-seed validation and "
            "over-processing penalties to reduce search-time overfitting."
        ),
    )
    parser.add_argument(
        "--proxy-seeds",
        default=None,
        help="Comma-separated split seeds for proxy scoring (overrides profile), e.g. 42,52,62",
    )
    parser.add_argument(
        "--proxy-clf-model",
        choices=["logreg", "linear_svm", "random_forest", "extra_trees", "knn", "hist_gbdt"],
        default="logreg",
        help="Proxy model used for classification tasks during search",
    )
    parser.add_argument(
        "--proxy-reg-model",
        choices=["ensemble", "linear", "random_forest"],
        default="ensemble",
        help="Proxy model used for regression tasks during search",
    )
    parser.add_argument(
        "--proxy-logreg-max-iter",
        type=int,
        default=3000,
        help="Max iterations for logistic-regression proxy model",
    )
    parser.add_argument(
        "--final-autogluon-topk",
        type=int,
        default=1,
        help="Re-evaluate top-k proxy pipelines with final AutoGluon (default: 1)",
    )
    parser.add_argument(
        "--dataset378-profile",
        choices=["off", "conservative", "scaling_only"],
        default="off",
        help=(
            "Apply dataset-specific option constraints only when dataset_id=378. "
            "Use conservative or scaling_only to reduce over-processing risk."
        ),
    )
    parser.add_argument(
        "--operator-param-search",
        action="store_true",
        help="Enable parameterized operator tokens (e.g., knn@k=7, pca@n=20) in search space.",
    )
    parser.add_argument(
        "--operator-param-grid",
        choices=["light", "full"],
        default="light",
        help="Grid size for parameterized operator search.",
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

    def _parse_pipeline_override(raw: Optional[str]) -> Optional[dict]:
        if raw is None:
            return None
        parsed = {}
        for token in str(raw).split(","):
            if "=" not in token:
                continue
            step, val = token.split("=", 1)
            step = step.strip()
            val = val.strip()
            if step:
                parsed[step] = val
        return parsed or None

    pipeline_override = _parse_pipeline_override(getattr(args, "pipeline_override", None))
    if pipeline_override and args.verbose:
        print(f"Pipeline override: {pipeline_override}")

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

    def _parse_dataset_id(raw: Any):
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)
        return s

    dataset_ids: List[Any] = []
    if args.dataset_ids:
        raw_tokens: List[str] = []
        for token in args.dataset_ids:
            raw_tokens.extend(str(token).split(","))
        dataset_ids = [_parse_dataset_id(tok) for tok in raw_tokens if str(tok).strip()]
        dataset_ids = [did for did in dataset_ids if did is not None]
    elif isinstance(args.dataset_id, str) and "," in args.dataset_id:
        # Backward-compatible convenience: allow comma lists in --dataset-id too.
        dataset_ids = [_parse_dataset_id(tok) for tok in args.dataset_id.split(",") if tok.strip()]
        dataset_ids = [did for did in dataset_ids if did is not None]
    else:
        did = _parse_dataset_id(args.dataset_id)
        if did is not None:
            dataset_ids = [did]

    if dataset_source in {"openml", "kaggle", "dummy"} and not dataset_ids:
        raise ValueError("--dataset-id or --dataset-ids is required for openml/kaggle/dummy source")
    if dataset_source == "csv" and not args.dataset_csv:
        raise ValueError("--dataset-csv is required for csv source")
    if dataset_source == "csv" and len(dataset_ids) > 1:
        raise ValueError("CSV source supports at most one dataset id (metadata lookup only)")
    if dataset_source == "csv" and not dataset_ids:
        dataset_ids = [None]

    recommender = MetaPipelineRecommender(perf, meta, pipeline_configs, verbose=args.verbose)
    if args.metric_path:
        recommender.load_metric(args.metric_path)

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
        if isinstance(cfg.get("step_order"), list):
            parts.append(f"step_order={cfg['step_order']}")
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

    def _load_dataset_for_run(dataset_id: Any):
        if dataset_source == "openml":
            return load_openml_dataset(
                dataset_id,
                verbose=args.verbose,
                local_data_folder=args.openml_local_folder,
            )
        if dataset_source == "kaggle":
            return load_kaggle_dataset(
                dataset_id,
                data_folder=args.kaggle_data_folder,
                target_column=args.kaggle_target_column,
                verbose=args.verbose,
            )
        if dataset_source == "csv":
            return load_csv_dataset(
                args.dataset_csv,
                target_column=args.target_column,
                dataset_id=dataset_id,
                verbose=args.verbose,
            )
        if dataset_source == "dummy":
            return load_dummy_dataset(dataset_id, verbose=args.verbose)
        raise ValueError(f"Unknown dataset source: {dataset_source}")

    def _build_run_options(dataset_id: Any):
        # Copy lists so per-run constraints do not mutate global defaults.
        options = {step: list(vals) for step, vals in DEFAULT_PIPELINE_OPTIONS.items()}
        profile_note = None

        if pipeline_override:
            for step, choice in pipeline_override.items():
                options[step] = [choice]
            profile_note = "pipeline_override"
            return options, profile_note

        if args.operator_param_search:
            # Parameterized operator tokens remain discrete choices and are scored inline by the same proxy.
            if args.operator_param_grid == "full":
                options["imputation"] = [
                    "none", "mean", "median", "most_frequent", "constant",
                    "knn@k=3", "knn@k=5", "knn@k=7", "knn@k=11",
                ]
                options["outlier_removal"] = [
                    "none",
                    "iqr@k=1.0", "iqr@k=1.5", "iqr@k=2.0",
                    "zscore@z=2.5", "zscore@z=3.0", "zscore@z=3.5",
                    "lof@n=10", "lof@n=20", "lof@n=30",
                    "isolation_forest@c=0.02", "isolation_forest@c=0.05", "isolation_forest@c=0.1",
                ]
                options["feature_selection"] = [
                    "none",
                    "variance_threshold@t=0.0", "variance_threshold@t=0.01", "variance_threshold@t=0.05",
                    "k_best@k=10", "k_best@k=20", "k_best@k=40",
                    "mutual_info@k=10", "mutual_info@k=20", "mutual_info@k=40",
                ]
                options["dimensionality_reduction"] = [
                    "none",
                    "pca@n=5", "pca@n=10", "pca@n=20",
                    "svd@n=5", "svd@n=10", "svd@n=20",
                ]
            else:
                options["imputation"] = [
                    "none", "mean", "median", "most_frequent", "constant",
                    "knn@k=3", "knn@k=7",
                ]
                options["outlier_removal"] = [
                    "none",
                    "iqr@k=1.5",
                    "zscore@z=3.0",
                    "lof@n=20",
                    "isolation_forest@c=0.05",
                ]
                options["feature_selection"] = [
                    "none",
                    "variance_threshold@t=0.01",
                    "k_best@k=20",
                    "mutual_info@k=20",
                ]
                options["dimensionality_reduction"] = [
                    "none",
                    "pca@n=10",
                    "svd@n=10",
                ]
            profile_note = f"operator_param_search={args.operator_param_grid}"

        if args.dataset378_profile == "off":
            return options, profile_note

        try:
            did = int(str(dataset_id))
        except Exception:
            did = None
        if did != 378:
            return options, profile_note

        if args.dataset378_profile == "conservative":
            options["imputation"] = ["none", "median", "most_frequent"]
            options["outlier_removal"] = ["none"]
            options["feature_selection"] = ["none", "variance_threshold"]
            options["dimensionality_reduction"] = ["none"]
            profile_note = (
                "dataset378_profile=conservative "
                "(imputation limited; outlier_removal and dim_reduction constrained)"
            )
            return options, profile_note

        if args.dataset378_profile == "scaling_only":
            options["imputation"] = ["none"]
            options["outlier_removal"] = ["none"]
            options["feature_selection"] = ["none"]
            options["dimensionality_reduction"] = ["none"]
            options["scaling"] = ["none", "standard", "minmax", "robust", "maxabs"]
            profile_note = (
                "dataset378_profile=scaling_only "
                "(only scaling varies; all other preprocessing steps fixed to none)"
            )
            return options, profile_note

        return options, profile_note

    def _adapt_options_to_dataset(options: dict, X: pd.DataFrame):
        if pipeline_override:
            return {k: list(v) for k, v in options.items()}, []
        notes = []
        out = {k: list(v) for k, v in options.items()}
        has_missing = bool(X.isna().to_numpy().any())
        if has_missing and "imputation" in out:
            before = len(out["imputation"])
            out["imputation"] = [v for v in out["imputation"] if base_operator_name(v) != "none"]
            if len(out["imputation"]) == 0:
                out["imputation"] = ["mean"]
            if len(out["imputation"]) != before:
                notes.append("removed imputation=none (dataset has missing values)")
        return out, notes

    def _build_proxy_settings() -> dict:
        if args.proxy_profile == "robust":
            settings = {
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
            tokens = [t.strip() for t in str(args.proxy_seeds).split(",") if t.strip()]
            parsed = []
            for token in tokens:
                try:
                    parsed.append(int(token))
                except ValueError:
                    pass
            if parsed:
                settings["split_seeds"] = parsed
        return settings

    output_dir = _get_output_dir()
    n_runs = len(dataset_ids)
    run_summaries = []
    search_enabled = args.use_aco or args.optimizer != "aco"
    proxy_settings = _build_proxy_settings()
    heuristic_top_k = int(args.heuristic_top_k) if args.heuristic_top_k is not None else int(args.k)
    if args.optimizer == "aco" and not args.use_aco:
        # Legacy behavior required --use-aco even when optimizer=aco.
        # Auto-enable to match user intent and avoid accidental prediction-only runs.
        search_enabled = True
        if args.verbose:
            print("Info: optimizer=aco selected; enabling search flow (legacy --use-aco is optional).")
    elif args.optimizer != "aco" and not args.use_aco and args.verbose:
        print("Info: --optimizer is non-ACO, enabling search flow (same as --use-aco).")
    if args.verbose:
        print(
            "Proxy profile: "
            f"{args.proxy_profile} (seeds={proxy_settings.get('split_seeds')}, "
            f"final_autogluon_topk={max(1, int(args.final_autogluon_topk))})"
        )
        print(
            "Search transfer/proxy setup: "
            f"dataset_weighting={args.dataset_weighting}, "
            f"heuristic_top_k={heuristic_top_k}, "
            f"heuristic_top_l={int(args.heuristic_top_l)}, "
            f"transfer_method={args.heuristic_transfer_method}, "
            f"sim_temperature={float(args.heuristic_similarity_temperature)}, "
            f"eta_floor={float(args.heuristic_eta_floor)}, "
            f"score_direction={args.score_direction}, "
            f"proxy_clf_model={proxy_settings.get('classification_model')}, "
            f"proxy_reg_model={proxy_settings.get('regression_model')}"
        )

    for run_idx, dataset_id in enumerate(dataset_ids, start=1):
        if n_runs > 1:
            print(f"\n=== Dataset {dataset_id} ({run_idx}/{n_runs}) ===")

        run_start = time.perf_counter()
        try:
            dataset = _load_dataset_for_run(dataset_id)
            if dataset is None or "X" not in dataset or "y" not in dataset:
                raise ValueError(f"Dataset loading failed for dataset_id={dataset_id}")

            X = dataset["X"]
            y = dataset["y"]
            run_options, run_profile_note = _build_run_options(dataset_id)
            run_options, run_option_notes = _adapt_options_to_dataset(run_options, X)
            if run_profile_note:
                print(f"  Applied profile: {run_profile_note}")
            for note in run_option_notes:
                print(f"  Auto option guard: {note}")
            test_dataset_df = X.copy()
            test_dataset_df["target"] = y

            def _mf_func(_df, _dataset=dataset):
                return extract_enhanced_metafeatures(_dataset, meta_features_df=meta)

            recommendation = recommender.recommend(
                new_dataset=test_dataset_df,
                target_column="target",
                options=run_options,
                k=args.k,
                eval_k=args.eval_k,
                use_autogluon=True,
                use_aco=search_enabled,
                aco_params={
                    "n_ants": args.n_ants,
                    "n_iterations": args.n_iterations,
                    "seed": args.seed,
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "evaporation": args.evaporation,
                    "dataset_weighting": args.dataset_weighting,
                    "heuristic_top_k": heuristic_top_k,
                    "heuristic_top_l": int(args.heuristic_top_l),
                    "heuristic_transfer_method": str(args.heuristic_transfer_method),
                    "heuristic_similarity_temperature": float(args.heuristic_similarity_temperature),
                    "heuristic_eta_floor": float(args.heuristic_eta_floor),
                    "score_direction": str(args.score_direction),
                    "query_dataset_id": dataset_id,
                    "ordering_quick_time_limit": args.ordering_quick_time_limit,
                    "dqn_epochs": args.dqn_epochs,
                    "dqn_batch_size": args.dqn_batch_size,
                    "dqn_lr": args.dqn_lr,
                    "dqn_gamma": args.dqn_gamma,
                    "dqn_target_update_interval": args.dqn_target_update,
                    "dqn_loss_fn": args.dqn_loss_fn,
                    "dqn_huber_delta": args.dqn_huber_delta,
                    "dqn_grad_clip_norm": args.dqn_grad_clip_norm,
                    "dqn_reward_clip": args.dqn_reward_clip,
                    "dqn_target_q_clip": args.dqn_target_q_clip,
                    "dqn_use_double_dqn": args.dqn_use_double_dqn,
                    "dqn_updates_per_episode": args.dqn_updates_per_episode,
                    "dqn_replay_warmup": args.dqn_replay_warmup,
                    "dqn_order_policy": args.dqn_order_policy,
                    "dqn_num_logic_orders": args.dqn_num_logic_orders,
                    "dqn_order_updates_per_episode": args.dqn_order_updates_per_episode,
                    "dqn_order_replay_warmup": args.dqn_order_replay_warmup,
                    "dqn_order_epsilon_start": args.dqn_order_epsilon_start,
                    "dqn_order_epsilon_end": args.dqn_order_epsilon_end,
                    "dqn_epsilon_start": args.dqn_epsilon_start,
                    "dqn_epsilon_end": args.dqn_epsilon_end,
                    "dqn_warmstart_weight": args.dqn_warmstart_weight,
                },
                time_limit_per_model=args.time_limit,
                metafeatures_func=_mf_func,
                search_ordering=args.search_ordering,
                num_orders=args.num_orders,
                order_strategy=args.order_strategy,
                optimizer=args.optimizer,
                sample_budget=args.sample_budget,
                proxy_settings=proxy_settings,
                final_autogluon_topk=args.final_autogluon_topk,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - run_start
            print(f"  Dataset {dataset_id} failed: {exc}")
            run_summaries.append(
                {
                    "dataset_id": dataset_id,
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": elapsed,
                }
            )
            continue

        aco_results = recommendation.get("aco_results") or []
        history = _build_history(recommendation.get("aco_history"), aco_results, args.n_ants, args.n_iterations)
        if history and (not recommendation.get("aco_history") or not isinstance(recommendation.get("aco_history"), list)):
            recommendation["aco_history"] = history

        if n_runs == 1:
            run_output_dir = output_dir
            dataset_tag = str(dataset_id) if dataset_id is not None else "single"
        else:
            dataset_tag = str(dataset_id) if dataset_id is not None else f"run{run_idx}"
            run_output_dir = os.path.join(output_dir, f"dataset_{dataset_tag}")
            os.makedirs(run_output_dir, exist_ok=True)

        rec_path = os.path.join(run_output_dir, "recommendation.json")
        recommendation["search_options"] = run_options
        recommendation["dataset_profile"] = args.dataset378_profile
        recommendation["search_hyperparams"] = {
            "k": int(args.k),
            "heuristic_top_k": int(heuristic_top_k),
            "heuristic_top_l": int(args.heuristic_top_l),
            "dataset_weighting": str(args.dataset_weighting),
            "heuristic_transfer_method": str(args.heuristic_transfer_method),
            "heuristic_similarity_temperature": float(args.heuristic_similarity_temperature),
            "heuristic_eta_floor": float(args.heuristic_eta_floor),
            "score_direction": str(args.score_direction),
            "optimizer": str(args.optimizer),
            "n_ants": int(args.n_ants),
            "n_iterations": int(args.n_iterations),
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "evaporation": float(args.evaporation),
            "proxy_clf_model": str(proxy_settings.get("classification_model")),
            "proxy_reg_model": str(proxy_settings.get("regression_model")),
            "proxy_split_seeds": list(proxy_settings.get("split_seeds", [])),
            "proxy_profile": str(args.proxy_profile),
            "operator_param_search": bool(args.operator_param_search),
            "operator_param_grid": str(args.operator_param_grid),
        }
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(recommendation, f, indent=2, default=str)

        history_path = None
        if history:
            history_path = os.path.join(run_output_dir, "aco_history.csv")
            pd.DataFrame(history).to_csv(history_path, index=False)

        plot_path = _save_history_plot(history, run_output_dir)

        pipeline_cfg = recommendation.get("pipeline_config") or {}
        if "recommended_performance" in recommendation:
            proxy_score = recommendation.get("recommended_performance")
        else:
            proxy_score = recommendation.get("expected_performance")
        final_eval = recommendation.get("final_evaluation", {})
        final_score = recommendation.get("final_performance", final_eval.get("score"))

        print("\nFinal recommendation")
        print(f"  Dataset: {dataset_tag}")
        if args.dataset378_profile != "off":
            print(f"  Dataset378 profile: {args.dataset378_profile}")
        if args.operator_param_search:
            print(f"  Operator-param profile: {args.operator_param_grid}")
        print(f"  Pipeline: {_format_pipeline(pipeline_cfg)}")
        if proxy_score is not None:
            print(f"  Proxy score: {float(proxy_score):.4f}")
        if final_score is not None and final_eval:
            print(f"  Final eval ({final_eval.get('method', 'unknown')}): {float(final_score):.4f}")
        print(f"  Optimizer: {recommendation.get('optimizer', args.optimizer)}")
        ordering_info = recommendation.get("ordering_search")
        if isinstance(ordering_info, dict) and ordering_info.get("enabled"):
            print(
                "  Ordering search: "
                f"strategy={ordering_info.get('strategy')} "
                f"orders={ordering_info.get('num_orders_evaluated')}"
            )
        print(f"  Saved recommendation: {rec_path}")
        if history_path:
            print(f"  Saved ACO history: {history_path}")
        if plot_path:
            print(f"  Saved ACO plot: {plot_path}")
        elif history:
            print("  ACO plot skipped (matplotlib not available)")

        elapsed = time.perf_counter() - run_start
        print(f"  Elapsed seconds: {elapsed:.2f}")

        logger.info("Saved recommendation to %s", rec_path)
        run_summaries.append(
            {
                "dataset_id": dataset_id,
                "status": "ok",
                "optimizer": recommendation.get("optimizer", args.optimizer),
                "proxy_score": proxy_score,
                "final_score": final_score,
                "final_method": final_eval.get("method") if isinstance(final_eval, dict) else None,
                "elapsed_seconds": elapsed,
                "recommendation_path": rec_path,
                "history_path": history_path,
                "plot_path": plot_path,
                "dataset_profile": args.dataset378_profile,
            }
        )

    if n_runs > 1:
        ok_runs = [r for r in run_summaries if r.get("status") == "ok"]
        times = [r["elapsed_seconds"] for r in ok_runs if isinstance(r.get("elapsed_seconds"), (int, float))]
        proxy_scores = [
            r["proxy_score"]
            for r in ok_runs
            if isinstance(r.get("proxy_score"), (int, float)) and np.isfinite(r.get("proxy_score"))
        ]
        final_scores = [
            r["final_score"]
            for r in ok_runs
            if isinstance(r.get("final_score"), (int, float)) and np.isfinite(r.get("final_score"))
        ]
        ag_scores = [
            r["final_score"]
            for r in ok_runs
            if r.get("final_method") == "autogluon"
            and isinstance(r.get("final_score"), (int, float))
            and np.isfinite(r.get("final_score"))
        ]

        avg_time = float(np.mean(times)) if times else None
        avg_proxy = float(np.mean(proxy_scores)) if proxy_scores else None
        avg_final = float(np.mean(final_scores)) if final_scores else None
        avg_ag = float(np.mean(ag_scores)) if ag_scores else None

        aggregate = {
            "num_requested": n_runs,
            "num_ok": len(ok_runs),
            "num_failed": n_runs - len(ok_runs),
            "avg_elapsed_seconds": avg_time,
            "avg_proxy_score": avg_proxy,
            "avg_final_score": avg_final,
            "avg_autogluon_score": avg_ag,
            "optimizer": args.optimizer,
        }
        summary_path = os.path.join(output_dir, "recommendations_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"aggregate": aggregate, "runs": run_summaries}, f, indent=2, default=str)
        print("\nAggregate summary")
        print(f"  Runs ok/failed: {aggregate['num_ok']}/{aggregate['num_failed']}")
        if avg_time is not None:
            print(f"  Avg elapsed seconds: {avg_time:.2f}")
        if avg_proxy is not None:
            print(f"  Avg proxy score: {avg_proxy:.4f}")
        if avg_final is not None:
            print(f"  Avg final score: {avg_final:.4f}")
        if avg_ag is not None:
            print(f"  Avg autogluon score: {avg_ag:.4f}")
        print(f"\nSaved multi-run summary: {summary_path}")


if __name__ == "__main__":
    main()
