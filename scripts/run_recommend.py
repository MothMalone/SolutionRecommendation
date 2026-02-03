"""Run pipeline recommendation from CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from automl_aco.config import DEFAULT_PIPELINE_OPTIONS
from automl_aco.data.metafeatures import extract_enhanced_metafeatures
from automl_aco.metalearning.recommender import MetaPipelineRecommender
from automl_aco.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline recommendation")
    parser.add_argument("--performance-matrix", required=True, help="Path to performance matrix CSV")
    parser.add_argument("--metafeatures", required=True, help="Path to metafeatures CSV")
    parser.add_argument("--pipeline-configs", required=False, help="Path to pipeline configs JSON")
    parser.add_argument("--dataset-csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--dataset-id", required=True, help="Dataset id for metafeature lookup")
    parser.add_argument("--use-aco", action="store_true", help="Enable ACO search")
    parser.add_argument("--k", type=int, default=5, help="Top-k similar datasets")
    parser.add_argument("--eval-k", type=int, default=3, help="Number of top pipelines to evaluate")
    parser.add_argument("--n-ants", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--metric-path", required=False, help="Optional metric path to load")
    return parser


def main() -> None:
    configure_logging()
    args = build_arg_parser().parse_args()

    perf = pd.read_csv(args.performance_matrix, index_col=0)
    meta = pd.read_csv(args.metafeatures, index_col=0)

    if args.pipeline_configs:
        with open(args.pipeline_configs, "r", encoding="utf-8") as f:
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

    df = pd.read_csv(args.dataset_csv)
    if args.target_column not in df.columns:
        raise ValueError(f"Target column {args.target_column} not found in dataset")

    X = df.drop(columns=[args.target_column])
    y = df[args.target_column]
    dataset = {
        "id": int(args.dataset_id) if str(args.dataset_id).isdigit() else args.dataset_id,
        "name": f"D_{args.dataset_id}",
        "X": X,
        "y": y,
    }

    recommender = MetaPipelineRecommender(perf, meta, pipeline_configs)
    if args.metric_path:
        recommender.load_metric(args.metric_path)

    def _mf_func(ds):
        return extract_enhanced_metafeatures(ds, meta_features_df=meta)

    recommendation = recommender.recommend(
        new_dataset=dataset,
        target_column=args.target_column,
        options=DEFAULT_PIPELINE_OPTIONS,
        k=args.k,
        eval_k=args.eval_k,
        use_autogluon=True,
        use_aco=args.use_aco,
        aco_params={"n_ants": args.n_ants, "n_iterations": args.n_iterations},
        time_limit_per_model=args.time_limit,
        metafeatures_func=_mf_func,
    )

    logger.info("Recommendation: %s", json.dumps(recommendation, default=str))


if __name__ == "__main__":
    main()
