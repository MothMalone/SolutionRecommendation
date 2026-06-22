"""Train and save the siamese regression metric."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from automl_aco.metalearning.recommender import MetaPipelineRecommender
from automl_aco.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train siamese regression metric")
    parser.add_argument("--performance-matrix", required=True, help="Path to performance matrix CSV")
    parser.add_argument("--metafeatures", required=True, help="Path to metafeatures CSV")
    parser.add_argument("--output", required=False, help="Output path for metric model")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric-objective",
        choices=["embedding_cosine", "projector_product"],
        default="embedding_cosine",
        help=(
            "embedding_cosine trains the embedding space directly; projector_product "
            "keeps the older projector(emb_i * emb_j) objective."
        ),
    )
    parser.add_argument(
        "--similarity-target",
        choices=[
            "rank_cosine",
            "row_zscore_cosine",
            "row_minmax_cosine",
            "legacy_global_zscore_cosine",
        ],
        default="rank_cosine",
        help="Ground-truth similarity target used to train the metric.",
    )
    parser.add_argument(
        "--score-direction",
        choices=["higher_is_better", "lower_is_better"],
        default="higher_is_better",
        help="Whether performance-matrix values are scores or losses.",
    )
    return parser


def main() -> None:
    configure_logging()
    args = build_arg_parser().parse_args()

    perf = pd.read_csv(args.performance_matrix, index_col=0)
    meta = pd.read_csv(args.metafeatures, index_col=0)

    recommender = MetaPipelineRecommender(perf, meta, pipeline_configs=[])
    recommender.train_metric(
        method="regression",
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        similarity_target=args.similarity_target,
        score_direction=args.score_direction,
        metric_objective=args.metric_objective,
    )

    output_path = args.output or recommender.default_metric_path()
    recommender.save_metric(output_path)
    logger.info("Saved metric to %s", output_path)


if __name__ == "__main__":
    main()
