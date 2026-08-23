#!/usr/bin/env python3
"""Leave-one-dataset-out evaluation of ACORec similarity variants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automl_aco.config import DEFAULT_PIPELINE_OPTIONS  # noqa: E402
from automl_aco.eval_ids import EVAL_ID_SET, normalize_id  # noqa: E402
from automl_aco.metalearning.metric import build_similarity_target_matrix  # noqa: E402
from automl_aco.metalearning.offline_eval import (  # noqa: E402
    eta_operator_spearman,
    retrieval_metrics,
    warm_start_regret,
)
from automl_aco.metalearning.recommender import MetaPipelineRecommender  # noqa: E402
from automl_aco.preprocessing.autodp import AUTODP_60_IDS  # noqa: E402


VARIANTS = {
    "cosine": None,
    "rank_mse": {"similarity_target": "rank_cosine", "metric_loss": "mse"},
    "rank_pearson": {"similarity_target": "rank_cosine", "metric_loss": "pearson"},
    "zscore_pearson": {"similarity_target": "row_zscore_cosine", "metric_loss": "pearson"},
    "rank_listwise": {"similarity_target": "rank_cosine", "metric_loss": "listwise_kl"},
}


def _load_inputs(args):
    perf = pd.read_csv(args.performance_matrix, index_col=0)
    normalized_columns = [normalize_id(column) for column in perf.columns]
    perf.columns = normalized_columns
    perf = perf.T.groupby(level=0, sort=False).mean().T

    meta = pd.read_csv(args.metafeatures)
    first = meta.columns[0]
    meta[first] = meta[first].map(normalize_id)
    meta = meta.drop_duplicates(first).set_index(first)

    pipeline_configs = json.loads(Path(args.pipeline_configs).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    query_ids = [normalize_id(value) for value in manifest["dataset_ids"]]
    return perf, meta, pipeline_configs, manifest, query_ids


def _target_similarity(query_id: str, reference_ids: list[str], joint_performance: pd.DataFrame):
    profiles = joint_performance[[query_id, *reference_ids]].T.to_numpy(dtype=float)
    matrix = build_similarity_target_matrix(profiles, similarity_target="rank_cosine")
    return {dataset_id: float(matrix[0, idx + 1]) for idx, dataset_id in enumerate(reference_ids)}


def evaluate_query(
    *,
    query_id: str,
    perf: pd.DataFrame,
    meta: pd.DataFrame,
    pipeline_configs,
    variants: list[str],
    args,
):
    forbidden = set(EVAL_ID_SET) | {str(value) for value in AUTODP_60_IDS}
    reference_ids = sorted(
        (set(perf.columns) & set(meta.index)) - forbidden - {query_id},
        key=lambda value: int(value) if value.isdigit() else value,
    )
    if query_id not in perf.columns or query_id not in meta.index:
        raise ValueError(f"Meta-dev query {query_id} is missing from matrix or metafeatures")

    joint = perf[[query_id, *reference_ids]].apply(pd.to_numeric, errors="coerce")
    joint_imputed = pd.DataFrame(
        SimpleImputer(strategy="mean").fit_transform(joint.T).T,
        index=joint.index,
        columns=joint.columns,
    )
    target_scores = _target_similarity(query_id, reference_ids, joint_imputed)
    query_scores = joint_imputed[query_id]
    reference_perf = joint_imputed[reference_ids]
    rows = []

    for variant in variants:
        settings = VARIANTS[variant]
        recommender = MetaPipelineRecommender(
            reference_perf,
            meta.loc[reference_ids],
            pipeline_configs,
            pipeline_options={key: list(value) for key, value in DEFAULT_PIPELINE_OPTIONS.items()},
        )
        if settings is not None:
            recommender.train_metric(
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                metric_objective="embedding_cosine",
                target_temperature=args.target_temperature,
                prediction_temperature=args.prediction_temperature,
                **settings,
            )
        query_frame = pd.DataFrame([meta.loc[query_id]]).reindex(
            columns=recommender.metafeatures_df.columns
        )
        query_scaled = recommender.scaler.transform(recommender.imputer.transform(query_frame)).ravel()
        similarities = dict(recommender._compute_dataset_similarities(query_scaled))
        ranking = sorted(similarities, key=lambda key: (-float(similarities[key]), str(key)))
        metrics = retrieval_metrics(similarities, target_scores, ks=(5, 10))
        eta = recommender._compute_aco_heuristic(
            query_scaled,
            {key: list(value) for key, value in DEFAULT_PIPELINE_OPTIONS.items()},
            top_k=args.neighbor_k,
            top_l=args.top_l,
            query_dataset_id=query_id,
        )
        rows.append(
            {
                "query_dataset_id": query_id,
                "variant": variant,
                **metrics,
                "warm_start_regret": warm_start_regret(
                    query_scores, reference_perf, ranking[: args.neighbor_k], top_l=args.top_l
                ),
                "eta_spearman": eta_operator_spearman(
                    eta, query_scores, pipeline_configs, DEFAULT_PIPELINE_OPTIONS
                ),
                "top_neighbors": json.dumps(ranking[: args.neighbor_k]),
                "query_excluded": query_id not in recommender.performance_matrix.columns,
                "seed": args.seed,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance-matrix", type=Path, default=ROOT / "data/openml/training_performance_matrix_autogluon.csv")
    parser.add_argument("--metafeatures", type=Path, default=ROOT / "data/openml/dataset_feats.csv")
    parser.add_argument("--pipeline-configs", type=Path, default=ROOT / "aco/pipeline_configs.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/openml/meta_dev18.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/similarity_meta_dev")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-temperature", type=float, default=0.1)
    parser.add_argument("--prediction-temperature", type=float, default=0.1)
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--top-l", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must satisfy 0 <= index < num-shards")

    perf, meta, configs, manifest, query_ids = _load_inputs(args)
    run_ids = query_ids[args.shard_index :: args.num_shards]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for query_id in run_ids:
        try:
            rows.extend(
                evaluate_query(
                    query_id=query_id,
                    perf=perf,
                    meta=meta,
                    pipeline_configs=configs,
                    variants=args.variants,
                    args=args,
                )
            )
        except Exception as exc:
            failures.append({"query_dataset_id": query_id, "error_type": type(exc).__name__, "error": str(exc)})

    stem = f"similarity_shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    pd.DataFrame(rows).to_csv(args.output_dir / f"{stem}.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / f"{stem}_failures.csv", index=False)
    if rows:
        summary = pd.DataFrame(rows).groupby("variant", as_index=False).agg(
            ndcg_at_10=("ndcg_at_10", "mean"),
            warm_start_regret=("warm_start_regret", "mean"),
            eta_spearman=("eta_spearman", "mean"),
        )
        summary.to_csv(args.output_dir / f"{stem}_summary.csv", index=False)
        print(summary.to_string(index=False))
    metadata = {
        "manifest": manifest,
        "query_ids": run_ids,
        "variants": args.variants,
        "seed": args.seed,
        "failures": len(failures),
    }
    (args.output_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
