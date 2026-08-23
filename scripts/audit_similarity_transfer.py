#!/usr/bin/env python3
"""Audit whether metafeature neighbors transfer behavioral pipeline information."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automl_aco.config import DEFAULT_PIPELINE_OPTIONS  # noqa: E402
from automl_aco.eval_ids import EVAL_ID_SET, normalize_id  # noqa: E402
from automl_aco.metalearning.metric import build_similarity_target_matrix  # noqa: E402
from automl_aco.metalearning.offline_eval import eta_operator_spearman, retrieval_metrics  # noqa: E402
from automl_aco.preprocessing.autodp import AUTODP_60_IDS  # noqa: E402
from automl_aco.search.heuristics import (  # noqa: E402
    aggregate_operator_heuristics,
    build_transfer_candidates,
    compute_similarity_weights,
    initialize_aco_with_transferred_eta,
    select_top_k_neighbors,
    select_top_l_pipelines_per_neighbor,
)
from automl_aco.utils.operator_spec import base_operator_name  # noqa: E402


def _load(args):
    perf = pd.read_csv(args.performance_matrix, index_col=0)
    perf.columns = [normalize_id(column) for column in perf.columns]
    perf = perf.T.groupby(level=0, sort=False).mean().T
    meta = pd.read_csv(args.metafeatures)
    id_column = meta.columns[0]
    meta[id_column] = meta[id_column].map(normalize_id)
    meta = meta.drop_duplicates(id_column).set_index(id_column)
    configs = json.loads(args.pipeline_configs.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    return perf, meta, configs, [normalize_id(value) for value in manifest["dataset_ids"]]


def _pipeline_probability(config, eta):
    probability = 1.0
    for step, operators in DEFAULT_PIPELINE_OPTIONS.items():
        values = np.asarray(eta[step], dtype=float)
        probs = values / values.sum()
        matches = [
            idx for idx, operator in enumerate(operators)
            if base_operator_name(operator) == base_operator_name(config.get(step))
        ]
        if not matches:
            return 0.0
        probability *= float(probs[matches].sum())
    return probability


def audit_query(query_id, perf, meta, configs, args):
    forbidden = set(EVAL_ID_SET) | {str(value) for value in AUTODP_60_IDS}
    reference_ids = sorted(
        (set(perf.columns) & set(meta.index)) - forbidden - {query_id},
        key=lambda value: int(value) if value.isdigit() else value,
    )
    joint = perf[[query_id, *reference_ids]].apply(pd.to_numeric, errors="coerce")
    joint = pd.DataFrame(
        SimpleImputer(strategy="mean").fit_transform(joint.T).T,
        index=joint.index,
        columns=joint.columns,
    )
    profiles = joint[[query_id, *reference_ids]].T.to_numpy(dtype=float)
    behavioral = build_similarity_target_matrix(profiles, similarity_target="rank_cosine")[0, 1:]
    behavioral_scores = dict(zip(reference_ids, behavioral.astype(float)))

    meta_reference = meta.loc[reference_ids].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    known = scaler.fit_transform(imputer.fit_transform(meta_reference))
    query = scaler.transform(imputer.transform(meta.loc[[query_id]].reindex(columns=meta_reference.columns)))
    predicted_values = cosine_similarity(known, query).ravel()
    predicted = dict(zip(reference_ids, predicted_values.astype(float)))
    predicted_ranking = sorted(reference_ids, key=lambda key: (-predicted[key], key))
    behavioral_ranking = sorted(reference_ids, key=lambda key: (-behavioral_scores[key], key))
    top_neighbors = select_top_k_neighbors(list(predicted.items()), args.neighbor_k, query_id)
    similarity_weights = compute_similarity_weights(
        top_neighbors, "similarity", args.similarity_temperature
    )
    top_pipelines = select_top_l_pipelines_per_neighbor(
        joint[reference_ids], top_neighbors, args.top_l
    )
    candidates = build_transfer_candidates(
        joint[reference_ids], top_neighbors, top_pipelines, similarity_weights
    )
    raw_eta = aggregate_operator_heuristics(candidates, configs, DEFAULT_PIPELINE_OPTIONS)
    eta = initialize_aco_with_transferred_eta(raw_eta, args.eta_floor)
    config_map = {str(config["name"]): config for config in configs if "name" in config}
    candidate_names = list(dict.fromkeys(str(row["pipeline"]) for row in candidates))
    query_scores = joint[query_id]
    descending_ranks = pd.Series(
        rankdata(-query_scores.to_numpy(dtype=float), method="min"), index=query_scores.index
    )
    usable_names = [name for name in candidate_names if name in query_scores.index]
    oracle = float(query_scores.max())
    transferred_best = float(query_scores.loc[usable_names].max()) if usable_names else np.nan
    operator_support = {}
    for step in DEFAULT_PIPELINE_OPTIONS:
        operator_support[step] = len({
            base_operator_name(config_map[name].get(step))
            for name in usable_names if name in config_map
        })
    recombined_space = int(np.prod([max(1, value) for value in operator_support.values()]))
    transferred_mass = float(sum(
        _pipeline_probability(config_map[name], eta)
        for name in usable_names if name in config_map
    ))
    weights = np.asarray(list(similarity_weights.values()), dtype=float)
    metrics = retrieval_metrics(predicted, behavioral_scores, ks=(5, 10))
    row = {
        "query_dataset_id": query_id,
        **metrics,
        "behavioral_similarity_top5_mean": float(np.mean([
            behavioral_scores[dataset_id] for dataset_id, _ in top_neighbors
        ])),
        "top5_behavioral_overlap": len(set(predicted_ranking[:5]) & set(behavioral_ranking[:5])) / 5.0,
        "max_neighbor_weight": float(weights.max()),
        "effective_neighbor_count": float(1.0 / np.sum(weights ** 2)),
        "candidate_records": len(candidates),
        "unique_candidate_pipelines": len(usable_names),
        "candidate_best_regret": oracle - transferred_best,
        "candidate_median_query_rank": float(descending_ranks.loc[usable_names].median()),
        "candidate_top1_hit": int(str(query_scores.idxmax()) in usable_names),
        "eta_spearman": eta_operator_spearman(
            eta, query_scores, configs, DEFAULT_PIPELINE_OPTIONS
        ),
        "mean_raw_eta_range": float(np.mean([
            np.ptp(np.asarray(values, dtype=float)) for values in raw_eta.values()
        ])),
        "recombined_operator_space": recombined_space,
        "recombination_expansion": recombined_space / max(1, len(usable_names)),
        "transferred_pipeline_initial_mass": transferred_mass,
        "predicted_neighbors": json.dumps(predicted_ranking[:5]),
        "behavioral_neighbors": json.dumps(behavioral_ranking[:5]),
    }
    neighbor_rows = [
        {
            "query_dataset_id": query_id,
            "neighbor_rank": rank,
            "neighbor_dataset_id": dataset_id,
            "metafeature_similarity": predicted[dataset_id],
            "behavioral_similarity": behavioral_scores[dataset_id],
            "similarity_weight": similarity_weights[dataset_id],
            "behavioral_rank": behavioral_ranking.index(dataset_id) + 1,
        }
        for rank, (dataset_id, _score) in enumerate(top_neighbors, 1)
    ]
    return row, neighbor_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance-matrix", type=Path, default=ROOT / "data/openml/training_performance_matrix_autogluon.csv")
    parser.add_argument("--metafeatures", type=Path, default=ROOT / "data/openml/dataset_feats.csv")
    parser.add_argument("--pipeline-configs", type=Path, default=ROOT / "aco/pipeline_configs.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/openml/meta_dev18.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/similarity_transfer_audit")
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--top-l", type=int, default=3)
    parser.add_argument("--similarity-temperature", type=float, default=1.0)
    parser.add_argument("--eta-floor", type=float, default=0.05)
    args = parser.parse_args()
    perf, meta, configs, query_ids = _load(args)
    rows, neighbor_rows, failures = [], [], []
    for query_id in query_ids:
        try:
            row, details = audit_query(query_id, perf, meta, configs, args)
            rows.append(row)
            neighbor_rows.extend(details)
        except Exception as exc:
            failures.append({"query_dataset_id": query_id, "error": f"{type(exc).__name__}: {exc}"})
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "query_audit.csv", index=False)
    pd.DataFrame(neighbor_rows).to_csv(args.output_dir / "neighbor_audit.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number])
    summary = {column: float(numeric[column].mean()) for column in numeric.columns}
    summary["queries"] = len(frame)
    summary["failures"] = len(failures)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
