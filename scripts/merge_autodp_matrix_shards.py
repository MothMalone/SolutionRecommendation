"""Merge and audit the 32 AutoDP performance-matrix shards."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from automl_aco.preprocessing.autodp import AUTODP_60_IDS  # noqa: E402


SHARD_RE = re.compile(r"\.part_(\d{4})_of_(\d{4})\.(?:txt|csv)$")
EXPECTED_SHARDS = 32
# Keep every non-holdout dataset that has at least one successful evaluation.
# Remaining missing cells are handled later by ACORec's performance imputer.
READY_MIN_AVAILABLE = 1


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp)
    temp.replace(path)


def load_pipeline_order(config_path: Path) -> list[str]:
    configs = json.loads(config_path.read_text(encoding="utf-8"))
    names = [str(config["name"]) for config in configs]
    if len(names) != 36 or len(set(names)) != 36:
        raise ValueError("Expected exactly 36 unique AutoDP reference pipelines")
    return names


def discover_shards(input_dir: Path) -> list[tuple[int, Path]]:
    matches: list[tuple[int, Path]] = []
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        match = SHARD_RE.search(path.name)
        if not match:
            continue
        shard_index, shard_total = map(int, match.groups())
        if shard_total != EXPECTED_SHARDS:
            raise ValueError(f"Unexpected shard total in {path.name}: {shard_total}")
        matches.append((shard_index, path))
    matches.sort()
    indices = [index for index, _ in matches]
    if indices != list(range(EXPECTED_SHARDS)):
        raise ValueError(f"Expected shard indices 0..31, got {indices}")
    return matches


def merge_shards(
    shards: list[tuple[int, Path]], pipeline_order: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    dataset_order: list[str] = []
    parts: list[tuple[int, Path, pd.DataFrame]] = []
    shard_rows: list[dict[str, object]] = []
    for shard_index, path in shards:
        part = pd.read_csv(path, index_col=0)
        if set(part.index.astype(str)) != set(pipeline_order):
            raise ValueError(f"Pipeline rows differ in {path.name}")
        if part.index.has_duplicates or part.columns.has_duplicates:
            raise ValueError(f"Duplicate row/column labels in {path.name}")
        part.index = part.index.astype(str)
        part.columns = part.columns.astype(str)
        part = part.apply(pd.to_numeric, errors="coerce")
        for column in part.columns:
            if column not in dataset_order:
                dataset_order.append(column)
        shard_rows.append(
            {
                "shard_index": shard_index,
                "file": path.name,
                "dataset_columns": int(part.shape[1]),
                "non_missing_cells": int(part.notna().sum().sum()),
                "blank_cells_in_file": int(part.isna().sum().sum()),
                "first_dataset": str(part.columns[0]),
                "last_dataset": str(part.columns[-1]),
            }
        )
        parts.append((shard_index, path, part))

    dataset_order.sort(key=lambda value: int(value.removeprefix("D_")))
    merged = pd.DataFrame(np.nan, index=pipeline_order, columns=dataset_order, dtype=float)
    source = pd.DataFrame("", index=pipeline_order, columns=dataset_order, dtype=object)
    conflicts: list[dict[str, object]] = []
    duplicate_agreements = 0
    for shard_index, path, part in parts:
        for pipeline in part.index:
            for dataset, value in part.loc[pipeline].items():
                if pd.isna(value):
                    continue
                existing = merged.at[pipeline, dataset]
                if pd.notna(existing):
                    if not np.isclose(float(existing), float(value), rtol=0, atol=1e-12):
                        conflicts.append(
                            {
                                "pipeline": pipeline,
                                "dataset": dataset,
                                "existing": float(existing),
                                "incoming": float(value),
                                "incoming_shard": shard_index,
                                "incoming_file": path.name,
                            }
                        )
                    else:
                        duplicate_agreements += 1
                    continue
                merged.at[pipeline, dataset] = float(value)
                source.at[pipeline, dataset] = path.name
    if conflicts:
        raise ValueError(f"Conflicting shard values: {conflicts[:5]}")
    return merged, source, shard_rows + [{"duplicate_agreements": duplicate_agreements}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "autodp_matrix")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "autodp_matrix" / "merged")
    parser.add_argument(
        "--pipeline-configs",
        type=Path,
        default=ROOT / "aco" / "pipeline_configs_autodp36.json",
    )
    parser.add_argument(
        "--metafeatures",
        type=Path,
        default=ROOT / "data" / "openml" / "dataset_feats.csv",
    )
    args = parser.parse_args()

    pipeline_order = load_pipeline_order(args.pipeline_configs)
    shards = discover_shards(args.input_dir)
    full, _source, shard_audit_raw = merge_shards(shards, pipeline_order)
    shard_audit = pd.DataFrame(shard_audit_raw[:-1])
    duplicate_agreements = int(shard_audit_raw[-1]["duplicate_agreements"])

    finite_values = full.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Merged matrix contains no finite scores")
    outside_unit = int(((finite_values < 0) | (finite_values > 1)).sum())
    if outside_unit:
        raise ValueError(f"Found {outside_unit} scores outside [0, 1]")

    holdout_columns = [
        f"D_{dataset_id}"
        for dataset_id in AUTODP_60_IDS
        if f"D_{dataset_id}" in full.columns
    ]
    reference_raw = full.drop(columns=holdout_columns)
    reference_available = reference_raw.notna().sum(axis=0)
    ready_columns = reference_available[reference_available >= READY_MIN_AVAILABLE].index
    ready = reference_raw.loc[:, ready_columns].copy()

    meta = pd.read_csv(args.metafeatures, index_col=0)
    meta_ids = {str(value).removeprefix("D_") for value in meta.index}
    ready_ids = {str(value).removeprefix("D_") for value in ready.columns}
    missing_metafeatures = sorted(ready_ids - meta_ids, key=int)
    if missing_metafeatures:
        raise ValueError(
            f"ACORec-ready datasets missing metafeatures: {missing_metafeatures}"
        )

    dataset_quality = pd.DataFrame(
        {
            "dataset": full.columns,
            "dataset_id": [int(value.removeprefix("D_")) for value in full.columns],
            "available_scores": full.notna().sum(axis=0).to_numpy(),
            "missing_scores": full.isna().sum(axis=0).to_numpy(),
            "coverage": full.notna().mean(axis=0).to_numpy(),
            "unique_observed_scores": full.nunique(axis=0, dropna=True).to_numpy(),
            "observed_score_std": full.std(axis=0, skipna=True, ddof=0).to_numpy(),
        }
    )
    holdout_set = set(holdout_columns)
    ready_set = set(ready.columns)
    dataset_quality["is_autodp60_holdout"] = dataset_quality["dataset"].isin(holdout_set)
    dataset_quality["included_in_ready_matrix"] = dataset_quality["dataset"].isin(ready_set)
    dataset_quality["exclusion_reason"] = ""
    dataset_quality.loc[
        dataset_quality["is_autodp60_holdout"], "exclusion_reason"
    ] = "AutoDP60 holdout (leakage guard)"
    dataset_quality.loc[
        (~dataset_quality["is_autodp60_holdout"])
        & (dataset_quality["available_scores"] == 0),
        "exclusion_reason",
    ] = "no successful evaluation"
    dataset_quality.loc[
        (~dataset_quality["is_autodp60_holdout"])
        & (dataset_quality["available_scores"] > 0)
        & (dataset_quality["available_scores"] < READY_MIN_AVAILABLE),
        "exclusion_reason",
    ] = f"fewer than {READY_MIN_AVAILABLE}/36 successful evaluations"

    pipeline_quality = pd.DataFrame(
        {
            "pipeline": full.index,
            "available_scores_full": full.notna().sum(axis=1).to_numpy(),
            "missing_scores_full": full.isna().sum(axis=1).to_numpy(),
            "coverage_full": full.notna().mean(axis=1).to_numpy(),
            "available_scores_ready": ready.notna().sum(axis=1).to_numpy(),
            "missing_scores_ready": ready.isna().sum(axis=1).to_numpy(),
            "coverage_ready": ready.notna().mean(axis=1).to_numpy(),
            "mean_score_ready": ready.mean(axis=1, skipna=True).to_numpy(),
            "score_std_ready": ready.std(axis=1, skipna=True, ddof=0).to_numpy(),
        }
    )

    baseline_name = "autodp_00_baseline"
    baseline = ready.loc[baseline_name]
    pipeline_delta_rows: list[dict[str, object]] = []
    for pipeline, scores in ready.iterrows():
        common = scores.notna() & baseline.notna()
        delta = scores.loc[common] - baseline.loc[common]
        pipeline_delta_rows.append(
            {
                "pipeline": pipeline,
                "paired_with_baseline": int(common.sum()),
                "mean_delta_vs_baseline": float(delta.mean()),
                "median_delta_vs_baseline": float(delta.median()),
                "wins_vs_baseline": int((delta > 1e-12).sum()),
                "ties_vs_baseline": int((delta.abs() <= 1e-12).sum()),
                "losses_vs_baseline": int((delta < -1e-12).sum()),
            }
        )
    pipeline_quality = pipeline_quality.merge(
        pd.DataFrame(pipeline_delta_rows), on="pipeline", how="left", validate="one_to_one"
    )

    dataset_performance_rows: list[dict[str, object]] = []
    for dataset in ready.columns:
        scores = ready[dataset].dropna()
        baseline_score = float(baseline[dataset])
        best_score = float(scores.max())
        winners = sorted(
            scores.index[np.isclose(scores.to_numpy(), best_score, rtol=0, atol=1e-12)]
        )
        dataset_performance_rows.append(
            {
                "dataset": dataset,
                "dataset_id": int(dataset.removeprefix("D_")),
                "observed_scores": int(len(scores)),
                "baseline_score": baseline_score,
                "best_score": best_score,
                "best_lift_vs_baseline": best_score - baseline_score,
                "best_pipelines": "|".join(winners),
                "baseline_is_best_or_tied": baseline_name in winners,
                "pipelines_beating_baseline": int((scores > baseline_score + 1e-12).sum()),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std(ddof=0)),
                "score_range": float(scores.max() - scores.min()),
            }
        )
    dataset_performance = pd.DataFrame(dataset_performance_rows)

    missing_positions = np.argwhere(ready.isna().to_numpy())
    missing_ready = pd.DataFrame(
        {
            "pipeline": [ready.index[row] for row, _ in missing_positions],
            "dataset": [ready.columns[column] for _, column in missing_positions],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "historical_performance_matrix_autodp36_full.csv"
    reference_path = args.output_dir / "training_performance_matrix_autodp36_reference_raw.csv"
    ready_path = args.output_dir / "training_performance_matrix_autodp36_ready.csv"
    atomic_csv(full, full_path)
    atomic_csv(reference_raw, reference_path)
    atomic_csv(ready, ready_path)
    dataset_quality.to_csv(args.output_dir / "autodp_matrix_quality_by_dataset.csv", index=False)
    pipeline_quality.to_csv(args.output_dir / "autodp_matrix_quality_by_pipeline.csv", index=False)
    dataset_performance.to_csv(
        args.output_dir / "autodp_matrix_performance_by_dataset.csv", index=False
    )
    shard_audit.to_csv(args.output_dir / "autodp_matrix_quality_by_shard.csv", index=False)
    missing_ready.to_csv(args.output_dir / "autodp_matrix_missing_cells_ready.csv", index=False)

    full_missing = int(full.isna().sum().sum())
    ready_missing = int(ready.isna().sum().sum())
    best_lifts = dataset_performance["best_lift_vs_baseline"]
    top_average_pipeline_row = pipeline_quality.sort_values(
        "mean_score_ready", ascending=False
    ).iloc[0]
    summary = {
        "source_shards": len(shards),
        "pipeline_count": int(full.shape[0]),
        "historical_dataset_count": int(full.shape[1]),
        "expected_jobs": int(full.size),
        "successful_jobs": int(full.notna().sum().sum()),
        "missing_jobs": full_missing,
        "overall_coverage": float(full.notna().mean().mean()),
        "conflicting_values": 0,
        "duplicate_equal_values": duplicate_agreements,
        "score_min": float(finite_values.min()),
        "score_max": float(finite_values.max()),
        "datasets_with_36_scores": int((full.notna().sum(axis=0) == 36).sum()),
        "datasets_with_zero_scores": int((full.notna().sum(axis=0) == 0).sum()),
        "datasets_with_1_to_30_scores": int(
            full.notna().sum(axis=0).between(1, 30).sum()
        ),
        "datasets_with_31_to_35_scores": int(
            full.notna().sum(axis=0).between(31, 35).sum()
        ),
        "present_autodp60_holdout_columns_removed": len(holdout_columns),
        "holdout_columns_removed": holdout_columns,
        "reference_raw_shape": list(reference_raw.shape),
        "ready_min_available_scores": READY_MIN_AVAILABLE,
        "ready_shape": list(ready.shape),
        "ready_missing_jobs": ready_missing,
        "ready_coverage": float(ready.notna().mean().mean()),
        "ready_datasets_with_metafeatures": len(ready_ids & meta_ids),
        "ready_datasets_missing_metafeatures": missing_metafeatures,
        "ready_observed_score_mean": float(np.nanmean(ready.to_numpy(dtype=float))),
        "ready_observed_score_median": float(np.nanmedian(ready.to_numpy(dtype=float))),
        "baseline_mean_score": float(baseline.mean()),
        "baseline_median_score": float(baseline.median()),
        "top_average_pipeline": str(top_average_pipeline_row["pipeline"]),
        "top_average_pipeline_mean_score": float(
            top_average_pipeline_row["mean_score_ready"]
        ),
        "oracle_best_mean_score": float(dataset_performance["best_score"].mean()),
        "oracle_best_mean_lift_vs_baseline": float(best_lifts.mean()),
        "oracle_best_median_lift_vs_baseline": float(best_lifts.median()),
        "datasets_where_any_pipeline_beats_baseline": int((best_lifts > 1e-12).sum()),
        "fraction_where_any_pipeline_beats_baseline": float(
            (best_lifts > 1e-12).mean()
        ),
        "datasets_where_baseline_is_best_or_tied": int(
            dataset_performance["baseline_is_best_or_tied"].sum()
        ),
        "median_dataset_score_std": float(dataset_performance["score_std"].median()),
        "median_dataset_score_range": float(dataset_performance["score_range"].median()),
        "ready_for_acorec": bool(
            ready.shape[0] == 36
            and ready.shape[1] > 0
            and not missing_metafeatures
            and int(ready.notna().sum(axis=1).min()) > 0
        ),
    }
    (args.output_dir / "autodp_matrix_merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = f"""# AutoDP performance-matrix audit

## Result

- Source shards: {len(shards)}/{EXPECTED_SHARDS}
- Full matrix: {full.shape[0]} pipelines x {full.shape[1]} datasets = {full.size:,} jobs
- Successful jobs: {int(full.notna().sum().sum()):,} ({full.notna().mean().mean():.2%})
- Missing jobs: {full_missing:,} ({full.isna().mean().mean():.2%})
- Conflicting duplicate values: 0
- Complete datasets: {int((full.notna().sum(axis=0) == 36).sum())}
- Completely empty datasets: {int((full.notna().sum(axis=0) == 0).sum())}

## Leakage and quality filtering

- Removed {len(holdout_columns)} AutoDP60 holdout columns present in the historical corpus.
- Kept every non-holdout dataset with at least one successful evaluation; only completely empty datasets were removed.
- ACORec-ready matrix: {ready.shape[0]} pipelines x {ready.shape[1]} datasets.
- Remaining missing cells: {ready_missing} ({ready.isna().mean().mean():.2%}); ACORec's row-mean performance imputer can handle them.
- Metafeature overlap: {len(ready_ids & meta_ids)}/{len(ready_ids)} ready datasets.

## Performance signal

- Baseline mean accuracy: {baseline.mean():.4f}.
- Best average pipeline: {top_average_pipeline_row['pipeline']} ({top_average_pipeline_row['mean_score_ready']:.4f}).
- Per-dataset oracle mean accuracy: {dataset_performance['best_score'].mean():.4f}.
- Mean oracle lift over baseline: {best_lifts.mean():.4f}.
- At least one pipeline beats baseline on {int((best_lifts > 1e-12).sum())}/{len(best_lifts)} datasets ({(best_lifts > 1e-12).mean():.1%}).
- Median within-dataset score range: {dataset_performance['score_range'].median():.4f}.

## Conclusion

The filtered matrix is ready for ACORec. Use `training_performance_matrix_autodp36_ready.csv`
with `aco/pipeline_configs_autodp36.json` and `data/openml/dataset_feats.csv`.
"""
    (args.output_dir / "autodp_matrix_audit_report.md").write_text(
        report, encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nACORec-ready matrix: {ready_path}")


if __name__ == "__main__":
    main()
