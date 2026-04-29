#!/usr/bin/env python3
"""Find OpenML datasets for imputation/scaling/outlier-focused experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automl_aco.data.openml_analysis import (
    evaluate_dataset_issues,
    get_cc18_dataset_ids,
    load_all_datasets_metadata,
    select_imputation_candidates,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze OpenML preprocessing candidates")
    parser.add_argument("--max-instances", type=int, default=10_000, help="Max instances for imputation candidate filter")
    parser.add_argument("--top-n", type=int, default=10, help="Rows to print for each ranking")
    parser.add_argument("--scale-threshold", type=float, default=1000.0, help="Scale ratio threshold")
    parser.add_argument("--outlier-threshold", type=float, default=0.05, help="Outlier ratio threshold")
    parser.add_argument(
        "--max-cc18-datasets",
        type=int,
        default=None,
        help="Optional limit for quick smoke runs (e.g., 20)",
    )
    parser.add_argument("--output-dir", default=None, help="Optional output dir for CSV exports")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    print("=" * 50)
    print("PART 1: FIND DATASETS THAT NEED IMPUTATION")
    print("=" * 50)
    all_datasets_meta = load_all_datasets_metadata()
    imputation_candidates = select_imputation_candidates(
        all_datasets_meta=all_datasets_meta,
        max_instances=int(args.max_instances),
    )
    print("\nTop datasets for Imputation:")
    print(
        imputation_candidates[
            ["did", "name", "NumberOfMissingValues", "NumberOfInstances"]
        ].head(int(args.top_n))
    )

    print("\n" + "=" * 50)
    print("PART 2: FIND DATASETS FOR SCALING & OUTLIER REMOVAL (CC18 SUITE)")
    print("=" * 50)
    print("Fetching OpenML-CC18 suite...")
    cc18_dataset_ids = get_cc18_dataset_ids()
    if args.max_cc18_datasets is not None:
        cc18_dataset_ids = cc18_dataset_ids[: int(args.max_cc18_datasets)]
    print(f"Loading and analyzing {len(cc18_dataset_ids)} datasets (first run may take longer)...")

    df_metrics_cc18 = evaluate_dataset_issues(
        dataset_ids=cc18_dataset_ids,
        verbose=bool(args.verbose),
    )
    needs_scaling = df_metrics_cc18[df_metrics_cc18["Scale_Ratio"] > float(args.scale_threshold)].sort_values(
        by="Scale_Ratio",
        ascending=False,
    )
    needs_outlier_removal = df_metrics_cc18[
        df_metrics_cc18["Outlier_Ratio"] > float(args.outlier_threshold)
    ].sort_values(by="Outlier_Ratio", ascending=False)

    print("\nTop datasets for Scaling (Scale Ratio > threshold):")
    print(needs_scaling[["ID", "Name", "Scale_Ratio"]].head(int(args.top_n)))

    print("\nTop datasets for Outlier Removal (Outlier Ratio > threshold):")
    print(needs_outlier_removal[["ID", "Name", "Outlier_Ratio"]].head(int(args.top_n)))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        imputation_candidates.to_csv(out_dir / "openml_imputation_candidates.csv", index=False)
        df_metrics_cc18.to_csv(out_dir / "openml_cc18_issue_metrics.csv", index=False)
        needs_scaling.to_csv(out_dir / "openml_cc18_scaling_candidates.csv", index=False)
        needs_outlier_removal.to_csv(out_dir / "openml_cc18_outlier_candidates.csv", index=False)
        print(f"\nSaved CSV outputs to: {out_dir}")


if __name__ == "__main__":
    main()
