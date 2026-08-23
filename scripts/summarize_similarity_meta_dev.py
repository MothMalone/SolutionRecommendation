#!/usr/bin/env python3
"""Merge sharded similarity LOO results and select the two offline finalists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    paths = sorted(
        path
        for path in args.input_dir.rglob("similarity_shard_*.csv")
        if not path.stem.endswith(("_failures", "_summary"))
    )
    if not paths:
        raise FileNotFoundError(f"No similarity_shard_*.csv under {args.input_dir}")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    key = [column for column in ("variant", "query_dataset_id") if column in frame]
    frame = frame.drop_duplicates(key, keep="last")
    numeric = [
        "ndcg_at_5", "ndcg_at_10", "overlap_at_5",
        "overlap_at_10", "warm_start_regret", "eta_spearman",
    ]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    summary = frame.groupby("variant", as_index=False)[numeric].mean(numeric_only=True)
    summary["folds"] = summary.variant.map(frame.groupby("variant").size())
    summary = summary.sort_values(
        ["ndcg_at_10", "warm_start_regret", "variant"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    finalists = summary.variant.head(2).tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "similarity_results_merged.csv", index=False)
    summary.to_csv(output_dir / "similarity_leaderboard.csv", index=False)
    selection = {
        "selection_rule": "NDCG@10 descending, then warm-start regret ascending",
        "finalists": finalists,
        "folds_expected_per_variant": 18,
        "complete": bool(len(summary) and summary.folds.min() >= 18),
    }
    (output_dir / "similarity_finalists.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
