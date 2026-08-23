#!/usr/bin/env python3
"""Aggregate ACORec accuracy ablations and apply the frozen selection rule."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from automl_aco.metalearning.offline_eval import paired_accuracy_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/acorec_meta_dev_ablation")
    parser.add_argument("--baseline", default="sim=rank_mse__search=global_control")
    parser.add_argument("--diffprep-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--factorial-old-sim", default="rank_mse")
    parser.add_argument("--factorial-new-sim", default="rank_listwise")
    parser.add_argument("--factorial-old-search", default="global_control")
    parser.add_argument("--factorial-new-search", default="improvement_mmas_stagnation")
    args = parser.parse_args()
    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in args.root.glob("sim=*__search=*/accuracy_results.csv"):
        frame = pd.read_csv(path)
        if not frame.empty:
            frame["combination"] = path.parent.name
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No accuracy_results.csv files under {args.root}")
    runs = pd.concat(frames, ignore_index=True)
    runs = runs[runs.status == "ok"].copy()
    runs["accuracy"] = pd.to_numeric(runs.accuracy, errors="coerce")
    per_dataset = runs.groupby(["combination", "dataset_id"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        accuracy_seed_std=("accuracy", "std"),
        tpot_fit_seconds=("tpot_fit_seconds", "mean"),
        unique_proxy_evaluations=("unique_proxy_evaluations", "mean"),
    )
    leaderboard = per_dataset.groupby("combination", as_index=False).agg(
        datasets=("dataset_id", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        median_accuracy=("accuracy", "median"),
        mean_seed_std=("accuracy_seed_std", "mean"),
        mean_tpot_fit_seconds=("tpot_fit_seconds", "mean"),
        mean_unique_proxy_evaluations=("unique_proxy_evaluations", "mean"),
    )
    leaderboard = leaderboard.sort_values(
        ["mean_accuracy", "mean_tpot_fit_seconds"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    leaderboard["within_0_005_of_best"] = (
        leaderboard.mean_accuracy.max() - leaderboard.mean_accuracy <= 0.005
    )
    eligible = leaderboard[leaderboard.within_0_005_of_best].sort_values(
        ["mean_tpot_fit_seconds", "mean_unique_proxy_evaluations"], kind="mergesort"
    )
    selected = str(eligible.iloc[0].combination)
    confirmation_finalists = leaderboard.combination.head(2).astype(str).tolist()

    comparisons = {}
    if args.baseline in set(per_dataset.combination):
        baseline = per_dataset[per_dataset.combination == args.baseline][["dataset_id", "accuracy"]]
        for combination in leaderboard.combination:
            candidate = per_dataset[per_dataset.combination == combination][["dataset_id", "accuracy"]]
            paired = candidate.merge(baseline, on="dataset_id", suffixes=("_candidate", "_baseline"))
            if len(paired):
                comparisons[combination] = paired_accuracy_summary(
                    paired.accuracy_candidate, paired.accuracy_baseline
                )

    final_vs_diffprep = None
    if args.diffprep_csv:
        diffprep = pd.read_csv(args.diffprep_csv)
        accuracy_column = "accuracy" if "accuracy" in diffprep.columns else "DiffPrep"
        chosen = per_dataset[per_dataset.combination == selected][["dataset_id", "accuracy"]]
        paired = chosen.merge(
            diffprep[["dataset_id", accuracy_column]], on="dataset_id", suffixes=("_acorec", "_diffprep")
        )
        final_vs_diffprep = paired_accuracy_summary(
            paired.accuracy, pd.to_numeric(paired[accuracy_column], errors="coerce")
        )
        final_vs_diffprep["target_delta"] = 0.03
        final_vs_diffprep["target_met"] = final_vs_diffprep["mean_accuracy_delta"] >= 0.03

    factorial_names = {
        "old_old": f"sim={args.factorial_old_sim}__search={args.factorial_old_search}",
        "new_old": f"sim={args.factorial_new_sim}__search={args.factorial_old_search}",
        "old_new": f"sim={args.factorial_old_sim}__search={args.factorial_new_search}",
        "new_new": f"sim={args.factorial_new_sim}__search={args.factorial_new_search}",
    }
    mean_by_combo = dict(zip(leaderboard.combination, leaderboard.mean_accuracy))
    factorial = None
    if all(name in mean_by_combo for name in factorial_names.values()):
        values = {key: float(mean_by_combo[name]) for key, name in factorial_names.items()}
        factorial = {
            "combinations": factorial_names,
            "mean_accuracy": values,
            "similarity_main_effect": 0.5 * (
                (values["new_old"] - values["old_old"])
                + (values["new_new"] - values["old_new"])
            ),
            "search_main_effect": 0.5 * (
                (values["old_new"] - values["old_old"])
                + (values["new_new"] - values["new_old"])
            ),
            "interaction_effect": (
                values["new_new"] - values["new_old"]
                - values["old_new"] + values["old_old"]
            ),
        }

    runs.to_csv(output_dir / "all_runs.csv", index=False)
    per_dataset.to_csv(output_dir / "per_dataset_accuracy.csv", index=False)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    summary = {
        "selection_rule": "max mean accuracy; within 0.005 choose lower TPOT runtime then fewer proxy evaluations",
        "selected_combination": selected,
        "five_minute_confirmation_finalists": confirmation_finalists,
        "baseline": args.baseline,
        "paired_comparisons": comparisons,
        "factorial_2x2": factorial,
        "final_vs_diffprep": final_vs_diffprep,
    }
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(leaderboard.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
