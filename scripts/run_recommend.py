"""Run pipeline recommendation from CLI."""
from __future__ import annotations

import argparse
import json
import os
import traceback
import re
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
    AUTODP_PIPELINE_OPTIONS,
    AUTODP_PREPROCESSOR_ORDER,
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
    NOTEBOOK_LEGACY_PIPELINE_OPTIONS,
)
from automl_aco.data.loaders import (
    load_openml_dataset,
    load_gitlab_openml_dataset,
    load_kaggle_dataset,
    load_dummy_dataset,
    load_csv_dataset,
)
from automl_aco.data.splits import split_train_val_test
from automl_aco.data.metafeatures import (
    compute_metafeatures_from_data,
    extract_enhanced_metafeatures,
)
from automl_aco.eval_ids import EVAL_IDS, holdout_ids, holdout_reference, normalize_id
from automl_aco.metalearning.recommender import MetaPipelineRecommender
from automl_aco.preprocessing.autodp import (
    AUTODP_60_IDS,
    AUTODP_REGRESSION_IDS,
    AUTODP_OPTIONS as AUTODP36_OPTIONS,
    DEFAULT_AUTODP_ORDER as AUTODP36_ORDER,
    exclude_holdout_columns as exclude_autodp60_holdout_columns,
)
from automl_aco.utils.operator_spec import base_operator_name
from automl_aco.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# ACO's option insertion order is also its default execution order.  Reorder
# the paper-space mapping to exactly match the fixed order used to build the
# AutoDP36 performance matrix.
AUTODP36_PIPELINE_OPTIONS = {
    step: list(AUTODP36_OPTIONS[step]) for step in AUTODP36_ORDER
}


def _cli_flag_was_passed(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline recommendation")
    parser.add_argument("--performance-matrix", required=False, help="Path to performance matrix CSV")
    parser.add_argument("--metafeatures", required=False, help="Path to metafeatures CSV")
    parser.add_argument(
        "--metafeatures-id-column",
        required=False,
        default=None,
        help=(
            "Optional column name in metafeatures CSV to use as dataset id "
            "(e.g., dataset_id, did). If omitted, auto-detected."
        ),
    )
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
        choices=["openml", "kaggle", "csv", "dummy", "local"],
        default=None,
        help=(
            "Dataset source. 'local' = read from --openml-local-folder (default test_data_local) "
            "with OpenML fetch as fallback. If omitted, inferred."
        ),
    )
    parser.add_argument("--dataset-csv", required=False, help="Path to dataset CSV (csv source only)")
    parser.add_argument("--target-column", default="target", help="Target column name")
    parser.add_argument("--dataset-id", required=False, help="Dataset id for metafeature lookup")
    parser.add_argument(
        "--recommend-on-train-val",
        action="store_true",
        help=(
            "Restrict recommendation/search to the externally fixed train+validation 80%%. "
            "The outer test rows are never passed to ACO/metafeatures/evaluators."
        ),
    )
    parser.add_argument(
        "--recommend-split-seed",
        type=int,
        default=42,
        help="Seed for the external 60/20/20 split when --recommend-on-train-val is enabled.",
    )
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
            "Optional local folder for OpenML CSV snapshots (e.g., 1520.csv) and the "
            "GitLab/Parquet download cache when --openml-backend gitlab/auto is selected."
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
        choices=["weighted_topk_topl", "legacy_weighted_average", "paper_flat_average"],
        default="weighted_topk_topl",
        help="Phase-2 heuristic transfer algorithm (paper_flat_average = literal Eq 6+7)",
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
        "--unobserved-operator-score",
        type=float,
        default=None,
        help=(
            "Opt-in eta score for operators absent from transferred pipelines. "
            "When set (e.g. 0.7), preserve this score after normalization to improve exploration."
        ),
    )
    parser.add_argument(
        "--score-direction",
        choices=["higher_is_better", "lower_is_better"],
        default="higher_is_better",
        help="Direction of performance-matrix values used in transfer.",
    )
    parser.add_argument("--eval-k", type=int, default=3, help="Number of top pipelines to evaluate")
    parser.add_argument(
        "--no-search",
        action="store_true",
        help=(
            "Disable optimizer search and only evaluate the recommender's retrieved/ranked candidate pipelines. "
            "Use this for retrieval-only vs search comparisons inside the same run_recommend protocol."
        ),
    )
    parser.add_argument(
        "--notebook-legacy-mode",
        action="store_true",
        help=(
            "Compatibility mode for the old notebook: legacy eta averaging, equality dataset weighting, "
            "old Markov ACO sampling, Markov smoothing 0.7, legacy metric target, and old encoding option space. "
            "The current self-query guard remains enabled."
        ),
    )
    parser.add_argument(
        "--legacy-notebook-aco",
        action="store_true",
        help=(
            "Use the old notebook ACO sampler exactly: raw Markov/marginal probability mixing "
            "and NumPy global RNG behavior."
        ),
    )
    parser.add_argument(
        "--notebook-legacy-options",
        action="store_true",
        help="Use the old notebook search space, including label/frequency encoding.",
    )
    parser.add_argument("--n-ants", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument(
        "--aco-total-ant-budget",
        type=int,
        default=None,
        help=(
            "Optional total number of ant draws across all iterations. The final iteration "
            "may be partial; for example n_ants=10 and budget=15 runs 10 then 5 ants. "
            "Unset preserves the normal n_ants * n_iterations behavior."
        ),
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="ACO alpha: pheromone importance")
    parser.add_argument("--beta", type=float, default=2.0, help="ACO beta: heuristic importance")
    parser.add_argument("--evaporation", type=float, default=0.2, help="ACO pheromone evaporation rate")
    parser.add_argument(
        "--top-k-pheromone",
        type=int,
        default=3,
        help="Top-k sampled pipelines per iteration used for pheromone reinforcement (ACO only)",
    )
    parser.add_argument(
        "--aco-weight-method",
        choices=["rank", "linear", "exponential", "reciprocal", "power_rank", "uniform"],
        default="rank",
        help="Weighting scheme for pheromone reinforcement among selected pipelines (ACO only)",
    )
    parser.add_argument(
        "--aco-mmas-bounds",
        action="store_true",
        help=(
            "Enable MMAS pheromone bounds [tau_min, tau_max] (anti-collapse). Auto tau_max=1/rho, "
            "tau_min=tau_min_ratio*tau_max unless --aco-tau-min/--aco-tau-max are given. "
            "Enabled automatically by --paper-faithful."
        ),
    )
    parser.add_argument("--aco-tau-min", type=float, default=None, help="Explicit MMAS tau_min (overrides auto)")
    parser.add_argument("--aco-tau-max", type=float, default=None, help="Explicit MMAS tau_max (overrides auto)")
    parser.add_argument(
        "--aco-tau-min-ratio",
        type=float,
        default=0.05,
        help="tau_min = ratio * tau_max when tau_min is auto (MMAS bounds)",
    )
    parser.add_argument(
        "--aco-markov-order",
        type=int,
        default=2,
        help="k-order context for conditional pheromone sampling (ACO only)",
    )
    parser.add_argument(
        "--aco-lambda-smooth",
        type=float,
        default=0.0,
        help=(
            "Interpolation weight between conditional/global pheromone sampling (ACO only). "
            "Default 0.0 disables Markov conditional influence."
        ),
    )
    parser.add_argument(
        "--interaction-prior-strength",
        type=float,
        default=0.0,
        help=(
            "Strength of fixed retrieval-conditioned pairwise operator priors. "
            "0 disables interaction-aware transfer and reproduces flat eta behavior."
        ),
    )
    parser.add_argument(
        "--interaction-prior-floor",
        type=float,
        default=0.2,
        help="Minimum multiplier for weak/unsupported historical operator pairs when interaction priors are enabled.",
    )
    parser.add_argument(
        "--protect-retrieval-incumbent",
        action="store_true",
        help=(
            "During search, always include the retrieval/no-search incumbent candidates in final evaluation. "
            "This makes search a safe refinement step instead of replacing the retrieval baseline."
        ),
    )
    parser.add_argument(
        "--operator-space",
        choices=["ours", "theirs", "autodp36"],
        default="ours",
        help=(
            "Which operator space ACORec searches. 'ours' = the 19-operator ACORec space. "
            "'theirs' = AutoDP's 22 operators reimplemented under our fit/transform discipline "
            "(see src/automl_aco/preprocessing/autodp_ops.py for the full deviation table). "
            "'theirs' adds a duplicate_removal step and has no dimensionality_reduction, because "
            "that is the shape of their space. NOTE: the reference performance matrix is written "
            "in OUR operator codes, so the no_search_retrieval transfer arm is disabled under "
            "'theirs' unless --transfer-arm-anyway is given. 'autodp36' = the standalone "
            "paper-style AutoDP space and matching 36-pipeline performance matrix generated "
            "by notebooks/build-performance-matrix-autodp.ipynb; its matrix/config paths are "
            "selected automatically unless explicitly overridden."
        ),
    )
    parser.add_argument(
        "--aco-search-fixes",
        action="store_true",
        help=(
            "Enable the isolated A/B search-correctness bundle: canonical cache keys, "
            "same-iteration deduplication, negative caching of invalid configs, and "
            "tie-aware rank reinforcement. It does not enable Markov pheromones."
        ),
    )
    parser.add_argument("--aco-canonical-cache", action="store_true")
    parser.add_argument("--aco-deduplicate-iteration", action="store_true")
    parser.add_argument("--aco-cache-invalid", action="store_true")
    parser.add_argument("--aco-tie-aware-rank", action="store_true")
    parser.add_argument(
        "--aco-refill-unique-ants",
        action="store_true",
        help="Keep drawing until n_ants unseen configurations are found or the attempt cap is reached.",
    )
    parser.add_argument("--aco-max-sampling-attempt-multiplier", type=int, default=100)
    parser.add_argument(
        "--aco-update-policy",
        choices=["global_elite", "iteration_elite", "improvement_only", "hybrid_elite"],
        default="global_elite",
    )
    parser.add_argument(
        "--aco-exploration-policy",
        choices=["none", "fixed", "stagnation"],
        default="none",
    )
    parser.add_argument("--aco-exploration-epsilon", type=float, default=0.1)
    parser.add_argument("--aco-exploration-initial-epsilon", type=float, default=0.05)
    parser.add_argument("--aco-exploration-step", type=float, default=0.05)
    parser.add_argument("--aco-exploration-max-epsilon", type=float, default=0.30)
    parser.add_argument(
        "--openml-backend",
        choices=["auto", "openml", "gitlab"],
        default="auto",
        help=(
            "Dataset backend for openml/local sources. auto tries local/OpenML first and "
            "GitLab/DataGit as fallback; openml disables GitLab fallback; gitlab reads the "
            "GitLab Parquet mirror directly and caches it under --openml-local-folder."
        ),
    )
    parser.add_argument(
        "--prepare-mode",
        choices=["leakfree", "native"],
        default="leakfree",
        help=(
            "How the chosen pipeline is FITTED for the final score. 'leakfree' = fit on train, "
            "transform test (ACORec's normal discipline). 'native' = fit on the FULL dataset, "
            "train and test together, mirroring AutoDP's published protocol, whose operators fit "
            "on concat(train, test). Use 'native' ONLY for the like-for-like AutoDP comparison: "
            "under it neither method's score measures generalisation, but both get the same "
            "information, which is what makes that comparison fair."
        ),
    )
    parser.add_argument(
        "--transfer-arm-anyway",
        action="store_true",
        help=(
            "Keep the no_search_retrieval transfer arm under --operator-space theirs. Off by "
            "default because the retrieved pipelines are coerced from our operator vocabulary "
            "into theirs, which silently produces degenerate seeds rather than an error."
        ),
    )
    parser.add_argument(
        "--baseline-only",
        choices=["off", "no_preprocessing", "autogluon_native", "no_search_retrieval", "light_preprocessing"],
        default="off",
        help=(
            "Skip ACORec entirely and just AutoGluon-evaluate a baseline on the same 0.6/0.2/0.2 "
            "split/seed, for apples-to-apples comparison. 'no_preprocessing' = all-none pipeline "
            "(onehot encoding only); 'autogluon_native' = raw data straight to AutoGluon; "
            "'no_search_retrieval' = the transfer-only pipeline (best pipeline of the nearest "
            "reference dataset, under the learned metric) — reports both it AND no_preprocessing."
        ),
    )
    parser.add_argument(
        "--baseline-fit-include-val",
        action="store_true",
        help=(
            # '%%' -- argparse runs help text through %-formatting, so a bare '%' raises
            # "ValueError: unsupported format character" and takes down --help entirely.
            "For --baseline-only, fit the final model on train+val (80%%) instead of train (60%%), "
            "matching the CV evaluator's final-fit so scores are directly comparable to CV-path runs."
        ),
    )
    parser.add_argument(
        "--noprep-penalty",
        type=float,
        default=0.0,
        help=(
            "Bias the CV gate AWAY from the trivial no-preprocessing candidate by this epsilon "
            "(selection only; reported score is the chosen pipeline's true test score). A real "
            "pipeline within epsilon of no-prep is chosen instead. 0 = off."
        ),
    )
    parser.add_argument(
        "--hybrid-floor",
        choices=["none", "light"],
        default="none",
        help=(
            "The conservative floor candidate in the hybrid/CV gate. 'none' = bare no-preprocessing "
            "(all-none + onehot). 'light' = normalization-only (scale + onehot, no imputation/structural) "
            "— a REAL pipeline that ties no-prep on AutoGluon, so the method never recommends 'no preprocessing'."
        ),
    )
    parser.add_argument(
        "--exclude-steps",
        default="",
        help=(
            "Comma-separated pipeline steps to remove from the operator space (e.g. "
            "'feature_selection,dimensionality_reduction' for the DiffPrep/their-space comparison)."
        ),
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="RQ2 heuristic-transfer ablation: uniform eta (no warm-start); ACO searches from scratch.",
    )
    parser.add_argument(
        "--global-prior-weight",
        type=float,
        default=0.0,
        help=(
            "Blend weight [0,1] for the global operator-quality prior learned from the reference "
            "matrix (suppresses operators that hurt AutoGluon on average, e.g. svd/lof/pca/knn). "
            "0 = neighbor transfer only; 0.3-0.5 stabilizes the weak neighbor signal. Leak-free."
        ),
    )
    parser.add_argument(
        "--hybrid-select",
        action="store_true",
        help=(
            "Overprocessing guard: add the no-preprocessing baseline as a final candidate and pick "
            "the winner by AutoGluon on the held-out VALIDATION split (leak-free), reporting its TEST "
            "score. Fixes cases where no-search beats search. Recommended for the strong config."
        ),
    )
    parser.add_argument(
        "--cv-select-folds",
        type=int,
        default=0,
        help=(
            "Select the final pipeline by k-fold cross-validation (low-variance signal) instead of a "
            "single validation split, then report the chosen pipeline's held-out test score. Cures the "
            "winner's curse. Requires --hybrid-select. Cost ~(k+1) AutoGluon fits/candidate; use with "
            "--final-autogluon-topk 1 to keep it affordable. 0 = off; 3 is a good default."
        ),
    )
    parser.add_argument(
        "--hybrid-select-margin",
        type=float,
        default=0.0,
        help=(
            "Validation margin for --hybrid-select: only override the no-preprocessing baseline when "
            "the search pipeline beats it on validation by >= this much; otherwise keep the baseline. "
            "A small positive value (e.g. 0.01) makes ACORec rarely lose to no-preprocessing."
        ),
    )
    parser.add_argument(
        "--retrieval-incumbent-topk",
        type=int,
        default=None,
        help="Number of retrieval-ranked incumbent candidates to protect (default: --eval-k).",
    )
    parser.add_argument(
        "--aco-early-stop-rounds",
        type=int,
        default=0,
        help="Stop ACO iterations early after N rounds without meaningful improvement (0 disables).",
    )
    parser.add_argument(
        "--aco-min-improvement",
        type=float,
        default=0.0,
        help="Minimum best-score improvement required to reset ACO early-stop patience.",
    )
    parser.add_argument(
        "--per-feature-independent-search",
        action="store_true",
        help=(
            "Enable sequential per-feature ACO search for independent operators "
            "(imputation/scaling/encoding), while global operators remain shared."
        ),
    )
    parser.add_argument(
        "--per-feature-steps",
        default="imputation,scaling,encoding",
        help="Comma-separated independent steps for per-feature search.",
    )
    parser.add_argument(
        "--per-feature-early-stop-rounds",
        type=int,
        default=2,
        help="Per-feature ACO early-stop patience in iterations (used when per-feature mode is enabled).",
    )
    parser.add_argument(
        "--per-feature-min-improvement",
        type=float,
        default=0.001,
        help="Per-feature ACO minimum improvement threshold for early stop.",
    )
    parser.add_argument(
        "--per-feature-feature-patience",
        type=int,
        default=0,
        help=(
            "Optional early stop across features: stop after this many consecutive features "
            "without improving best proxy score (0 disables)."
        ),
    )
    parser.add_argument(
        "--per-feature-feature-min-improvement",
        type=float,
        default=0.0,
        help="Minimum improvement threshold used by per-feature feature-level patience.",
    )
    parser.add_argument(
        "--per-feature-max-features",
        type=int,
        default=0,
        help="Optional cap on number of features processed in per-feature mode (0 uses all features).",
    )
    parser.add_argument(
        "--optimizer",
        choices=[
            "aco",
            "dqn",
            "retrieval_local",
            "random",
            "ga",
            "sa",
            "greedy",
            "mcts",
            "beam",
            "tpe",
            "exhaustive",
        ],
        default="aco",
        help=(
            "Search optimizer. ACO uses n-ants*n-iterations; DQN/others use sample-budget. "
            "retrieval_local mutates retrieved full-pipeline incumbents."
        ),
    )
    parser.add_argument("--sample-budget", type=int, default=100, help="Config evaluation budget for non-ACO optimizers")
    parser.add_argument(
        "--retrieval-local-neighbor-k",
        type=int,
        default=1,
        help="Number of nearest non-self datasets used to seed optimizer=retrieval_local.",
    )
    parser.add_argument(
        "--retrieval-local-top-l",
        type=int,
        default=1,
        help="Number of top historical full pipelines per retrieved neighbor for optimizer=retrieval_local.",
    )
    parser.add_argument(
        "--retrieval-local-radius",
        type=int,
        choices=[1, 2],
        default=1,
        help="Maximum local mutation radius around retrieved pipelines for optimizer=retrieval_local.",
    )
    parser.add_argument(
        "--retrieval-local-random-candidates",
        type=int,
        default=0,
        help="Optional extra random configs added after local mutations for optimizer=retrieval_local.",
    )
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
        "--train-metric-inline",
        action="store_true",
        help="Train the Siamese regression metric before recommendation instead of requiring a saved --metric-path.",
    )
    parser.add_argument(
        "--no-train-metric-inline",
        action="store_true",
        help=(
            "Disable inline Siamese metric training even when --notebook-legacy-mode would enable it. "
            "Use this to keep legacy ACO/options while falling back to raw metafeature cosine retrieval."
        ),
    )
    parser.add_argument("--metric-hidden-dim", type=int, default=64)
    parser.add_argument("--metric-embed-dim", type=int, default=64)
    parser.add_argument("--metric-epochs", type=int, default=100)
    parser.add_argument("--metric-lr", type=float, default=1e-3)
    parser.add_argument(
        "--metric-loss",
        choices=["mse", "pearson", "listwise_kl"],
        default="mse",
        help=(
            "Siamese training loss. 'pearson' (1 - batch correlation) is collapse-resistant and "
            "optimizes ranking agreement for top-K retrieval; 'mse' is the original (collapses on "
            "the low-variance rank_cosine target). Recommended strong config: --metric-loss pearson."
        ),
    )
    parser.add_argument("--metric-weight-decay", type=float, default=0.0, help="Adam weight decay for the Siamese metric")
    parser.add_argument("--metric-target-temperature", type=float, default=0.1)
    parser.add_argument("--metric-prediction-temperature", type=float, default=0.1)
    parser.add_argument(
        "--metric-objective",
        choices=["embedding_cosine", "projector_product"],
        default="embedding_cosine",
        help=(
            "Siamese metric objective. embedding_cosine trains the embedding space directly; "
            "projector_product keeps the older projector(emb_i * emb_j) objective."
        ),
    )
    parser.add_argument(
        "--metric-similarity-target",
        choices=[
            "rank_cosine",
            "row_zscore_cosine",
            "row_minmax_cosine",
            "legacy_global_zscore_cosine",
        ],
        default="rank_cosine",
        help="Ground-truth dataset-similarity target when training the Siamese metric inline.",
    )
    parser.add_argument(
        "--save-trained-metric",
        required=False,
        help="Optional path to save a metric trained with --train-metric-inline.",
    )
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
        "--shard",
        default=None,
        help="Split dataset IDs across sessions as 'i/n' (1-indexed), e.g. '1/4' runs the first quarter.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Run a shard's datasets concurrently via N subprocesses (each capped to 1 thread). 1 = serial.",
    )
    parser.add_argument(
        "--local-test-mode",
        action="store_true",
        help=(
            "Pre-flight smoke path: serial, CPU, small search budget (n_ants=4, n_iterations=3, "
            "short time limit). NOT the final numbers. Overrides budget flags unless passed explicitly."
        ),
    )
    parser.add_argument(
        "--tar-outputs",
        action="store_true",
        help="After all datasets finish, tar the output directory to <output-dir>.tar.gz (for Kaggle).",
    )
    parser.add_argument(
        "--skip-aco-plot",
        action="store_true",
        help="Skip saving the ACO progress PNG for local/headless environments with fragile Matplotlib font caches.",
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
        "--autogluon-profile",
        choices=["best_quality", "medium_quality", "local_rf_xt"],
        default="best_quality",
        help=(
            "AutoGluon fit profile for final candidate evaluation. "
            "Use best_quality for final Kaggle reporting; local_rf_xt is a stable local smoke-test profile."
        ),
    )
    parser.add_argument(
        "--require-autogluon",
        dest="require_autogluon",
        action="store_true",
        help="Require AutoGluon for evaluation and fail fast if unavailable (default).",
    )
    parser.add_argument(
        "--allow-autogluon-fallback",
        dest="require_autogluon",
        action="store_false",
        help="Allow fallback to simple-model final evaluation if AutoGluon is unavailable.",
    )
    parser.set_defaults(require_autogluon=True)
    parser.add_argument(
        "--no-autogluon",
        dest="no_autogluon",
        action="store_true",
        help=(
            "Skip the final AutoGluon stage entirely and report the pipeline ACO PROPOSES from its "
            "proxy ranking. Diagnostic only: with --hybrid-select the final AutoGluon CV gate is "
            "what CHOOSES among the ACO winner, the transfer-only pipeline and the no-preprocessing "
            "floor, so the reported pipeline here is ACO's proposal, not the method's final "
            "recommendation, and final_evaluation.method will read 'proxy'. Use it to inspect which "
            "operators the search favours without paying for AutoGluon."
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
    parser.add_argument(
        "--paper-faithful",
        action="store_true",
        help=(
            "Reproducibility preset: restore the literal paper method as the baseline. Sets "
            "metric_objective=projector_product (Eq 2/5 learned projector), "
            "heuristic_transfer_method=paper_flat_average + heuristic_top_l=1 (Eq 6/7), "
            "aco_weight_method=linear (Eq 10 per-iteration min-max reward), Markov mixing off "
            "(lambda_smooth=0). Override any individual flag after this to ablate."
        ),
    )
    parser.add_argument(
        "--disable-leakage-holdout",
        action="store_true",
        help=(
            "DANGER: do NOT remove the 24 evaluation IDs from the reference set before "
            "training/normalizing/neighbor retrieval. Reproduces the old leaky behavior for "
            "comparison only. Any number produced this way is contaminated and not reportable."
        ),
    )
    parser.add_argument(
        "--reference-holdout-ids",
        nargs="*",
        default=[],
        help=(
            "Additional dataset IDs removed from metric training and retrieval for "
            "leave-one-dataset-out meta-validation."
        ),
    )
    return parser


def _strip_parallel_flags(argv: List[str]) -> List[str]:
    """Drop dataset-selection and parallel flags from a copy of argv so each worker subprocess
    can be given a single --dataset-ids and --workers 1."""
    value_flags = {"--workers", "--shard", "--dataset-id", "--output-dir"}
    zero_flags = {"--tar-outputs"}
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in value_flags:
            i += 2  # skip flag and its single value
            continue
        if tok in zero_flags:
            i += 1
            continue
        if tok == "--dataset-ids":
            i += 1
            while i < len(argv) and not str(argv[i]).startswith("--"):
                i += 1  # consume the nargs=+ values
            continue
        out.append(tok)
        i += 1
    return out


def _dispatch_workers(dataset_ids: List[Any], output_dir: str, workers: int) -> None:
    """Run one subprocess per dataset, ``workers`` at a time, each capped to a single thread so
    AutoGluon does not oversubscribe (~2 cores/Kaggle session). Each dataset writes to its own
    ``<output_dir>/dataset_<id>`` subdir; a dataset is skipped if its recommendation.json already
    exists (resumable). Subprocesses inherit the same flags minus parallel/selection flags."""
    import concurrent.futures
    import subprocess

    base = [sys.executable, os.path.abspath(__file__)] + _strip_parallel_flags(sys.argv[1:])
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = "1"

    def _child_dir(did: Any) -> str:
        return os.path.join(output_dir, f"dataset_{did}")

    pending = []
    for did in dataset_ids:
        if os.path.exists(os.path.join(_child_dir(did), "recommendation.json")):
            print(f"[workers] dataset {did}: SKIP (output exists)")
        else:
            pending.append(did)

    def _run_one(did: Any) -> Tuple[Any, int]:
        child_dir = _child_dir(did)
        os.makedirs(child_dir, exist_ok=True)
        cmd = base + ["--dataset-ids", str(did), "--workers", "1", "--output-dir", child_dir]
        proc = subprocess.run(cmd, env=env)
        return did, proc.returncode

    print(f"[workers] dispatching {len(pending)} datasets across {workers} subprocess workers "
          f"({len(dataset_ids) - len(pending)} already done)")
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for did, rc in pool.map(_run_one, pending):
            status = "ok" if rc == 0 else f"FAILED(rc={rc})"
            print(f"[workers] dataset {did}: {status}")
            if rc != 0:
                failures.append(did)
    if failures:
        print(f"[workers] {len(failures)} dataset(s) failed: {failures}")


def _tar_output_dir(output_dir: str) -> Optional[str]:
    import tarfile

    tar_path = str(output_dir).rstrip("/") + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_dir, arcname=os.path.basename(str(output_dir).rstrip("/")))
    print(f"[tar] wrote {tar_path}")
    return tar_path


def main() -> None:
    configure_logging()
    args = build_arg_parser().parse_args()
    is_autodp36 = str(args.operator_space) == "autodp36"
    if is_autodp36 and args.notebook_legacy_options:
        raise ValueError("--operator-space autodp36 cannot be combined with --notebook-legacy-options")
    if is_autodp36 and args.operator_param_search:
        raise ValueError(
            "--operator-param-search is defined for ACORec's original operators, not autodp36"
        )
    metric_target_explicit = _cli_flag_was_passed("--metric-similarity-target")

    if args.notebook_legacy_mode:
        args.notebook_legacy_options = True
        args.heuristic_transfer_method = "legacy_weighted_average"
        args.dataset_weighting = "equality"
        args.legacy_notebook_aco = True
        args.aco_lambda_smooth = 0.7
        args.proxy_logreg_max_iter = 1000
        if not metric_target_explicit:
            args.metric_similarity_target = "legacy_global_zscore_cosine"
        args.metric_hidden_dim = 32
        args.metric_embed_dim = 32
        if not args.metric_path:
            args.train_metric_inline = True

    if args.paper_faithful:
        if args.notebook_legacy_mode:
            raise ValueError("--paper-faithful and --notebook-legacy-mode are mutually exclusive.")
        # Each value is applied only if the user did not pass that flag explicitly, so
        # `--paper-faithful --heuristic-top-l 2` ablates a single component on top of the baseline.
        if not _cli_flag_was_passed("--metric-objective"):
            args.metric_objective = "projector_product"
        if not _cli_flag_was_passed("--heuristic-transfer-method"):
            args.heuristic_transfer_method = "paper_flat_average"
        if not _cli_flag_was_passed("--heuristic-top-l"):
            args.heuristic_top_l = 1
        if not _cli_flag_was_passed("--aco-weight-method"):
            args.aco_weight_method = "linear"
        if not _cli_flag_was_passed("--aco-lambda-smooth"):
            args.aco_lambda_smooth = 0.0
        if not _cli_flag_was_passed("--aco-mmas-bounds"):
            args.aco_mmas_bounds = True  # paper claims MMAS; bounds are its defining mechanism
        if not args.metric_path:
            args.train_metric_inline = True

    # Avoid the silent cosine fallback: if the user tuned any Siamese-training flag, they intend to
    # USE the learned metric, so enable inline training (unless a saved metric or explicit opt-out).
    _metric_train_flags = ("--metric-loss", "--metric-weight-decay", "--metric-objective",
                           "--metric-similarity-target", "--metric-epochs", "--metric-hidden-dim",
                           "--metric-embed-dim", "--metric-lr")
    if (not args.metric_path and not args.no_train_metric_inline
            and any(_cli_flag_was_passed(f) for f in _metric_train_flags)):
        args.train_metric_inline = True

    if args.no_train_metric_inline:
        args.train_metric_inline = False

    if args.local_test_mode:
        # Smoke/pre-flight: small budget, serial. Explicit budget flags still win.
        if not _cli_flag_was_passed("--n-ants"):
            args.n_ants = 4
        if not _cli_flag_was_passed("--n-iterations"):
            args.n_iterations = 3
        if not _cli_flag_was_passed("--time-limit"):
            args.time_limit = 60
        if not _cli_flag_was_passed("--workers"):
            args.workers = 1
        print("[local-test-mode] smoke budget: n_ants=%d n_iterations=%d time_limit=%d (NOT final numbers)"
              % (args.n_ants, args.n_iterations, args.time_limit))

    if args.metric_path and args.train_metric_inline:
        raise ValueError("Use either --metric-path or --train-metric-inline, not both.")

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
    elif is_autodp36:
        performance_matrix_path = pick_existing(
            "AutoDP36 performance matrix",
            [
                str(ROOT / "autodp_matrix" / "merged" / "training_performance_matrix_autodp36_ready.csv"),
                os.path.join(
                    args.kaggle_root,
                    "autodp_matrix",
                    "merged",
                    "training_performance_matrix_autodp36_ready.csv",
                ),
            ],
        )
    else:
        if use_kaggle:
            repo_perf_primary = os.path.join(args.kaggle_root, "data", "openml", "training_performance_matrix_autogluon.csv")
            repo_perf_legacy = os.path.join(args.kaggle_root, "aco", "training_performance_matrix_autogluon.csv")
            performance_matrix_path = pick_existing(
                "performance matrix",
                [repo_perf_primary, repo_perf_legacy] + KAGGLE_TRAIN_PERF_PATHS,
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
            # Prefer the broader openml metafeature file when both exist.
            repo_meta_primary = os.path.join(args.kaggle_root, "data", "openml", "dataset_feats.csv")
            repo_meta_secondary = os.path.join(args.kaggle_root, "aco", "dataset_feats.csv")
            metafeatures_path = pick_existing(
                "metafeatures",
                [
                    repo_meta_primary,
                    repo_meta_secondary,
                    str(ROOT / "data" / "openml" / "dataset_feats.csv"),
                    KAGGLE_METAFEATURES_PATH,
                ],
            )
        else:
            metafeatures_path = pick_existing(
                "metafeatures",
                ["dataset_feats.csv", LOCAL_METAFEATURES_PATH, "data/openml/dataset_feats.csv", "Data/openml/dataset_feats.csv"],
            )

    if args.pipeline_configs:
        pipeline_configs_path = args.pipeline_configs
    elif is_autodp36:
        pipeline_configs_path = pick_existing(
            "AutoDP36 pipeline configs",
            [
                str(ROOT / "aco" / "pipeline_configs_autodp36.json"),
                os.path.join(args.kaggle_root, "aco", "pipeline_configs_autodp36.json"),
            ],
        )
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
    autodp60_columns_removed: List[str] = []
    if is_autodp36:
        # Defense in depth: even if a caller overrides the ready matrix with the
        # historical/full file, AutoDP's 60 evaluation datasets never enter the
        # metric learner or retrieval reference.
        perf, autodp60_columns_removed = exclude_autodp60_holdout_columns(perf)
    meta_raw = pd.read_csv(metafeatures_path)
    if args.verbose:
        print(f"Loaded performance matrix: {performance_matrix_path}")
        print(f"Loaded metafeatures: {metafeatures_path}")
        print(f"Loaded pipeline configs: {pipeline_configs_path}")
        if is_autodp36:
            print(
                "AutoDP36 profile: "
                f"matrix={perf.shape[0]}x{perf.shape[1]}, "
                f"AutoDP60 columns removed defensively={len(autodp60_columns_removed)}"
            )

    def _normalize_id(val: object) -> str:
        if pd.isna(val):
            return ""
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        if isinstance(val, (float, np.floating)):
            f = float(val)
            if np.isfinite(f) and abs(f - round(f)) <= 1e-9:
                return str(int(round(f)))
            return str(val).strip()

        s = str(val).strip()
        float_like = re.fullmatch(r"([0-9]+)\.0+", s)
        if float_like:
            return float_like.group(1)

        prefixed = re.fullmatch(r"(?i)(?:d|dataset|openml)[_\-: ]*([0-9]+)", s)
        if prefixed:
            return prefixed.group(1)
        return s

    def _meta_overlap_count(meta_index_like: pd.Series, perf_df: pd.DataFrame) -> int:
        perf_norm = {_normalize_id(c) for c in perf_df.columns}
        vals = meta_index_like.astype(str).map(_normalize_id)
        return len(set(vals) & perf_norm)

    def _maybe_set_meta_index(meta_df: pd.DataFrame, perf_df: pd.DataFrame) -> pd.DataFrame:
        perf_norm = {_normalize_id(c) for c in perf_df.columns}
        best_overlap = _meta_overlap_count(pd.Series(meta_df.index.astype(str)), perf_df)
        best_source = ("index", None)

        candidate_cols = list(meta_df.columns)
        explicit_col = str(args.metafeatures_id_column).strip() if args.metafeatures_id_column else None
        if explicit_col:
            if explicit_col not in meta_df.columns:
                raise ValueError(
                    f"--metafeatures-id-column={explicit_col!r} not found in metafeatures columns: "
                    f"{list(meta_df.columns)[:15]}"
                )
            candidate_cols = [explicit_col]
        else:
            prioritized = ["dataset_id", "did", "openml_id", "id", "Dataset", "dataset", "Unnamed: 0"]
            candidate_cols = [c for c in prioritized if c in meta_df.columns] + [
                c for c in meta_df.columns if c not in prioritized
            ]

        for col in candidate_cols:
            overlap = _meta_overlap_count(meta_df[col], perf_df)
            if overlap > best_overlap:
                best_overlap = overlap
                best_source = ("column", col)

        if best_source[0] == "column" and best_source[1] is not None and best_overlap > 0:
            best_col = str(best_source[1])
            if args.verbose:
                print(f"Using metafeatures id column: {best_col} (overlap={best_overlap})")
            meta_df = meta_df.set_index(best_col)
        elif args.verbose:
            print(f"Using metafeatures index as dataset id (overlap={best_overlap})")

        perf_count = len(perf_norm)
        overlap_ratio = float(best_overlap) / max(perf_count, 1)
        low_absolute = perf_count >= 200 and best_overlap < 100
        low_relative = overlap_ratio < 0.25
        if low_absolute or low_relative:
            raise ValueError(
                "Low metafeature/performance alignment detected. "
                f"overlap={best_overlap}, perf_datasets={perf_count}, ratio={overlap_ratio:.3f}. "
                "This usually means the metafeatures file index/ID column is wrong "
                "(for example, a row-number index was loaded instead of dataset IDs). "
                "Pass the correct file or use --metafeatures-id-column."
            )
        return meta_df

    meta = _maybe_set_meta_index(meta_raw, perf)

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

    def _infer_pipeline_config_from_name(pipeline_name: str) -> dict:
        # Heuristic backfill for historical pipeline names present in the
        # performance matrix but absent from pipeline_configs.json.
        # This keeps Phase-2 transfer informative instead of collapsing to
        # uniform eta when name-to-config mapping is missing.
        token = str(pipeline_name).strip().lower()
        cfg = {
            "name": str(pipeline_name),
            "imputation": "none",
            "scaling": "none",
            "encoding": "onehot",
            "feature_selection": "none",
            "outlier_removal": "none",
            "dimensionality_reduction": "none",
        }

        if "knn" in token:
            cfg["imputation"] = "knn"
        elif "mostfreq" in token or "most_frequent" in token:
            cfg["imputation"] = "most_frequent"
        elif "constant" in token:
            cfg["imputation"] = "constant"
        elif "median" in token:
            cfg["imputation"] = "median"
        elif "mean" in token:
            cfg["imputation"] = "mean"

        if "no_scale" in token:
            cfg["scaling"] = "none"
        elif "robust" in token:
            cfg["scaling"] = "robust"
        elif "minmax" in token:
            cfg["scaling"] = "minmax"
        elif "maxabs" in token:
            cfg["scaling"] = "maxabs"
        elif "standard" in token:
            cfg["scaling"] = "standard"
        elif "uniform" in token or "quantile" in token:
            # Quantile/uniform-style transforms are not explicit operators in
            # current search space; minmax is the closest scale-normalizing proxy.
            cfg["scaling"] = "minmax"

        if "mutualinfo" in token or "mutual_info" in token:
            cfg["feature_selection"] = "mutual_info"
        elif "kbest" in token or "k_best" in token:
            cfg["feature_selection"] = "k_best"
        elif "variance" in token:
            cfg["feature_selection"] = "variance_threshold"

        if "iforest" in token or "isolation" in token:
            cfg["outlier_removal"] = "isolation_forest"
        elif "zscore" in token:
            cfg["outlier_removal"] = "zscore"
        elif "iqr" in token:
            cfg["outlier_removal"] = "iqr"
        elif "lof" in token:
            cfg["outlier_removal"] = "lof"

        if "pca" in token:
            cfg["dimensionality_reduction"] = "pca"
        elif "svd" in token:
            cfg["dimensionality_reduction"] = "svd"

        return cfg

    existing_names = {str(cfg.get("name")) for cfg in pipeline_configs if isinstance(cfg, dict) and "name" in cfg}
    missing_pipeline_names = [str(name) for name in perf.index if str(name) not in existing_names]
    if missing_pipeline_names:
        inferred = [_infer_pipeline_config_from_name(name) for name in missing_pipeline_names]
        pipeline_configs = list(pipeline_configs) + inferred
        if args.verbose:
            sample = ", ".join(missing_pipeline_names[:8])
            print(
                "Augmented pipeline configs for unmatched performance-matrix rows: "
                f"+{len(inferred)} inferred (sample: {sample})"
            )

    dataset_source = args.dataset_source
    if dataset_source == "local":
        # 'local' is an alias: read from a local folder, fall back to OpenML fetch.
        dataset_source = "openml"
        if not args.openml_local_folder:
            args.openml_local_folder = "test_data_local"
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

    # --shard "i/n": deterministically split the dataset IDs across N sessions (1-indexed).
    if args.shard:
        try:
            shard_i, shard_n = (int(x) for x in str(args.shard).split("/"))
        except Exception as exc:
            raise ValueError(f"--shard must be 'i/n' with integers, got {args.shard!r}") from exc
        if not (1 <= shard_i <= shard_n):
            raise ValueError(f"--shard i/n requires 1 <= i <= n, got {args.shard!r}")
        ordered = sorted(dataset_ids, key=lambda d: str(d))
        dataset_ids = [d for idx, d in enumerate(ordered) if idx % shard_n == (shard_i - 1)]
        print(f"[shard {shard_i}/{shard_n}] this session runs {len(dataset_ids)} datasets: {dataset_ids}")

    # Leakage prevention: hold the 30 evaluation IDs out of the reference used to TRAIN /
    # NORMALIZE / RETRIEVE. `meta` (full table) is retained for target-row metafeature lookup,
    # since a new dataset's metafeatures are read from the same precomputed file. The current
    # query is additionally excluded at inference via aco_params["query_dataset_id"].
    holdout_report = None
    if args.disable_leakage_holdout:
        print(
            "\n*** WARNING: --disable-leakage-holdout set. The 30 eval IDs remain in the "
            "reference set; results are CONTAMINATED and must not be reported. ***\n"
        )
        perf_ref, meta_ref = perf, meta
    else:
        perf_ref, meta_ref, holdout_report = holdout_reference(perf, meta, verbose=args.verbose)
        if args.verbose:
            print(
                f"[leakage-holdout] reference now disjoint from {len(EVAL_IDS)} eval IDs: "
                f"perf {holdout_report['perf_cols_before']}->{holdout_report['perf_cols_after']} cols, "
                f"meta {holdout_report['meta_rows_before']}->{holdout_report['meta_rows_after']} rows."
            )

    extra_holdouts = {normalize_id(value) for value in args.reference_holdout_ids}
    if extra_holdouts:
        perf_ref, meta_ref, extra_holdout_report = holdout_ids(
            perf_ref, meta_ref, extra_holdouts
        )
        if args.verbose:
            print(
                f"[meta-validation-holdout] removed IDs={sorted(extra_holdouts)} "
                f"(perf_cols={extra_holdout_report['perf_columns_removed']}, "
                f"meta_rows={extra_holdout_report['metafeature_rows_removed']})"
            )

    if is_autodp36:
        recommender_options = {
            step: list(values) for step, values in AUTODP36_PIPELINE_OPTIONS.items()
        }
    elif str(args.operator_space) == "theirs":
        recommender_options = {
            step: list(values) for step, values in AUTODP_PIPELINE_OPTIONS.items()
        }
    elif args.notebook_legacy_options:
        recommender_options = {
            step: list(values) for step, values in NOTEBOOK_LEGACY_PIPELINE_OPTIONS.items()
        }
    else:
        recommender_options = {
            step: list(values) for step, values in DEFAULT_PIPELINE_OPTIONS.items()
        }
    recommender = MetaPipelineRecommender(
        perf_ref,
        meta_ref,
        pipeline_configs,
        pipeline_options=recommender_options,
        verbose=args.verbose,
    )
    if args.metric_path:
        recommender.load_metric(args.metric_path)
        if args.verbose:
            print(f"Loaded Siamese metric: {args.metric_path}")
    elif args.train_metric_inline:
        if args.verbose:
            print(
                "Training Siamese metric inline: "
                f"hidden_dim={int(args.metric_hidden_dim)}, "
                f"embed_dim={int(args.metric_embed_dim)}, "
                f"epochs={int(args.metric_epochs)}, "
                f"target={args.metric_similarity_target}, "
                f"objective={args.metric_objective}"
            )
        recommender.train_metric(
            method="regression",
            hidden_dim=int(args.metric_hidden_dim),
            embed_dim=int(args.metric_embed_dim),
            epochs=int(args.metric_epochs),
            lr=float(args.metric_lr),
            seed=int(args.seed),
            similarity_target=str(args.metric_similarity_target),
            score_direction=str(args.score_direction),
            metric_objective=str(args.metric_objective),
            metric_loss=str(args.metric_loss),
            weight_decay=float(args.metric_weight_decay),
            target_temperature=float(args.metric_target_temperature),
            prediction_temperature=float(args.metric_prediction_temperature),
        )
        if args.save_trained_metric:
            saved_metric_path = recommender.save_metric(args.save_trained_metric)
            if args.verbose:
                print(f"Saved trained Siamese metric: {saved_metric_path}")

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
        for step in _active_options().keys():
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

        iters = [h.get("iteration", idx + 1) for idx, h in enumerate(history)]
        raw_scores = [h.get("best_score") for h in history]

        filled = []
        last = None
        for score in raw_scores:
            if score is None or (isinstance(score, (int, float)) and not np.isfinite(score)):
                filled.append(last)
            else:
                last = float(score)
                filled.append(last)

        if all(val is None for val in filled):
            return None

        scores = [np.nan if val is None else val for val in filled]

        plt.figure(figsize=(6, 4))
        plt.plot(iters, scores, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Best Score")
        plt.title("ACO Best Score per Iteration")
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(output_dir, "aco_progress.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path

    def _load_dataset_for_run(dataset_id: Any):
        if dataset_source == "openml":
            evaluation_test_ids = {int(value) for value in EVAL_IDS}
            if is_autodp36:
                evaluation_test_ids.update(int(value) for value in AUTODP_60_IDS)
            autodp_test_ids = sorted(evaluation_test_ids)
            autodp_regression_ids = list(AUTODP_REGRESSION_IDS) if is_autodp36 else None
            backend = str(getattr(args, "openml_backend", "auto"))
            if backend == "gitlab":
                return load_gitlab_openml_dataset(
                    dataset_id,
                    test_dataset_ids=autodp_test_ids,
                    regression_dataset_ids=autodp_regression_ids,
                    verbose=args.verbose,
                    cache_dir=args.openml_local_folder,
                    max_samples_if_test=100000,
                )

            loaded = load_openml_dataset(
                dataset_id,
                test_dataset_ids=autodp_test_ids,
                regression_dataset_ids=autodp_regression_ids,
                verbose=args.verbose,
                local_data_folder=args.openml_local_folder,
                max_samples_if_test=100000,
            )
            if loaded is not None or backend == "openml":
                return loaded
            if args.verbose:
                print(f"OpenML failed for D_{dataset_id}; trying GitLab/DataGit mirror")
            return load_gitlab_openml_dataset(
                dataset_id,
                test_dataset_ids=autodp_test_ids,
                regression_dataset_ids=autodp_regression_ids,
                verbose=args.verbose,
                cache_dir=args.openml_local_folder,
                max_samples_if_test=100000,
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

    def _active_options():
        """The operator space this run is searching, before per-run restriction."""
        if is_autodp36:
            return AUTODP36_PIPELINE_OPTIONS
        if str(getattr(args, "operator_space", "ours")) == "theirs":
            return AUTODP_PIPELINE_OPTIONS
        if getattr(args, "notebook_legacy_options", False):
            return NOTEBOOK_LEGACY_PIPELINE_OPTIONS
        return DEFAULT_PIPELINE_OPTIONS

    def _active_step_order():
        if is_autodp36:
            return list(AUTODP36_ORDER)
        return (
            list(AUTODP_PREPROCESSOR_ORDER)
            if str(getattr(args, "operator_space", "ours")) == "theirs"
            else None
        )

    def _build_run_options(dataset_id: Any):
        # Copy lists so per-run constraints do not mutate global defaults.
        if is_autodp36:
            base_options = AUTODP36_PIPELINE_OPTIONS
        elif str(getattr(args, "operator_space", "ours")) == "theirs":
            # AutoDP's operator space, reimplemented leak-free. See
            # src/automl_aco/preprocessing/autodp_ops.py for the full deviation table.
            base_options = AUTODP_PIPELINE_OPTIONS
        elif args.notebook_legacy_options:
            base_options = NOTEBOOK_LEGACY_PIPELINE_OPTIONS
        else:
            base_options = DEFAULT_PIPELINE_OPTIONS
        options = {step: list(vals) for step, vals in base_options.items()}

        # Under the native (transductive) protocol the pipeline is fitted on the full frame and
        # applied via transform(), which never deletes rows -- so a row-dropping operator would be
        # a silent no-op. Remove them from the search rather than let them be selected and do
        # nothing. This SHRINKS the native search space (see docs/ARMS.md) and must be disclosed.
        if str(getattr(args, "prepare_mode", "leakfree")) == "native":
            from automl_aco.search.evaluation import _ROW_DROPPING_OPERATORS
            dropped = []
            for step, vals in list(options.items()):
                keep = [v for v in vals if str(v) not in _ROW_DROPPING_OPERATORS]
                dropped += [v for v in vals if str(v) in _ROW_DROPPING_OPERATORS]
                options[step] = keep or ["none"]
            if dropped and getattr(args, "verbose", False):
                print(f"  [prepare-mode native] excluded row-dropping operators: {sorted(set(dropped))}")
        # Operator-space restriction (e.g. DiffPrep "their space" drops feature engineering ops).
        if getattr(args, "exclude_steps", ""):
            for _st in [s.strip() for s in str(args.exclude_steps).split(",") if s.strip()]:
                if _st == "encoding":
                    continue  # encoding is required for AutoGluon input; never drop it
                options.pop(_st, None)
        profile_note = None

        if pipeline_override:
            for step, choice in pipeline_override.items():
                options[step] = [choice]
            profile_note = "pipeline_override"
            return options, profile_note

        if args.notebook_legacy_options:
            profile_note = "notebook_legacy_options"

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

        # NOTE: a per-dataset `--dataset378-profile` flag (option constraints keyed to eval ID
        # 378) was removed as an evaluation-integrity violation: tuning option spaces to a
        # specific test dataset is indefensible under review. No dataset-ID-keyed branches remain.
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

    # --workers K: fan out one subprocess per dataset, then (optionally) tar and exit.
    if int(args.workers) > 1 and n_runs > 1:
        _dispatch_workers(dataset_ids, output_dir, int(args.workers))
        if args.tar_outputs:
            _tar_output_dir(output_dir)
        return

    run_summaries = []
    search_enabled = False if args.no_search else (args.use_aco or args.optimizer != "aco")
    proxy_settings = _build_proxy_settings()
    heuristic_top_k = int(args.heuristic_top_k) if args.heuristic_top_k is not None else int(args.k)
    if args.no_search:
        if args.verbose:
            print("Info: --no-search selected; optimizer search is disabled.")
    elif args.optimizer == "aco" and not args.use_aco:
        # Legacy behavior required --use-aco even when optimizer=aco.
        # Auto-enable to match user intent and avoid accidental prediction-only runs.
        search_enabled = True
        if args.verbose:
            print("Info: optimizer=aco selected; enabling search flow (legacy --use-aco is optional).")
    elif args.optimizer != "aco" and not args.use_aco and args.verbose:
        print("Info: --optimizer is non-ACO, enabling search flow (same as --use-aco).")
    if args.verbose:
        if args.notebook_legacy_mode:
            print(
                "Notebook legacy mode enabled: "
                "legacy eta averaging, equality weighting, lambda_smooth=0.7, "
                "old ACO sampler, old option space, and leave-one-out self-query guard."
            )
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
            f"unobserved_operator_score={args.unobserved_operator_score}, "
            f"top_k_pheromone={int(args.top_k_pheromone)}, "
            f"weight_method={args.aco_weight_method}, "
            f"markov_order={int(args.aco_markov_order)}, "
            f"lambda_smooth={float(args.aco_lambda_smooth)}, "
            f"interaction_prior_strength={float(args.interaction_prior_strength)}, "
            f"interaction_prior_floor={float(args.interaction_prior_floor)}, "
            f"protect_retrieval_incumbent={bool(args.protect_retrieval_incumbent)}, "
            f"autogluon_profile={args.autogluon_profile}, "
            f"aco_early_stop_rounds={int(args.aco_early_stop_rounds)}, "
            f"aco_min_improvement={float(args.aco_min_improvement)}, "
            f"per_feature_independent_search={bool(args.per_feature_independent_search)}, "
            f"legacy_notebook_aco={bool(args.legacy_notebook_aco)}, "
            f"retrieval_local_neighbor_k={int(args.retrieval_local_neighbor_k)}, "
            f"retrieval_local_top_l={int(args.retrieval_local_top_l)}, "
            f"retrieval_local_radius={int(args.retrieval_local_radius)}, "
            f"retrieval_local_random_candidates={int(args.retrieval_local_random_candidates)}, "
            f"score_direction={args.score_direction}, "
            f"require_autogluon={bool(args.require_autogluon)}, "
            f"proxy_clf_model={proxy_settings.get('classification_model')}, "
            f"proxy_reg_model={proxy_settings.get('regression_model')}"
        )
        if args.per_feature_independent_search:
            print(
                "Per-feature mode setup: "
                f"steps={args.per_feature_steps}, "
                f"iter_early_stop={int(args.per_feature_early_stop_rounds)}, "
                f"iter_min_improvement={float(args.per_feature_min_improvement)}, "
                f"feature_patience={int(args.per_feature_feature_patience)}, "
                f"feature_min_improvement={float(args.per_feature_feature_min_improvement)}, "
                f"max_features={int(args.per_feature_max_features)}"
            )

    for run_idx, dataset_id in enumerate(dataset_ids, start=1):
        if n_runs > 1:
            print(f"\n=== Dataset {dataset_id} ({run_idx}/{n_runs}) ===")

        # Resumable: skip if this dataset's recommendation already exists.
        _tag = str(dataset_id) if dataset_id is not None else ("single" if n_runs == 1 else f"run{run_idx}")
        _expected_dir = output_dir if n_runs == 1 else os.path.join(output_dir, f"dataset_{_tag}")
        if os.path.exists(os.path.join(_expected_dir, "recommendation.json")):
            print(f"  Skip {dataset_id}: output already exists at {_expected_dir}")
            continue

        run_start = time.perf_counter()
        try:
            dataset = _load_dataset_for_run(dataset_id)
            if dataset is None or "X" not in dataset or "y" not in dataset:
                raise ValueError(f"Dataset loading failed for dataset_id={dataset_id}")

            X = dataset["X"]
            y = dataset["y"]
            # The default historical behavior passes the full table to the
            # recommender. For a leak-free experiment, explicitly construct
            # the outer split first and give ACO/metafeatures only train+val.
            if args.recommend_on_train_val:
                (
                    X_train_outer,
                    y_train_outer,
                    X_val_outer,
                    y_val_outer,
                    X_test_outer,
                    y_test_outer,
                ) = split_train_val_test(X, y, seed=int(args.recommend_split_seed))
                X_for_search = pd.concat(
                    [X_train_outer, X_val_outer], axis=0, ignore_index=True
                )
                y_for_search = pd.concat(
                    [y_train_outer, y_val_outer], axis=0, ignore_index=True
                )
                dataset_for_search = dict(dataset)
                dataset_for_search["X"] = X_for_search
                dataset_for_search["y"] = y_for_search
            else:
                X_for_search = X
                y_for_search = y
                dataset_for_search = dataset
            run_options, run_profile_note = _build_run_options(dataset_id)
            run_options, run_option_notes = _adapt_options_to_dataset(run_options, X_for_search)
            if run_profile_note:
                print(f"  Applied profile: {run_profile_note}")
            for note in run_option_notes:
                print(f"  Auto option guard: {note}")
            test_dataset_df = X_for_search.copy()
            test_dataset_df["target"] = y_for_search
            if args.recommend_on_train_val:
                # Preserve the canonical outer split for the proxy evaluator.
                # The ACO input contains only outer train+validation rows, but
                # the proxy must not split those 80% rows a second time.
                test_dataset_df.attrs["_acorec_fixed_proxy_split"] = {
                    "seed": int(args.recommend_split_seed),
                    "X_train": X_train_outer.reset_index(drop=True),
                    "y_train": y_train_outer.reset_index(drop=True),
                    "X_val": X_val_outer.reset_index(drop=True),
                    "y_val": y_val_outer.reset_index(drop=True),
                }

            def _query_metafeatures(_dataset=dataset_for_search):
                if args.recommend_on_train_val:
                    # Precomputed OpenML rows include target-dependent and
                    # landmarking features calculated on the full dataset.
                    # Recompute from the permitted outer train+validation
                    # rows so heuristic transfer cannot inspect outer test.
                    return compute_metafeatures_from_data(
                        _dataset["X"],
                        _dataset["y"],
                        seed=int(args.recommend_split_seed),
                    )
                return extract_enhanced_metafeatures(_dataset, meta_features_df=meta)

            if args.baseline_only != "off":
                from automl_aco.search.evaluation import evaluate_candidates_autogluon
                fit_inc_val = bool(args.baseline_fit_include_val)
                if args.baseline_only == "light_preprocessing":
                    # Validate the "light floor": best normalization (scale + onehot, NO imputation,
                    # NO structural) vs the bare no-prep pipeline, same 80%-fit. If light ties no-prep,
                    # the light floor costs ~nothing while never being "no preprocessing".
                    scale_step = "normalization" if is_autodp36 else "scaling"
                    scale_default = "zscore" if is_autodp36 else "standard"
                    scalers = [
                        s for s in run_options.get(scale_step, []) if str(s).lower() != "none"
                    ] or [scale_default]
                    light_cands = []
                    for s in scalers:
                        c = {step: "none" for step in _active_options()}
                        c[scale_step] = s
                        c["encoding"] = "none" if is_autodp36 else "onehot"
                        c["name"] = f"light_{s}"
                        light_cands.append(c)
                    no_prep = {step: "none" for step in _active_options()}
                    no_prep["encoding"] = "none" if is_autodp36 else "onehot"
                    no_prep["name"] = "no_preprocessing"
                    _bc, _bs, b_results, _b = evaluate_candidates_autogluon(
                        dataset=test_dataset_df, target_column="target",
                        candidate_configs=[*light_cands, no_prep],
                        time_limit_per_model=int(args.time_limit),
                        autogluon_profile=str(args.autogluon_profile), verbose=args.verbose,
                        fit_include_val=fit_inc_val,
                    )
                    scores = {cfg.get("name"): float(sc) for cfg, sc in b_results}
                    np_score = scores.get("no_preprocessing")
                    light_scores = {k: v for k, v in scores.items() if k.startswith("light_")}
                    if not light_scores:
                        raise RuntimeError(f"light_preprocessing eval produced no result for {dataset_id}")
                    best_light_name = max(light_scores, key=light_scores.get)
                    best_light = light_scores[best_light_name]
                    recommendation = {
                        "dataset_id": dataset_id, "baseline_only": args.baseline_only,
                        "fit_include_val": fit_inc_val,
                        "best_light_name": best_light_name, "best_light_score": best_light,
                        "light_scores": light_scores, "no_preprocessing_score": np_score,
                        "final_evaluation": {"method": "autogluon", "score": float(best_light)},
                    }
                    _tag2 = str(dataset_id) if dataset_id is not None else ("single" if n_runs == 1 else f"run{run_idx}")
                    run_out = output_dir if n_runs == 1 else os.path.join(output_dir, f"dataset_{_tag2}")
                    os.makedirs(run_out, exist_ok=True)
                    with open(os.path.join(run_out, "recommendation.json"), "w", encoding="utf-8") as f:
                        json.dump(recommendation, f, indent=2, default=str)
                    _d = (best_light - np_score) if np_score is not None else float("nan")
                    print(f"  [baseline-only:light_preprocessing] {dataset_id} -> best_light={best_light:.4f}"
                          f" ({best_light_name}) no_prep={np_score if np_score is None else round(np_score,4)}"
                          f" delta={_d:+.4f}")
                    run_summaries.append({"dataset_id": dataset_id, "final_score": float(best_light),
                                          "autogluon_score": float(best_light), "status": "ok",
                                          "elapsed_seconds": time.perf_counter() - run_start})
                    continue
                if args.baseline_only == "no_search_retrieval":
                    # Transfer-only pipeline: best pipeline of the nearest reference dataset (learned
                    # metric, query excluded). Reported alongside no_preprocessing for a clean swap
                    # decision; both fit on the same split so the comparison is apples-to-apples.
                    def _bl_mf(_df):
                        return _query_metafeatures()
                    ns_cfg, ns_row, ns_neighbors = recommender.retrieval_no_search_pipeline(
                        new_dataset=test_dataset_df,
                        metafeatures_func=_bl_mf,
                        options=run_options,
                        neighbor_k=1,
                        top_l=1,
                        score_direction=str(args.score_direction),
                        query_dataset_id=dataset_id,
                    )
                    if ns_cfg is None:
                        raise RuntimeError(f"No retrieval neighbor available for {dataset_id}")
                    no_prep = {step: "none" for step in _active_options()}
                    no_prep["encoding"] = "none" if is_autodp36 else "onehot"
                    no_prep["name"] = "no_preprocessing"
                    _b_cfg, _b_best, b_results, _b = evaluate_candidates_autogluon(
                        dataset=test_dataset_df, target_column="target",
                        candidate_configs=[ns_cfg, no_prep],
                        time_limit_per_model=int(args.time_limit),
                        autogluon_profile=str(args.autogluon_profile), verbose=args.verbose,
                        fit_include_val=fit_inc_val,
                    )
                    scores = {cfg.get("name"): float(sc) for cfg, sc in b_results}
                    ns_score = scores.get("no_search_retrieval")
                    np_score = scores.get("no_preprocessing")
                    if ns_score is None:
                        raise RuntimeError(f"no_search_retrieval eval produced no result for {dataset_id}")
                    recommendation = {
                        "dataset_id": dataset_id,
                        "pipeline_config": ns_cfg,
                        "final_evaluation": {"method": "autogluon", "score": float(ns_score)},
                        "baseline_only": args.baseline_only,
                        "fit_include_val": fit_inc_val,
                        "no_search_retrieval_score": ns_score,
                        "no_preprocessing_score": np_score,
                        "neighbor": ns_row,
                        "neighbors": [[str(d), float(s)] for d, s in (ns_neighbors or [])][:3],
                    }
                    _tag2 = str(dataset_id) if dataset_id is not None else ("single" if n_runs == 1 else f"run{run_idx}")
                    run_out = output_dir if n_runs == 1 else os.path.join(output_dir, f"dataset_{_tag2}")
                    os.makedirs(run_out, exist_ok=True)
                    with open(os.path.join(run_out, "recommendation.json"), "w", encoding="utf-8") as f:
                        json.dump(recommendation, f, indent=2, default=str)
                    _delta = (ns_score - np_score) if (np_score is not None) else float("nan")
                    print(f"  [baseline-only:no_search_retrieval] {dataset_id} -> "
                          f"no_search={ns_score:.4f} no_prep={np_score if np_score is None else round(np_score,4)} "
                          f"delta={_delta:+.4f} pipeline={ns_cfg.get('name')}")
                    run_summaries.append({"dataset_id": dataset_id, "final_score": float(ns_score),
                                          "autogluon_score": float(ns_score), "status": "ok",
                                          "elapsed_seconds": time.perf_counter() - run_start})
                    continue
                if args.baseline_only == "autogluon_native":
                    # Raw data straight to AutoGluon (identity pipeline: AG's own preprocessing only).
                    base_cfg = {step: "none" for step in _active_options()}
                    base_cfg["encoding"] = "none"
                    base_cfg["name"] = "autogluon_native"
                else:
                    base_cfg = {step: "none" for step in _active_options()}
                    base_cfg["encoding"] = "none" if is_autodp36 else "onehot"
                    base_cfg["name"] = "no_preprocessing"
                b_cfg, b_score, b_results, _b = evaluate_candidates_autogluon(
                    dataset=test_dataset_df, target_column="target",
                    candidate_configs=[base_cfg],
                    time_limit_per_model=int(args.time_limit),
                    autogluon_profile=str(args.autogluon_profile), verbose=args.verbose,
                    fit_include_val=fit_inc_val,
                )
                if b_cfg is None or not np.isfinite(b_score):
                    raise RuntimeError(f"Baseline AutoGluon eval produced no valid result for {dataset_id}")
                recommendation = {
                    "dataset_id": dataset_id,
                    "pipeline_config": b_cfg,
                    "final_evaluation": {"method": "autogluon", "score": float(b_score)},
                    "baseline_only": args.baseline_only,
                    "fit_include_val": fit_inc_val,
                }
                _tag2 = str(dataset_id) if dataset_id is not None else ("single" if n_runs == 1 else f"run{run_idx}")
                run_out = output_dir if n_runs == 1 else os.path.join(output_dir, f"dataset_{_tag2}")
                os.makedirs(run_out, exist_ok=True)
                with open(os.path.join(run_out, "recommendation.json"), "w", encoding="utf-8") as f:
                    json.dump(recommendation, f, indent=2, default=str)
                print(f"  [baseline-only:{args.baseline_only}] {dataset_id} -> autogluon test={b_score:.4f}")
                run_summaries.append({"dataset_id": dataset_id, "final_score": float(b_score),
                                      "autogluon_score": float(b_score), "status": "ok",
                                      "elapsed_seconds": time.perf_counter() - run_start})
                continue

            def _mf_func(_df):
                return _query_metafeatures()

            recommendation = recommender.recommend(
                new_dataset=test_dataset_df,
                target_column="target",
                options=run_options,
                k=args.k,
                eval_k=args.eval_k,
                use_autogluon=not bool(getattr(args, "no_autogluon", False)),
                use_aco=search_enabled,
                aco_params={
                    "n_ants": args.n_ants,
                    "n_iterations": args.n_iterations,
                    "total_ant_budget": args.aco_total_ant_budget,
                    "seed": args.seed,
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "evaporation": args.evaporation,
                    "top_k_pheromone": int(args.top_k_pheromone),
                    "weight_method": str(args.aco_weight_method),
                    "markov_order": int(args.aco_markov_order),
                    "lambda_smooth": float(args.aco_lambda_smooth),
                    "canonical_cache_keys": bool(
                        args.aco_search_fixes or args.aco_canonical_cache
                    ),
                    "deduplicate_iteration": bool(
                        args.aco_search_fixes or args.aco_deduplicate_iteration
                    ),
                    "cache_invalid_configs": bool(
                        args.aco_search_fixes or args.aco_cache_invalid
                    ),
                    "tie_aware_rank_weights": bool(
                        args.aco_search_fixes or args.aco_tie_aware_rank
                    ),
                    "refill_unique_ants": bool(args.aco_refill_unique_ants),
                    "max_sampling_attempt_multiplier": int(
                        args.aco_max_sampling_attempt_multiplier
                    ),
                    "update_policy": str(args.aco_update_policy),
                    "exploration_policy": str(args.aco_exploration_policy),
                    "exploration_epsilon": float(args.aco_exploration_epsilon),
                    "exploration_initial_epsilon": float(args.aco_exploration_initial_epsilon),
                    "exploration_step": float(args.aco_exploration_step),
                    "exploration_max_epsilon": float(args.aco_exploration_max_epsilon),
                    "mmas_bounds": bool(args.aco_mmas_bounds),
                    "tau_min": args.aco_tau_min,
                    "tau_max": args.aco_tau_max,
                    "tau_min_ratio": float(args.aco_tau_min_ratio),
                    "interaction_prior_strength": float(args.interaction_prior_strength),
                    "interaction_prior_floor": float(args.interaction_prior_floor),
                    "protect_retrieval_incumbent": bool(args.protect_retrieval_incumbent),
                    "hybrid_select": bool(args.hybrid_select),
                    "prepare_mode": str(getattr(args, "prepare_mode", "leakfree")),
                    # The reference performance matrix is written in OUR operator codes. Under
                    # --operator-space theirs the retrieved pipelines would be coerced into their
                    # vocabulary by _coerce_pipeline_to_options, producing degenerate seeds
                    # silently rather than an error, so the transfer arm is dropped by default.
                    "hybrid_include_no_search": (
                        True if str(getattr(args, "operator_space", "ours")) != "theirs"
                        else bool(getattr(args, "transfer_arm_anyway", False))
                    ),
                    "hybrid_floor": str(args.hybrid_floor),
                    "noprep_penalty": float(getattr(args, "noprep_penalty", 0.0)),
                    "hybrid_select_margin": float(args.hybrid_select_margin),
                    "cv_select_folds": int(args.cv_select_folds),
                    "no_warm_start": bool(args.no_warm_start),
                    "global_prior_weight": float(args.global_prior_weight),
                    "retrieval_incumbent_topk": int(args.retrieval_incumbent_topk or args.eval_k),
                    "early_stop_rounds": int(args.aco_early_stop_rounds),
                    "min_improvement": float(args.aco_min_improvement),
                    "dataset_weighting": args.dataset_weighting,
                    "heuristic_top_k": heuristic_top_k,
                    "heuristic_top_l": int(args.heuristic_top_l),
                    "heuristic_transfer_method": str(args.heuristic_transfer_method),
                    "heuristic_similarity_temperature": float(args.heuristic_similarity_temperature),
                    "heuristic_eta_floor": float(args.heuristic_eta_floor),
                    "unobserved_operator_score": (
                        None
                        if args.unobserved_operator_score is None
                        else float(args.unobserved_operator_score)
                    ),
                    "score_direction": str(args.score_direction),
                    "require_autogluon": bool(args.require_autogluon),
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
                    "per_feature_independent_search": bool(args.per_feature_independent_search),
                    "per_feature_steps": str(args.per_feature_steps),
                    "per_feature_early_stop_rounds": int(args.per_feature_early_stop_rounds),
                    "per_feature_min_improvement": float(args.per_feature_min_improvement),
                    "per_feature_feature_patience": int(args.per_feature_feature_patience),
                    "per_feature_feature_min_improvement": float(args.per_feature_feature_min_improvement),
                    "per_feature_max_features": int(args.per_feature_max_features),
                    "legacy_notebook_aco": bool(args.legacy_notebook_aco),
                    "retrieval_local_neighbor_k": int(args.retrieval_local_neighbor_k),
                    "retrieval_local_top_l": int(args.retrieval_local_top_l),
                    "retrieval_local_radius": int(args.retrieval_local_radius),
                    "retrieval_local_random_candidates": int(args.retrieval_local_random_candidates),
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
                autogluon_profile=str(args.autogluon_profile),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - run_start
            print(f"  Dataset {dataset_id} failed: {exc}")
            if os.environ.get("ACOREC_TRACEBACK"):
                traceback.print_exc()
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
        recommendation["dataset_id"] = dataset_id
        recommendation["operator_space"] = str(args.operator_space)
        recommendation["recommendation_protocol"] = {
            "recommend_on_train_val": bool(args.recommend_on_train_val),
            "recommend_split_seed": int(args.recommend_split_seed),
            "test_used_during_search": False if args.recommend_on_train_val else None,
            "search_rows": int(len(y_for_search)),
            "proxy_split": (
                "explicit_outer_split" if args.recommend_on_train_val else "internal_split"
            ),
            "proxy_train_rows": (
                int(len(y_train_outer)) if args.recommend_on_train_val else None
            ),
            "proxy_validation_rows": (
                int(len(y_val_outer)) if args.recommend_on_train_val else None
            ),
            "query_metafeatures_source": (
                "computed_from_outer_train_validation"
                if args.recommend_on_train_val
                else "precomputed_or_full_dataset"
            ),
            "outer_test_rows": int(len(y_test_outer)) if args.recommend_on_train_val else None,
        }
        recommendation["reference_assets"] = {
            "performance_matrix": str(performance_matrix_path),
            "pipeline_configs": str(pipeline_configs_path),
            "metafeatures": str(metafeatures_path),
            "autodp60_columns_removed_defensively": list(autodp60_columns_removed),
        }
        recommendation["search_options"] = run_options
        recommendation["leakage_holdout"] = holdout_report
        recommendation["search_hyperparams"] = {
            "k": int(args.k),
            "heuristic_top_k": int(heuristic_top_k),
            "heuristic_top_l": int(args.heuristic_top_l),
            "dataset_weighting": str(args.dataset_weighting),
            "heuristic_transfer_method": str(args.heuristic_transfer_method),
            "heuristic_similarity_temperature": float(args.heuristic_similarity_temperature),
            "heuristic_eta_floor": float(args.heuristic_eta_floor),
            "unobserved_operator_score": (
                None
                if args.unobserved_operator_score is None
                else float(args.unobserved_operator_score)
            ),
            "score_direction": str(args.score_direction),
            "require_autogluon": bool(args.require_autogluon),
            "autogluon_profile": str(args.autogluon_profile),
            "optimizer": str(args.optimizer),
            "n_ants": int(args.n_ants),
            "n_iterations": int(args.n_iterations),
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "evaporation": float(args.evaporation),
            "top_k_pheromone": int(args.top_k_pheromone),
            "weight_method": str(args.aco_weight_method),
            "markov_order": int(args.aco_markov_order),
            "lambda_smooth": float(args.aco_lambda_smooth),
            "aco_search_fixes": bool(args.aco_search_fixes),
            "canonical_cache_keys": bool(
                args.aco_search_fixes or args.aco_canonical_cache
            ),
            "deduplicate_iteration": bool(
                args.aco_search_fixes or args.aco_deduplicate_iteration
            ),
            "cache_invalid_configs": bool(
                args.aco_search_fixes or args.aco_cache_invalid
            ),
            "tie_aware_rank_weights": bool(
                args.aco_search_fixes or args.aco_tie_aware_rank
            ),
            "refill_unique_ants": bool(args.aco_refill_unique_ants),
            "max_sampling_attempt_multiplier": int(
                args.aco_max_sampling_attempt_multiplier
            ),
            "update_policy": str(args.aco_update_policy),
            "exploration_policy": str(args.aco_exploration_policy),
            "exploration_epsilon": float(args.aco_exploration_epsilon),
            "exploration_initial_epsilon": float(args.aco_exploration_initial_epsilon),
            "exploration_step": float(args.aco_exploration_step),
            "exploration_max_epsilon": float(args.aco_exploration_max_epsilon),
            "interaction_prior_strength": float(args.interaction_prior_strength),
            "interaction_prior_floor": float(args.interaction_prior_floor),
            "protect_retrieval_incumbent": bool(args.protect_retrieval_incumbent),
            "retrieval_incumbent_topk": int(args.retrieval_incumbent_topk or args.eval_k),
            "aco_early_stop_rounds": int(args.aco_early_stop_rounds),
            "aco_min_improvement": float(args.aco_min_improvement),
            "per_feature_independent_search": bool(args.per_feature_independent_search),
            "per_feature_steps": str(args.per_feature_steps),
            "per_feature_early_stop_rounds": int(args.per_feature_early_stop_rounds),
            "per_feature_min_improvement": float(args.per_feature_min_improvement),
            "per_feature_feature_patience": int(args.per_feature_feature_patience),
            "per_feature_feature_min_improvement": float(args.per_feature_feature_min_improvement),
            "per_feature_max_features": int(args.per_feature_max_features),
            "legacy_notebook_aco": bool(args.legacy_notebook_aco),
            "retrieval_local_neighbor_k": int(args.retrieval_local_neighbor_k),
            "retrieval_local_top_l": int(args.retrieval_local_top_l),
            "retrieval_local_radius": int(args.retrieval_local_radius),
            "retrieval_local_random_candidates": int(args.retrieval_local_random_candidates),
            "proxy_clf_model": str(proxy_settings.get("classification_model")),
            "proxy_reg_model": str(proxy_settings.get("regression_model")),
            "proxy_split_seeds": list(proxy_settings.get("split_seeds", [])),
            "proxy_profile": str(args.proxy_profile),
            "operator_param_search": bool(args.operator_param_search),
            "operator_param_grid": str(args.operator_param_grid),
            "no_search": bool(args.no_search),
            "notebook_legacy_mode": bool(args.notebook_legacy_mode),
            "notebook_legacy_options": bool(args.notebook_legacy_options),
            "train_metric_inline": bool(args.train_metric_inline),
            "metric_path": str(args.metric_path) if args.metric_path else None,
            "metric_hidden_dim": int(args.metric_hidden_dim),
            "metric_embed_dim": int(args.metric_embed_dim),
            "metric_epochs": int(args.metric_epochs),
            "metric_lr": float(args.metric_lr),
            "metric_objective": str(args.metric_objective),
            "metric_similarity_target": str(args.metric_similarity_target),
            "metric_target_temperature": float(args.metric_target_temperature),
            "metric_prediction_temperature": float(args.metric_prediction_temperature),
            "reference_holdout_ids": sorted(extra_holdouts),
            "save_trained_metric": str(args.save_trained_metric) if args.save_trained_metric else None,
            "skip_aco_plot": bool(args.skip_aco_plot),
        }
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(recommendation, f, indent=2, default=str)

        history_path = None
        if history:
            history_path = os.path.join(run_output_dir, "aco_history.csv")
            pd.DataFrame(history).to_csv(history_path, index=False)

        plot_path = None if args.skip_aco_plot else _save_history_plot(history, run_output_dir)

        pipeline_cfg = recommendation.get("pipeline_config") or {}
        if "recommended_performance" in recommendation:
            proxy_score = recommendation.get("recommended_performance")
        else:
            proxy_score = recommendation.get("expected_performance")
        final_eval = recommendation.get("final_evaluation", {})
        final_score = recommendation.get("final_performance", final_eval.get("score"))

        print("\nFinal recommendation")
        print(f"  Dataset: {dataset_tag}")
        if args.operator_param_search:
            print(f"  Operator-param profile: {args.operator_param_grid}")
        print(f"  Pipeline: {_format_pipeline(pipeline_cfg)}")
        per_feature_info = recommendation.get("per_feature_search")
        if isinstance(per_feature_info, dict) and per_feature_info.get("enabled"):
            pf_log = per_feature_info.get("log") or []
            print(f"  Per-feature search: enabled (features_logged={len(pf_log)})")
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
                "autogluon_profile": str(args.autogluon_profile),
                "elapsed_seconds": elapsed,
                "recommendation_path": rec_path,
                "history_path": history_path,
                "plot_path": plot_path,
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

    if args.tar_outputs:
        _tar_output_dir(output_dir)


if __name__ == "__main__":
    main()
