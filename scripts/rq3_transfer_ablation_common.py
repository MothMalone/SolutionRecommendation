"""Leak-free RQ3 ablations for transfer-neighborhood size and ACO ants.

This module deliberately keeps the ACORec core untouched.  It runs the existing
``run_recommend.py`` entry point with the search restricted to the externally
fixed train+validation split, then evaluates the frozen recommendation once on
the outer test split.  The two thin command-line wrappers select the parameter
being ablated:

* ``num_retrieved_datasets``: ``--heuristic-top-k`` (K), with H fixed;
* ``num_selected_pipelines``: ``--heuristic-top-l`` (H), with K fixed;
* ``num_ants``: ``--n-ants``, with K and H fixed;
* ``total_ant_budget``: total ant draws, with 10 ants per full iteration;
* ``pheromone_weight_method``: ``--aco-weight-method``, with the remaining
  ACO search parameters fixed.
* ``aco_update_policy``: ``global_elite``, ``iteration_elite``, or
  ``hybrid_elite`` update policy, with the ACO weighting method fixed.
* ``aco_alpha_beta``: the relative influence of pheromone (alpha) and the
  transferred heuristic (beta), with all other ACO settings fixed.

Each dataset/variant is an independent checkpoint, so a Kaggle session can be
sharded and resumed without repeating successful runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_STEPS = (
    "imputation",
    "scaling",
    "encoding",
    "feature_selection",
    "outlier_removal",
    "dimensionality_reduction",
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _parse_int_list(value: Any) -> List[int]:
    """Parse comma/space separated positive integers while preserving order."""
    if value is None:
        return []
    tokens: Iterable[str]
    if isinstance(value, (list, tuple)):
        tokens = [str(item) for item in value]
    else:
        tokens = str(value).replace(",", " ").split()
    result: List[int] = []
    seen = set()
    for token in tokens:
        if not str(token).strip():
            continue
        number = int(str(token).strip())
        if number < 1:
            raise ValueError(f"Ablation values must be positive, got {number}")
        if number not in seen:
            result.append(number)
            seen.add(number)
    return result


def _parse_pheromone_weight_methods(value: Any) -> List[str]:
    """Parse and validate pheromone reinforcement weighting methods."""
    tokens = str(value).replace(",", " ").split() if value is not None else []
    allowed = {"exponential", "rank", "uniform"}
    result: List[str] = []
    for token in tokens:
        method = token.strip().lower()
        if not method:
            continue
        if method not in allowed:
            raise ValueError(
                f"Unsupported pheromone weight method {method!r}; "
                f"expected one of {sorted(allowed)}"
            )
        if method not in result:
            result.append(method)
    return result


def _parse_update_policies(value: Any) -> List[str]:
    """Parse and validate ACO pheromone update policies."""
    tokens = str(value).replace(",", " ").split() if value is not None else []
    allowed = {"global_elite", "iteration_elite", "hybrid_elite"}
    result: List[str] = []
    for token in tokens:
        policy = token.strip().lower()
        if not policy:
            continue
        if policy not in allowed:
            raise ValueError(
                f"Unsupported ACO update policy {policy!r}; "
                f"expected one of {sorted(allowed)}"
            )
        if policy not in result:
            result.append(policy)
    return result


_ALPHA_BETA_VARIANTS: Mapping[str, Tuple[float, float]] = {
    "heuristic_only": (0.0, 2.0),
    "pheromone_only": (1.0, 0.0),
    "current": (1.0, 2.0),
    "balanced": (1.0, 1.0),
    "pheromone_strong": (2.0, 1.0),
}


def _parse_alpha_beta_variants(value: Any) -> List[str]:
    """Parse named alpha/beta configurations used by the RQ3 diagnostic."""
    tokens = str(value).replace(",", " ").split() if value is not None else []
    result: List[str] = []
    for token in tokens:
        name = token.strip().lower()
        if not name:
            continue
        if name not in _ALPHA_BETA_VARIANTS:
            raise ValueError(
                f"Unsupported alpha/beta variant {name!r}; "
                f"expected one of {sorted(_ALPHA_BETA_VARIANTS)}"
            )
        if name not in result:
            result.append(name)
    return result


def _parse_dataset_ids(raw: Optional[Sequence[Any]]) -> List[str]:
    if not raw:
        return []
    result: List[str] = []
    seen = set()
    for token in raw:
        for piece in str(token).replace(",", " ").split():
            value = piece.strip()
            if not value:
                continue
            # Dataset IDs are integer identifiers, but keeping a string here
            # avoids accidental float formatting in command lines and paths.
            if value.endswith(".0") and value[:-2].isdigit():
                value = value[:-2]
            if value not in seen:
                result.append(value)
                seen.add(value)
    return result


def _load_dataset_ids(args: argparse.Namespace, root: Path) -> List[str]:
    explicit = _parse_dataset_ids(args.dataset_ids)
    if explicit:
        return explicit
    manifest = Path(args.manifest) if args.manifest else root / "data" / "openml" / "meta_dev18.json"
    if not manifest.exists():
        raise FileNotFoundError(
            "No dataset IDs were supplied and the meta-dev manifest is missing: "
            f"{manifest}. Pass --dataset-ids explicitly."
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return _parse_dataset_ids(payload.get("dataset_ids", []))


def _operator_base(value: Any) -> str:
    text = str(value).strip().lower()
    try:
        # Keep this helper optional so the diagnostic script remains usable in
        # a lightweight environment before the repository package is imported.
        from automl_aco.utils.operator_spec import base_operator_name

        return str(base_operator_name(text)).strip().lower()
    except Exception:
        return text.split("@", 1)[0].strip().lower()


def _count_operator_value(value: Any) -> int:
    """Count active operator assignments, including per-feature mappings."""
    if isinstance(value, Mapping):
        return sum(_count_operator_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_operator_value(item) for item in value)
    if value is None:
        return 0
    text = str(value).strip()
    if not text or _operator_base(text) in {"none", "identity", "passthrough"}:
        return 0
    return 1


def count_active_operator_assignments(config: Mapping[str, Any]) -> int:
    """Return active operator assignments in one selected pipeline.

    A scalar six-stage pipeline contributes at most six.  A per-feature config
    contributes one for every non-identity feature/operator assignment.  This
    is intentionally different from counting the fixed number of pipeline
    stages, which would not be informative for an H ablation.
    """
    if not isinstance(config, Mapping):
        return 0
    return int(sum(_count_operator_value(config.get(step)) for step in _PIPELINE_STEPS))


def selected_pipeline_operator_stats(recommendation: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract operator-complexity diagnostics from a recommendation artifact."""
    selected = recommendation.get("pipeline_config")
    selected_count = count_active_operator_assignments(selected or {})

    candidate_configs: List[Mapping[str, Any]] = []
    raw_candidates = recommendation.get("aco_results") or []
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            candidate = None
            if isinstance(item, (list, tuple)) and item and isinstance(item[0], Mapping):
                candidate = item[0]
            elif isinstance(item, Mapping):
                candidate = item
            if candidate is not None:
                candidate_configs.append(candidate)
    counts = [count_active_operator_assignments(candidate) for candidate in candidate_configs]
    return {
        "selected_pipeline_active_operator_count": int(selected_count),
        # Short alias used by the RQ3 tables.  It counts non-identity operator
        # assignments, not the fixed six stage slots.
        "selected_pipeline_operator_count": int(selected_count),
        "selected_candidate_count": int(len(counts)),
        "selected_candidate_operator_count_total": int(sum(counts)),
        "selected_candidates_active_operator_count_total": int(sum(counts)),
        "selected_candidates_active_operator_count_mean": (
            float(np.mean(counts)) if counts else None
        ),
    }


def _last_history_values(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "aco_history.csv"
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if frame.empty:
        return {}
    last = frame.iloc[-1].to_dict()

    def value(*names: str) -> Any:
        for name in names:
            if name in last and pd.notna(last[name]):
                return last[name]
        return None

    return {
        "proxy_unique_evaluations": value("cache_size", "cumulative_unique_evaluations"),
        "proxy_evaluation_requests": value("cumulative_evaluation_request_count", "evaluation_request_count"),
        "proxy_duplicate_draws": value("cumulative_duplicate_draw_count", "duplicate_draw_count"),
        "search_best_score": value("global_best_score", "best_score"),
        "search_no_improve_rounds": value("no_improve_rounds"),
        "search_step_entropy": value("step_entropy"),
        "search_effective_epsilon": value("effective_epsilon"),
        "search_pheromone_deposit_count": value("pheromone_deposit_count"),
        "search_global_improved": value("global_improved"),
    }


def _git_commit(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def build_parser(ablation: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--performance-matrix", type=Path, default=None)
    parser.add_argument("--metafeatures", type=Path, default=None)
    parser.add_argument("--pipeline-configs", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="Dataset manifest used when --dataset-ids is omitted")
    parser.add_argument("--dataset-ids", nargs="*", default=None)
    parser.add_argument("--dataset-source", choices=["openml", "kaggle"], default="openml")
    parser.add_argument("--openml-backend", choices=["auto", "openml", "gitlab"], default="gitlab")
    parser.add_argument("--openml-local-folder", type=Path, default=None)
    parser.add_argument("--kaggle-data-folder", type=Path, default=None)
    parser.add_argument("--kaggle-target-column", default="target")
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset cache used by search and final evaluation")
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/rq3_transfer_ablation"))
    parser.add_argument(
        "--variant-values",
        default=None,
        help="Comma/space separated ablation values or named strategy variants",
    )
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--fixed-h", type=int, default=3)
    parser.add_argument("--search-k", type=int, default=5, help="Fixed ACO candidate/neighbor parameter; K ablation uses --heuristic-top-k")
    parser.add_argument("--n-ants", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument(
        "--early-stop-rounds",
        "--aco-early-stop-rounds",
        dest="early_stop_rounds",
        type=int,
        default=0,
        help="Stop ACO after this many iterations without proxy improvement; 0 disables.",
    )
    parser.add_argument(
        "--min-improvement",
        "--aco-min-improvement",
        dest="min_improvement",
        type=float,
        default=0.0,
        help="Minimum proxy-score increase that resets early-stop patience.",
    )
    parser.add_argument("--top-k-pheromone", type=int, default=3, help="Fixed number of elite candidates reinforced by pheromone")
    parser.add_argument(
        "--aco-weight-method",
        choices=["rank", "linear", "exponential", "reciprocal", "power_rank", "uniform"],
        default="rank",
    )
    parser.add_argument("--aco-markov-order", type=int, default=2)
    parser.add_argument("--aco-lambda-smooth", type=float, default=0.0)
    parser.add_argument("--aco-alpha", type=float, default=1.0, help="Fixed pheromone exponent outside alpha/beta ablations")
    parser.add_argument("--aco-beta", type=float, default=2.0, help="Fixed heuristic exponent outside alpha/beta ablations")
    parser.add_argument(
        "--aco-update-policy",
        choices=["global_elite", "iteration_elite", "improvement_only", "hybrid_elite"],
        default="global_elite",
    )
    parser.add_argument("--aco-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--proxy-profile", choices=["default", "robust"], default="default")
    parser.add_argument("--proxy-clf-model", default="logreg")
    parser.add_argument("--eval-k", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--evaluator", choices=["autogluon", "tpot"], default="autogluon")
    parser.add_argument("--final-time-limit", type=int, default=300, help="AutoGluon time limit per frozen pipeline")
    parser.add_argument("--autogluon-presets", default="best_quality")
    parser.add_argument("--tpot-time-mins", type=int, default=5)
    parser.add_argument("--tpot-max-eval-time-mins", type=int, default=1)
    parser.add_argument("--tpot-jobs", type=int, default=1)
    parser.add_argument("--tpot-memory-limit", default="5GB")
    parser.add_argument("--tpot-population-size", type=int, default=20)
    parser.add_argument("--metric-path", type=Path, default=None)
    parser.add_argument("--train-metric-inline", action="store_true")
    parser.add_argument("--metric-epochs", type=int, default=100)
    parser.add_argument("--metric-objective", default=None)
    parser.add_argument("--metric-similarity-target", default=None)
    parser.add_argument("--metric-target-temperature", type=float, default=None)
    parser.add_argument("--metric-prediction-temperature", type=float, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true", help="Re-run both search and final evaluation")
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Run ACO search and collect proxy diagnostics without final evaluator",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--dataset-shard-index", type=int, default=0)
    parser.add_argument("--num-dataset-shards", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(ablation=ablation)
    return parser


def _resolve_paths(args: argparse.Namespace, root: Path) -> Dict[str, Path]:
    data_dir = Path(args.data_dir or args.openml_local_folder or root / "test_data_local")
    return {
        "performance_matrix": Path(args.performance_matrix or root / "data" / "openml" / "training_performance_matrix_autogluon.csv"),
        "metafeatures": Path(args.metafeatures or root / "data" / "openml" / "dataset_feats.csv"),
        "pipeline_configs": Path(args.pipeline_configs or root / "aco" / "pipeline_configs.json"),
        "data_dir": data_dir,
    }


def _variant_values(args: argparse.Namespace) -> List[Any]:
    if args.ablation == "pheromone_weight_method":
        values = _parse_pheromone_weight_methods(
            args.variant_values or "exponential,rank,uniform"
        )
    elif args.ablation == "aco_update_policy":
        values = _parse_update_policies(
            args.variant_values or "global_elite,iteration_elite,hybrid_elite"
        )
    elif args.ablation == "aco_alpha_beta":
        values = _parse_alpha_beta_variants(
            args.variant_values or "heuristic_only,pheromone_only,current,balanced,pheromone_strong"
        )
    elif args.variant_values:
        values = _parse_int_list(args.variant_values)
    elif args.ablation == "num_retrieved_datasets":
        values = [1, 3, 5]
    elif args.ablation == "num_ants":
        values = [5, 10, 15, 20]
    elif args.ablation == "total_ant_budget":
        values = [5, 10, 15, 20, 25, 30]
    else:
        values = [1, 2, 3]
    if not values:
        raise ValueError("At least one ablation value is required")
    return values


def _variant_name(ablation: str, parameter_value: Any) -> str:
    """Return a compact, unambiguous directory name for an ablation value."""
    if ablation == "pheromone_weight_method":
        method = str(parameter_value).strip().lower()
        if method not in {"exponential", "rank", "uniform"}:
            raise ValueError(f"Unsupported pheromone weight method: {method}")
        return f"W_{method}"
    if ablation == "aco_update_policy":
        policy = str(parameter_value).strip().lower()
        if policy not in {"global_elite", "iteration_elite", "hybrid_elite"}:
            raise ValueError(f"Unsupported ACO update policy: {policy}")
        return f"U_{policy}"
    if ablation == "aco_alpha_beta":
        name = str(parameter_value).strip().lower()
        if name not in _ALPHA_BETA_VARIANTS:
            raise ValueError(f"Unsupported alpha/beta variant: {name}")
        alpha, beta = _ALPHA_BETA_VARIANTS[name]
        return f"AB_{name}_a{alpha:g}_b{beta:g}"
    prefix = {
        "num_retrieved_datasets": "K",
        "num_selected_pipelines": "H",
        "num_ants": "A",
        "total_ant_budget": "B",
    }.get(ablation)
    if prefix is None:
        raise ValueError(f"Unsupported RQ3 ablation: {ablation}")
    return f"{prefix}{int(parameter_value)}"


def _build_search_command(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    dataset_id: str,
    run_dir: Path,
    parameter_value: Any,
) -> List[str]:
    root = Path(args.root)
    update_policy = str(args.aco_update_policy)
    alpha, beta = float(args.aco_alpha), float(args.aco_beta)
    if args.ablation == "num_retrieved_datasets":
        transfer_k, transfer_h = parameter_value, int(args.fixed_h)
        n_ants = int(args.n_ants)
    elif args.ablation == "num_selected_pipelines":
        transfer_k, transfer_h = int(args.fixed_k), parameter_value
        n_ants = int(args.n_ants)
    elif args.ablation == "num_ants":
        transfer_k, transfer_h = int(args.fixed_k), int(args.fixed_h)
        n_ants = int(parameter_value)
        weight_method = str(args.aco_weight_method)
    elif args.ablation == "total_ant_budget":
        transfer_k, transfer_h = int(args.fixed_k), int(args.fixed_h)
        n_ants = int(args.n_ants)
        weight_method = str(args.aco_weight_method)
    elif args.ablation == "pheromone_weight_method":
        transfer_k, transfer_h = int(args.fixed_k), int(args.fixed_h)
        n_ants = int(args.n_ants)
        weight_method = str(parameter_value).strip().lower()
    elif args.ablation == "aco_update_policy":
        transfer_k, transfer_h = int(args.fixed_k), int(args.fixed_h)
        n_ants = int(args.n_ants)
        weight_method = str(args.aco_weight_method)
        update_policy = str(parameter_value).strip().lower()
    elif args.ablation == "aco_alpha_beta":
        transfer_k, transfer_h = int(args.fixed_k), int(args.fixed_h)
        n_ants = int(args.n_ants)
        weight_method = str(args.aco_weight_method)
        alpha, beta = _ALPHA_BETA_VARIANTS[str(parameter_value).strip().lower()]
    else:
        raise ValueError(f"Unsupported RQ3 ablation: {args.ablation}")

    if args.ablation != "pheromone_weight_method":
        weight_method = str(args.aco_weight_method)

    command = [
        sys.executable,
        str(root / "scripts" / "run_recommend.py"),
        "--operator-space", "ours",
        "--performance-matrix", str(paths["performance_matrix"]),
        "--metafeatures", str(paths["metafeatures"]),
        "--pipeline-configs", str(paths["pipeline_configs"]),
        "--dataset-source", str(args.dataset_source),
        "--dataset-ids", str(dataset_id),
        "--optimizer", "aco",
        "--use-aco",
        "--no-autogluon",
        "--recommend-on-train-val",
        "--recommend-split-seed", str(int(args.split_seed)),
        "--reference-holdout-ids", str(dataset_id),
        "--openml-backend", str(args.openml_backend),
        "--k", str(max(1, int(args.search_k))),
        "--eval-k", str(max(1, int(args.eval_k))),
        "--heuristic-top-k", str(max(1, transfer_k)),
        "--heuristic-top-l", str(max(1, transfer_h)),
        "--n-ants", str(max(1, n_ants)),
        "--n-iterations", str(
            max(1, math.ceil(int(parameter_value) / max(1, n_ants)))
            if args.ablation == "total_ant_budget"
            else max(1, int(args.n_iterations))
        ),
        "--top-k-pheromone", str(max(1, int(args.top_k_pheromone))),
        "--aco-weight-method", weight_method,
        "--aco-markov-order", str(max(1, int(args.aco_markov_order))),
        "--aco-lambda-smooth", str(float(args.aco_lambda_smooth)),
        "--aco-update-policy", update_policy,
        "--alpha", str(float(alpha)),
        "--beta", str(float(beta)),
        "--aco-early-stop-rounds", str(max(0, int(args.early_stop_rounds))),
        "--aco-min-improvement", str(max(0.0, float(args.min_improvement))),
        "--seed", str(int(args.aco_seed)),
        "--proxy-profile", str(args.proxy_profile),
        "--proxy-clf-model", str(args.proxy_clf_model),
        "--workers", "1",
        "--output-dir", str(run_dir),
        "--openml-local-folder", str(paths["data_dir"]),
        "--skip-aco-plot",
        "--aco-search-fixes",
    ]
    if args.ablation == "total_ant_budget":
        command.extend(["--aco-total-ant-budget", str(max(1, int(parameter_value)))])
    if args.dataset_source == "kaggle":
        command.extend(["--kaggle-data-folder", str(args.kaggle_data_folder or paths["data_dir"]), "--kaggle-target-column", str(args.kaggle_target_column)])
    if args.metric_path:
        command.extend(["--metric-path", str(args.metric_path)])
    elif args.train_metric_inline:
        command.extend(["--train-metric-inline", "--metric-epochs", str(int(args.metric_epochs))])
    if args.metric_objective:
        command.extend(["--metric-objective", str(args.metric_objective)])
    if args.metric_similarity_target:
        command.extend(["--metric-similarity-target", str(args.metric_similarity_target)])
    if args.metric_target_temperature is not None:
        command.extend(["--metric-target-temperature", str(float(args.metric_target_temperature))])
    if args.metric_prediction_temperature is not None:
        command.extend(["--metric-prediction-temperature", str(float(args.metric_prediction_temperature))])
    if args.verbose:
        command.append("--verbose")
    return command


def _build_evaluation_command(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    dataset_id: str,
    run_dir: Path,
    evaluation_path: Path,
    recommendation_path: Path,
) -> List[str]:
    root = Path(args.root)
    if args.evaluator == "autogluon":
        return [
            sys.executable,
            str(root / "scripts" / "evaluate_acorec_autogluon.py"),
            "--recommendation-json", str(recommendation_path),
            "--dataset-id", str(dataset_id),
            "--data-dir", str(paths["data_dir"]),
            "--output-json", str(evaluation_path),
            "--max-samples", str(int(args.max_samples)),
            "--split-seed", str(int(args.split_seed)),
            "--time-limit", str(max(1, int(args.final_time_limit))),
            "--presets", str(args.autogluon_presets),
            *( ["--force"] if args.force else [] ),
        ]
    return [
        sys.executable,
        str(root / "scripts" / "evaluate_acorec_tpot.py"),
        "--recommendation-json", str(recommendation_path),
        "--dataset-id", str(dataset_id),
        "--data-dir", str(paths["data_dir"]),
        "--output-json", str(evaluation_path),
        "--max-samples", str(int(args.max_samples)),
        "--split-seed", str(int(args.split_seed)),
        "--tpot-seed", "1",
        "--max-time-mins", str(max(1, int(args.tpot_time_mins))),
        "--max-eval-time-mins", str(max(1, int(args.tpot_max_eval_time_mins))),
        "--n-jobs", str(max(1, int(args.tpot_jobs))),
        "--memory-limit", str(args.tpot_memory_limit),
        "--population-size", str(max(2, int(args.tpot_population_size))),
        "--verbose", "1",
        *( ["--force"] if args.force else [] ),
    ]


def _run_process(command: Sequence[str], root: Path, env: Mapping[str, str]) -> Tuple[int, float]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=root, env=dict(env), check=False)
    return int(completed.returncode), float(time.perf_counter() - started)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_rows(rows: List[Dict[str, Any]], suite_dir: Path) -> None:
    suite_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(suite_dir / "results.csv", index=False)
    (suite_dir / "results.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def _load_rows(suite_dir: Path) -> List[Dict[str, Any]]:
    path = suite_dir / "results.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
    except Exception:
        return []


def _summary(suite_dir: Path, rows: List[Dict[str, Any]], args: argparse.Namespace, dataset_ids: List[str], values: List[Any]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        summary = pd.DataFrame()
    else:
        successful = frame[frame["status"].eq("ok")].copy()
        records: List[Dict[str, Any]] = []
        for variant, group in frame.groupby("variant", sort=False):
            ok = group[group["status"].eq("ok")]

            def mean(column: str) -> Optional[float]:
                if column not in ok:
                    return None
                values_numeric = pd.to_numeric(ok[column], errors="coerce").dropna()
                return float(values_numeric.mean()) if not values_numeric.empty else None

            def median(column: str) -> Optional[float]:
                if column not in ok:
                    return None
                values_numeric = pd.to_numeric(ok[column], errors="coerce").dropna()
                return float(values_numeric.median()) if not values_numeric.empty else None

            records.append(
                {
                    "ablation": args.ablation,
                    "variant": variant,
                    "parameter_value": group["parameter_value"].iloc[0],
                    "requested_datasets": int(len(dataset_ids)),
                    "runs": int(len(group)),
                    "successful_runs": int(len(ok)),
                    "failed_runs": int(len(group) - len(ok)),
                    "mean_accuracy": mean("accuracy"),
                    "median_accuracy": median("accuracy"),
                    "mean_f1_macro": mean("f1_macro"),
                    "median_f1_macro": median("f1_macro"),
                    # The two validation quantities are intentionally kept
                    # separate: search_best_score is the proxy used by ACO;
                    # evaluator_validation_score is the evaluator's own
                    # train/validation score before its untouched outer test.
                    "mean_search_validation_score": mean("search_best_score"),
                    "mean_evaluator_validation_score": mean("evaluator_validation_score"),
                    "mean_balanced_accuracy": mean("balanced_accuracy"),
                    "mean_search_seconds": mean("search_wall_clock_seconds"),
                    "mean_evaluation_seconds": mean("evaluation_wall_clock_seconds"),
                    "mean_total_seconds": mean("total_wall_clock_seconds"),
                    "mean_selected_pipeline_operator_count": mean("selected_pipeline_operator_count"),
                    "median_total_seconds": median("total_wall_clock_seconds"),
                    "mean_selected_pipeline_active_operator_count": mean("selected_pipeline_active_operator_count"),
                    "mean_selected_candidates_active_operator_count_total": mean("selected_candidates_active_operator_count_total"),
                    "mean_proxy_unique_evaluations": mean("proxy_unique_evaluations"),
                    "git_commit": _git_commit(Path(args.root)),
                }
            )
        summary = pd.DataFrame(records)
    summary.to_csv(suite_dir / "summary.csv", index=False)
    metadata = {
        "ablation": args.ablation,
        "dataset_ids": dataset_ids,
        "variant_values": values,
        "fixed_k": int(args.fixed_k),
        "fixed_h": int(args.fixed_h),
        "top_k_pheromone": int(args.top_k_pheromone),
        "aco_weight_method": str(args.aco_weight_method),
        "aco_markov_order": int(args.aco_markov_order),
        "aco_lambda_smooth": float(args.aco_lambda_smooth),
        "aco_alpha": float(args.aco_alpha),
        "aco_beta": float(args.aco_beta),
        "aco_update_policy": str(args.aco_update_policy),
        "n_iterations": int(args.n_iterations),
        "evaluator": args.evaluator,
        "split_seed": int(args.split_seed),
        "aco_seed": int(args.aco_seed),
        "git_commit": _git_commit(Path(args.root)),
        "protocol": (
            "search on externally fixed train+validation; no final evaluator"
            if args.search_only
            else "search on externally fixed train+validation; frozen pipeline evaluated once on untouched outer test"
        ),
    }
    (suite_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nSaved:")
    print(" ", suite_dir / "results.csv")
    print(" ", suite_dir / "summary.csv")
    if not summary.empty:
        print(summary.to_string(index=False))


def run_ablation(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    paths = _resolve_paths(args, root)
    for key in ("performance_matrix", "metafeatures", "pipeline_configs"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing {key}: {paths[key]}")
    paths["data_dir"].mkdir(parents=True, exist_ok=True)

    dataset_ids = _load_dataset_ids(args, root)
    if not dataset_ids:
        raise ValueError("No dataset IDs to run")
    if not 0 <= int(args.dataset_shard_index) < int(args.num_dataset_shards):
        raise ValueError("dataset-shard-index must satisfy 0 <= index < num-dataset-shards")
    dataset_ids = dataset_ids[int(args.dataset_shard_index) :: int(args.num_dataset_shards)]
    values = _variant_values(args)
    suite_dir = Path(args.output_root).resolve() / str(args.ablation)
    if args.dry_run:
        for value in values:
            variant = _variant_name(args.ablation, value)
            for dataset_id in dataset_ids:
                run_dir = suite_dir / variant / f"dataset_{dataset_id}"
                recommendation_path = run_dir / "recommendation.json"
                evaluation_path = run_dir / (
                    "autogluon_evaluation.json" if args.evaluator == "autogluon" else "tpot_evaluation.json"
                )
                print("SEARCH:", " ".join(str(part) for part in _build_search_command(args, paths, dataset_id, run_dir, value)))
                print("EVAL:", " ".join(str(part) for part in _build_evaluation_command(args, paths, dataset_id, run_dir, evaluation_path, recommendation_path)))
        return 0
    rows = _load_rows(suite_dir) if args.resume and not args.force else []
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {
        (str(row.get("variant")), str(row.get("dataset_id"))): row for row in rows
    }

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    for value in values:
        variant = _variant_name(args.ablation, value)
        for dataset_id in dataset_ids:
            key = (variant, str(dataset_id))
            existing = by_key.get(key)
            run_dir = suite_dir / variant / f"dataset_{dataset_id}"
            recommendation_path = run_dir / "recommendation.json"
            evaluation_name = "autogluon_evaluation.json" if args.evaluator == "autogluon" else "tpot_evaluation.json"
            evaluation_path = run_dir / evaluation_name
            if (
                args.resume
                and not args.force
                and existing
                and str(existing.get("status")) == "ok"
                and recommendation_path.exists()
                and (args.search_only or evaluation_path.exists())
            ):
                print(f"SKIP {variant} dataset={dataset_id} (already successful)")
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            recommendation_path = run_dir / "recommendation.json"
            evaluation_path = run_dir / evaluation_name
            started = time.perf_counter()
            row: Dict[str, Any] = {
                "ablation": args.ablation,
                "variant": variant,
                "parameter_value": (
                    str(value)
                    if args.ablation in {"pheromone_weight_method", "aco_update_policy", "aco_alpha_beta"}
                    else int(value)
                ),
                "n_ants": int(value if args.ablation == "num_ants" else args.n_ants),
                "total_ant_budget": (
                    int(value) if args.ablation == "total_ant_budget" else None
                ),
                "fixed_k": int(args.fixed_k),
                "fixed_h": int(args.fixed_h),
                "top_k_pheromone": int(args.top_k_pheromone),
                "aco_weight_method": (
                    str(value) if args.ablation == "pheromone_weight_method" else str(args.aco_weight_method)
                ),
                "aco_update_policy": (
                    str(value) if args.ablation == "aco_update_policy" else str(args.aco_update_policy)
                ),
                "aco_markov_order": int(args.aco_markov_order),
                "aco_lambda_smooth": float(args.aco_lambda_smooth),
                "aco_alpha": (
                    float(_ALPHA_BETA_VARIANTS[str(value)][0])
                    if args.ablation == "aco_alpha_beta" else float(args.aco_alpha)
                ),
                "aco_beta": (
                    float(_ALPHA_BETA_VARIANTS[str(value)][1])
                    if args.ablation == "aco_alpha_beta" else float(args.aco_beta)
                ),
                "n_iterations": (
                    int(math.ceil(int(value) / max(1, int(args.n_ants))))
                    if args.ablation == "total_ant_budget"
                    else int(args.n_iterations)
                ),
                "early_stop_rounds": int(args.early_stop_rounds),
                "min_improvement": float(args.min_improvement),
                "dataset_id": str(dataset_id),
                "split_seed": int(args.split_seed),
                "aco_seed": int(args.aco_seed),
                "evaluator": args.evaluator,
                "recommend_on_train_val": True,
                "outer_test_used_during_search": False,
                "status": "failed",
                "recommendation_path": str(recommendation_path),
                "evaluation_path": str(evaluation_path),
                "git_commit": _git_commit(root),
            }
            try:
                if args.force or not recommendation_path.exists():
                    search_command = _build_search_command(args, paths, str(dataset_id), run_dir, value)
                    print(f"\n[{variant}] dataset={dataset_id} search")
                    if args.verbose:
                        print(" ", " ".join(str(part) for part in search_command))
                    search_return, search_seconds = _run_process(search_command, root, env)
                else:
                    search_return, search_seconds = 0, 0.0
                    print(f"\n[{variant}] dataset={dataset_id} reuse recommendation")
                row.update({"search_return_code": int(search_return), "search_wall_clock_seconds": float(search_seconds)})
                if search_return != 0 and not recommendation_path.exists():
                    raise RuntimeError(f"run_recommend returned {search_return} and produced no recommendation")

                recommendation = _load_json(recommendation_path)
                if not recommendation:
                    raise RuntimeError(f"Could not parse recommendation: {recommendation_path}")
                row.update(selected_pipeline_operator_stats(recommendation))
                row.update(_last_history_values(run_dir))

                if args.search_only:
                    # Search-only runs intentionally do not invoke AutoGluon or
                    # TPOT.  The recommendation and ACO history are the only
                    # required artifacts for proxy/search-cost diagnostics.
                    row["evaluation_path"] = None
                    row["evaluation_return_code"] = None
                    row["evaluation_wall_clock_seconds"] = None
                    row["evaluation_status"] = "not_run"
                    row["status"] = "ok"
                    continue

                if args.force or not evaluation_path.exists() or _load_json(evaluation_path).get("status") != "ok":
                    evaluation_command = _build_evaluation_command(args, paths, str(dataset_id), run_dir, evaluation_path, recommendation_path)
                    print(f"[{variant}] dataset={dataset_id} {args.evaluator} evaluation")
                    if args.verbose:
                        print(" ", " ".join(str(part) for part in evaluation_command))
                    eval_return, eval_seconds = _run_process(evaluation_command, root, env)
                else:
                    eval_return, eval_seconds = 0, 0.0
                    print(f"[{variant}] dataset={dataset_id} reuse evaluation")
                row["evaluation_return_code"] = int(eval_return)
                row["evaluation_wall_clock_seconds"] = float(eval_seconds)
                evaluation = _load_json(evaluation_path)
                row.update(
                    {
                        "accuracy": _safe_float(evaluation.get("accuracy")),
                        "balanced_accuracy": _safe_float(evaluation.get("balanced_accuracy")),
                        "f1_macro": _safe_float(evaluation.get("f1_macro")),
                        "evaluator_validation_score": _safe_float(
                            evaluation.get("validation_score")
                        ),
                        "final_score": _safe_float(evaluation.get("score")),
                        "evaluator_reported_seconds": _safe_float(evaluation.get("total_seconds", evaluation.get("fit_seconds"))),
                        "evaluation_status": evaluation.get("status"),
                        "split_fingerprints": evaluation.get("split_fingerprints"),
                    }
                )
                if eval_return != 0 or evaluation.get("status") != "ok":
                    raise RuntimeError(str(evaluation.get("error") or f"{args.evaluator} evaluator returned {eval_return}"))
                row["status"] = "ok"
            except Exception as exc:
                row.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:4000]})
                print(f"FAILED {variant} dataset={dataset_id}: {exc}")
            finally:
                row["total_wall_clock_seconds"] = float(time.perf_counter() - started)
                by_key[key] = row
                rows = [by_key[item] for item in sorted(by_key)]
                _save_rows(rows, suite_dir)

    rows = [by_key[item] for item in sorted(by_key)]
    _save_rows(rows, suite_dir)
    _summary(suite_dir, rows, args, dataset_ids, values)
    expected_variants = {_variant_name(args.ablation, value) for value in values}
    return 0 if all(
        str(row.get("status")) == "ok"
        for row in rows
        if str(row.get("variant")) in expected_variants
        and str(row.get("dataset_id")) in set(dataset_ids)
    ) else 2
