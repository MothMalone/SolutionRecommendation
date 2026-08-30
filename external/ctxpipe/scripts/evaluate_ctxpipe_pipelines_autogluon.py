#!/usr/bin/env python3
"""Evaluate CtxPipe-suggested pipelines with AutoGluon.

This script:
1) Reads pipeline suggestions from CtxPipe `pipelines.tsv`.
2) Selects one "final" suggestion per dataset (default: best CtxPipe reward).
3) Replays the suggested preprocessing primitives.
4) Fits AutoGluon on transformed train data and reports score on transformed test data.

Split behavior matches SoluRec-style split utility in `ctxpipe/solrec_split.py`.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
import tempfile
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

# Allow running via `python scripts/...` by adding repo root to import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import comp
from ctxpipe.env.primitives.imputercat import ImputerCatPrim
from ctxpipe.env.primitives.primitive import Primitive
from ctxpipe.solrec_split import split_train_val_test

try:
    from pandas.errors import SettingWithCopyWarning
except Exception:  # pragma: no cover
    SettingWithCopyWarning = Warning  # type: ignore

# Some legacy primitive implementations assign into sliced DataFrames.
# Suppress this warning noise during replay/evaluation.
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None


@dataclass
class PipelineRow:
    tag: str
    dataset: str
    sequence_raw: str
    sequence: List[str]
    ctx_reward: Optional[float]


def _load_autogluon():
    try:
        from autogluon.features.generators import IdentityFeatureGenerator  # type: ignore
        from autogluon.tabular import TabularPredictor  # type: ignore
    except Exception as exc:
        raise RuntimeError("AutoGluon not available in environment") from exc
    return TabularPredictor, IdentityFeatureGenerator


def _detect_problem_type(y: pd.Series) -> Tuple[str, str]:
    unique_classes = y.nunique()
    if np.issubdtype(y.dtype, np.number) and unique_classes > 50:
        return "regression", "r2"
    if unique_classes == 2:
        return "binary", "accuracy"
    return "multiclass", "accuracy"


def _parse_sequence(sequence_raw: str) -> List[str]:
    # Format commonly looks like: [<ImputerMedian>, <OneHotEncoder>, ...]
    names = re.findall(r"<([^>]+)>", sequence_raw)
    if names:
        return [name.strip() for name in names if name.strip()]

    # Fallback: list of names as Python literal, e.g. ["ImputerMean", "OneHotEncoder"]
    try:
        value = ast.literal_eval(sequence_raw)
        if isinstance(value, list):
            out = [str(v).strip() for v in value if str(v).strip()]
            if out:
                return out
    except Exception:
        pass

    return []


def _parse_reward(raw: str) -> Optional[float]:
    try:
        return float(raw)
    except Exception:
        return None


def load_pipeline_rows(path: Path) -> List[PipelineRow]:
    rows: List[PipelineRow] = []
    if not path.exists():
        raise FileNotFoundError(f"pipelines.tsv not found: {path}")

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        tag, dataset, sequence_raw, reward_raw = parts[0], parts[1], parts[2], parts[3]
        sequence = _parse_sequence(sequence_raw)
        rows.append(
            PipelineRow(
                tag=tag,
                dataset=dataset,
                sequence_raw=sequence_raw,
                sequence=sequence,
                ctx_reward=_parse_reward(reward_raw),
            )
        )
    return rows


def select_final_pipelines(
    rows: Sequence[PipelineRow], strategy: str = "best_ctx_reward"
) -> Dict[str, PipelineRow]:
    by_dataset: Dict[str, List[PipelineRow]] = {}
    for row in rows:
        by_dataset.setdefault(row.dataset, []).append(row)

    selected: Dict[str, PipelineRow] = {}
    for dataset, candidates in by_dataset.items():
        if strategy == "first":
            selected[dataset] = candidates[0]
            continue
        if strategy == "latest_tag":
            def _tag_key(r: PipelineRow) -> Tuple[int, str]:
                m = re.search(r"(\d+)$", r.tag)
                return (int(m.group(1)) if m else -1, r.tag)

            selected[dataset] = sorted(candidates, key=_tag_key, reverse=True)[0]
            continue

        # default: best_ctx_reward, break ties by latest tag
        def _score_key(r: PipelineRow) -> Tuple[float, int, str]:
            reward = r.ctx_reward if r.ctx_reward is not None else float("-inf")
            m = re.search(r"(\d+)$", r.tag)
            tag_num = int(m.group(1)) if m else -1
            return (reward, tag_num, r.tag)

        selected[dataset] = sorted(candidates, key=_score_key, reverse=True)[0]

    return selected


def _build_primitive_registry() -> Dict[str, Primitive]:
    templates: List[Primitive] = []
    templates.extend(comp.imputernums)
    templates.append(ImputerCatPrim())
    templates.extend(comp.encoders)
    templates.extend(comp.fpreprocessings)
    templates.extend(comp.fengines)
    templates.extend(comp.fselections)
    templates.append(Primitive())

    registry: Dict[str, Primitive] = {}
    for prim in templates:
        registry.setdefault(prim.name, prim)
    return registry


def _instantiate_step(name: str, registry: Dict[str, Primitive]) -> Primitive:
    if name in ("blank", "Primitive", "none", "None"):
        return Primitive()
    if name not in registry:
        raise KeyError(f"Unknown primitive name: {name}")
    template = registry[name]
    # Re-instantiate by class to avoid shared fitted state between datasets.
    return template.__class__()  # type: ignore[call-arg]


def _replay_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    step_names: Sequence[str],
    registry: Dict[str, Primitive],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_x = X_train.reset_index(drop=True).copy()
    train_y = y_train.reset_index(drop=True).copy()
    test_x = X_test.reset_index(drop=True).copy()

    for name in step_names:
        step = _instantiate_step(name, registry)
        train_x, test_x = step.transform(train_x, test_x, train_y)
        if not isinstance(train_x, pd.DataFrame):
            train_x = pd.DataFrame(train_x)
        if not isinstance(test_x, pd.DataFrame):
            test_x = pd.DataFrame(test_x)
        train_x = train_x.reset_index(drop=True)
        test_x = test_x.reset_index(drop=True)

    return train_x, train_y, test_x


def _align_test_columns_to_train(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    train_aligned = X_train.copy()
    test_aligned = X_test.copy()

    # Keep a stable string-based schema for downstream model interfaces.
    train_aligned.columns = train_aligned.columns.map(str)
    test_aligned.columns = test_aligned.columns.map(str)

    train_cols = list(train_aligned.columns)
    test_cols = set(test_aligned.columns)
    missing_in_test = [c for c in train_cols if c not in test_cols]
    extra_in_test = [c for c in test_aligned.columns if c not in set(train_cols)]

    # AutoGluon requires TEST schema to match TRAIN schema exactly.
    test_aligned = test_aligned.reindex(columns=train_cols, fill_value=0)

    return train_aligned, test_aligned, len(missing_in_test), len(extra_in_test)


def _load_dataset(dataset_dir: Path) -> Tuple[pd.DataFrame, str]:
    data_csv = dataset_dir / "data.csv"
    info_json = dataset_dir / "info.json"
    if not data_csv.exists():
        raise FileNotFoundError(f"Missing data.csv: {data_csv}")
    if not info_json.exists():
        raise FileNotFoundError(f"Missing info.json: {info_json}")

    info = json.loads(info_json.read_text())
    target_column = info.get("label", "target")

    df = pd.read_csv(data_csv).infer_objects()
    if target_column not in df.columns:
        raise ValueError(f"target column '{target_column}' not found in {data_csv}")

    # Keep parity with SoluRec loader behavior for target validity.
    mask = ~pd.isna(df[target_column])
    df = df[mask].reset_index(drop=True)

    return df, target_column


def evaluate_row_with_autogluon(
    row: PipelineRow,
    dataset_prefix: Path,
    registry: Dict[str, Primitive],
    time_limit_per_model: int,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    TabularPredictor, IdentityFeatureGenerator = _load_autogluon()

    dataset_dir = dataset_prefix / row.dataset
    df, target_column = _load_dataset(dataset_dir)

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    problem_type, eval_metric = _detect_problem_type(y)

    X_train, y_train, _X_val, _y_val, X_test, y_test = split_train_val_test(
        X,
        y,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )

    X_train_p, y_train_p, X_test_p = _replay_pipeline(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        step_names=row.sequence,
        registry=registry,
    )
    X_train_p, X_test_p, n_missing_test_cols, n_extra_test_cols = _align_test_columns_to_train(
        X_train_p, X_test_p
    )
    y_test_p = y_test.reset_index(drop=True)

    if X_train_p.shape[0] == 0:
        raise RuntimeError("pipeline produced empty TRAIN data")
    if X_test_p.shape[0] == 0:
        raise RuntimeError("pipeline produced empty TEST data")
    if len(X_test_p) != len(y_test_p):
        raise RuntimeError("TEST X/y length mismatch after replay")

    train_df = X_train_p.copy()
    train_df[target_column] = y_train_p.reset_index(drop=True)
    test_df = X_test_p.copy()

    temp_dir = Path(tempfile.gettempdir()) / f"ag_ctxpipe_{uuid.uuid4().hex}"
    try:
        predictor = TabularPredictor(
            label=target_column,
            path=str(temp_dir),
            problem_type=problem_type,
            eval_metric=eval_metric,
            verbosity=2 if verbose else 0,
        )
        predictor.fit(
            train_data=train_df,
            time_limit=time_limit_per_model,
            presets="best_quality",
            # Match ACORec's downstream exactly: disable dynamic_stacking (DyStack). Under a tight
            # time_limit DyStack burns the budget on its internal sub-fit and then crashes
            # ("Time remaining: -Xs") or yields nondeterministic scores run-to-run. Disabling it
            # keeps the best_quality model set, bounds the fit, and makes baseline vs ACORec scores
            # comparable on the SAME settings.
            dynamic_stacking=False,
            feature_generator=IdentityFeatureGenerator(),
            raise_on_no_models_fitted=False,
        )
        try:
            if len(predictor.model_names()) == 0:
                raise RuntimeError("AutoGluon fitted no models")
        except Exception:
            # Some versions may not expose model_names at this point.
            pass

        preds = predictor.predict(test_df)
        if problem_type == "regression":
            ag_score = float(r2_score(y_test_p, preds))
        else:
            ag_score = float(accuracy_score(y_test_p, preds))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "dataset": row.dataset,
        "tag": row.tag,
        "target_column": target_column,
        "problem_type": problem_type,
        "eval_metric": eval_metric,
        "ctx_reward": row.ctx_reward,
        "ag_score": ag_score,
        "sequence": row.sequence,
        "sequence_raw": row.sequence_raw,
        "n_train": int(X_train_p.shape[0]),
        "n_test": int(X_test_p.shape[0]),
        "n_features_train": int(X_train_p.shape[1]),
        "n_features_test": int(X_test_p.shape[1]),
        "n_missing_test_cols_filled": int(n_missing_test_cols),
        "n_extra_test_cols_dropped": int(n_extra_test_cols),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CtxPipe suggested pipelines using AutoGluon."
    )
    parser.add_argument(
        "--pipelines-tsv",
        type=str,
        default="exp/ctxpipe-3linear/pipelines.tsv",
        help="Path to CtxPipe pipelines TSV.",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        required=True,
        help="Dataset root containing <dataset>/data.csv and <dataset>/info.json.",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="best_ctx_reward",
        choices=["best_ctx_reward", "latest_tag", "first"],
        help="How to pick one final suggested pipeline per dataset.",
    )
    parser.add_argument(
        "--dataset-ids-file",
        type=str,
        default="",
        help="Optional list of OpenML IDs to include (matches folders openml_<id>).",
    )
    parser.add_argument(
        "--time-limit-per-model",
        type=int,
        default=300,
        help="AutoGluon fit time limit per dataset in seconds.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (for split parity; val is not used in fit).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Split seed.",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=0,
        help="Optional cap on number of datasets to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="exp/ctxpipe-3linear/autogluon_eval/results.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="exp/ctxpipe-3linear/autogluon_eval/results.json",
        help="Output JSON summary file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logs.",
    )
    return parser.parse_args()


def _load_dataset_filter(path: str) -> Optional[set]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset ids file not found: {p}")
    ids = []
    for token in re.split(r"[\s,]+", p.read_text().strip()):
        if token:
            ids.append(token)
    # Convert IDs into CtxPipe folder names.
    return {f"openml_{int(x)}" for x in ids}


def main() -> None:
    args = parse_args()
    pipelines_tsv = Path(args.pipelines_tsv)
    dataset_prefix = Path(args.dataset_prefix)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)

    rows = load_pipeline_rows(pipelines_tsv)
    if not rows:
        raise SystemExit(f"No valid rows parsed from {pipelines_tsv}")

    selected = select_final_pipelines(rows, strategy=args.selection)
    dataset_filter = _load_dataset_filter(args.dataset_ids_file)
    selected_rows = list(selected.values())
    missing_requested: List[str] = []
    if dataset_filter is not None:
        selected_rows = [r for r in selected_rows if r.dataset in dataset_filter]
        present = {r.dataset for r in selected_rows}
        missing_requested = sorted(dataset_filter - present)

    selected_rows.sort(key=lambda r: r.dataset)
    if args.max_datasets > 0:
        selected_rows = selected_rows[: args.max_datasets]

    registry = _build_primitive_registry()
    records: List[Dict[str, Any]] = []

    for idx, row in enumerate(selected_rows, start=1):
        if args.verbose:
            print(f"[{idx}/{len(selected_rows)}] dataset={row.dataset} tag={row.tag}")
        try:
            record = evaluate_row_with_autogluon(
                row=row,
                dataset_prefix=dataset_prefix,
                registry=registry,
                time_limit_per_model=args.time_limit_per_model,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                split_seed=args.split_seed,
                verbose=args.verbose,
            )
            record["status"] = "ok"
            record["error"] = ""
        except Exception as exc:
            record = {
                "dataset": row.dataset,
                "tag": row.tag,
                "ctx_reward": row.ctx_reward,
                "ag_score": np.nan,
                "sequence": row.sequence,
                "sequence_raw": row.sequence_raw,
                "status": "error",
                "error": str(exc),
            }
        records.append(record)

    for ds in missing_requested:
        records.append(
            {
                "dataset": ds,
                "tag": "",
                "ctx_reward": np.nan,
                "ag_score": np.nan,
                "sequence": [],
                "sequence_raw": "",
                "status": "error",
                "error": "no_ctxpipe_pipeline_found_for_requested_dataset",
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_csv, index=False)

    summary = {
        "n_total_selected": len(selected_rows),
        "n_requested_missing_pipeline": len(missing_requested),
        "n_ok": int((results_df["status"] == "ok").sum()) if not results_df.empty else 0,
        "n_error": int((results_df["status"] == "error").sum()) if not results_df.empty else 0,
        "mean_ag_score_ok": float(results_df.loc[results_df["status"] == "ok", "ag_score"].mean())
        if not results_df.empty
        else np.nan,
        "selection": args.selection,
        "time_limit_per_model": args.time_limit_per_model,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "pipelines_tsv": str(pipelines_tsv),
        "dataset_prefix": str(dataset_prefix),
        "output_csv": str(output_csv),
    }
    output_json.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, default=str)
    )

    print(f"Saved CSV:  {output_csv}")
    print(f"Saved JSON: {output_json}")
    if missing_requested:
        print("Missing requested datasets in pipelines.tsv:")
        for ds in missing_requested:
            print(f"  - {ds}")
    print(
        f"Done. selected={summary['n_total_selected']} ok={summary['n_ok']} error={summary['n_error']}"
    )


if __name__ == "__main__":
    main()
