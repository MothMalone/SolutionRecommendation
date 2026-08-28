#!/usr/bin/env python3
"""Shardable AutoDP-on-ACORec-operators benchmark with estimator-only TPOT final evaluation.

AutoDP's MCTS runs in its pinned legacy environment. Its winning ACORec-space preprocessing chain
is then evaluated by TPOT 1.1.0 on the shared outer 60/20/20 split: AutoDP search/preprocessing
fit and TPOT CV use the 60% training partition; validation is unused; final scoring is once on the
20% test partition. Results are resumable JSONL records, one per dataset.

Example:
    python scripts/adp_bench_tpot.py --ids 1066 --adp-meta-corpus data/adp_ourops_corpus \
        --out outputs/adp_ourops_tpot.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from adp_bench import _export_dataset, _run_autodp, _shard  # noqa: E402
from eval_autodatapre_tpot import score_prepared  # noqa: E402
from automl_aco.eval_ids import EVAL_IDS  # noqa: E402


MODE = "tpot_leakfree"


def _read_successful(out_path: str) -> set[tuple[str, str]]:
    """Return only records that should suppress a rerun.

    The benchmark appends failures as durable records so they remain visible in the
    output.  They must not, however, prevent a later retry after a transient Kaggle
    or dependency failure.
    """
    successful = set()
    path = Path(out_path)
    if not path.exists():
        return successful
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if record.get("status") in {"ok", "apply_failed_returned_raw"}:
                successful.add((str(record.get("dataset_id")), str(record.get("mode"))))
    return successful


def _append(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _planned_ids(args) -> list[str]:
    ids = [token for token in " ".join(args.ids).replace(",", " ").split() if token]
    if not ids:
        ids = list(EVAL_IDS)
    return _shard(ids, args.shard) if args.shard else ids


def _validate_corpus(path: Path) -> None:
    for name in ("Metafeature.csv", "label.csv"):
        if not (path / name).is_file():
            raise SystemExit(f"--adp-meta-corpus {path} is missing {name}")


def run(args) -> None:
    corpus = Path(args.adp_meta_corpus).resolve()
    _validate_corpus(corpus)
    out_path = Path(args.out).resolve()
    ids = _planned_ids(args)
    done = _read_successful(str(out_path))
    todo = [dataset_id for dataset_id in ids if (str(dataset_id), MODE) not in done]
    print(
        f"[plan] {len(ids)} dataset(s); {len(done)} completed records in {out_path}; "
        f"{len(todo)} AutoDP-ACORec-space + TPOT run(s) pending"
    )
    print(f"[corpus] {corpus}")

    for ordinal, dataset_id in enumerate(todo, 1):
        started = time.time()
        scratch = Path(tempfile.mkdtemp(prefix=f"adp_ourops_tpot_{dataset_id}_"))
        model = None
        try:
            csv_path = Path(args.data_dir) / f"{dataset_id}.csv" if args.data_dir else None
            if csv_path is None or not csv_path.is_file():
                csv_path = Path(_export_dataset(
                    str(dataset_id), str(Path(args.data_dir) if args.data_dir else scratch),
                    args.openml_local_folder, args.verbose,
                ))
            print(f"\n[{ordinal}/{len(todo)}] dataset {dataset_id}: AutoDP MCTS on ACORec operators", flush=True)
            prepared_dir, return_code = _run_autodp(
                args.adp_python,
                str(csv_path),
                str(dataset_id),
                MODE,
                str(scratch),
                args.cap_seconds,
                args.runtime,
                args.seed,
                operator_space="ours",
                meta_corpus=str(corpus),
            )
            prepared_dir = Path(prepared_dir)
            failed_path = prepared_dir / "autodp_failed.json"
            if return_code != 0 or failed_path.exists():
                detail = f"AutoDP exited {return_code}"
                if failed_path.exists():
                    detail = json.loads(failed_path.read_text(encoding="utf-8")).get("detail", detail)
                record = {
                    "dataset_id": str(dataset_id), "mode": MODE, "status": "autodp_timeout" if failed_path.exists() else "autodp_error",
                    "score": None, "score_kept": None, "pipeline": None,
                    "autodp_seconds": round(time.time() - started, 2), "tpot_seconds": None,
                    "total_seconds": round(time.time() - started, 2), "detail": detail,
                    "adp_meta_corpus": str(corpus),
                }
            else:
                result, model = score_prepared(
                    str(csv_path), str(prepared_dir), split_seed=args.seed, tpot_seed=args.tpot_seed,
                    max_time_mins=args.max_time_mins, max_eval_time_mins=args.max_eval_time_mins,
                    n_jobs=args.n_jobs, memory_limit=args.memory_limit,
                    population_size=args.population_size, max_cv_folds=args.max_cv_folds,
                    verbose=args.verbose_tpot,
                )
                record = {
                    "dataset_id": str(dataset_id), "mode": MODE, "status": result["autodp_status"],
                    "score": result["score_full"], "score_kept": result["score_kept"],
                    "metric": result["eval_metric"], "problem_type": result["problem_type"],
                    "pipeline": result["autodp_pipeline"], "autodp_seconds": result["autodp_search_seconds"],
                    "tpot_seconds": result["tpot_eval_seconds"], "total_seconds": round(time.time() - started, 2),
                    "test_coverage": result["test_coverage"], "n_features_scored": result["n_features_scored"],
                    "autodp_converged": result["autodp_converged"], "autodp_hit_cap": result["autodp_hit_cap"],
                    "tpot_pipeline": result["selected_estimator"], "tpot_cv_folds": result["cv_folds"],
                    "adp_meta_corpus": str(corpus),
                }
                print(
                    f"[ok] {dataset_id}: {record['metric']}={record['score']:.4f}, "
                    f"AutoDP={record['autodp_seconds']}s TPOT={record['tpot_seconds']}s",
                    flush=True,
                )
        except Exception as exc:
            record = {
                "dataset_id": str(dataset_id), "mode": MODE, "status": "error",
                "score": None, "score_kept": None, "pipeline": None,
                "autodp_seconds": None, "tpot_seconds": None,
                "total_seconds": round(time.time() - started, 2),
                "error": f"{type(exc).__name__}: {exc}", "adp_meta_corpus": str(corpus),
            }
            print(f"[error] {dataset_id}: {record['error']}", flush=True)
            if args.verbose:
                traceback.print_exc()
        finally:
            _append(out_path, record)
            del model
            gc.collect()
            shutil.rmtree(scratch, ignore_errors=True)

    print(f"[done] results appended to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", default=None, help="dataset ids; default: all canonical eval ids")
    parser.add_argument("--shard", default="", metavar="I/N", help="round-robin shard, e.g. 2/5")
    parser.add_argument("--out", default="outputs/adp_ourops_tpot.jsonl")
    parser.add_argument("--adp-meta-corpus", required=True, help="directory containing Metafeature.csv and label.csv")
    parser.add_argument("--adp-python", default=str(ROOT / ".venv-autodp" / "bin" / "python"))
    parser.add_argument("--data-dir", default=None, help="reuse exported <id>.csv files here")
    parser.add_argument("--openml-local-folder", default=None)
    parser.add_argument("--cap-seconds", type=float, default=1800.0, help="AutoDP wall-clock watchdog per dataset")
    parser.add_argument("--runtime", type=float, default=None, help="explicit AutoDP MCTS time budget; default: convergence")
    parser.add_argument("--seed", type=int, default=42, help="shared outer-split seed")
    parser.add_argument("--tpot-seed", type=int, default=1)
    parser.add_argument("--max-time-mins", type=int, default=5)
    parser.add_argument("--max-eval-time-mins", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--memory-limit", default="5GB")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--max-cv-folds", type=int, default=5)
    parser.add_argument("--verbose-tpot", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not Path(args.adp_python).is_file():
        raise SystemExit(
            f"No AutoDP Python at {args.adp_python}. Build it with bash scripts/setup_autodp_env.sh "
            "or pass --adp-python."
        )
    run(args)


if __name__ == "__main__":
    main()
