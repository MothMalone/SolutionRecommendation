#!/usr/bin/env python3
"""AutoDP arm of the ACORec-vs-AutoDP comparison, scored by estimator-only TPOT — shardable.

The TPOT analogue of ``scripts/adp_bench.py``: one command, one JSONL, resumable, shardable. Per
dataset it (1) runs AutoDP's search in the pinned ``.venv-autodp`` env via
``scripts/run_autodatapre.py`` (persisted under ``--prepared-root`` so it is done once), then
(2) scores the prepared frame with estimator-only TPOT via ``scripts/evaluate_autodp_tpot.py`` in
the TPOT env. The two steps CANNOT share an interpreter (numpy<1.24 vs numpy>=1.25), so this driver
runs in the base environment and shells out to both.

Emits ``adp_bench``-compatible compact records (``evaluator: "tpot"``), so
``python scripts/adp_bench.py --summarize 'arms_*_tpot_*.jsonl'`` produces the table with a
``fair/tpot`` column next to ``fair/h2o`` and ``fair``.

Run one shard:
    python scripts/run_autodp_tpot_arm.py --shard 1/5 \
        --data-dir /kaggle/working/eval_all \
        --prepared-root /kaggle/working/adp_prepared \
        --operator-space ours --adp-meta-corpus data/adp_ourops_corpus \
        --adp-python /tmp/adpenv/bin/python --tpot-libs /tmp/tpotlibs \
        --out /kaggle/working/arms_1-adp-ourops_tpot_1of5.jsonl

Summarize:
    python scripts/adp_bench.py --summarize '/kaggle/working/arms_1-adp-ourops_tpot_*.jsonl'
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

import pandas as pd  # noqa: E402

from automl_aco.eval_ids import EVAL_IDS  # noqa: E402


def _read_done(out_path: str) -> set:
    """(dataset_id, mode) pairs that produced a REAL score. Failed / errored rows are retried on
    rerun (the AutoDP search is cached under --prepared-root, so the retry is cheap)."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("score") is not None or rec.get("status") == "ok":
                done.add((str(rec.get("dataset_id")), rec.get("mode")))
    return done


def _append(out_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _shard(ids, spec: str):
    part, total = (int(x) for x in spec.split("/"))
    if not (1 <= part <= total):
        raise SystemExit(f"--shard {spec}: part must be within 1..{total}")
    return [d for i, d in enumerate(ids) if i % total == (part - 1)]


def _export_dataset(did: str, dest_dir: str, local_folder, verbose: bool) -> str:
    cmd = [sys.executable, os.path.join(_HERE, "export_eval_datasets.py"),
           "--ids", str(did), "--out-dir", dest_dir]
    if local_folder:
        cmd += ["--openml-local-folder", local_folder]
    if verbose:
        cmd.append("--verbose")
    subprocess.run(cmd, check=True)
    path = os.path.join(dest_dir, f"{did}.csv")
    if not os.path.exists(path):
        raise RuntimeError(f"export produced no CSV for {did}")
    return path


def _run_autodp_search(args, csv_path: str, did: str, mode: str) -> tuple[str, int]:
    """AutoDP search in .venv-autodp, persisted under --prepared-root. Returns (prepared_dir, rc)."""
    space_tag = mode if args.operator_space == "theirs" else f"{mode}_ourops"
    prepared_dir = os.path.join(args.prepared_root, space_tag, f"dataset_{did}")
    if args.skip_search or os.path.exists(os.path.join(prepared_dir, "prepared.csv")):
        return prepared_dir, 0

    cmd = [args.adp_python, os.path.join(_HERE, "run_autodatapre.py"),
           "--dataset-csv", csv_path, "--dataset-id", str(did), "--mode", mode,
           "--cap-seconds", str(args.cap_seconds), "--seed", str(args.seed),
           "--out-dir", args.prepared_root]
    if args.operator_space == "ours":
        cmd += ["--operator-space", "ours"]
        if args.adp_meta_corpus:
            cmd += ["--adp-meta-corpus", str(args.adp_meta_corpus)]
        if args.adp_family_order and args.adp_family_order != "prior":
            cmd += ["--adp-family-order", str(args.adp_family_order)]
    env = dict(os.environ)
    # Must NOT carry the base env's packages -- they outrank the venv site-packages and drag
    # numpy>=1.26 into AutoDP, which needs numpy<1.24.
    env["PYTHONPATH"] = os.path.join(_REPO, "src")
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(cmd, env=env)
    return prepared_dir, proc.returncode


def _run_tpot_score(args, csv_path: str, prepared_dir: str, did: str, out_json: str) -> None:
    """Score the prepared frame with estimator-only TPOT, in the TPOT env."""
    cmd = [args.tpot_python, os.path.join(_HERE, "evaluate_autodp_tpot.py"),
           "--dataset-csv", csv_path, "--prepared-dir", prepared_dir,
           "--dataset-id", str(did), "--output-json", out_json, "--force",
           "--split-seed", str(args.split_seed), "--tpot-seed", str(args.tpot_seed),
           "--max-time-mins", str(args.max_time_mins),
           "--max-eval-time-mins", str(args.max_eval_time_mins),
           "--n-jobs", str(args.n_jobs), "--memory-limit", args.memory_limit,
           "--population-size", str(args.population_size),
           "--max-cv-folds", str(args.max_cv_folds), "--verbose", str(args.verbose)]
    env = dict(os.environ)
    pp = [p for p in (args.tpot_libs, os.path.join(_REPO, "src")) if p]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(cmd, env=env)
    if not os.path.exists(out_json):
        raise RuntimeError(f"TPOT scorer produced no output (rc={proc.returncode})")


def _compact_record(did: str, mode: str, res: dict, n_rows: int, n_features: int,
                    t_start: float) -> dict:
    pipe = res.get("autodp_pipeline") or []
    return {
        "dataset_id": str(did),
        "mode": mode,
        "status": res.get("autodp_status"),
        "score": res.get("score_full"),
        "score_kept": res.get("score_kept"),
        "metric": res.get("eval_metric"),
        "problem_type": res.get("problem_type"),
        "pipeline": res.get("autodp_pipeline"),
        "evaluator": "tpot",
        "evaluator_meta": {"tpot_estimator": res.get("tpot_estimator"),
                           "selected_estimator": res.get("selected_estimator"),
                           "cv_folds": res.get("cv_folds"),
                           "tpot_knobs": res.get("tpot_knobs")},
        "autodp_seconds": res.get("autodp_search_seconds"),
        "autogluon_seconds": res.get("eval_seconds"),
        "eval_seconds": res.get("eval_seconds"),
        "total_seconds": round(time.time() - t_start, 1),
        "test_coverage": res.get("test_coverage"),
        "n_rows": n_rows,
        "n_features": n_features,
        "n_features_scored": res.get("n_features_scored"),
        "autodp_converged": res.get("autodp_converged"),
        "autodp_hit_cap": res.get("autodp_hit_cap"),
        "search_split": res.get("search_split"),
        "internal_scorer_seed": res.get("internal_scorer_seed"),
        "leakfree_cbe": res.get("leakfree_cbe"),
        "search_iteration_exceptions": res.get("search_iteration_exceptions"),
        "search_iteration_exception_kinds": res.get("search_iteration_exception_kinds"),
        "residual_encoding_applied": res.get("residual_encoding_applied"),
        "empty_pipeline": len(pipe) <= 1,
        "dead_search": bool(res.get("dead_search", False)),
        "dead_search_none_profit_evals": res.get("dead_search_none_profit_evals"),
        "dataset_csv": res.get("dataset_csv"),
    }


def run(args) -> None:
    ids = ([t for t in " ".join(args.ids).replace(",", " ").split() if t]
           if args.ids else list(EVAL_IDS))
    if args.shard:
        ids = _shard(ids, args.shard)
    mode = "native" if args.protocol == "native" else "fair"

    done = _read_done(args.out)
    todo = [d for d in ids if (str(d), mode) not in done]
    print(f"[plan] {len(ids)} dataset(s) [{mode}]; {len(done)} already in {args.out}; "
          f"{len(todo)} to run", flush=True)

    for did in todo:
        t_start = time.time()
        scratch = tempfile.mkdtemp(prefix=f"adptpot_{did}_")
        try:
            csv_path = os.path.join(args.data_dir, f"{did}.csv") if args.data_dir else None
            if not (csv_path and os.path.exists(csv_path)):
                csv_path = _export_dataset(did, args.data_dir or scratch,
                                           args.openml_local_folder, args.verbose)
            shape = pd.read_csv(csv_path, nrows=1).shape[1]
            n_rows = sum(1 for _ in open(csv_path)) - 1
            print(f"\n=== {did} [{mode}] {n_rows} rows x {shape - 1} features "
                  f"({time.strftime('%H:%M:%S')}) ===", flush=True)

            prepared_dir, rc = _run_autodp_search(args, csv_path, did, mode)
            if rc != 0 or not os.path.exists(os.path.join(prepared_dir, "prepared.csv")):
                _append(args.out, {
                    "dataset_id": str(did), "mode": mode, "status": "autodp_error",
                    "score": None, "evaluator": "tpot", "pipeline": None,
                    "autodp_seconds": round(time.time() - t_start, 1), "autogluon_seconds": None,
                    "total_seconds": round(time.time() - t_start, 1),
                    "error": f"AutoDP search rc={rc}, no prepared.csv at {prepared_dir}",
                })
                print(f"[error] {did}: AutoDP search produced no prepared frame (rc={rc})",
                      flush=True)
                continue

            out_json = os.path.join(scratch, "tpot_evaluation.json")
            _run_tpot_score(args, csv_path, prepared_dir, did, out_json)
            res = json.loads(open(out_json).read())

            if res.get("status") != "ok":
                _append(args.out, {
                    "dataset_id": str(did), "mode": mode,
                    "status": res.get("status", "error"), "score": None, "evaluator": "tpot",
                    "pipeline": None, "autodp_seconds": None, "autogluon_seconds": None,
                    "total_seconds": round(time.time() - t_start, 1),
                    "error": f"{res.get('error_type')}: {res.get('error')}",
                    "dataset_csv": res.get("dataset_csv"),
                })
                print(f"[fail] {did}: {res.get('error_type')}: {res.get('error')}", flush=True)
                continue

            record = _compact_record(did, mode, res, n_rows, shape - 1, t_start)
            _append(args.out, record)
            n_exc = record.get("search_iteration_exceptions") or 0
            exc_flag = f" !! {n_exc} SEARCH-ITERATION EXCEPTIONS" if n_exc else ""
            dead_flag = " !! DEAD SEARCH -> raw frame" if record["dead_search"] else ""
            cov_flag = ("" if (record.get("test_coverage") or 1) >= 0.999
                        else f" !! coverage={record['test_coverage']:.2f}")
            print(f"[ok] {did} [{mode}] [tpot] {record['metric']}={record['score']:.4f} "
                  f"kept={record['score_kept']:.4f} pipeline={record['pipeline']} "
                  f"adp={record['autodp_seconds']}s eval={record['eval_seconds']}s "
                  f"total={record['total_seconds']}s{exc_flag}{dead_flag}{cov_flag}", flush=True)
        except Exception as exc:
            _append(args.out, {
                "dataset_id": str(did), "mode": mode, "status": "error", "score": None,
                "evaluator": "tpot", "pipeline": None, "autodp_seconds": None,
                "autogluon_seconds": None, "total_seconds": round(time.time() - t_start, 1),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[error] {did} [{mode}]: {exc}", flush=True)
            if args.verbose:
                traceback.print_exc()
        finally:
            import shutil
            shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n[done] results appended to {args.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*", default=None, help="dataset ids (default: all 30 EVAL_IDS)")
    ap.add_argument("--shard", default=None, metavar="I/N", help="round-robin I-th of N shards")
    ap.add_argument("--protocol", choices=["native", "leakfree"], default="leakfree",
                    help="leakfree (default) -> AutoDP --mode fair; native -> --mode native")
    ap.add_argument("--data-dir", default=None,
                    help="dir of exported <id>.csv (the SAME files the ACORec arm reads)")
    ap.add_argument("--openml-local-folder", default=None,
                    help="fallback OpenML mount for exporting a missing <id>.csv")
    ap.add_argument("--prepared-root", default="outputs/adp_prepared",
                    help="persist AutoDP search output here; reused across evaluators / reruns")
    ap.add_argument("--skip-search", action="store_true",
                    help="assume --prepared-root already holds every prepared.csv; score only")
    ap.add_argument("--operator-space", choices=["theirs", "ours"], default="ours",
                    help="ours (default) = AutoDP MCTS over ACORec's operator space (arm 1-adp-ourops)")
    ap.add_argument("--adp-meta-corpus", default=None,
                    help="retrained meta-learner corpus, e.g. data/adp_ourops_corpus")
    ap.add_argument("--adp-family-order", choices=["prior", "all"], default="prior")
    ap.add_argument("--adp-python",
                    default=os.path.join(_REPO, ".venv-autodp", "bin", "python"),
                    help="interpreter for the pinned AutoDP env")
    ap.add_argument("--tpot-python", default=sys.executable,
                    help="interpreter for the TPOT env (default: this python)")
    ap.add_argument("--tpot-libs", default=None,
                    help="dir to prepend to PYTHONPATH for the TPOT step, e.g. /tmp/tpotlibs")
    ap.add_argument("--out", default="arms_1-adp-ourops_tpot.jsonl", help="output JSONL")
    ap.add_argument("--cap-seconds", type=float, default=5400.0,
                    help="wall-clock watchdog for the AutoDP search per dataset")
    ap.add_argument("--seed", type=int, default=42, help="AutoDP search seed")
    # TPOT knobs -- defaults come from scripts/_tpot_eval.py via evaluate_autodp_tpot.py
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--tpot-seed", type=int, default=1)
    ap.add_argument("--max-time-mins", type=int, default=5)
    ap.add_argument("--max-eval-time-mins", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--memory-limit", default="5GB")
    ap.add_argument("--population-size", type=int, default=20)
    ap.add_argument("--max-cv-folds", type=int, default=5)
    ap.add_argument("--verbose", type=int, default=2)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
