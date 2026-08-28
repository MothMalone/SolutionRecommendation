#!/usr/bin/env python3
"""One command, one output file: the AutoDP baseline for a set of datasets.

Runs the whole thing per dataset -- export, AutoDP search, AutoGluon scoring -- and appends ONE
line per (dataset, mode) to a single JSONL file. Every intermediate artefact goes to a scratch
directory that is deleted afterwards, so nothing accumulates in the working directory and the repo
does not have to be copied anywhere writable.

Each output line carries what you asked for, plus what is needed to defend the number:
    dataset_id, mode, score (AutoGluon test accuracy / R2), pipeline (the operators AutoDP chose),
    autodp_seconds, autogluon_seconds, total_seconds, status, coverage, rows/features.

Runs in the MAIN environment (the one with AutoGluon) and shells out to the pinned AutoDP
environment for the search step -- they cannot share an interpreter, since AutoDP needs
numpy<1.24 / pandas<2.0.

Resumable and shardable. It skips any (dataset, mode) already present in the output file, so
re-running after a session timeout continues, and separate shards can run concurrently in
different Kaggle sessions as long as each writes its OWN file.

Run one dataset:
    python scripts/adp_bench.py --ids 248 --out /kaggle/working/autodp_248.jsonl

Run a shard:
    python scripts/adp_bench.py --shard 1/4 --out /kaggle/working/autodp_s1.jsonl

Merge shards into a table:
    python scripts/adp_bench.py --summarize /kaggle/working/autodp_*.jsonl --out-md summary.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

import pandas as pd

from automl_aco.eval_ids import EVAL_IDS


# --------------------------------------------------------------------------------------- helpers


def _read_done(out_path: str) -> set:
    """(dataset_id, mode) pairs already recorded, so a rerun continues instead of repeating."""
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
            done.add((str(rec.get("dataset_id")), rec.get("mode")))
    return done


def _append(out_path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _shard(ids, spec: str):
    """'2/4' -> the 2nd of 4 contiguous, size-balanced shards (1-indexed)."""
    part, total = (int(x) for x in spec.split("/"))
    if not (1 <= part <= total):
        raise SystemExit(f"--shard {spec}: part must be within 1..{total}")
    return [d for i, d in enumerate(ids) if i % total == (part - 1)]


def _export_dataset(did: str, dest_dir: str, local_folder, verbose: bool) -> str:
    """Materialise <id>.csv exactly as run_recommend.py builds the frame in memory."""
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


def _run_autodp(adp_python: str, csv_path: str, did: str, mode: str, scratch: str,
                cap_seconds: float, runtime, seed: int, operator_space: str = "theirs",
                 meta_corpus=None) -> str:
    """Run the AutoDP search in its own environment. Returns the dir holding prepared.csv."""
    cmd = [adp_python, os.path.join(_HERE, "run_autodatapre.py"),
           "--dataset-csv", csv_path, "--dataset-id", str(did), "--mode", mode,
           "--cap-seconds", str(cap_seconds), "--seed", str(seed), "--out-dir", scratch]
    if operator_space == "ours":
        # AutoDP's MCTS searching ACORec's operator space (scripts/autodp_our_space.py).
        cmd += ["--operator-space", "ours"]
        if meta_corpus:
            cmd += ["--adp-meta-corpus", str(meta_corpus)]
    if runtime is not None:
        cmd += ["--runtime", str(runtime)]
    env = dict(os.environ)
    # PYTHONPATH must NOT carry the main env's packages: it outranks the venv's own site-packages
    # and would drag numpy>=1.26 / pandas 2.x into AutoDP, which needs numpy<1.24 / pandas<2.0.
    env["PYTHONPATH"] = os.path.join(_REPO, "src")
    env["MPLBACKEND"] = "Agg"
    # Not check=True: a wall-clock kill is an expected outcome that the caller records as a
    # timeout row, not an exception to surface as a stack trace.
    proc = subprocess.run(cmd, env=env)
    # run_autodatapre.py namespaces the output directory by operator space so that
    # AutoDP's native and ACORec-space artifacts cannot collide.
    space_tag = mode if operator_space == "theirs" else f"{mode}_ourops"
    return os.path.join(scratch, space_tag, f"dataset_{did}"), proc.returncode


# ------------------------------------------------------------------------------------------ run


def import_dir(args) -> int:
    """Adopt results from an earlier run_autodatapre_all.sh tree into the JSONL file.

    Lets a batch that already spent hours under outputs/autodp/<mode>/dataset_<id>/ carry over
    instead of being recomputed. Existing (dataset, mode) pairs in the output file win.
    """
    done = _read_done(args.out)
    added = 0
    for p in sorted(glob.glob(os.path.join(args.import_dir, "*", "dataset_*", "autodp_eval.json"))):
        with open(p) as f:
            r = json.load(f)
        key = (str(r.get("dataset_id")), r.get("mode"))
        if key in done:
            continue
        _append(args.out, {
            "dataset_id": key[0], "mode": key[1], "status": r.get("autodp_status"),
            "score": r.get("score_full"), "score_kept": r.get("score_kept"),
            "metric": r.get("eval_metric"), "problem_type": r.get("problem_type"),
            "pipeline": r.get("autodp_pipeline"),
            "autodp_seconds": r.get("autodp_search_seconds"),
            "autogluon_seconds": r.get("autogluon_eval_seconds"),
            "total_seconds": r.get("total_seconds"),
            "test_coverage": r.get("test_coverage"),
            "n_features_scored": r.get("n_features_scored"),
            "autodp_converged": r.get("autodp_converged"),
            "autodp_hit_cap": r.get("autodp_hit_cap"),
            "imported_from": p,
        })
        done.add(key)
        added += 1
    for p in sorted(glob.glob(os.path.join(args.import_dir, "*", "dataset_*", "autodp_failed.json"))):
        with open(p) as f:
            r = json.load(f)
        key = (str(r.get("dataset_id")), r.get("mode"))
        if key in done:
            continue
        _append(args.out, {
            "dataset_id": key[0], "mode": key[1], "status": "timeout", "score": None,
            "pipeline": None, "autodp_seconds": None, "autogluon_seconds": None,
            "total_seconds": None, "detail": r.get("detail"),
            "cap_seconds": r.get("cap_seconds"), "imported_from": p,
        })
        done.add(key)
        added += 1
    print(f"[import] adopted {added} result(s) from {args.import_dir} into {args.out}")
    return added


def run(args) -> None:
    from eval_autodatapre import score_prepared  # imported late: needs AutoGluon in this env

    ids = [t for t in " ".join(args.ids).replace(",", " ").split() if t] if args.ids else list(EVAL_IDS)
    if args.shard:
        ids = _shard(ids, args.shard)
    modes = [m for m in " ".join(args.modes).replace(",", " ").split() if m]

    done = _read_done(args.out)
    todo = [(d, m) for d in ids for m in modes if (str(d), m) not in done]
    print(f"[plan] {len(ids)} dataset(s) x {len(modes)} mode(s); {len(done)} already in {args.out}; "
          f"{len(todo)} to run", flush=True)
    if not todo:
        return

    for did, mode in todo:
        t_start = time.time()
        scratch = tempfile.mkdtemp(prefix=f"adp_{did}_{mode}_")
        try:
            csv_path = os.path.join(args.data_dir, f"{did}.csv") if args.data_dir else None
            if not (csv_path and os.path.exists(csv_path)):
                csv_path = _export_dataset(did, args.data_dir or scratch, args.openml_local_folder,
                                           args.verbose)
            shape = pd.read_csv(csv_path, nrows=1).shape[1]
            n_rows = sum(1 for _ in open(csv_path)) - 1
            print(f"\n=== {did} [{mode}] {n_rows} rows x {shape - 1} features "
                  f"({time.strftime('%H:%M:%S')}) ===", flush=True)

            prepared_dir, rc = _run_autodp(args.adp_python, csv_path, did, mode, scratch,
                                           args.cap_seconds, args.runtime, args.seed,
                                           args.operator_space, args.adp_meta_corpus)
            failed = os.path.join(prepared_dir, "autodp_failed.json")
            if os.path.exists(failed) or rc != 0:
                info = {}
                if os.path.exists(failed):
                    with open(failed) as f:
                        info = json.load(f)
                _append(args.out, {
                    "dataset_id": str(did), "mode": mode,
                    "status": "timeout" if info else "autodp_error",
                    "score": None, "pipeline": None,
                    "autodp_seconds": round(time.time() - t_start, 1), "autogluon_seconds": None,
                    "total_seconds": round(time.time() - t_start, 1),
                    "n_rows": n_rows, "n_features": shape - 1,
                    "detail": info.get("detail") or f"AutoDP exited {rc}; see the log above",
                    "cap_seconds": args.cap_seconds,
                })
                print(f"[{'timeout' if info else 'autodp_error'}] {did} [{mode}]: "
                      f"no prepared dataset produced", flush=True)
                continue

            res = score_prepared(csv_path, prepared_dir, time_limit=args.time_limit,
                                 autogluon_profile=args.autogluon_profile, seed=args.seed,
                                 verbose=args.verbose)
            record = {
                "dataset_id": str(did),
                "mode": mode,
                "status": res["autodp_status"],
                "score": res["score_full"],           # comparable to ours: deleted test rows count as wrong
                "score_kept": res["score_kept"],       # generous: only the rows AutoDP kept
                "metric": res["eval_metric"],
                "problem_type": res["problem_type"],
                "pipeline": res["autodp_pipeline"],
                "autodp_seconds": res["autodp_search_seconds"],
                "autogluon_seconds": res["autogluon_eval_seconds"],
                "total_seconds": round(time.time() - t_start, 1),
                "test_coverage": res["test_coverage"],
                "n_rows": n_rows,
                "n_features": shape - 1,
                "n_features_scored": res["n_features_scored"],
                "autodp_converged": res["autodp_converged"],
                "autodp_hit_cap": res["autodp_hit_cap"],
            }
            _append(args.out, record)
            print(f"[ok] {did} [{mode}] {record['metric']}={record['score']:.4f} "
                  f"pipeline={record['pipeline']} "
                  f"adp={record['autodp_seconds']}s ag={record['autogluon_seconds']}s "
                  f"total={record['total_seconds']}s", flush=True)
        except Exception as exc:
            _append(args.out, {
                "dataset_id": str(did), "mode": mode, "status": "error", "score": None,
                "pipeline": None, "autodp_seconds": None, "autogluon_seconds": None,
                "total_seconds": round(time.time() - t_start, 1), "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[error] {did} [{mode}]: {exc}", flush=True)
            if args.verbose:
                traceback.print_exc()
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n[done] results appended to {args.out}", flush=True)


# ------------------------------------------------------------------------------------ summarize


def summarize(args) -> None:
    paths = []
    for pat in args.summarize:
        paths.extend(sorted(glob.glob(pat)))
    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no records found in {paths}")

    modes = sorted({r["mode"] for r in records}, key=lambda m: {"native": 0, "fair": 1}.get(m, 2))
    by_id: dict = {}
    for r in records:
        by_id.setdefault(str(r["dataset_id"]), {})[r["mode"]] = r

    lines = ["# AutoDP (autodatapre 0.1.12) — score, pipeline, runtime\n"]
    lines.append("Scores are AutoGluon test performance (accuracy / R²) on the seed-42 0.6/0.2/0.2 "
                 "split, fit on train+val (80%), predicting the same 20% test rows — the identical "
                 "protocol our own numbers use. `native` = AutoDP's published API, whose search sees "
                 "the full dataset including our test rows; `fair` = its search restricted to our "
                 "80% train+val. An empty pipeline means AutoDP selected no preprocessing operator, "
                 "so its output is the raw data.\n")
    header = ["Dataset", "task"]
    for m in modes:
        header += [f"{m} score", f"{m} pipeline", f"{m} time"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    sums: dict = {}
    counts: dict = {}
    tsums: dict = {}
    for did in sorted(by_id, key=lambda s: (len(s), s)):
        any_r = next(iter(by_id[did].values()))
        row = [did, str(any_r.get("problem_type", "—")).replace("multiclass", "multi").replace("binary", "bin")]
        for m in modes:
            r = by_id[did].get(m)
            if r is None:
                row += ["—", "—", "—"]
                continue
            secs = r.get("total_seconds")
            secs_txt = "—" if secs is None else f"{float(secs):.0f}s"
            if r.get("score") is None:
                row += [r.get("status", "fail"), "—", secs_txt]
                continue
            pipe = r.get("pipeline") or []
            ops = ", ".join(pipe[1:]) if len(pipe) > 1 else "(none)"
            row += [f"{r['score']:.4f}", ops, secs_txt]
            sums[m] = sums.get(m, 0.0) + r["score"]
            counts[m] = counts.get(m, 0) + 1
            tsums[m] = tsums.get(m, 0.0) + float(secs or 0)
        lines.append("| " + " | ".join(row) + " |")

    mean_row = ["**mean**", ""]
    for m in modes:
        mean_row += [f"**{sums[m] / counts[m]:.4f}**" if counts.get(m) else "—", "",
                     f"**{tsums[m] / counts[m]:.0f}s**" if counts.get(m) else "—"]
    lines.append("| " + " | ".join(mean_row) + " |")

    bad = [r for r in records if r.get("score") is None]
    if bad:
        lines.append("\n**No score** (excluded from the means above): "
                     + ", ".join(f"{r['dataset_id']} ({r['mode']}, {r.get('status')})" for r in bad))
    low_cov = [r for r in records if (r.get("test_coverage") or 1.0) < 0.999]
    if low_cov:
        lines.append("\n**AutoDP deleted test rows** here; `score` counts those rows as wrong: "
                     + ", ".join(f"{r['dataset_id']} ({r['mode']}, cov={r['test_coverage']:.2f})"
                                 for r in low_cov))
    lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write(text)
        print(f"Wrote {args.out_md}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*", default=None, help="dataset ids (default: all 23 eval ids)")
    ap.add_argument("--shard", default=None, metavar="I/N", help="run the I-th of N shards, e.g. 2/4")
    ap.add_argument("--modes", nargs="*", default=["native", "fair"])
    ap.add_argument("--out", default="autodp_results.jsonl", help="the single output file (JSONL)")
    ap.add_argument("--cap-seconds", type=float, default=900.0,
                    help="wall-clock watchdog per dataset; AutoDP otherwise runs to its own convergence")
    ap.add_argument("--runtime", type=float, default=None,
                    help="explicit AutoDP runTime budget; omit to use its run-to-convergence default")
    ap.add_argument("--time-limit", type=int, default=300, help="AutoGluon time_limit")
    ap.add_argument("--autogluon-profile", default="best_quality",
                    choices=["best_quality", "medium_quality", "local_rf_xt"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adp-python", default=os.path.join(_REPO, ".venv-autodp", "bin", "python"),
                    help="interpreter of the pinned AutoDP env (see scripts/setup_autodp_env.sh)")
    ap.add_argument("--data-dir", default=None,
                    help="keep exported <id>.csv here and reuse them; default: scratch, deleted after")
    ap.add_argument("--openml-local-folder", default=None,
                    help="folder of <id>.csv to fall back on when OpenML is unreachable")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--import-dir", default=None,
                    help="adopt results from an earlier run_autodatapre_all.sh output dir "
                         "(outputs/autodp) into --out, then continue with whatever is missing")
    ap.add_argument("--adp-meta-corpus", default=None,
                    help="Retrained meta-learner corpus (scripts/build_adp_meta_corpus.py).")
    ap.add_argument("--operator-space", choices=["theirs", "ours"], default="theirs",
                    help="which operator space AutoDP's MCTS searches. 'theirs' = its own; "
                         "'ours' = ACORec's space via scripts/autodp_our_space.py.")
    ap.add_argument("--summarize", nargs="*", default=None, metavar="JSONL",
                    help="render a table from result files instead of running anything")
    ap.add_argument("--out-md", default=None, help="with --summarize: also write the table here")
    args = ap.parse_args()

    if args.summarize:
        summarize(args)
        return
    if args.import_dir:
        import_dir(args)
        if not args.ids and not args.shard:
            return  # import-only
    if not os.path.exists(args.adp_python):
        raise SystemExit(f"no AutoDP interpreter at {args.adp_python}\n"
                         f"  build it first:  bash scripts/setup_autodp_env.sh\n"
                         f"  or point --adp-python at an existing one")
    run(args)


if __name__ == "__main__":
    main()
