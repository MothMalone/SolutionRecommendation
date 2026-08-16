#!/usr/bin/env python3
"""Shardable driver for the ACORec-vs-AutoDP cross-comparison arms.

One JSONL line per (arm, dataset) with the score, the pipeline found and the wall time -- the same
compact output shape as ``adp_bench.py``, so the two can be concatenated and reported together.

THE ARMS
--------
The grid is (data x operators x method). ``--arm`` selects a cell:

  arm            data     operators  method    what it answers
  ------------   ------   ---------  -------   ---------------------------------------------
  0-adp-baseline ours     theirs     AutoDP    the CORRECTED AutoDP baseline (re-run, see below)
  1-adp-ourops   ours     ours       AutoDP    can their search do better in OUR space?
  2-aco-ourops   theirs   ours       ACORec    us on their datasets, our operators
  2-adp-ourops   theirs   ours       AutoDP    them on their datasets, our operators
  3-aco-theirops theirs   theirs     ACORec    us in THEIR space on THEIR data
  1-aco-theirops ours     theirs     ACORec    us in their space on our data (the clean control)

ACORec arms run through ``run_recommend.py``; AutoDP arms shell out to ``adp_bench.py``, which owns
the two-environment split (AutoDP needs numpy<1.24 and must never share an interpreter with
AutoGluon).

SHARDING
--------
``--shard I/N`` splits the dataset list round-robin, so any number of Kaggle notebooks can run
concurrently and the outputs concatenate. Completed rows are skipped on rerun, so a notebook that
times out can simply be restarted.

    python scripts/run_arms.py --arm 3-aco-theirops --shard 1/4 --out arms.jsonl

Print the commands for all shards of an arm without running anything:

    python scripts/run_arms.py --arm 3-aco-theirops --shards 4 --print-commands
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Our 23 evaluation datasets (the RQ3 set).
OUR_DATASETS = [
    "248", "1066", "1164", "1047", "862", "40663", "1054", "1387", "876", "18", "1520",
    "1548", "184", "378", "381", "1485", "14", "27", "29", "31", "1049", "1050", "1063",
]

# AutoDP's evaluation datasets, transcribed from their paper's table.
#
# !! UNVERIFIED !! These were read off the paper and have NOT been confirmed against the OpenML
# API. Several are OCR-risky (8335 / 43723 / 42493). Verify every id before reporting any arm-2
# number, and check the overlap against the 901-dataset reference library -- 184 and 31 are known
# to appear in both, and any overlap must be held out of the library for arm 2 to be leak-free.
THEIR_DATASETS = [
    "42493", "43723", "8335", "40945", "1461", "31", "42178", "184", "40701", "1590",
]

ARMS = {
    # The corrected AutoDP baseline. The published AutoDP column was produced in an environment
    # missing category_encoders (an undeclared dependency), which made every search return an
    # EMPTY pipeline. Every one of those numbers is invalid and this arm replaces them.
    "0-adp-baseline": dict(method="autodp", data="ours",   ops="theirs"),
    "1-adp-ourops":   dict(method="autodp", data="ours",   ops="ours"),
    "1-aco-theirops": dict(method="acorec", data="ours",   ops="theirs"),
    "2-aco-ourops":   dict(method="acorec", data="theirs", ops="ours"),
    "2-adp-ourops":   dict(method="autodp", data="theirs", ops="ours"),
    "3-aco-theirops": dict(method="acorec", data="theirs", ops="theirs"),
}


def datasets_for(arm: str) -> list:
    return OUR_DATASETS if ARMS[arm]["data"] == "ours" else THEIR_DATASETS


def shard(items, spec: str):
    if not spec:
        return items
    i, n = (int(x) for x in spec.split("/"))
    if not (1 <= i <= n):
        raise SystemExit(f"--shard must be I/N with 1 <= I <= N, got {spec}")
    return [x for j, x in enumerate(items) if j % n == (i - 1)]


def done_ids(out_path: Path, arm: str, protocol: str) -> set:
    if not out_path.exists():
        return set()
    seen = set()
    for line in out_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        # Key on protocol too: the same dataset under native and leakfree are different
        # results, and skipping the second one silently loses half the comparison.
        if (row.get("arm") == arm and row.get("status") == "ok"
                and str(row.get("protocol", protocol)) == str(protocol)):
            seen.add(str(row.get("dataset_id")))
    return seen


def acorec_cmd(dataset_id: str, csv_path: Path, ops: str, workdir: Path, args) -> list:
    cmd = [
        sys.executable, str(REPO / "scripts" / "run_recommend.py"),
        "--dataset-source", "csv",
        "--dataset-csv", str(csv_path),
        "--target-column", args.target_column,
        "--dataset-id", str(dataset_id),
        "--operator-space", ops,
        "--use-aco", "--hybrid-select",
        "--n-ants", str(args.n_ants),
        "--n-iterations", str(args.n_iterations),
        "--cv-select-folds", "3",
        "--prepare-mode", "native" if args.protocol == "native" else "leakfree",
        "--time-limit", str(args.time_limit),
        "--output-dir", str(workdir),
    ]
    if args.extra:
        cmd += shlex.split(args.extra)
    return cmd


def autodp_cmd(dataset_id: str, ops: str, out_dir: Path, args) -> list:
    # adp_bench's flags are --ids / --modes (plural), NOT --datasets / --mode.
    cmd = [
        sys.executable, str(REPO / "scripts" / "adp_bench.py"),
        "--ids", str(dataset_id),
        "--modes", "native" if args.protocol == "native" else "fair",
        "--out", str(out_dir / f"adp_{dataset_id}.jsonl"),
        "--time-limit", str(args.time_limit),
        # adp_bench's --data-dir is scratch for exported CSVs; the folder of ready-made <id>.csv
        # is --openml-local-folder, which is what --data-dir means on this side. Kaggle is
        # offline, so this is the only way it finds the data.
        "--openml-local-folder", str(args.data_dir),
        "--operator-space", ops if ops == "ours" else "theirs",
    ]
    if args.extra:
        cmd += shlex.split(args.extra)
    return cmd


def read_acorec_result(workdir: Path) -> dict:
    rec = workdir / "recommendation.json"
    if not rec.exists():
        return {"status": "no_output"}
    data = json.loads(rec.read_text())
    final = data.get("final_evaluation", {}) or {}
    cfg = dict(data.get("pipeline_config", {}) or {})
    # Which gate arm won. NOT read from ag_candidate_scores: that dict omits one candidate (the
    # floor can win without appearing there -- see docs/ARMS.md open issue 2). Derived from the
    # recommended pipeline's own shape instead.
    ag = data.get("ag_candidate_scores") or {}
    name = str(cfg.get("name", "") or "")
    steps = {k: v for k, v in cfg.items() if k not in ("name", "step_order")}
    non_none = {k: v for k, v in steps.items() if str(v) != "none" and k != "encoding"}
    if name.startswith("no_search_retrieval"):
        selected = "no_search_retrieval"
    elif name.startswith("light_"):
        selected = name
    elif not non_none:
        selected = "no_preprocessing"
    else:
        selected = "aco"
    return {
        "status": "ok",
        "score": final.get("score"),
        # "autogluon" = a real AutoGluon score. "proxy" / "autogluon_failed" mean the number in
        # this column is NOT an AutoGluon score -- audit before averaging.
        "eval_method": final.get("method"),
        "selected_candidate": selected,
        "step_order": cfg.pop("step_order", None),
        "pipeline": cfg,
        "ag_candidate_scores": ag or None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), required=False)
    ap.add_argument("--shard", default="", help="I/N round-robin shard")
    ap.add_argument("--datasets", default="",
                    help="comma-separated dataset ids to run (overrides the arm's default list)")
    ap.add_argument("--shards", type=int, default=0, help="with --print-commands: emit all N shards")
    ap.add_argument("--print-commands", action="store_true")
    ap.add_argument("--per-dataset", action="store_true",
                    help="with --print-commands: one command per dataset (the default "
                         "when --shards is not given)")
    ap.add_argument("--summarize", default="",
                    help="glob of arm JSONL files; print a table and exit")
    ap.add_argument("--out", default="arms.jsonl")
    ap.add_argument("--data-dir", default="test_data_local",
                    help="directory holding <dataset_id>.csv")
    ap.add_argument("--target-column", default="target")
    ap.add_argument("--time-limit", type=int, default=300, help="AutoGluon seconds per fit")
    ap.add_argument("--n-ants", type=int, default=10)
    ap.add_argument("--n-iterations", type=int, default=10)
    ap.add_argument("--protocol", choices=["native", "leakfree"], default="native",
                    help="native (default) = AutoDP's published transductive protocol, and ACORec "
                         "prepared the SAME way, so both see the same information. leakfree = "
                         "fit-on-train for both.")
    ap.add_argument("--extra", default="", help="extra args passed through to the inner script")
    ap.add_argument("--scratch", default="", help="scratch dir (default: alongside --out)")
    args = ap.parse_args()

    if args.summarize:
        return summarize(args.summarize)

    if args.print_commands:
        arms = [args.arm] if args.arm else sorted(ARMS)
        for arm in arms:
            ds = [d.strip() for d in args.datasets.split(",") if d.strip()] or datasets_for(arm)
            if args.per_dataset or not args.shards:
                # One dataset per command: the fastest way to see a first result, and a Kaggle
                # notebook that dies takes exactly one dataset down with it.
                print(f"\n# ---- arm {arm}: {len(ds)} runs, one dataset each ----")
                for d in ds:
                    print(f"python scripts/run_arms.py --arm {arm} --datasets {d} "
                          f"--out arms_{arm}.jsonl --protocol {args.protocol} "
                          f"--time-limit {args.time_limit}")
            else:
                n = args.shards
                print(f"\n# ---- arm {arm}: {len(ds)} datasets over {n} shards ----")
                for i in range(1, n + 1):
                    print(f"python scripts/run_arms.py --arm {arm} --shard {i}/{n} "
                          f"--out arms_{arm}_{i}of{n}.jsonl --protocol {args.protocol} "
                          f"--time-limit {args.time_limit}")
        return 0

    if not args.arm:
        ap.error("--arm is required unless --print-commands is given")

    spec = ARMS[args.arm]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch) if args.scratch else out_path.parent / f"_scratch_{args.arm}"
    scratch.mkdir(parents=True, exist_ok=True)

    base = [d.strip() for d in args.datasets.split(",") if d.strip()] or datasets_for(args.arm)
    ids = shard(base, args.shard)
    already = done_ids(out_path, args.arm, args.protocol)
    todo = [i for i in ids if i not in already]

    print(f"[arms] arm={args.arm} method={spec['method']} data={spec['data']} ops={spec['ops']}")
    print(f"[arms] shard={args.shard or 'all'}  {len(todo)} to run, {len(already)} already done")
    if spec["data"] == "theirs":
        print("[arms] WARNING: THEIR_DATASETS ids are transcribed from the paper and UNVERIFIED. "
              "Confirm against OpenML and hold out any overlap with the reference library "
              "before reporting.")

    for n, ds in enumerate(todo, 1):
        csv_path = Path(args.data_dir) / f"{ds}.csv"
        print(f"\n[{n}/{len(todo)}] dataset {ds}", flush=True)
        t0 = time.time()
        row = {"arm": args.arm, "dataset_id": ds, "method": spec["method"],
               "data": spec["data"], "operators": spec["ops"], "protocol": args.protocol}

        if spec["method"] == "acorec":
            if not csv_path.exists():
                row.update(status="missing_csv", error=str(csv_path))
                _append(out_path, row)
                print(f"     SKIP: {csv_path} not found")
                continue
            workdir = scratch / f"{args.arm}_{ds}"
            cmd = acorec_cmd(ds, csv_path, spec["ops"], workdir, args)
            rc = subprocess.run(cmd, cwd=REPO).returncode
            row.update(read_acorec_result(workdir) if rc == 0 else
                       {"status": "failed", "returncode": rc})
        else:
            cmd = autodp_cmd(ds, spec["ops"], scratch, args)
            rc = subprocess.run(cmd, cwd=REPO).returncode
            inner = scratch / f"adp_{ds}.jsonl"
            if rc == 0 and inner.exists() and inner.read_text().strip():
                row.update(json.loads(inner.read_text().strip().splitlines()[-1]))
                row.setdefault("status", "ok")
            else:
                row.update({"status": "failed", "returncode": rc})

        row["total_seconds"] = round(time.time() - t0, 1)
        _append(out_path, row)
        print(f"     status={row.get('status')} score={row.get('score')} "
              f"time={row['total_seconds']}s")

    print(f"\n[arms] wrote {out_path}")
    return 0


def summarize(pattern: str) -> int:
    """Print one table across any number of arm JSONL files."""
    import glob
    rows = []
    for f in sorted(glob.glob(pattern)):
        for line in Path(f).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        print(f"no rows matched {pattern}")
        return 1
    print(f"{'arm':16} {'dataset':9} {'proto':9} {'score':>8} {'eval':16} {'sel':18} {'secs':>7}")
    print("-" * 92)
    by_arm = {}
    for r in sorted(rows, key=lambda r: (r.get("arm", ""), str(r.get("dataset_id")))):
        sc = r.get("score")
        print(f"{str(r.get('arm'))[:16]:16} {str(r.get('dataset_id'))[:9]:9} "
              f"{str(r.get('protocol', '-'))[:9]:9} "
              f"{(f'{sc:.4f}' if isinstance(sc, (int, float)) else str(r.get('status'))):>8} "
              f"{str(r.get('eval_method', '-'))[:16]:16} "
              f"{str(r.get('selected_candidate', '-'))[:18]:18} "
              f"{r.get('total_seconds', 0):>7}")
        if isinstance(sc, (int, float)):
            by_arm.setdefault(r.get("arm"), []).append(sc)
    print("-" * 92)
    for arm, scores in sorted(by_arm.items()):
        print(f"{arm:16} n={len(scores):<3} mean={sum(scores)/len(scores):.4f}")
    bad = [r for r in rows if r.get("eval_method") not in (None, "autogluon")]
    if bad:
        print(f"\n!! {len(bad)} row(s) are NOT AutoGluon scores "
              f"(eval_method in {sorted({str(r.get('eval_method')) for r in bad})}) -- exclude before averaging")
    return 0


def _append(path: Path, row: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
