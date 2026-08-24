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
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The evaluation set. Single source of truth -- do NOT re-declare it here.
#
# This list used to be maintained separately and had silently drifted from eval_ids.EVAL_IDS by
# three substitutions (1049/1050/1063 present, 2/382/993 absent), so an arm could run datasets
# that were never held out of the reference library. Importing removes that failure mode.
sys.path.insert(0, str(REPO / "src"))
from automl_aco.eval_ids import EVAL_IDS  # noqa: E402

OUR_DATASETS = list(EVAL_IDS)

# AutoDP's evaluation datasets, transcribed from their paper's table.
#
# !! UNVERIFIED !! These were read off the paper and have NOT been confirmed against the OpenML
# API. Several are OCR-risky (8335 / 43723 / 42493). Verify every id before reporting any arm-2
# number.
#
# LEAKAGE: these are NOT in EVAL_IDS, so holdout_reference() does not touch them by default --
# yet six of the ten are in the shipped reference library:
#
#     performance-matrix columns: 1461, 1590, 184, 31, 40701           (5/10)
#     metafeature rows:           the same five, plus 40945            (6/10)
#
# 184 and 31 used to be protected because they were in EVAL_IDS_LEGACY_23; the eval set moved to
# 30 different ids and that protection silently lapsed. Every ACORec arm on `theirs` data
# therefore passes --holdout-ids (see acorec_cmd) so the recommender cannot retrieve the target
# dataset's own best pipeline.
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


# The deployed ACORec configuration (docs/EXPERIMENTS.md "REF"). The arms exist to vary ONE thing
# -- the operator space or the data -- so every other knob must match the configuration the paper
# reports, or the arms measure something nobody claimed.
#
# `--train-metric-inline` is the one that silently mattered: without it (and without any
# --metric-* flag) run_recommend leaves `train_metric_inline=False` and falls back to plain COSINE
# similarity on raw metafeatures. That is the RQ2.1 *ablation*, not ACORec. Every arm run before
# this was added measured the ablation.
#
# `--require-autogluon` is the second: without it an AutoGluon failure quietly falls back to the
# LogReg proxy and still writes a score, which is how `autogluon_failed` rows ended up mixed into
# docs/local_full_run.
ACOREC_REF_FLAGS = [
    "--train-metric-inline",
    "--metric-loss", "pearson",
    "--metric-weight-decay", "1e-4",
    "--metric-objective", "embedding_cosine",
    "--aco-mmas-bounds",
    "--aco-weight-method", "linear",
    "--hybrid-select",
    "--final-autogluon-topk", "1",
    "--proxy-seeds", "42,52,62",
    "--cv-select-folds", "3",
    "--optimizer", "aco",
    "--require-autogluon",
    "--autogluon-profile", "best_quality",
]


def acorec_cmd(dataset_id: str, csv_path: Path, ops: str, workdir: Path, args,
               data: str = "ours") -> list:
    cmd = [
        sys.executable, str(REPO / "scripts" / "run_recommend.py"),
        "--dataset-source", "csv",
        "--dataset-csv", str(csv_path),
        "--target-column", args.target_column,
        "--dataset-id", str(dataset_id),
        "--operator-space", ops,
        "--use-aco",
        "--n-ants", str(args.n_ants),
        "--n-iterations", str(args.n_iterations),
        "--prepare-mode", "native" if args.protocol == "native" else "leakfree",
        "--time-limit", str(args.time_limit),
        "--seed", str(args.seed),
        "--output-dir", str(workdir),
    ]
    # Arms on THEIR data evaluate on ids that are not in EVAL_IDS, so the default holdout leaves
    # six of the ten inside the reference library. Hold out the whole list, not just the dataset
    # being run: the neighbour pool and the Siamese see all of them.
    if data == "theirs":
        cmd += ["--holdout-ids", ",".join(THEIR_DATASETS)]
    if args.acorec_config == "ref":
        cmd += ACOREC_REF_FLAGS
    else:
        cmd += ["--hybrid-select", "--cv-select-folds", "3"]
    # Trailing extras win, so an ablation can override any REF flag above.
    if args.extra:
        cmd += shlex.split(args.extra)
    if args.acorec_extra:
        cmd += shlex.split(args.acorec_extra)
    return cmd


def autodp_cmd(dataset_id: str, ops: str, out_dir: Path, args) -> list:
    # adp_bench's flags are --ids / --modes (plural), NOT --datasets / --mode.
    cmd = [
        sys.executable, str(REPO / "scripts" / "adp_bench.py"),
        "--ids", str(dataset_id),
        "--modes", "native" if args.protocol == "native" else "fair",
        "--out", str(out_dir / f"adp_{dataset_id}.jsonl"),
        "--time-limit", str(args.time_limit),
        "--cap-seconds", str(args.cap_seconds),
        # adp_bench's --data-dir is scratch for exported CSVs; the folder of ready-made <id>.csv
        # is --openml-local-folder, which is what --data-dir means on this side. Kaggle is
        # offline, so this is the only way it finds the data.
        "--openml-local-folder", str(args.data_dir),
        "--operator-space", ops if ops == "ours" else "theirs",
    ]
    # Default the retrained meta-learner corpus onto the ours-ops arms (verified disjoint from
    # both EVAL_IDS and THEIR_DATASETS -- docs/ARMS.md Part 4). --adp-meta-corpus "" opts back out.
    corpus = getattr(args, "adp_meta_corpus", None)
    if ops == "ours" and corpus is None:
        corpus = str(REPO / "data" / "adp_ourops_corpus")
    if ops == "ours" and corpus:
        cmd += ["--adp-meta-corpus", corpus]
    if args.extra:
        cmd += shlex.split(args.extra)
    if args.adp_extra:
        cmd += shlex.split(args.adp_extra)
    return cmd


def check_extra_flags(cmd_head: list, extra: str, label: str) -> None:
    """Fail fast if an --extra flag is not accepted by the tool it will be handed to.

    `--extra` reaches whichever tool the arm uses, so a string written for AutoDP arms
    (`--adp-python`, `--cap-seconds`) makes run_recommend exit 2 on EVERY dataset -- 30 rows of
    `status: failed` an hour later. One `--help` parse up front turns that into an immediate
    error. Use --acorec-extra / --adp-extra to keep one command template across all arms.
    """
    if not extra:
        return
    try:
        proc = subprocess.run(cmd_head + ["--help"], capture_output=True, text=True, timeout=120)
        helptext = proc.stdout
    except Exception:
        return  # never block a run on the checker itself
    # A failed or empty --help must NOT read as "every flag is unknown" -- that would abort a valid
    # run. (run_recommend's --help did crash for a while: a literal '%' in a help string made
    # argparse raise "unsupported format character", so stdout came back empty.)
    if proc.returncode != 0 or len(helptext) < 200:
        print(f"[arms] note: could not read {label} --help (rc={proc.returncode}); "
              f"skipping passthrough-flag validation")
        return
    unknown = [tok for tok in shlex.split(extra)
               if tok.startswith("--") and tok.split("=")[0] not in helptext]
    if unknown:
        raise SystemExit(
            f"[arms] {label} does not accept {unknown} (from --extra/{label}-extra).\n"
            f"        These would fail on every dataset. If they are meant for the other tool, "
            f"pass them via --acorec-extra or --adp-extra instead of --extra."
        )


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
    cfg.pop("name", None)  # a repr of the config itself; pure noise in the results file
    return {
        "status": "ok",
        "score": final.get("score"),
        # "autogluon" = a real AutoGluon score. "proxy" / "autogluon_failed" mean the number in
        # this column is NOT an AutoGluon score -- audit before averaging.
        "eval_method": final.get("method"),
        "selected_candidate": selected,
        "step_order": cfg.pop("step_order", None),
        "pipeline": cfg,
        # Per-candidate gate scores are verbose (config-repr keys) and incomplete; keep them out of
        # the results file. Rerun with --keep-scratch if you need them for a specific dataset.
        "n_gate_candidates": len(ag) or None,
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
    ap.add_argument("--data-dir", default="data/eval_datasets",
                    help="directory holding <dataset_id>.csv. Default holds all 30 EVAL_IDS "
                         "(scripts/export_eval_datasets.py + export_diffprep_datasets.py); the old "
                         "default, test_data_local/, only has the legacy 23 and is missing the 17 "
                         "DiffPrep ids, which made every arm on those ids record missing_csv.")
    ap.add_argument("--target-column", default="target")
    ap.add_argument("--time-limit", type=int, default=300, help="AutoGluon seconds per fit")
    # REF budget (docs/EXPERIMENTS.md), not the old 10/10.
    ap.add_argument("--n-ants", type=int, default=4)
    ap.add_argument("--n-iterations", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--acorec-config", choices=["ref", "minimal"], default="ref",
                    help="ref (default) = the deployed ACORec config from docs/EXPERIMENTS.md, "
                         "including the trained Siamese and --require-autogluon. minimal = the "
                         "old bare invocation, which silently used cosine similarity.")
    ap.add_argument("--acorec-extra", default="",
                    help="Extra flags for run_recommend.py only (ACORec arms).")
    ap.add_argument("--adp-extra", default="",
                    help="Extra flags for adp_bench.py only (AutoDP arms).")
    ap.add_argument("--adp-meta-corpus", default=None,
                    help="Corpus dir from scripts/build_adp_meta_corpus.py: retrains AutoDP's "
                         "1-NN meta-learner over ACORec's operator space instead of aliasing. "
                         "Default: data/adp_ourops_corpus for the ours-ops arms (verified disjoint "
                         "from both EVAL_IDS and THEIR_DATASETS -- docs/ARMS.md Part 4). Pass "
                         "--adp-meta-corpus '' to opt out and fall back to aliasing.")
    ap.add_argument("--ack-autodp-not-retrained", action="store_true",
                    help="Required for arms running AutoDP over ACORec's operator space if "
                         "--adp-meta-corpus is explicitly cleared. Acknowledges that its "
                         "meta-learner and value model are then applied out-of-domain via "
                         "operator-code aliasing rather than retrained.")
    ap.add_argument("--cap-seconds", type=float, default=900.0,
                    help="AutoDP arms only: wall-clock search watchdog forwarded to adp_bench.py "
                         "(its own default). Previously not forwarded at all, so every AutoDP arm "
                         "silently ran at adp_bench's 900s default with no way to change it here.")
    ap.add_argument("--protocol", choices=["native", "leakfree"], default="leakfree",
                    help="leakfree (default) = fit-on-train discipline for both methods, and "
                         "AutoDP's search scores on OUR seed-42 split rather than its own unseeded "
                         "internal one (scripts/autodp_protocol.py). native = AutoDP's published "
                         "transductive protocol, literally unpatched, kept only as a disclosure "
                         "column -- NOT a reported number for either method.")
    ap.add_argument("--extra", default="", help="extra args passed through to the inner script")
    ap.add_argument("--scratch", default="",
                    help="scratch dir. Default: a temp dir OUTSIDE --out's folder, deleted as it "
                         "goes, so /kaggle/working ends up holding only the results file.")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="keep per-dataset working dirs (recommendation.json etc) for debugging")
    args = ap.parse_args()

    if args.summarize:
        return summarize(args.summarize)

    if args.print_commands:
        arms = [args.arm] if args.arm else sorted(ARMS)
        for arm in arms:
            ds = [d.strip() for d in args.datasets.split(",") if d.strip()] or datasets_for(arm)
            if ARMS[arm]["method"] == "autodp" and ARMS[arm]["ops"] == "ours":
                print(f"\n# NOTE: arm {arm} additionally requires either "
                      "--adp-meta-corpus DIR (meta-learner retrained over our operators, "
                      "built by scripts/build_adp_meta_corpus.py) or "
                      "--ack-autodp-not-retrained (fall back to operator-code aliasing).")
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
    # Deliberately NOT under out_path.parent: on Kaggle that is /kaggle/working, and the whole
    # point is that the only thing left there is the results file.
    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix=f"arms_{args.arm}_"))
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

    # AutoDP over OUR operator space. Its meta-learner (get_CLA_meta_task_order) is a plain 1-NN
    # over two CSVs, so it CAN be retrained over our operators -- that is what
    # scripts/build_adp_meta_corpus.py produces, and --adp-meta-corpus is now the expected path.
    #
    # Its value model (model_CLA.pickle) is deliberately NOT retrained. get_Estimate builds a
    # MultiHeadAttention with fresh random weights on every call and never persists them (no
    # torch.save/load exists anywhere in the package), so the MLP is queried through a different
    # random projection each time: measured signal/noise 0.076 (between-pipeline sd 0.0235 vs
    # within-pipeline sd 0.3114, 4 pipelines x 40 calls). Refitting it would fit one projection and
    # be queried through another. It is also not load-bearing -- MCTS evaluates every expanded node
    # for real (mctsdata.getAcc) and drop_unpromising prunes on those real scores alone.
    # See docs/DATASET_CHANGE_AND_RQ3.md.
    if spec["method"] == "autodp" and spec["ops"] == "ours":
        # None (unset) defaults to the verified corpus; "" is an explicit opt-out.
        corpus_arg = (str(REPO / "data" / "adp_ourops_corpus") if args.adp_meta_corpus is None
                     else args.adp_meta_corpus)
        if corpus_arg:
            corpus = Path(corpus_arg)
            for required in ("Metafeature.csv", "label.csv"):
                if not (corpus / required).exists():
                    ap.error(f"--adp-meta-corpus {corpus} has no {required}; build it with "
                             f"scripts/build_adp_meta_corpus.py")
            print(f"[arms] meta-learner RETRAINED over our operators from {corpus}")
            print("[arms] DISCLOSE WHEN REPORTING: the value estimator is not retrained "
                  "(noise-dominated by construction; see docs).")
        elif args.ack_autodp_not_retrained:
            print("[arms] DISCLOSE WHEN REPORTING: AutoDP's meta-learner is NOT retrained here "
                  "and reaches our operators only via aliasing; pca/svd are aliased to TB.")
        else:
            ap.error(
                f"arm {args.arm} runs AutoDP over ACORec's operator space. Pass "
                "--adp-meta-corpus DIR to use a meta-learner retrained over our operators "
                "(build it with scripts/build_adp_meta_corpus.py), or "
                "--ack-autodp-not-retrained to fall back to operator-code aliasing and "
                "disclose it when reporting."
            )

    # Validate passthrough flags against the tool that will receive them, before running anything.
    if spec["method"] == "acorec":
        head = [sys.executable, str(REPO / "scripts" / "run_recommend.py")]
        check_extra_flags(head, args.extra, "run_recommend.py")
        check_extra_flags(head, args.acorec_extra, "run_recommend.py")
        print(f"[arms] acorec-config={args.acorec_config}"
              + ("  (Siamese TRAINED, AutoGluon required)" if args.acorec_config == "ref"
                 else "  WARNING: 'minimal' uses COSINE similarity, not the trained Siamese"))
    else:
        head = [sys.executable, str(REPO / "scripts" / "adp_bench.py")]
        check_extra_flags(head, args.extra, "adp_bench.py")
        check_extra_flags(head, args.adp_extra, "adp_bench.py")

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
            cmd = acorec_cmd(ds, csv_path, spec["ops"], workdir, args, data=spec["data"])
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
        if not args.keep_scratch:
            for junk in scratch.glob(f"*{ds}*"):
                shutil.rmtree(junk, ignore_errors=True) if junk.is_dir() else junk.unlink(missing_ok=True)
        n_exc = row.get("search_iteration_exceptions") or 0
        exc_flag = f"  !! {n_exc} search-iteration exceptions (result may be unevaluated)" if n_exc else ""
        print(f"     status={row.get('status')} score={row.get('score')} "
              f"time={row['total_seconds']}s{exc_flag}")

    if not args.keep_scratch and not args.scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"\n[arms] wrote {out_path}  (only file produced; scratch removed)")
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
    empty = [r for r in rows if r.get("empty_pipeline")]
    if empty:
        print(f"\n!! {len(empty)} AutoDP row(s) have an EMPTY pipeline -- AutoDP selected no "
              f"preprocessing, so the score is the RAW frame, not a search result: "
              + ", ".join(f"{r.get('arm')}/{r.get('dataset_id')}" for r in empty))
    salvaged = [r for r in rows if r.get("salvaged_from_checkpoint")]
    if salvaged:
        print(f"\n!! {len(salvaged)} AutoDP row(s) were SALVAGED from a cap-killed search "
              f"(best completed iteration, not a converged search) -- disclose when reporting: "
              + ", ".join(f"{r.get('arm')}/{r.get('dataset_id')}" for r in salvaged))
    crashed = [r for r in rows if (r.get("search_iteration_exceptions") or 0) > 0]
    if crashed:
        print(f"\n!! {len(crashed)} AutoDP row(s) had search-iteration exceptions -- their MCTS "
              f"loop swallows exceptions in a bare except:, so these pipelines may never have "
              f"been genuinely evaluated. Treat with caution before averaging: "
              + ", ".join(f"{r.get('arm')}/{r.get('dataset_id')} "
                          f"({r['search_iteration_exceptions']} exc)" for r in crashed))
    return 0


def _append(path: Path, row: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
