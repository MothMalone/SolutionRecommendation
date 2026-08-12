#!/usr/bin/env python3
"""Aggregate the AutoDP baseline into an accuracy + runtime table (stage 4).

Reads stage-3 ``autodp_eval.json`` files under ``--input-dir`` (both ``native/`` and ``fair/`` if
present) and, for every ``--ours LABEL=DIR``, the ``recommendation.json`` files our own runs write.
Every score is AutoGluon test performance (accuracy for classification, R2 for regression) on the
identical seed-42 0.6/0.2/0.2 split with the same 80%-fit protocol, so the columns compare directly.

Runtime is AutoDP's own search time (its MCTS to convergence), NOT the AutoGluon scoring time --
scoring is our harness and is the same work for every method. ``--runtime-column total`` switches
to search + AutoGluon instead.

Examples:
    python scripts/report_autodatapre.py --input-dir outputs/autodp
    python scripts/report_autodatapre.py --input-dir outputs/autodp \\
        --ours "ACORec=outputs/final_run" --ours "no-prep=outputs/bq_baseline"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional


def _fmt(v: Optional[float], nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _fmt_secs(v: Optional[float]) -> str:
    return "—" if v is None else (f"{v:.0f}s" if v < 600 else f"{v / 60:.1f}m")


def _load_ours(path: str) -> Dict[str, Dict[str, Any]]:
    """dataset_id -> {score, seconds} from recommendation.json files under ``path``."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(glob.glob(os.path.join(path, "**", "recommendation.json"), recursive=True)):
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception:
            continue
        score = (rec.get("final_evaluation") or {}).get("score")
        out[str(rec.get("dataset_id"))] = {
            "score": float(score) if score is not None else None,
            "seconds": rec.get("elapsed_seconds"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="outputs/autodp",
                    help="root holding <mode>/dataset_<id>/autodp_eval.json")
    ap.add_argument("--ours", action="append", default=[], metavar="LABEL=DIR",
                    help="repeatable: a run directory of ours to put alongside")
    ap.add_argument("--autodp-score", choices=["score_full", "score_kept"], default="score_full",
                    help="score_full counts test rows AutoDP deleted as wrong (comparable to ours); "
                         "score_kept scores only the rows it kept (generous to AutoDP)")
    ap.add_argument("--runtime-column", choices=["search", "total"], default="search")
    ap.add_argument("--out", default=None, help="default: <input-dir>/AUTODP_REPORT.md")
    args = ap.parse_args()

    ours: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in args.ours:
        if "=" not in spec:
            raise SystemExit(f"--ours expects LABEL=DIR, got {spec!r}")
        label, path = spec.split("=", 1)
        ours[label] = _load_ours(path)
        print(f"[ours] {label}: {len(ours[label])} datasets from {path}")

    per_mode: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for p in sorted(glob.glob(os.path.join(args.input_dir, "*", "dataset_*", "autodp_eval.json"))):
        try:
            with open(p) as f:
                res = json.load(f)
        except Exception:
            continue
        per_mode.setdefault(res.get("mode", "native"), {})[str(res.get("dataset_id"))] = res

    # Datasets AutoDP never finished. Shown as "timeout" rather than dropped, so a hard row is not
    # quietly missing from the mean.
    failures: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for p in sorted(glob.glob(os.path.join(args.input_dir, "*", "dataset_*", "autodp_failed.json"))):
        try:
            with open(p) as f:
                res = json.load(f)
        except Exception:
            continue
        failures.setdefault(res.get("mode", "native"), {})[str(res.get("dataset_id"))] = res

    if not per_mode and not ours and not failures:
        raise SystemExit(f"no results found under {args.input_dir}")

    all_modes = set(per_mode) | set(failures)
    modes = [m for m in ("native", "fair") if m in all_modes] + \
            sorted(m for m in all_modes if m not in ("native", "fair"))
    ids = sorted({d for m in per_mode.values() for d in m} | {d for o in ours.values() for d in o}
                 | {d for m in failures.values() for d in m},
                 key=lambda s: (len(s), s))

    time_key = "autodp_search_seconds" if args.runtime_column == "search" else "total_seconds"
    time_label = "time" if args.runtime_column == "search" else "time(+AG)"

    header = ["Dataset", "task"]
    for m in modes:
        header += [f"AutoDP {m}", f"{m} {time_label}"]
    for label in ours:
        header += [label, f"{label} time"]

    lines: List[str] = ["# AutoDP (autodatapre 0.1.12) — accuracy and runtime\n"]
    lines.append(
        "Scores are AutoGluon test performance (accuracy for classification, R² for regression) on "
        "the identical seed-42 0.6/0.2/0.2 split, fit on train+val (80%), predicting the same 20% "
        "test rows. AutoDP's MCTS search, its pretrained meta-learner and its internal NB/LDA/RF "
        "scoring signal run as published; only the final scorer is swapped to AutoGluon, exactly as "
        "for our pipelines.\n")
    lines.append(
        f"AutoDP score column = `{args.autodp_score}`. **native** = its published API, whose search "
        "sees the full dataset including our test rows (transductive, generous to AutoDP). **fair** "
        "= its search restricted to our 80% train+val, the protocol our method is held to. "
        + ("Runtime is AutoDP's own search time to convergence; AutoGluon scoring time is excluded "
           "because it is our harness and identical work for every method.\n"
           if args.runtime_column == "search" else
           "Runtime is AutoDP's search time plus AutoGluon scoring.\n"))

    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    tsums: Dict[str, float] = {}
    tcounts: Dict[str, int] = {}
    wins: Dict[str, List[int]] = {}

    for did in ids:
        any_res = next((per_mode[m][did] for m in modes if did in per_mode.get(m, {})), None)
        task = str((any_res or {}).get("problem_type", "—")).replace("multiclass", "multi").replace("binary", "bin")
        row = [did, task]
        adp_scores: Dict[str, Optional[float]] = {}
        for m in modes:
            r = per_mode.get(m, {}).get(did)
            v = r.get(args.autodp_score) if r else None
            adp_scores[m] = v
            if r is None and did in failures.get(m, {}):
                row.append("timeout")
                row.append(f"&gt;{_fmt_secs(failures[m][did].get('cap_seconds'))}")
                continue
            flag = " ⚠️" if (r and r.get("autodp_status") != "ok") else ""
            row.append(_fmt(v) + flag)
            secs = r.get(time_key) if r else None
            row.append(_fmt_secs(secs))
            if v is not None:
                sums[m] = sums.get(m, 0.0) + v
                counts[m] = counts.get(m, 0) + 1
            if secs is not None:
                tsums[m] = tsums.get(m, 0.0) + float(secs)
                tcounts[m] = tcounts.get(m, 0) + 1
        for label, table in ours.items():
            entry = table.get(did) or {}
            v, secs = entry.get("score"), entry.get("seconds")
            row.append(_fmt(v))
            row.append(_fmt_secs(float(secs)) if secs is not None else "—")
            if v is not None:
                sums[label] = sums.get(label, 0.0) + v
                counts[label] = counts.get(label, 0) + 1
                if secs is not None:
                    tsums[label] = tsums.get(label, 0.0) + float(secs)
                    tcounts[label] = tcounts.get(label, 0) + 1
                for m in modes:
                    if adp_scores.get(m) is not None:
                        w = wins.setdefault(f"{label} vs AutoDP {m}", [0, 0, 0])
                        d = v - adp_scores[m]
                        w[0 if d > 1e-9 else (2 if d < -1e-9 else 1)] += 1
        lines.append("| " + " | ".join(row) + " |")

    mean_row = ["**mean**", ""]
    for key in list(modes) + list(ours.keys()):
        mean_row.append(f"**{sums[key] / counts[key]:.4f}**" if counts.get(key) else "—")
        mean_row.append(f"**{_fmt_secs(tsums[key] / tcounts[key])}**" if tcounts.get(key) else "—")
    lines.append("| " + " | ".join(mean_row) + " |")

    if wins:
        lines.append("\n**Head-to-head** (W/T/L on datasets where both scored):\n")
        for key, (w, t, l) in sorted(wins.items()):
            lines.append(f"- `{key}`: **{w}W** / {t}T / {l}L")

    notes: List[str] = []
    timed_out = [f"{d} ({m}, cap {r.get('cap_seconds')}s)" for m in modes
                 for d, r in failures.get(m, {}).items()]
    if timed_out:
        notes.append("**AutoDP never finished** on these, so they carry no score and are excluded "
                     "from its mean and from the head-to-head above: " + ", ".join(timed_out)
                     + ". AutoDP checks its time budget only between search iterations, so one slow "
                       "operator (`AD` dedup is O(n²) string comparisons, `LOF`, `MICE`) can overrun "
                       "any budget on a large frame. Rerun a single dataset with a larger "
                       "`--cap-seconds` to chase a real number.")
    degraded = [f"{d} ({m})" for m in modes for d, r in per_mode.get(m, {}).items()
                if r.get("autodp_status") != "ok"]
    if degraded:
        notes.append("⚠️ AutoDP's winning pipeline crashed on apply, so its own code path returned "
                     "the RAW frame (its entry point hides this behind a bare `except:`): "
                     + ", ".join(degraded))
    low_cov = [f"{d} ({m}, cov={r['test_coverage']:.2f})" for m in modes
               for d, r in per_mode.get(m, {}).items() if r.get("test_coverage", 1.0) < 0.999]
    if low_cov:
        notes.append("AutoDP deleted test rows here, so `score_kept` would be measured on an easier "
                     "subset than ours: " + ", ".join(low_cov))
    no_op = [f"{d} ({m})" for m in modes for d, r in per_mode.get(m, {}).items()
             if isinstance(r.get("autodp_pipeline"), list) and len(r["autodp_pipeline"]) <= 1]
    if no_op:
        notes.append("AutoDP selected NO preprocessing operator (classifier choice only), so its "
                     "output equals the raw data: " + ", ".join(no_op))
    capped = [f"{d} ({m})" for m in modes for d, r in per_mode.get(m, {}).items()
              if r.get("autodp_hit_cap")]
    if capped:
        notes.append("AutoDP's convergence rule never fired, so the run was cut at the wall-clock "
                     "cap and retried with an explicit budget: " + ", ".join(capped))
    if notes:
        lines.append("\n## Notes\n")
        lines += [f"- {n}" for n in notes]
    lines.append("")

    out = args.out or os.path.join(args.input_dir, "AUTODP_REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
