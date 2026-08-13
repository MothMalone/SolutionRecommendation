#!/usr/bin/env python3
"""Where does the ACORec-vs-AutoDP gap actually come from?

Joins the paired accuracies with an AutoDP PIPELINE CENSUS (what AutoDP's MCTS actually selected
per dataset) and splits the datasets into three groups:

  both-no-op   AutoDP selected no operator AND the scores are identical -> both methods handed
               AutoGluon the same raw frame. These rows are evidence for neither method.
  adp-no-op    AutoDP selected no operator, scores differ -> its score IS the no-preprocessing
               baseline (same harness, same split, same AutoGluon settings, raw frame in), so the
               gap measures ACORec's gain over raw data, not over AutoDP's preprocessing.
  contested    AutoDP selected at least one operator -> a genuine like-for-like comparison of two
               preprocessing pipelines.

The point: a mean over all datasets silently mixes these three, and only `contested` is a
preprocessing-vs-preprocessing result.

The census comes from running AutoDP's search alone (no scoring) -- see the --census argument.
Element 0 of an AutoDP pipeline is the classifier it picked for its internal scoring signal, so
the OPERATORS are pipeline[1:]; a length of 0 or 1 means no preprocessing at all.
"""
from __future__ import annotations

import argparse
import json
import re

# Paired accuracies from the AutoGluon evaluation (native protocol), in the reported row order.
PAIRED = [
    ("248", 0.713, 0.713), ("1066", 0.828, 0.862), ("1164", 0.730, 0.703),
    ("1047", 0.889, 0.921), ("862", 0.824, 0.824), ("40663", 0.620, 0.684),
    ("1054", 0.813, 0.813), ("1387", 0.798, 0.788), ("876", 0.650, 0.700),
    ("18", 0.758, 0.813), ("1520", 0.656, 0.688), ("1548", 0.686, 0.684),
    ("184", 0.650, 0.649), ("378", 0.814, 0.814), ("381", 0.819, 0.803),
    ("1485", 0.885, 0.887), ("14", 0.840, 0.875), ("27", 0.863, 0.877),
    ("29", 0.805, 0.862), ("31", 0.710, 0.790),
]

TIE_EPS = 1e-9


def parse_census(path: str) -> dict:
    """id -> {'pipeline': [...], 'search_s': float, 'converged': bool} from the census log."""
    out = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        did = parts[0].strip()
        if parts[1].strip().startswith("TIMEOUT") or parts[1].strip().startswith("SKIP"):
            out[did] = {"pipeline": None, "search_s": None, "converged": None}
            continue
        try:
            pipeline = json.loads(parts[1].strip().replace("'", '"'))
        except Exception:
            continue
        secs = conv = None
        for p in parts[2:]:
            m = re.match(r"search=([\d.]+)s", p.strip())
            if m:
                secs = float(m.group(1))
            m = re.match(r"converged=(\w+)", p.strip())
            if m:
                conv = m.group(1) == "True"
        out[did] = {"pipeline": pipeline, "search_s": secs, "converged": conv}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True, help="AutoDP pipeline census log (id<TAB>pipeline<TAB>...)")
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    census = parse_census(args.census)

    rows = []
    for did, adp, aco in PAIRED:
        c = census.get(did, {})
        pipe = c.get("pipeline")
        ops = None if pipe is None else pipe[1:]
        delta = aco - adp
        if ops is None:
            group = "unknown"
        elif len(ops) == 0:
            group = "both-no-op" if abs(delta) < 1e-3 else "adp-no-op"
        else:
            group = "contested"
        rows.append({"id": did, "adp": adp, "aco": aco, "delta": delta, "ops": ops,
                     "group": group, "search_s": c.get("search_s"), "converged": c.get("converged")})

    lines = ["# Where the ACORec-vs-AutoDP gap comes from\n"]
    lines.append("`ops` = the preprocessing operators AutoDP's MCTS selected (its pipeline minus "
                 "element 0, which is the classifier it uses for its own internal scoring). An empty "
                 "`ops` means AutoDP returned the **raw dataset**, so its score is a "
                 "no-preprocessing baseline rather than a preprocessing result.\n")
    lines.append("| Dataset | AutoDP | ACORec | Δ | AutoDP ops | search | converged | group |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        ops_txt = "—" if r["ops"] is None else (", ".join(r["ops"]) if r["ops"] else "**(none)**")
        secs = "—" if r["search_s"] is None else f"{r['search_s']:.1f}s"
        conv = "—" if r["converged"] is None else ("yes" if r["converged"] else "no")
        lines.append(f"| {r['id']} | {r['adp']:.3f} | {r['aco']:.3f} | {r['delta']:+.3f} | "
                     f"{ops_txt} | {secs} | {conv} | {r['group']} |")

    lines.append("\n## By group\n")
    lines.append("| group | n | mean AutoDP | mean ACORec | mean Δ | reading |")
    lines.append("|---|---|---|---|---|---|")
    readings = {
        "both-no-op": "neither method preprocessed — evidence for nobody",
        "adp-no-op": "ACORec's gain over RAW data (AutoDP declined to act)",
        "contested": "genuine preprocessing vs preprocessing",
        "unknown": "AutoDP census incomplete (timed out)",
    }
    for g in ("both-no-op", "adp-no-op", "contested", "unknown"):
        sub = [r for r in rows if r["group"] == g]
        if not sub:
            continue
        n = len(sub)
        lines.append(f"| {g} | {n} | {sum(r['adp'] for r in sub) / n:.4f} | "
                     f"{sum(r['aco'] for r in sub) / n:.4f} | "
                     f"{sum(r['delta'] for r in sub) / n:+.4f} | {readings[g]} |")
    n = len(rows)
    lines.append(f"| **all** | {n} | **{sum(r['adp'] for r in rows) / n:.4f}** | "
                 f"**{sum(r['aco'] for r in rows) / n:.4f}** | "
                 f"**{sum(r['delta'] for r in rows) / n:+.4f}** | the headline number |")

    wins = sum(1 for r in rows if r["delta"] > TIE_EPS)
    losses = sum(1 for r in rows if r["delta"] < -TIE_EPS)
    ties = n - wins - losses
    lines.append(f"\n**Paired record:** ACORec {wins}W / {ties}T / {losses}L over {n} datasets.")

    try:
        from scipy.stats import wilcoxon
        d = [r["delta"] for r in rows if abs(r["delta"]) > TIE_EPS]
        stat, p = wilcoxon(d)
        lines.append(f"\n**Wilcoxon signed-rank** (ties dropped, n={len(d)}): W={stat:.1f}, p={p:.4f}"
                     + ("  — significant at 0.05." if p < 0.05 else
                        "  — NOT significant at 0.05, so the mean gap alone will not carry the claim."))
    except Exception as exc:
        lines.append(f"\n(Wilcoxon unavailable: {exc})")

    noop = [r for r in rows if r["ops"] is not None and len(r["ops"]) == 0]
    acted = [r for r in rows if r["ops"]]
    lines.append(f"\n**AutoDP behaviour:** selected NO preprocessing on **{len(noop)}/{len(noop) + len(acted)}** "
                 f"datasets with a completed search; it acted on {len(acted)} "
                 f"({', '.join(r['id'] for r in acted) or 'none'}).")
    fast = [r for r in rows if r["converged"] and (r["search_s"] or 0) < 5]
    if fast:
        lines.append(f"On {len(fast)} of them its convergence rule fired in under 5 seconds "
                     f"({', '.join(r['id'] for r in fast)}) — it stopped early, it was not starved of budget.")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write(text)
        print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
