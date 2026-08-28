#!/usr/bin/env python3
"""Emit the ACORec-parity dataset id list for the AutoDP meta-corpus rebuild.

The AutoDP 1-NN family-order meta-learner (arm 1, `1-adp-ourops`) is fit on
`data/adp_ourops_corpus`. ACORec's similarity metric is fit on the 901-column AutoGluon
performance matrix (leak-free: 879 after the 30 eval ids are held out). To put the two on a
comparable footing, rebuild the AutoDP corpus over the SAME reference datasets ACORec uses.

This script takes the `D_<id>` columns of the performance matrix, keeps the ones that are
classification (`NumberOfClasses >= 2`) and not extreme-width (`NumberOfFeatures <= --max-features`,
default 1000 -- matching `build_adp_meta_corpus.py`), and removes the 30 `EVAL_IDS` and the 10
`THEIR_DATASETS` (5 of which ARE perf-matrix columns, so this filter is mandatory or
`build_adp_meta_corpus.py --ids` aborts in `assert_disjoint`).

    python scripts/adp_parity_ref_ids.py > /tmp/ref_ids.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from automl_aco.eval_ids import EVAL_ID_SET, assert_disjoint, normalize_id  # noqa: E402
from run_arms import THEIR_DATASETS  # noqa: E402

PERF = REPO / "data" / "openml" / "training_performance_matrix_autogluon.csv"
FEATS = REPO / "data" / "openml" / "dataset_feats.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-features", type=int, default=1000)
    ap.add_argument("--stats", action="store_true", help="print counts to stderr")
    args = ap.parse_args()

    header = PERF.read_text().splitlines()[0].split(",")
    perf_ids = [normalize_id(c[2:]) for c in header if c.startswith("D_")]

    feats = pd.read_csv(FEATS, index_col=0)
    feats.index = [normalize_id(str(i)) for i in feats.index]

    exclude = set(EVAL_ID_SET) | {normalize_id(i) for i in THEIR_DATASETS}
    kept, dropped_missing, dropped_reg, dropped_wide, dropped_excl = [], 0, 0, 0, 0
    for did in perf_ids:
        if did in exclude:
            dropped_excl += 1
            continue
        if did not in feats.index:
            dropped_missing += 1
            continue
        row = feats.loc[did]
        if "NumberOfClasses" in feats.columns and not (row["NumberOfClasses"] >= 2):
            dropped_reg += 1
            continue
        if ("NumberOfFeatures" in feats.columns and args.max_features
                and row["NumberOfFeatures"] > args.max_features):
            dropped_wide += 1
            continue
        kept.append(did)

    # dedupe, keep order
    seen = set()
    kept = [d for d in kept if not (d in seen or seen.add(d))]
    assert_disjoint(kept, context="adp parity ref ids", extra_ids=list(THEIR_DATASETS))

    if args.stats:
        print(f"perf-matrix D_ columns : {len(perf_ids)}", file=sys.stderr)
        print(f"  excluded (eval/their): {dropped_excl}", file=sys.stderr)
        print(f"  not in dataset_feats : {dropped_missing}", file=sys.stderr)
        print(f"  regression           : {dropped_reg}", file=sys.stderr)
        print(f"  > {args.max_features} features       : {dropped_wide}", file=sys.stderr)
        print(f"  KEPT                 : {len(kept)}", file=sys.stderr)

    print(",".join(kept))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
