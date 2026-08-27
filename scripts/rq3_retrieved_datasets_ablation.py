#!/usr/bin/env python3
"""RQ3 ablation over the number of retrieved datasets (K)."""
from __future__ import annotations

from rq3_transfer_ablation_common import build_parser, run_ablation


def main() -> int:
    parser = build_parser(
        "num_retrieved_datasets",
        "RQ3: ablate the number of retrieved datasets used for heuristic transfer",
    )
    args = parser.parse_args()
    return run_ablation(args)


if __name__ == "__main__":
    raise SystemExit(main())

