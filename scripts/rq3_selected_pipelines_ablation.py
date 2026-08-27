#!/usr/bin/env python3
"""RQ3 ablation over the number of selected pipelines per retrieved dataset (H)."""
from __future__ import annotations

from rq3_transfer_ablation_common import build_parser, run_ablation


def main() -> int:
    parser = build_parser(
        "num_selected_pipelines",
        "RQ3: ablate the number of selected pipelines transferred per retrieved dataset",
    )
    args = parser.parse_args()
    return run_ablation(args)


if __name__ == "__main__":
    raise SystemExit(main())

