"""Run RQ3 with a fixed 10-ant colony and a varying cumulative ant budget."""
from __future__ import annotations

from rq3_transfer_ablation_common import build_parser, run_ablation


def main() -> int:
    parser = build_parser(
        "total_ant_budget",
        "RQ3: vary cumulative ant draws while keeping 10 ants per full ACO iteration",
    )
    args = parser.parse_args()
    return run_ablation(args)


if __name__ == "__main__":
    raise SystemExit(main())
