"""Run the RQ3 ACO pheromone update-policy ablation.

The three variants compare the existing global-elite policy, an
iteration-elite policy, and a hybrid policy that reserves one top-k slot for
the global best and uses the remaining slots for the current iteration elite.
"""
from __future__ import annotations

from rq3_transfer_ablation_common import build_parser, run_ablation


def main() -> int:
    parser = build_parser(
        "aco_update_policy",
        "RQ3: compare global, iteration, and hybrid ACO pheromone updates",
    )
    args = parser.parse_args()
    return run_ablation(args)


if __name__ == "__main__":
    raise SystemExit(main())
