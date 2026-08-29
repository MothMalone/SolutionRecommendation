"""Run the RQ3 ablation for pheromone reinforcement weighting strategy."""
from __future__ import annotations

from rq3_transfer_ablation_common import build_parser, run_ablation


def main() -> int:
    parser = build_parser(
        "pheromone_weight_method",
        "RQ3: vary exponential, rank, and uniform pheromone reinforcement weighting",
    )
    args = parser.parse_args()
    return run_ablation(args)


if __name__ == "__main__":
    raise SystemExit(main())
