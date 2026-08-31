"""Run the RQ3 ACO pheromone/heuristic influence ablation."""
from __future__ import annotations

from rq3_transfer_ablation_common import build_parser, run_ablation


def main() -> int:
    parser = build_parser(
        "aco_alpha_beta",
        "RQ3: compare the relative influence of ACO pheromone (alpha) and heuristic (beta)",
    )
    args = parser.parse_args()
    return run_ablation(args)


if __name__ == "__main__":
    raise SystemExit(main())
