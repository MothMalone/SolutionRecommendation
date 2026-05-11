#!/usr/bin/env python3
"""RQ3 sensitivity: pheromone reinforcement strategy."""
from __future__ import annotations

try:
    from .rq3_sensitivity_common import build_common_parser, run_sensitivity_suite
except ImportError:  # pragma: no cover - script execution fallback
    from rq3_sensitivity_common import build_common_parser, run_sensitivity_suite


def main() -> int:
    parser = build_common_parser("RQ3 sensitivity: pheromone reinforcement strategy")
    args = parser.parse_args()

    variants = [
        # Sweep weight method with fixed top-k, order, and smoothing.
        {"name": "w_rank_k3_m2_l0p7", "flags": ["--top-k-pheromone", "3", "--aco-weight-method", "rank", "--aco-markov-order", "2", "--aco-lambda-smooth", "0.7"]},  # reference setting for this suite
        {"name": "w_uniform_k3_m2_l0p7", "flags": ["--top-k-pheromone", "3", "--aco-weight-method", "uniform", "--aco-markov-order", "2", "--aco-lambda-smooth", "0.7"]},
        {"name": "w_exp_k3_m2_l0p7", "flags": ["--top-k-pheromone", "3", "--aco-weight-method", "exponential", "--aco-markov-order", "2", "--aco-lambda-smooth", "0.7"]},
        # Sweep top-k pheromone with fixed method, order, and smoothing.
        {"name": "k1_rank_m2_l0p7", "flags": ["--top-k-pheromone", "1", "--aco-weight-method", "rank", "--aco-markov-order", "2", "--aco-lambda-smooth", "0.7"]},
        {"name": "k3_rank_m2_l0p7", "flags": ["--top-k-pheromone", "3", "--aco-weight-method", "rank", "--aco-markov-order", "2", "--aco-lambda-smooth", "0.7"]},
        {"name": "k5_rank_m2_l0p7", "flags": ["--top-k-pheromone", "5", "--aco-weight-method", "rank", "--aco-markov-order", "2", "--aco-lambda-smooth", "0.7"]},
    ]
    return run_sensitivity_suite(args=args, suite_name="rq3_pheromone_update", variants=variants)


if __name__ == "__main__":
    raise SystemExit(main())
