#!/usr/bin/env python3
"""RQ3 sensitivity: transfer weighting, similarity temperature, and eta floor."""
from __future__ import annotations

try:
    from .rq3_sensitivity_common import build_common_parser, run_sensitivity_suite
except ImportError:  # pragma: no cover - script execution fallback
    from rq3_sensitivity_common import build_common_parser, run_sensitivity_suite


def main() -> int:
    parser = build_common_parser("RQ3 sensitivity: transfer weighting / temperature / eta floor")
    args = parser.parse_args()

    variants = [
        {"name": "sim_T0p25_floor0p05", "flags": ["--dataset-weighting", "similarity", "--heuristic-similarity-temperature", "0.25", "--heuristic-eta-floor", "0.05"]},
        {"name": "sim_T1_floor0p05", "flags": ["--dataset-weighting", "similarity", "--heuristic-similarity-temperature", "1.0", "--heuristic-eta-floor", "0.05"]},  # default-like
        {"name": "sim_T2_floor0p05", "flags": ["--dataset-weighting", "similarity", "--heuristic-similarity-temperature", "2.0", "--heuristic-eta-floor", "0.05"]},
        {"name": "sim_T1_floor0p01", "flags": ["--dataset-weighting", "similarity", "--heuristic-similarity-temperature", "1.0", "--heuristic-eta-floor", "0.01"]},
        {"name": "sim_T1_floor0p10", "flags": ["--dataset-weighting", "similarity", "--heuristic-similarity-temperature", "1.0", "--heuristic-eta-floor", "0.10"]},
        {"name": "equal_floor0p05", "flags": ["--dataset-weighting", "equality", "--heuristic-similarity-temperature", "1.0", "--heuristic-eta-floor", "0.05"]},
    ]
    return run_sensitivity_suite(args=args, suite_name="rq3_transfer_weighting", variants=variants)


if __name__ == "__main__":
    raise SystemExit(main())
