#!/usr/bin/env python3
"""RQ3 sensitivity: ACO dynamics (alpha, beta, evaporation)."""
from __future__ import annotations

try:
    from .rq3_sensitivity_common import build_common_parser, run_sensitivity_suite
except ImportError:  # pragma: no cover - script execution fallback
    from rq3_sensitivity_common import build_common_parser, run_sensitivity_suite


def main() -> int:
    parser = build_common_parser("RQ3 sensitivity: ACO alpha/beta/evaporation")
    args = parser.parse_args()

    variants = [
        {"name": "a0p5_b1_r0p2", "flags": ["--alpha", "0.5", "--beta", "1.0", "--evaporation", "0.2"]},
        {"name": "a1_b2_r0p2", "flags": ["--alpha", "1.0", "--beta", "2.0", "--evaporation", "0.2"]},  # default-like
        {"name": "a2_b2_r0p2", "flags": ["--alpha", "2.0", "--beta", "2.0", "--evaporation", "0.2"]},
        {"name": "a1_b3_r0p2", "flags": ["--alpha", "1.0", "--beta", "3.0", "--evaporation", "0.2"]},
        {"name": "a1_b2_r0p1", "flags": ["--alpha", "1.0", "--beta", "2.0", "--evaporation", "0.1"]},
        {"name": "a1_b2_r0p4", "flags": ["--alpha", "1.0", "--beta", "2.0", "--evaporation", "0.4"]},
    ]
    return run_sensitivity_suite(args=args, suite_name="rq3_aco_dynamics", variants=variants)


if __name__ == "__main__":
    raise SystemExit(main())
