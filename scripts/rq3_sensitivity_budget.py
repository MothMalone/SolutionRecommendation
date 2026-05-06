#!/usr/bin/env python3
"""RQ3 sensitivity: search budget (ants x iterations)."""
from __future__ import annotations

try:
    from .rq3_sensitivity_common import build_common_parser, run_sensitivity_suite
except ImportError:  # pragma: no cover - script execution fallback
    from rq3_sensitivity_common import build_common_parser, run_sensitivity_suite


def main() -> int:
    parser = build_common_parser("RQ3 sensitivity: ACO budget")
    args = parser.parse_args()

    variants = [
        {"name": "ants5_iter5", "flags": ["--n-ants", "5", "--n-iterations", "5"]},
        {"name": "ants10_iter10", "flags": ["--n-ants", "10", "--n-iterations", "10"]},  # default-like
        {"name": "ants15_iter10", "flags": ["--n-ants", "15", "--n-iterations", "10"]},
        {"name": "ants10_iter15", "flags": ["--n-ants", "10", "--n-iterations", "15"]},
        {"name": "ants20_iter10", "flags": ["--n-ants", "20", "--n-iterations", "10"]},
    ]
    return run_sensitivity_suite(args=args, suite_name="rq3_budget", variants=variants)


if __name__ == "__main__":
    raise SystemExit(main())
