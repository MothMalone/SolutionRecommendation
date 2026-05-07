#!/usr/bin/env python3
"""RQ3 sensitivity: heuristic transfer neighborhood size (top-K) and per-neighbor top-L."""
from __future__ import annotations

try:
    from .rq3_sensitivity_common import build_common_parser, run_sensitivity_suite
except ImportError:  # pragma: no cover - script execution fallback
    from rq3_sensitivity_common import build_common_parser, run_sensitivity_suite


def main() -> int:
    parser = build_common_parser("RQ3 sensitivity: transfer top-K/top-L")
    args = parser.parse_args()

    variants = [
        # Sweep K with fixed L=3.
        {"name": "K3_L3", "flags": ["--heuristic-top-k", "3", "--heuristic-top-l", "3"]},
        {"name": "K5_L3", "flags": ["--heuristic-top-k", "5", "--heuristic-top-l", "3"]},  # default-like
        {"name": "K10_L3", "flags": ["--heuristic-top-k", "10", "--heuristic-top-l", "3"]},
        {"name": "K20_L3", "flags": ["--heuristic-top-k", "20", "--heuristic-top-l", "3"]},
        # Sweep L with fixed K=5.
        {"name": "K5_L1", "flags": ["--heuristic-top-k", "5", "--heuristic-top-l", "1"]},
        {"name": "K5_L5", "flags": ["--heuristic-top-k", "5", "--heuristic-top-l", "5"]},
    ]
    return run_sensitivity_suite(args=args, suite_name="rq3_transfer_neighbors", variants=variants)


if __name__ == "__main__":
    raise SystemExit(main())
