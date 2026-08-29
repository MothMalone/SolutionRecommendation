"""Run the RQ3 ablation that varies the number of ACO ants per iteration."""
from __future__ import annotations

from rq3_transfer_ablation_common import build_parser, run_ablation


def main() -> int:
    parser = build_parser(
        "num_ants",
        "RQ3: vary the number of ants sampled per ACO iteration with K and H fixed",
    )
    args = parser.parse_args()
    return run_ablation(args)


if __name__ == "__main__":
    raise SystemExit(main())
