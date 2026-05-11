#!/usr/bin/env python3
"""RQ3 ablation: feature-mode constraints vs joint search.

This suite compares three modes under the same ACO/search budget:
1) joint_all (default ACORec joint preprocessing search space)
2) per_feature_rules_global_fixed (global operators fixed by rule, independent
   operators still optimized by ACO)
3) per_feature_independent_only (disable global operators; optimize only
   independent operators)

Notes:
- `feature_selection` is treated as a global operator.
- This suite compares constrained global modes against the joint baseline.
- True per-feature operator search now requires passing per-feature operator
  maps in the pipeline config (supported in `Preprocessor`).
"""
from __future__ import annotations

try:
    from .rq3_sensitivity_common import build_common_parser, run_sensitivity_suite
except ImportError:  # pragma: no cover - script execution fallback
    from rq3_sensitivity_common import build_common_parser, run_sensitivity_suite


def main() -> int:
    parser = build_common_parser("RQ3 ablation: per-feature preprocessing modes")
    parser.add_argument(
        "--global-dimred",
        choices=["pca", "svd"],
        default="pca",
        help="Fixed dimensionality reduction operator for the rule-based global mode.",
    )
    parser.add_argument(
        "--global-outlier",
        choices=["iqr", "zscore", "lof", "isolation_forest"],
        default="isolation_forest",
        help="Fixed outlier operator for the rule-based global mode.",
    )
    parser.add_argument(
        "--global-feature-selection",
        choices=["variance_threshold", "k_best", "mutual_info"],
        default="k_best",
        help="Fixed feature-selection operator for the rule-based global mode.",
    )
    parser.add_argument(
        "--rule-search-ordering",
        action="store_true",
        help=(
            "Enable order search in the rule-based global mode. "
            "This explores valid step orders with the selected strategy."
        ),
    )
    parser.add_argument(
        "--rule-order-strategy",
        choices=["fixed", "random", "heuristic", "scored", "all"],
        default="heuristic",
        help="Order proposal strategy used when --rule-search-ordering is enabled.",
    )
    parser.add_argument(
        "--rule-num-orders",
        type=int,
        default=10,
        help="Number of candidate orders used when --rule-search-ordering is enabled.",
    )
    args = parser.parse_args()

    rule_override = (
        f"dimensionality_reduction={args.global_dimred},"
        f"outlier_removal={args.global_outlier},"
        f"feature_selection={args.global_feature_selection}"
    )
    independent_only_override = (
        "dimensionality_reduction=none,"
        "outlier_removal=none,"
        "feature_selection=none"
    )

    rule_flags = ["--pipeline-override", rule_override]
    if args.rule_search_ordering:
        rule_flags.extend(
            [
                "--search-ordering",
                "--order-strategy",
                str(args.rule_order_strategy),
                "--num-orders",
                str(max(1, int(args.rule_num_orders))),
            ]
        )

    variants = [
        {"name": "joint_all", "flags": []},
        {"name": "per_feature_rules_global_fixed", "flags": rule_flags + ["--per-feature-independent-search"]},
        {
            "name": "per_feature_independent_only",
            "flags": ["--pipeline-override", independent_only_override, "--per-feature-independent-search"],
        },
    ]
    return run_sensitivity_suite(
        args=args,
        suite_name="rq3_per_feature_modes",
        variants=variants,
    )


if __name__ == "__main__":
    raise SystemExit(main())
