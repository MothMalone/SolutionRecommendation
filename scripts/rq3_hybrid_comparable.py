"""Comparable hybrid candidate selection under the original recommendation protocol.

This script is intentionally apples-to-apples with the existing search/no-search
runs: it uses the full target dataset and the same internal AutoGluon evaluator
used by MetaPipelineRecommender. It does not introduce an extra outer holdout.

Use this when you want to compare hybrid against the previous LaTeX table. For a
stricter generalization test, use rq3_hybrid_outer_holdout.py instead.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automl_aco.metalearning.recommender import MetaPipelineRecommender
from automl_aco.search.evaluation import evaluate_candidates_autogluon

_BASE_PATH = Path(__file__).resolve().with_name("rq3_hybrid_outer_holdout.py")
_SPEC = importlib.util.spec_from_file_location("rq3_hybrid_outer_holdout_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load helper script at {_BASE_PATH}")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)


def _cli_flag_was_passed(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def _source_priority(source: str) -> int:
    order = {
        "no_search_selected": 0,
        "search_selected": 1,
        "no_search_inner_topk": 2,
        "search_inner_topk": 3,
    }
    return order.get(source, 99)


def _build_parser():
    parser = base.build_parser()
    parser.description = "Comparable hybrid validation using the original internal evaluation protocol"
    parser.set_defaults(output_dir="/kaggle/working/rq3_hybrid_comparable")
    parser.add_argument(
        "--hybrid-selection-margin",
        type=float,
        default=0.0,
        help=(
            "Require the best non-baseline hybrid candidate to beat the better selected "
            "pipeline by this margin. If not, keep the better selected search/no-search pipeline."
        ),
    )
    parser.add_argument(
        "--hybrid-tie-breaker",
        choices=["no_search", "search"],
        default="no_search",
        help="Which selected baseline to keep when search and no-search are tied within --hybrid-selection-margin.",
    )
    return parser


def _candidate_configs_for_eval(
    candidates: Mapping[str, Tuple[str, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    ordered = sorted(candidates.items(), key=lambda item: (_source_priority(item[1][0]), item[0]))
    configs: List[Dict[str, Any]] = []
    for sig, (source, cfg) in ordered:
        c = dict(cfg)
        c.setdefault("name", f"{source}_{sig}")
        c["_hybrid_signature"] = sig
        c["_hybrid_source"] = source
        configs.append(c)
    return configs


def _lookup_score_by_sig(results: List[Tuple[Dict[str, Any], float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cfg, score in results:
        sig = cfg.get("_hybrid_signature")
        if sig is None:
            continue
        out[str(sig)] = float(score)
    return out


def main() -> int:
    args = _build_parser().parse_args()
    metric_target_explicit = _cli_flag_was_passed("--metric-similarity-target")

    if args.notebook_legacy_mode:
        args.notebook_legacy_options = True
        args.heuristic_transfer_method = "legacy_weighted_average"
        args.dataset_weighting = "equality"
        args.legacy_notebook_aco = True
        args.aco_lambda_smooth = 0.7
        args.proxy_logreg_max_iter = 1000
        if not metric_target_explicit:
            args.metric_similarity_target = "legacy_global_zscore_cosine"
        args.metric_hidden_dim = 32
        args.metric_embed_dim = 32
        if not args.metric_path:
            args.train_metric_inline = True
    if args.no_train_metric_inline:
        args.train_metric_inline = False

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    openml_local = Path(args.openml_local_folder or root / "test_data_local")
    kaggle_data = Path(args.kaggle_data_folder or root / "test_data_local")

    perf = pd.read_csv(args.performance_matrix, index_col=0)
    raw_meta = pd.read_csv(args.metafeatures)
    meta = base._maybe_set_meta_index(raw_meta, perf, args.metafeatures_id_column)
    configs = base._load_pipeline_configs(Path(args.pipeline_configs), perf, bool(args.allow_inferred_pipeline_configs))

    recommender = MetaPipelineRecommender(perf, meta, configs, verbose=bool(args.verbose))
    if args.metric_path:
        recommender.load_metric(args.metric_path)
    elif args.train_metric_inline:
        if args.verbose:
            print(
                "Training Siamese metric inline: "
                f"hidden_dim={args.metric_hidden_dim}, embed_dim={args.metric_embed_dim}, "
                f"epochs={args.metric_epochs}, target={args.metric_similarity_target}, "
                f"objective={args.metric_objective}"
            )
        recommender.train_metric(
            method="regression",
            hidden_dim=int(args.metric_hidden_dim),
            embed_dim=int(args.metric_embed_dim),
            epochs=int(args.metric_epochs),
            lr=float(args.metric_lr),
            seed=int(args.seed),
            similarity_target=str(args.metric_similarity_target),
            score_direction="higher_is_better",
            metric_objective=str(args.metric_objective),
        )

    proxy_settings = base._build_proxy_settings(args)
    dataset_ids = base._parse_dataset_ids(args.dataset_ids)
    rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    for run_idx, dataset_id in enumerate(dataset_ids, start=1):
        start = time.perf_counter()
        if args.verbose:
            print(f"\n=== Comparable hybrid dataset {dataset_id} ({run_idx}/{len(dataset_ids)}) ===")
        try:
            dataset, df = base._load_dataset_as_frame(
                dataset_id,
                dataset_source=str(args.dataset_source),
                openml_local_folder=openml_local,
                kaggle_data_folder=kaggle_data,
                kaggle_target_column=str(args.kaggle_target_column),
                verbose=bool(args.verbose),
            )
            options, option_notes = base._adapt_options_to_dataset(base._build_options(args), df.drop(columns=["target"]))
            for note in option_notes:
                if args.verbose:
                    print(f"  Auto option guard: {note}")

            def mf_func(_df: pd.DataFrame, _dataset: Mapping[str, Any] = dataset) -> Dict[str, Any]:
                return base._lookup_metafeatures(_dataset, meta)

            aco_params = base._build_aco_params(args, dataset_id)
            no_search_rec = recommender.recommend(
                new_dataset=df,
                target_column="target",
                options=options,
                k=max(1, int(args.k)),
                eval_k=max(1, int(args.eval_k)),
                use_autogluon=True,
                use_aco=False,
                aco_params=aco_params,
                time_limit_per_model=int(args.time_limit),
                metafeatures_func=mf_func,
                proxy_settings=proxy_settings,
                final_autogluon_topk=max(1, int(args.final_autogluon_topk)),
            )
            search_rec = recommender.recommend(
                new_dataset=df,
                target_column="target",
                options=options,
                k=max(1, int(args.k)),
                eval_k=max(1, int(args.eval_k)),
                use_autogluon=True,
                use_aco=True,
                optimizer="aco",
                aco_params=aco_params,
                time_limit_per_model=int(args.time_limit),
                metafeatures_func=mf_func,
                proxy_settings=proxy_settings,
                final_autogluon_topk=max(1, int(args.final_autogluon_topk)),
            )

            candidates = base._collect_candidates(
                no_search_rec=no_search_rec,
                search_rec=search_rec,
                options=options,
                include_inner_topk=bool(args.include_inner_topk_candidates),
                inner_topk=int(args.inner_topk_candidates),
            )
            if not candidates:
                raise RuntimeError("No hybrid candidates collected")

            eval_configs = _candidate_configs_for_eval(candidates)
            ag_best_cfg, ag_best_score, ag_results, ag_unsorted = evaluate_candidates_autogluon(
                df,
                "target",
                eval_configs,
                time_limit_per_model=int(args.time_limit),
                verbose=bool(args.verbose),
            )
            if ag_best_cfg is None or not ag_results or not np.isfinite(ag_best_score):
                raise RuntimeError("No candidate produced valid comparable AutoGluon score")

            score_by_sig = _lookup_score_by_sig(ag_results)
            no_sig = base._config_signature(no_search_rec.get("pipeline_config") or {}, options)
            search_sig = base._config_signature(search_rec.get("pipeline_config") or {}, options)
            no_score = float(score_by_sig.get(no_sig, np.nan))
            search_score = float(score_by_sig.get(search_sig, np.nan))
            valid = [(sig, source, cfg, score_by_sig.get(sig, np.nan)) for sig, (source, cfg) in candidates.items()]
            valid = [item for item in valid if np.isfinite(item[3])]
            margin = float(args.hybrid_selection_margin)
            selector_like = [
                (sig, source, cfg, float(score), "comparable_autogluon", "ok", "")
                for sig, source, cfg, score in valid
            ]
            (
                best_sig,
                best_source,
                best_cfg,
                best_score,
                _best_metric,
                selector_decision_reason,
                selector_best_minus_baseline,
            ) = base._select_hybrid_candidate(
                selector_like,
                no_sig=no_sig,
                search_sig=search_sig,
                margin=margin,
                tie_breaker=str(args.hybrid_tie_breaker),
            )
            valid.sort(key=lambda item: (-float(item[3]), _source_priority(item[1]), item[0]))

            for sig, source, cfg, score in valid:
                candidate_rows.append(
                    {
                        "dataset_id": str(dataset_id),
                        "candidate_source": source,
                        "signature": sig,
                        "comparable_autogluon_score": float(score),
                        "selected_by_hybrid": bool(sig == best_sig),
                        "hybrid_selection_margin": margin,
                        "selector_decision_reason": selector_decision_reason,
                        "pipeline_config": base._display_config(cfg, options),
                    }
                )

            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "status": "ok",
                    "error": "",
                    "candidate_count": int(len(candidates)),
                    "no_search_original_final_score": no_search_rec.get("final_performance", np.nan),
                    "no_search_original_final_method": (no_search_rec.get("final_evaluation") or {}).get("method", ""),
                    "no_search_comparable_score": no_score,
                    "no_search_pipeline": base._display_config(no_search_rec.get("pipeline_config") or {}, options),
                    "search_original_final_score": search_rec.get("final_performance", np.nan),
                    "search_original_final_method": (search_rec.get("final_evaluation") or {}).get("method", ""),
                    "search_proxy_score": search_rec.get("recommended_performance", np.nan),
                    "search_comparable_score": search_score,
                    "search_pipeline": base._display_config(search_rec.get("pipeline_config") or {}, options),
                    "hybrid_comparable_score": float(best_score),
                    "hybrid_source": best_source,
                    "hybrid_selection_margin": margin,
                    "hybrid_tie_breaker": str(args.hybrid_tie_breaker),
                    "selector_decision_reason": selector_decision_reason,
                    "selector_best_minus_baseline": selector_best_minus_baseline,
                    "hybrid_pipeline": base._display_config(best_cfg, options),
                    "hybrid_minus_no_search": float(best_score - no_score) if np.isfinite(no_score) else np.nan,
                    "hybrid_minus_search": float(best_score - search_score) if np.isfinite(search_score) else np.nan,
                    "search_minus_no_search": float(search_score - no_score) if np.isfinite(search_score) and np.isfinite(no_score) else np.nan,
                    "search_same_as_no_search": bool(search_sig == no_sig),
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
            )
            if args.verbose:
                print(f"Dataset {dataset_id} failed: {exc}")

    summary_df = pd.DataFrame(rows)
    candidate_df = pd.DataFrame(candidate_rows)
    summary_df.to_csv(out_dir / "hybrid_comparable_summary.csv", index=False)
    candidate_df.to_csv(out_dir / "hybrid_comparable_candidates.csv", index=False)

    ok = summary_df[summary_df.get("status", "") == "ok"].copy() if not summary_df.empty else pd.DataFrame()
    aggregate: Dict[str, Any] = {"n_datasets": int(ok["dataset_id"].nunique()) if not ok.empty else 0}
    if not ok.empty:
        for col in [
            "no_search_comparable_score",
            "search_comparable_score",
            "hybrid_comparable_score",
            "hybrid_minus_no_search",
            "hybrid_minus_search",
            "search_minus_no_search",
        ]:
            aggregate[f"mean_{col}"] = float(pd.to_numeric(ok[col], errors="coerce").mean())
        aggregate["hybrid_beats_no_search_rate"] = float((pd.to_numeric(ok["hybrid_minus_no_search"], errors="coerce") > 0).mean())
        aggregate["hybrid_beats_search_rate"] = float((pd.to_numeric(ok["hybrid_minus_search"], errors="coerce") > 0).mean())
        aggregate["search_beats_no_search_rate"] = float((pd.to_numeric(ok["search_minus_no_search"], errors="coerce") > 0).mean())
        aggregate["hybrid_source_counts"] = ok["hybrid_source"].value_counts(dropna=False).to_dict()

    with (out_dir / "hybrid_comparable_aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(pd.DataFrame([aggregate]).to_string(index=False))
    print(f"Saved comparable hybrid outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
