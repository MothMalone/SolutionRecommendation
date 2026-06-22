"""Hybrid validation with fixed train/test and a split validation set.

Protocol:
1. Use the original notebook split: train / validation / test.
2. Keep train fixed and keep test fixed.
3. Split only the original validation set into val_search and val_selector.
4. Use train -> val_search for ACO/proxy search.
5. Use train -> val_selector to choose between no-search and search candidates.
6. Use train -> test for the final reported AutoGluon score.

This is intended to be comparable to the previous configuration while avoiding
using the same validation signal both for search and for choosing whether search
or no-search should be trusted.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from automl_aco.data.splits import split_train_val_test
from automl_aco.metalearning.recommender import MetaPipelineRecommender
from automl_aco.preprocessing.preprocessor import Preprocessor
from automl_aco.search.aco import search_pipelines_aco
from automl_aco.search.evaluation import _detect_problem_type, _normalize_proxy_settings, _proxy_penalty, _step_is_none
from automl_aco.search.heuristics import select_top_k_neighbors

_BASE_PATH = Path(__file__).resolve().with_name("rq3_hybrid_outer_holdout.py")
_SPEC = importlib.util.spec_from_file_location("rq3_hybrid_outer_holdout_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load helper script at {_BASE_PATH}")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)


def _cli_flag_was_passed(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def _df_from_xy(X: pd.DataFrame, y: pd.Series, target_column: str = "target") -> pd.DataFrame:
    df = X.reset_index(drop=True).copy()
    df[target_column] = y.reset_index(drop=True)
    return df


def _split_validation(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    selector_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if len(X_val) != len(y_val):
        raise ValueError("X_val/y_val length mismatch")
    if len(X_val) < 2:
        raise ValueError("Validation split is too small to divide into val_search and val_selector")
    frac = min(max(float(selector_fraction), 0.05), 0.95)
    rng = np.random.RandomState(int(seed))
    indices = rng.permutation(len(X_val))
    n_selector = int(round(len(indices) * frac))
    n_selector = min(max(1, n_selector), len(indices) - 1)
    selector_idx = indices[:n_selector]
    search_idx = indices[n_selector:]
    return (
        X_val.iloc[search_idx].reset_index(drop=True),
        y_val.iloc[search_idx].reset_index(drop=True),
        X_val.iloc[selector_idx].reset_index(drop=True),
        y_val.iloc[selector_idx].reset_index(drop=True),
    )


def _make_preprocessor(cfg: Mapping[str, Any]) -> Preprocessor:
    step_order = cfg.get("step_order")
    pre_cfg = {k: v for k, v in dict(cfg).items() if not str(k).startswith("__") and k != "step_order"}
    if isinstance(step_order, list) and step_order:
        return Preprocessor(pre_cfg, step_order=step_order)
    return Preprocessor(pre_cfg)


def _score_fixed_simple(
    X_train_p: pd.DataFrame,
    y_train_p: pd.Series,
    X_eval_p: pd.DataFrame,
    y_eval_p: pd.Series,
    *,
    problem_type: str,
    proxy_cfg: Mapping[str, Any],
) -> Optional[float]:
    if problem_type == "regression":
        model_name = str(proxy_cfg.get("regression_model", "ensemble")).lower().strip()
        models: List[Any]
        if model_name == "linear":
            models = [LinearRegression()]
        elif model_name == "random_forest":
            models = [RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)]
        else:
            models = [
                LinearRegression(),
                RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
            ]
        scores: List[float] = []
        for model in models:
            try:
                model.fit(X_train_p, y_train_p)
                pred = model.predict(X_eval_p)
                scores.append(float(r2_score(y_eval_p, pred)))
            except Exception:
                continue
        if not scores:
            return None
        return float(np.mean(scores)) if model_name == "ensemble" else float(scores[0])

    model_name = str(proxy_cfg.get("classification_model", "logreg")).lower().strip()
    if model_name != "logreg":
        # The comparable fixed-split experiment keeps the default fast proxy.
        # Other proxy models can be added later, but logreg is the tested path.
        model_name = "logreg"
    best = -np.inf
    for C in (0.01, 0.1, 1.0):
        for class_weight in (None, "balanced"):
            try:
                clf = LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    class_weight=class_weight,
                    max_iter=max(200, int(proxy_cfg.get("logreg_max_iter", 3000))),
                    random_state=42,
                )
                clf.fit(X_train_p, y_train_p)
                pred = clf.predict(X_eval_p)
                best = max(best, float(accuracy_score(y_eval_p, pred)))
            except Exception:
                continue
    return float(best) if np.isfinite(best) else None


def _evaluate_fixed_simple(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate_configs: Sequence[Dict[str, Any]],
    *,
    proxy_settings: Optional[Dict[str, Any]],
    verbose: bool,
) -> Tuple[Optional[Dict[str, Any]], float, List[Tuple[Dict[str, Any], float]], List[Tuple[Dict[str, Any], float]]]:
    proxy_cfg = _normalize_proxy_settings(proxy_settings)
    X_train = train_df.drop(columns=["target"]).copy()
    y_train = train_df["target"].reset_index(drop=True)
    X_eval = eval_df.drop(columns=["target"]).copy()
    y_eval = eval_df["target"].reset_index(drop=True)
    problem_type, _metric = _detect_problem_type(pd.concat([y_train, y_eval], ignore_index=True))
    missing_ratio = float(X_train.isna().to_numpy().mean()) if X_train.shape[1] else 0.0
    has_missing = bool(missing_ratio > 0.0)
    n_features_before = int(X_train.shape[1])
    results: List[Tuple[Dict[str, Any], float]] = []

    for cfg in candidate_configs:
        c = dict(cfg)
        c.setdefault("name", str(c))
        name = str(c.get("name"))
        if has_missing and _step_is_none(c, "imputation"):
            if verbose:
                print(f"    x {name} skipped: missing values require imputation")
            continue
        try:
            pre = _make_preprocessor(c)
            transformed = pre.fit_transform(X_train, y_train)
            if isinstance(transformed, tuple):
                X_train_p, y_train_p = transformed
            else:
                X_train_p = transformed
                y_train_p = y_train
            X_eval_p = pre.transform(X_eval)
            y_eval_p = y_eval.reset_index(drop=True)
            if X_train_p.shape[0] == 0 or X_eval_p.shape[0] == 0:
                continue
            if len(X_train_p) != len(y_train_p) or len(X_eval_p) != len(y_eval_p):
                continue
            raw_score = _score_fixed_simple(
                X_train_p,
                y_train_p.reset_index(drop=True),
                X_eval_p,
                y_eval_p,
                problem_type=problem_type,
                proxy_cfg=proxy_cfg,
            )
            if raw_score is None:
                continue
            penalty = _proxy_penalty(
                cfg=c,
                n_rows_before=len(X_train),
                n_rows_after=len(X_train_p),
                n_features_before=n_features_before,
                missing_ratio=missing_ratio,
                proxy_cfg=proxy_cfg,
            )
            score = float(raw_score - penalty)
            results.append((c, score))
            if verbose:
                print(f"    ok {name} -> {score:.4f}")
        except Exception as exc:
            if verbose:
                print(f"    x {name}: {exc}")
            continue

    if not results:
        return None, np.nan, [], []
    unsorted = results.copy()
    results.sort(key=lambda item: item[1], reverse=True)
    return results[0][0], float(results[0][1]), results, unsorted


def _evaluate_fixed_autogluon_many(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidates: Mapping[str, Tuple[str, Dict[str, Any]]],
    *,
    time_limit: int,
    verbose: bool,
) -> Dict[str, Tuple[float, str, str]]:
    scores: Dict[str, Tuple[float, str, str]] = {}
    for sig, (_source, cfg) in candidates.items():
        try:
            score, metric = base._evaluate_candidate_outer_holdout(
                train_df=train_df,
                holdout_df=eval_df,
                target_column="target",
                cfg=cfg,
                time_limit=int(time_limit),
                verbose=verbose,
            )
            scores[sig] = (float(score), metric, "")
        except Exception as exc:
            scores[sig] = (np.nan, "", str(exc))
    return scores


def _evaluate_fixed_simple_many(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidates: Mapping[str, Tuple[str, Dict[str, Any]]],
    *,
    proxy_settings: Optional[Dict[str, Any]],
    verbose: bool,
) -> Dict[str, Tuple[float, str, str]]:
    cfgs: List[Dict[str, Any]] = []
    sig_by_name: Dict[str, str] = {}
    for sig, (source, cfg) in candidates.items():
        c = dict(cfg)
        name = f"{source}_{sig}"
        c["name"] = name
        c["__fixed_sig__"] = sig
        sig_by_name[name] = sig
        cfgs.append(c)
    _best, _score, results, _unsorted = _evaluate_fixed_simple(
        train_df,
        eval_df,
        cfgs,
        proxy_settings=proxy_settings,
        verbose=verbose,
    )
    scores: Dict[str, Tuple[float, str, str]] = {sig: (np.nan, "proxy", "not_evaluated") for sig in candidates}
    for cfg, score in results:
        sig = str(cfg.get("__fixed_sig__") or sig_by_name.get(str(cfg.get("name")), ""))
        if sig:
            scores[sig] = (float(score), "proxy", "")
    return scores


def _new_metafeatures_scaled(
    recommender: MetaPipelineRecommender,
    dataset: Mapping[str, Any],
    meta: pd.DataFrame,
) -> np.ndarray:
    new_mf = base._lookup_metafeatures(dataset, meta)
    new_mf_df = pd.DataFrame([new_mf]).reindex(columns=recommender.metafeatures_df.columns, fill_value=0)
    return recommender.scaler.transform(recommender.imputer.transform(new_mf_df)).ravel()


def _retrieval_candidates(
    recommender: MetaPipelineRecommender,
    *,
    new_mf_scaled: np.ndarray,
    options: Mapping[str, Sequence[str]],
    k: int,
    eval_k: int,
    query_dataset_id: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sims = recommender._compute_dataset_similarities(new_mf_scaled)
    top_neighbors = select_top_k_neighbors(sims, top_k=max(1, int(k)), query_dataset_id=query_dataset_id)
    if not top_neighbors:
        raise RuntimeError("No eligible similar datasets found after self-query exclusion")
    top_datasets = [ds for ds, _sim in top_neighbors]
    top_sims = np.asarray([sim for _ds, sim in top_neighbors], dtype=float)
    if top_sims.sum() == 0:
        top_sims = np.ones_like(top_sims)
    perf_subset = recommender.performance_matrix.loc[:, top_datasets].fillna(0)
    candidate_perfs = pd.Series(
        np.average(perf_subset.values, axis=1, weights=top_sims),
        index=recommender.performance_matrix.index,
    )
    names = [str(n) for n in candidate_perfs.sort_values(ascending=False).index.tolist()[: max(1, int(eval_k))]]
    cfg_by_name = {str(cfg.get("name")): cfg for cfg in recommender.pipeline_configs if isinstance(cfg, dict)}
    out: List[Dict[str, Any]] = []
    for name in names:
        raw = cfg_by_name.get(name)
        if raw is None:
            continue
        cfg = recommender._coerce_pipeline_to_options(raw, options, name=name, step_order=list(options.keys()))
        out.append(cfg)
    neighbor_rows = [
        {"neighbor_dataset": str(ds), "similarity": float(sim), "rank": rank}
        for rank, (ds, sim) in enumerate(top_neighbors, start=1)
    ]
    return out, neighbor_rows


def _candidate_signature(cfg: Mapping[str, Any], options: Mapping[str, Sequence[str]]) -> str:
    return base._config_signature(dict(cfg), options)


def _put_candidate(
    candidates: Dict[str, Tuple[str, Dict[str, Any]]],
    *,
    source: str,
    cfg: Dict[str, Any],
    options: Mapping[str, Sequence[str]],
) -> None:
    sig = _candidate_signature(cfg, options)
    if sig not in candidates:
        candidates[sig] = (source, dict(cfg))


def _pick_best(
    candidates: Mapping[str, Tuple[str, Dict[str, Any]]],
    scores: Mapping[str, Tuple[float, str, str]],
    allowed_sources: Optional[Sequence[str]] = None,
) -> Tuple[str, str, Dict[str, Any], float]:
    allowed = set(allowed_sources) if allowed_sources is not None else None
    valid: List[Tuple[str, str, Dict[str, Any], float]] = []
    for sig, (source, cfg) in candidates.items():
        if allowed is not None and source not in allowed:
            continue
        score = float(scores.get(sig, (np.nan, "", ""))[0])
        if np.isfinite(score):
            valid.append((sig, source, cfg, score))
    if not valid:
        raise RuntimeError("No valid candidates for selection")
    valid.sort(key=lambda item: item[3], reverse=True)
    return valid[0]


def build_parser():
    parser = base.build_parser()
    parser.description = "Hybrid with fixed train/test and split validation"
    parser.set_defaults(output_dir="/kaggle/working/rq3_hybrid_fixed_validation_split")
    parser.add_argument("--val-selector-fraction", type=float, default=0.5)
    parser.add_argument("--val-split-seed", type=int, default=2026)
    parser.add_argument("--selector-backend", choices=["simple", "autogluon"], default="autogluon")
    parser.add_argument("--search-candidate-topk", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
    openml_local = Path(args.openml_local_folder or root / "test_data_local")
    kaggle_data = Path(args.kaggle_data_folder or root / "test_data_local")
    summary_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []

    for run_idx, dataset_id in enumerate(dataset_ids, start=1):
        start = time.perf_counter()
        if args.verbose:
            print(f"\n=== Fixed validation hybrid dataset {dataset_id} ({run_idx}/{len(dataset_ids)}) ===")
        try:
            dataset, df = base._load_dataset_as_frame(
                dataset_id,
                dataset_source=str(args.dataset_source),
                openml_local_folder=openml_local,
                kaggle_data_folder=kaggle_data,
                kaggle_target_column=str(args.kaggle_target_column),
                verbose=bool(args.verbose),
            )
            X = df.drop(columns=["target"]).copy()
            y = df["target"].copy()
            X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, seed=int(args.seed))
            X_val_search, y_val_search, X_val_selector, y_val_selector = _split_validation(
                X_val,
                y_val,
                selector_fraction=float(args.val_selector_fraction),
                seed=int(args.val_split_seed),
            )
            train_df = _df_from_xy(X_train, y_train)
            val_search_df = _df_from_xy(X_val_search, y_val_search)
            val_selector_df = _df_from_xy(X_val_selector, y_val_selector)
            test_df = _df_from_xy(X_test, y_test)
            search_context_df = pd.concat([train_df, val_search_df], ignore_index=True)

            options, option_notes = base._adapt_options_to_dataset(base._build_options(args), X_train)
            for note in option_notes:
                if args.verbose:
                    print(f"  Auto option guard: {note}")

            new_mf_scaled = _new_metafeatures_scaled(recommender, dataset, meta)
            aco_params = base._build_aco_params(args, dataset_id)

            no_candidates = _retrieval_candidates(
                recommender,
                new_mf_scaled=new_mf_scaled,
                options=options,
                k=max(1, int(args.k)),
                eval_k=max(1, int(args.eval_k)),
                query_dataset_id=dataset_id,
            )[0]
            if not no_candidates:
                raise RuntimeError("No no-search candidates found")

            if bool(args.per_feature_independent_search):
                old_eval = recommender._evaluate_candidates_with_simple_models

                def fixed_eval(_dataset, _target_column, candidate_configs, proxy_settings=None):
                    return _evaluate_fixed_simple(
                        train_df,
                        val_search_df,
                        candidate_configs,
                        proxy_settings=proxy_settings,
                        verbose=bool(args.verbose),
                    )

                recommender._evaluate_candidates_with_simple_models = fixed_eval  # type: ignore[method-assign]
                try:
                    aco_results, aco_unsorted, aco_history, _feature_log = recommender._search_pipelines_aco_per_feature(
                        new_dataset=search_context_df,
                        target_column="target",
                        new_metafeatures=new_mf_scaled,
                        options=options,
                        proxy_settings=proxy_settings,
                        n_pipelines=max(1, int(args.search_candidate_topk)),
                        n_ants=int(args.n_ants),
                        n_iterations=int(args.n_iterations),
                        seed=int(args.seed),
                        alpha=float(args.alpha),
                        beta=float(args.beta),
                        evaporation=float(args.evaporation),
                        top_k_pheromone=int(args.top_k_pheromone),
                        weight_method=str(args.aco_weight_method),
                        markov_order=int(args.aco_markov_order),
                        lambda_smooth=float(args.aco_lambda_smooth),
                        use_all_iter_pipelines=False,
                        step_order=list(options.keys()),
                        dataset_weighting=str(args.dataset_weighting),
                        heuristic_top_k=int(args.heuristic_top_k),
                        heuristic_top_l=int(args.heuristic_top_l),
                        heuristic_similarity_temperature=float(args.similarity_temperature),
                        heuristic_eta_floor=float(args.eta_floor),
                        heuristic_transfer_method=str(args.heuristic_transfer_method),
                        score_direction="higher_is_better",
                        query_dataset_id=dataset_id,
                        time_limit_per_model=int(args.time_limit),
                        metafeatures_func=lambda _df: base._lookup_metafeatures(dataset, meta),
                        per_feature_steps=[s.strip() for s in str(args.per_feature_steps).split(",") if s.strip()],
                        per_feature_early_stop_rounds=int(args.per_feature_early_stop_rounds),
                        per_feature_min_improvement=float(args.per_feature_min_improvement),
                        per_feature_feature_patience=int(args.per_feature_feature_patience),
                        per_feature_feature_min_improvement=float(args.per_feature_feature_min_improvement),
                        per_feature_max_features=int(args.per_feature_max_features),
                    )
                finally:
                    recommender._evaluate_candidates_with_simple_models = old_eval  # type: ignore[method-assign]
            else:
                eta = recommender._compute_aco_heuristic(
                    new_mf_scaled,
                    options,
                    dataset_weighting=str(args.dataset_weighting),
                    top_k=max(1, int(args.heuristic_top_k)),
                    use_top_pipelines_from_metric=True,
                    recommend_kwargs={
                        "new_dataset": search_context_df,
                        "target_column": "target",
                        "options": options,
                        "k": 5,
                        "eval_k": 3,
                        "use_aco": False,
                        "time_limit_per_model": int(args.time_limit),
                        "use_autogluon": False,
                        "metafeatures_func": lambda _df: base._lookup_metafeatures(dataset, meta),
                    },
                    top_l=max(1, int(args.heuristic_top_l)),
                    similarity_temperature=float(args.similarity_temperature),
                    eta_floor=float(args.eta_floor),
                    heuristic_transfer_method=str(args.heuristic_transfer_method),
                    score_direction="higher_is_better",
                    query_dataset_id=dataset_id,
                )

                def eval_for_aco(sampled_configs: List[Dict[str, Any]]):
                    with_order = []
                    for cfg in sampled_configs:
                        c = dict(cfg)
                        c["step_order"] = list(options.keys())
                        with_order.append(c)
                    return _evaluate_fixed_simple(
                        train_df,
                        val_search_df,
                        with_order,
                        proxy_settings=proxy_settings,
                        verbose=bool(args.verbose),
                    )

                result = search_pipelines_aco(
                    options=options,
                    evaluate_fn=eval_for_aco,
                    eta=eta,
                    n_pipelines=max(1, int(args.search_candidate_topk)),
                    n_ants=int(args.n_ants),
                    n_iterations=int(args.n_iterations),
                    seed=int(args.seed),
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    evaporation=float(args.evaporation),
                    top_k_pheromone=int(args.top_k_pheromone),
                    use_all_iter_pipelines=False,
                    weight_method=str(args.aco_weight_method),
                    markov_order=int(args.aco_markov_order),
                    lambda_smooth=float(args.aco_lambda_smooth),
                    early_stop_rounds=int(args.aco_early_stop_rounds),
                    min_improvement=float(args.aco_min_improvement),
                    verbose=bool(args.verbose),
                    return_history=True,
                    legacy_notebook_aco=bool(args.legacy_notebook_aco),
                )
                aco_results, aco_unsorted, aco_history = result

            for hist in aco_history:
                row = dict(hist)
                row["dataset_id"] = str(dataset_id)
                history_rows.append(row)

            candidates: Dict[str, Tuple[str, Dict[str, Any]]] = {}
            for cfg in no_candidates:
                _put_candidate(candidates, source="no_search_candidate", cfg=cfg, options=options)
            for cfg, _score in list(aco_results)[: max(1, int(args.search_candidate_topk))]:
                c = dict(cfg)
                c.setdefault("step_order", list(options.keys()))
                _put_candidate(candidates, source="search_candidate", cfg=c, options=options)
            if bool(args.include_inner_topk_candidates):
                for cfg, _score in list(aco_unsorted)[: max(0, int(args.inner_topk_candidates))]:
                    c = dict(cfg)
                    c.setdefault("step_order", list(options.keys()))
                    _put_candidate(candidates, source="search_inner_topk", cfg=c, options=options)

            if str(args.selector_backend) == "autogluon":
                selector_scores = _evaluate_fixed_autogluon_many(
                    train_df,
                    val_selector_df,
                    candidates,
                    time_limit=int(args.time_limit),
                    verbose=bool(args.verbose),
                )
            else:
                selector_scores = _evaluate_fixed_simple_many(
                    train_df,
                    val_selector_df,
                    candidates,
                    proxy_settings=proxy_settings,
                    verbose=bool(args.verbose),
                )

            no_sig, no_source, no_cfg, no_selector = _pick_best(
                candidates,
                selector_scores,
                allowed_sources=["no_search_candidate"],
            )
            search_sig, search_source, search_cfg, search_selector = _pick_best(
                candidates,
                selector_scores,
                allowed_sources=["search_candidate", "search_inner_topk"],
            )
            hybrid_sig, hybrid_source, hybrid_cfg, hybrid_selector = _pick_best(candidates, selector_scores)

            final_candidates = {
                no_sig: (no_source, no_cfg),
                search_sig: (search_source, search_cfg),
                hybrid_sig: (hybrid_source, hybrid_cfg),
            }
            final_scores = _evaluate_fixed_autogluon_many(
                train_df,
                test_df,
                final_candidates,
                time_limit=int(args.time_limit),
                verbose=bool(args.verbose),
            )
            no_final = float(final_scores.get(no_sig, (np.nan, "", ""))[0])
            search_final = float(final_scores.get(search_sig, (np.nan, "", ""))[0])
            hybrid_final = float(final_scores.get(hybrid_sig, (np.nan, "", ""))[0])

            for sig, (source, cfg) in candidates.items():
                sel_score, sel_metric, sel_error = selector_scores.get(sig, (np.nan, "", ""))
                final_score, final_metric, final_error = final_scores.get(sig, (np.nan, "", ""))
                candidate_rows.append(
                    {
                        "dataset_id": str(dataset_id),
                        "signature": sig,
                        "source": source,
                        "selector_score": sel_score,
                        "selector_metric": sel_metric,
                        "selector_error": sel_error,
                        "final_target_score": final_score,
                        "final_target_metric": final_metric,
                        "final_target_error": final_error,
                        "selected_as_no_search": bool(sig == no_sig),
                        "selected_as_search": bool(sig == search_sig),
                        "selected_as_hybrid": bool(sig == hybrid_sig),
                        "pipeline_config": base._display_config(cfg, options),
                    }
                )

            summary_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "status": "ok",
                    "error": "",
                    "train_rows": int(len(train_df)),
                    "val_search_rows": int(len(val_search_df)),
                    "val_selector_rows": int(len(val_selector_df)),
                    "target_test_rows": int(len(test_df)),
                    "selector_backend": str(args.selector_backend),
                    "candidate_count": int(len(candidates)),
                    "no_search_selector_score": float(no_selector),
                    "search_selector_score": float(search_selector),
                    "hybrid_selector_score": float(hybrid_selector),
                    "no_search_final_target_score": no_final,
                    "search_final_target_score": search_final,
                    "hybrid_final_target_score": hybrid_final,
                    "search_minus_no_search_final": float(search_final - no_final) if np.isfinite(search_final) and np.isfinite(no_final) else np.nan,
                    "hybrid_minus_no_search_final": float(hybrid_final - no_final) if np.isfinite(hybrid_final) and np.isfinite(no_final) else np.nan,
                    "hybrid_minus_search_final": float(hybrid_final - search_final) if np.isfinite(hybrid_final) and np.isfinite(search_final) else np.nan,
                    "no_search_pipeline": base._display_config(no_cfg, options),
                    "search_pipeline": base._display_config(search_cfg, options),
                    "hybrid_source": hybrid_source,
                    "hybrid_pipeline": base._display_config(hybrid_cfg, options),
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
            )
            if args.verbose:
                print(f"Dataset {dataset_id} failed: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    history_df = pd.DataFrame(history_rows)
    summary_df.to_csv(out_dir / "hybrid_fixed_validation_summary.csv", index=False)
    candidate_df.to_csv(out_dir / "hybrid_fixed_validation_candidates.csv", index=False)
    history_df.to_csv(out_dir / "hybrid_fixed_validation_aco_history.csv", index=False)

    ok = summary_df[summary_df.get("status", "") == "ok"].copy() if not summary_df.empty else pd.DataFrame()
    aggregate: Dict[str, Any] = {"n_datasets": int(ok["dataset_id"].nunique()) if not ok.empty else 0}
    if not ok.empty:
        for col in [
            "no_search_final_target_score",
            "search_final_target_score",
            "hybrid_final_target_score",
            "search_minus_no_search_final",
            "hybrid_minus_no_search_final",
            "hybrid_minus_search_final",
        ]:
            aggregate[f"mean_{col}"] = float(pd.to_numeric(ok[col], errors="coerce").mean())
        aggregate["hybrid_beats_no_search_rate"] = float((pd.to_numeric(ok["hybrid_minus_no_search_final"], errors="coerce") > 0).mean())
        aggregate["hybrid_beats_search_rate"] = float((pd.to_numeric(ok["hybrid_minus_search_final"], errors="coerce") > 0).mean())
        aggregate["search_beats_no_search_rate"] = float((pd.to_numeric(ok["search_minus_no_search_final"], errors="coerce") > 0).mean())
        aggregate["hybrid_source_counts"] = ok["hybrid_source"].value_counts(dropna=False).to_dict()

    with (out_dir / "hybrid_fixed_validation_aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(pd.DataFrame([aggregate]).to_string(index=False))
    print(f"Saved fixed-validation hybrid outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
