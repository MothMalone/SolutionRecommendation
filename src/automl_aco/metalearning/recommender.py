"""Meta-learning pipeline recommender (ported from notebook)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import DEFAULT_ORDERING_CONSTRAINTS, DEFAULT_PIPELINE_OPTIONS
from ..utils.logging import get_logger
from ..metalearning.metric import train_siamese_regression_metric, save_metric as save_metric_file, load_metric as load_metric_file
from ..metalearning.dqn_policy import (
    DQNPolicyConfig,
    WarmStartDQNPolicy,
    WarmStartOrderPolicy,
    build_action_offsets,
)
from ..search.heuristics import compute_aco_heuristic
from ..search.aco import search_pipelines_aco
from ..search.optimizers import search_pipelines_with_optimizer
from ..search.evaluation import evaluate_candidates_simple, evaluate_candidates_autogluon
from ..search.ordering import OrderSearchConfig, all_topological_orders, heuristic_score_order, propose_orders

logger = get_logger(__name__)

try:  # optional AutoGluon availability flag
    from autogluon.tabular import TabularPredictor  # type: ignore
    AUTOGLUON_AVAILABLE = True
except Exception:  # pragma: no cover
    AUTOGLUON_AVAILABLE = False


def _autogluon_runtime_error() -> Optional[str]:
    try:
        import numpy as _np
    except Exception as exc:
        return f"NumPy import failed: {exc}"
    try:
        major = int(str(_np.__version__).split(".")[0])
    except Exception:
        major = 0
    if major >= 2:
        return f"AutoGluon requires NumPy < 2.0 (found {_np.__version__})"
    try:
        from autogluon.tabular import TabularPredictor as _TabularPredictor  # noqa: F401
        from autogluon.features.generators import IdentityFeatureGenerator as _IdentityFeatureGenerator  # noqa: F401
    except Exception as exc:
        return str(exc)
    return None


def _is_autogluon_unavailable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = (
        "autogluon not available",
        "requires numpy < 2.0",
        "no module named 'autogluon'",
        "numpy<2",
    )
    return any(m in msg for m in markers)


class MetaPipelineRecommender:
    def __init__(
        self,
        performance_matrix: pd.DataFrame,
        metafeatures_df: pd.DataFrame,
        pipeline_configs: List[Dict[str, Any]],
        pipeline_options: Optional[Dict[str, List[str]]] = None,
        verbose: bool = False,
    ):
        self.performance_matrix = performance_matrix.copy()
        self.metafeatures_df = metafeatures_df.copy()
        self.verbose = verbose

        def _normalize_id(val: Any) -> str:
            s = str(val).strip()
            if s.startswith("D_"):
                s = s[2:]
            if s.startswith("Dataset_"):
                s = s.split("_", 1)[1]
            return s

        perf_cols = list(self.performance_matrix.columns)
        meta_idx = list(self.metafeatures_df.index)

        perf_norm_map: Dict[str, Any] = {}
        for c in perf_cols:
            norm = _normalize_id(c)
            if norm not in perf_norm_map:
                perf_norm_map[norm] = c

        meta_norm_map: Dict[str, Any] = {}
        for i in meta_idx:
            norm = _normalize_id(i)
            if norm not in meta_norm_map:
                meta_norm_map[norm] = i

        common_norm = set(perf_norm_map.keys()) & set(meta_norm_map.keys())
        if not common_norm:
            perf_sample = perf_cols[:10]
            meta_sample = meta_idx[:10]
            raise ValueError(
                "No common datasets between performance_matrix and metafeatures_df. "
                f"Perf sample: {perf_sample} | Meta sample: {meta_sample}"
            )

        def _sort_key(x: str):
            return int(x) if x.isdigit() else x

        common_norm_sorted = sorted(common_norm, key=_sort_key)
        perf_common = [perf_norm_map[k] for k in common_norm_sorted]
        meta_common = [meta_norm_map[k] for k in common_norm_sorted]

        if self.verbose:
            print(f"Aligned datasets: {len(common_norm_sorted)}")
            if len(common_norm_sorted) < 10:
                print(f"Common dataset ids: {common_norm_sorted}")

        self.performance_matrix = self.performance_matrix.loc[:, perf_common]
        self.metafeatures_df = self.metafeatures_df.loc[meta_common, :]

        # Normalize aligned identifiers so perf columns and metafeatures index match
        norm_index = [str(k) for k in common_norm_sorted]
        self.performance_matrix.columns = norm_index
        self.metafeatures_df.index = norm_index

        self.pipeline_configs = pipeline_configs
        self.pipeline_options = pipeline_options or dict(DEFAULT_PIPELINE_OPTIONS)

        self.metafeatures_df = self._sanitize_numeric_frame(self.metafeatures_df, frame_name="metafeatures")
        self.performance_matrix = self._sanitize_numeric_frame(self.performance_matrix, frame_name="performance_matrix")

        if self.performance_matrix.notna().sum().sum() == 0:
            raise ValueError(
                "Performance matrix has no scores for the aligned datasets. "
                "This usually means the metafeatures file does not match the performance matrix dataset IDs."
            )
        if self.metafeatures_df.notna().sum().sum() == 0:
            raise ValueError("Metafeatures are all missing/non-finite after numeric sanitization.")

        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = MinMaxScaler()
        self.metafeatures_imputed = self.imputer.fit_transform(self.metafeatures_df)
        self.metafeatures_scaled = self.scaler.fit_transform(self.metafeatures_imputed)

        self.perf_imputer = SimpleImputer(strategy="mean")
        self.performance_matrix_imputed = pd.DataFrame(
            self.perf_imputer.fit_transform(self.performance_matrix.T).T,
            index=self.performance_matrix.index,
            columns=self.performance_matrix.columns,
        )

        self.embedder = None
        self.projector = None
        self.metric_type = None
        self.metric_params = None

    def _sanitize_numeric_frame(self, frame: pd.DataFrame, frame_name: str) -> pd.DataFrame:
        """Coerce to numeric and replace non-finite values so sklearn imputers can fit safely."""
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        inf_mask = np.isinf(numeric.to_numpy(dtype=float, copy=False))
        inf_count = int(inf_mask.sum())
        if inf_count:
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
        if self.verbose and inf_count:
            print(f"Sanitized {inf_count} inf values in {frame_name} -> NaN")
        return numeric

    def encode_pipeline_config(self, pipe_config: Dict[str, Any], options: Optional[Dict[str, List[str]]] = None) -> np.ndarray:
        opts = options or self.pipeline_options
        if not opts:
            raise ValueError("pipeline options must be provided to encode pipeline configs")
        encoded: List[int] = []
        for step in opts:
            values = opts[step]
            if step not in pipe_config:
                onehot = [0] * len(values)
            else:
                onehot = [1 if pipe_config[step] == v else 0 for v in values]
            encoded.extend(onehot)
        return np.array(encoded, dtype=float)

    def train_metric(self, method: str = "regression", **kwargs):
        if method != "regression":
            raise ValueError("Only 'regression' metric training is implemented")
        model = train_siamese_regression_metric(
            metafeatures_df=self.metafeatures_df,
            performance_matrix_imputed=self.performance_matrix_imputed,
            hidden_dim=kwargs.get("hidden_dim", 64),
            embed_dim=kwargs.get("embed_dim", 64),
            epochs=kwargs.get("epochs", 100),
            lr=kwargs.get("lr", 1e-3),
            seed=kwargs.get("seed", 42),
            similarity_target=kwargs.get("similarity_target", "rank_cosine"),
            score_direction=kwargs.get("score_direction", "higher_is_better"),
        )
        self.embedder = model.embedder
        self.projector = model.projector
        self.metric_type = "regression"
        self.metric_params = model.params
        return model

    def _get_output_dir(self) -> str:
        import os
        if os.path.isdir("/kaggle/working"):
            return "/kaggle/working"
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def default_metric_path(self) -> str:
        import os
        return os.path.join(self._get_output_dir(), "siamese_metric.pt")

    def save_metric(self, path: Optional[str] = None) -> str:
        if self.metric_type != "regression" or self.embedder is None or self.projector is None:
            raise ValueError("No trained regression metric to save")
        if path is None:
            path = self.default_metric_path()
        model = type("Tmp", (), {"embedder": self.embedder, "projector": self.projector, "params": self.metric_params})
        return save_metric_file(model, path)

    def load_metric(self, path: Optional[str] = None, map_location: str = "cpu") -> str:
        if path is None:
            path = self.default_metric_path()
        model = load_metric_file(path, map_location=map_location)
        self.embedder = model.embedder
        self.projector = model.projector
        self.metric_type = "regression"
        self.metric_params = model.params
        return path

    def _compute_dataset_similarities(self, new_metafeatures: np.ndarray) -> List[Tuple[Any, float]]:
        sims: List[Tuple[Any, float]] = []
        if self.metric_type == "regression" and self.embedder is not None and self.projector is not None:
            try:
                import torch
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("torch is required for metric similarity") from exc

            with torch.no_grad():
                known_mf_scaled = self.scaler.transform(self.imputer.transform(self.metafeatures_df))
                known_tensor = torch.tensor(known_mf_scaled, dtype=torch.float32)
                new_tensor = torch.tensor(new_metafeatures.reshape(1, -1), dtype=torch.float32)

                emb_known = self.embedder(known_tensor)
                emb_new = self.embedder(new_tensor).squeeze(0)

                emb_known = emb_known / (emb_known.norm(dim=1, keepdim=True) + 1e-8)
                emb_new = emb_new / (emb_new.norm() + 1e-8)

                for ds_id, h_known in zip(self.metafeatures_df.index, emb_known):
                    inter = (emb_new * h_known).unsqueeze(0)
                    sim = float(self.projector(inter).item())
                    sims.append((ds_id, sim))
            return sims

        known = self.metafeatures_scaled
        cosines = cosine_similarity(known, new_metafeatures.reshape(1, -1)).ravel()
        return list(zip(self.metafeatures_df.index, cosines))

    def _compute_aco_heuristic(
        self,
        new_metafeatures: np.ndarray,
        options: Dict[str, List[str]],
        dataset_weighting: str = "similarity",
        top_k: int = 10,
        use_top_pipelines_from_metric: bool = True,
        recommend_kwargs: Optional[Dict[str, Any]] = None,
        top_l: int = 3,
        similarity_temperature: float = 1.0,
        eta_floor: float = 0.05,
        heuristic_transfer_method: str = "weighted_topk_topl",
        score_direction: str = "higher_is_better",
        query_dataset_id: Optional[Any] = None,
    ) -> Dict[str, np.ndarray]:
        dataset_similarity_scores = dict(self._compute_dataset_similarities(new_metafeatures))
        return compute_aco_heuristic(
            performance_matrix=self.performance_matrix_imputed,
            metafeatures_df=self.metafeatures_df,
            pipeline_configs=self.pipeline_configs,
            options=options,
            new_metafeatures=new_metafeatures,
            dataset_weighting=dataset_weighting,
            top_k=top_k,
            use_top_pipelines_from_metric=use_top_pipelines_from_metric,
            recommend_func=self.recommend if use_top_pipelines_from_metric else None,
            recommend_kwargs=recommend_kwargs,
            metafeatures_scaled=self.metafeatures_scaled,
            dataset_similarity_scores=dataset_similarity_scores,
            top_l=top_l,
            similarity_temperature=similarity_temperature,
            eta_floor=eta_floor,
            heuristic_transfer_method=heuristic_transfer_method,
            score_direction=score_direction,
            query_dataset_id=query_dataset_id,
            verbose=self.verbose,
        )

    def _search_pipelines_aco(
        self,
        new_dataset: Any,
        target_column: str,
        new_metafeatures: np.ndarray,
        options: Dict[str, List[str]],
        proxy_settings: Optional[Dict[str, Any]] = None,
        n_pipelines: int = 3,
        n_ants: int = 3,
        n_iterations: int = 5,
        seed: int = 42,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.2,
        dataset_weighting: str = "similarity",
        heuristic_top_k: int = 10,
        heuristic_top_l: int = 3,
        heuristic_similarity_temperature: float = 1.0,
        heuristic_eta_floor: float = 0.05,
        heuristic_transfer_method: str = "weighted_topk_topl",
        score_direction: str = "higher_is_better",
        query_dataset_id: Optional[Any] = None,
        time_limit_per_model: int = 3000,
        local_search: bool = False,
        metafeatures_func=None,
        top_k_pheromone: int = 3,
        average_pheromone_update: bool = False,
        use_all_iter_pipelines: bool = False,
        weight_method: str = "rank",
        markov_order: int = 2,
        lambda_smooth: float = 0.7,
        step_order: Optional[List[str]] = None,
    ):
        eta = self._compute_aco_heuristic(
            new_metafeatures,
            options,
            dataset_weighting=dataset_weighting,
            top_k=max(1, int(heuristic_top_k)),
            use_top_pipelines_from_metric=True,
            recommend_kwargs={
                "new_dataset": new_dataset,
                "target_column": target_column,
                "options": options,
                "k": 5,
                "eval_k": 3,
                "use_aco": False,
                "time_limit_per_model": time_limit_per_model,
                "use_autogluon": False,
                "metafeatures_func": metafeatures_func,
            },
            top_l=max(1, int(heuristic_top_l)),
            similarity_temperature=float(heuristic_similarity_temperature),
            eta_floor=float(heuristic_eta_floor),
            heuristic_transfer_method=str(heuristic_transfer_method),
            score_direction=str(score_direction),
            query_dataset_id=query_dataset_id,
        )
        if self.verbose:
            print("Phase 3 handoff: received transferred eta_norm for ACO sampling.")

        def _evaluate(sampled_configs):
            if step_order:
                sampled_with_order = []
                for cfg in sampled_configs:
                    cfg_with_order = dict(cfg)
                    cfg_with_order["step_order"] = list(step_order)
                    sampled_with_order.append(cfg_with_order)
                return self._evaluate_candidates_with_simple_models(
                    new_dataset,
                    target_column,
                    sampled_with_order,
                    proxy_settings=proxy_settings,
                )
            return self._evaluate_candidates_with_simple_models(
                new_dataset,
                target_column,
                sampled_configs,
                proxy_settings=proxy_settings,
            )

        result = search_pipelines_aco(
            options=options,
            evaluate_fn=_evaluate,
            eta=eta,
            n_pipelines=n_pipelines,
            n_ants=n_ants,
            n_iterations=n_iterations,
            seed=seed,
            alpha=alpha,
            beta=beta,
            evaporation=evaporation,
            top_k_pheromone=top_k_pheromone,
            use_all_iter_pipelines=use_all_iter_pipelines,
            weight_method=weight_method,
            markov_order=markov_order,
            lambda_smooth=lambda_smooth,
            verbose=self.verbose,
            return_history=True,
        )
        if isinstance(result, tuple) and len(result) == 3:
            return result
        final, unsorted = result
        return final, unsorted, []

    def _evaluate_candidates_with_autogluon(self, dataset, target_column, candidate_configs, time_limit_per_model=300):
        return evaluate_candidates_autogluon(
            dataset=dataset,
            target_column=target_column,
            candidate_configs=candidate_configs,
            time_limit_per_model=time_limit_per_model,
            verbose=self.verbose,
        )

    def _evaluate_candidates_with_simple_models(
        self,
        dataset,
        target_column,
        candidate_configs,
        proxy_settings: Optional[Dict[str, Any]] = None,
    ):
        return evaluate_candidates_simple(
            dataset=dataset,
            target_column=target_column,
            candidate_configs=candidate_configs,
            proxy_settings=proxy_settings,
            verbose=self.verbose,
        )

    def _search_pipelines_optimizer(
        self,
        optimizer: str,
        new_dataset: Any,
        target_column: str,
        options: Dict[str, List[str]],
        proxy_settings: Optional[Dict[str, Any]] = None,
        n_pipelines: int = 3,
        sample_budget: int = 100,
        seed: int = 42,
        step_order: Optional[List[str]] = None,
    ):
        def _evaluate(sampled_configs):
            if step_order:
                sampled_with_order = []
                for cfg in sampled_configs:
                    cfg_with_order = dict(cfg)
                    cfg_with_order["step_order"] = list(step_order)
                    sampled_with_order.append(cfg_with_order)
                return self._evaluate_candidates_with_simple_models(
                    new_dataset,
                    target_column,
                    sampled_with_order,
                    proxy_settings=proxy_settings,
                )
            return self._evaluate_candidates_with_simple_models(
                new_dataset,
                target_column,
                sampled_configs,
                proxy_settings=proxy_settings,
            )

        return search_pipelines_with_optimizer(
            optimizer=optimizer,
            options=options,
            evaluate_fn=_evaluate,
            sample_budget=sample_budget,
            seed=seed,
            n_pipelines=n_pipelines,
            verbose=self.verbose,
        )

    def _search_pipelines_dqn(
        self,
        new_dataset: Any,
        target_column: str,
        new_metafeatures: np.ndarray,
        options: Dict[str, List[str]],
        proxy_settings: Optional[Dict[str, Any]] = None,
        n_pipelines: int = 3,
        sample_budget: int = 100,
        seed: int = 42,
        dataset_weighting: str = "similarity",
        heuristic_top_k: int = 10,
        heuristic_top_l: int = 3,
        heuristic_similarity_temperature: float = 1.0,
        heuristic_eta_floor: float = 0.05,
        heuristic_transfer_method: str = "weighted_topk_topl",
        score_direction: str = "higher_is_better",
        query_dataset_id: Optional[Any] = None,
        time_limit_per_model: int = 3000,
        metafeatures_func=None,
        dqn_params: Optional[Dict[str, Any]] = None,
        enable_internal_order_policy: bool = True,
    ):
        params = dqn_params or {}
        dqn_cfg = DQNPolicyConfig(
            hidden_dim=int(params.get("dqn_hidden_dim", 128)),
            lr=float(params.get("dqn_lr", 3e-4)),
            gamma=float(params.get("dqn_gamma", 0.95)),
            epochs=int(params.get("dqn_epochs", 80)),
            batch_size=int(params.get("dqn_batch_size", 64)),
            target_update_interval=int(params.get("dqn_target_update_interval", 5)),
            warmstart_weight=float(params.get("dqn_warmstart_weight", 0.5)),
            loss_fn=str(params.get("dqn_loss_fn", "huber")),
            huber_delta=float(params.get("dqn_huber_delta", 1.0)),
            grad_clip_norm=float(params.get("dqn_grad_clip_norm", 5.0)),
            reward_clip=float(params.get("dqn_reward_clip", 1.0)),
            target_q_clip=float(params.get("dqn_target_q_clip", 5.0)),
            use_double_dqn=bool(params.get("dqn_use_double_dqn", True)),
        )

        base_order = list(options.keys())
        constraints = [
            (a, b)
            for a, b in DEFAULT_ORDERING_CONSTRAINTS
            if a in base_order and b in base_order
        ]

        order_policy_mode = str(params.get("dqn_order_policy", "ctxpipe")).lower().strip()
        max_logic_orders = int(params.get("dqn_num_logic_orders", 6))

        def _build_ctxpipe_like_orders() -> List[List[str]]:
            all_orders = all_topological_orders(base_order, constraints, limit=None)
            fixed_prefix = [s for s in ("imputation", "encoding") if s in base_order]
            if fixed_prefix:
                filtered = [o for o in all_orders if o[: len(fixed_prefix)] == fixed_prefix]
                all_orders = filtered if filtered else all_orders
            all_orders.sort(key=heuristic_score_order, reverse=True)
            return all_orders[: max(1, max_logic_orders)]

        if enable_internal_order_policy and order_policy_mode == "ctxpipe":
            logic_orders = _build_ctxpipe_like_orders()
        else:
            logic_orders = [base_order]

        policies_by_order: List[WarmStartDQNPolicy] = []
        eta_by_order: List[Dict[str, np.ndarray]] = []
        replay_by_order: List[List[List[Dict[str, Any]]]] = []
        for order in logic_orders:
            ordered_options = {step: options[step] for step in order}
            _offsets, history_dim = build_action_offsets(ordered_options)
            state_dim = int(len(new_metafeatures) + history_dim)
            policies_by_order.append(
                WarmStartDQNPolicy(
                    options=ordered_options,
                    state_dim=state_dim,
                    config=dqn_cfg,
                )
            )
            replay_by_order.append([[] for _ in order])
            eta_by_order.append(
                self._compute_aco_heuristic(
                    new_metafeatures,
                    ordered_options,
                    dataset_weighting=dataset_weighting,
                    top_k=max(1, int(heuristic_top_k)),
                    use_top_pipelines_from_metric=True,
                    recommend_kwargs={
                        "new_dataset": new_dataset,
                        "target_column": target_column,
                        "options": ordered_options,
                        "k": 5,
                        "eval_k": 3,
                        "use_aco": False,
                        "time_limit_per_model": time_limit_per_model,
                        "use_autogluon": False,
                        "metafeatures_func": metafeatures_func,
                    },
                    top_l=max(1, int(heuristic_top_l)),
                    similarity_temperature=float(heuristic_similarity_temperature),
                    eta_floor=float(heuristic_eta_floor),
                    heuristic_transfer_method=str(heuristic_transfer_method),
                    score_direction=str(score_direction),
                    query_dataset_id=query_dataset_id,
                )
            )

        order_prior = np.ones(len(logic_orders), dtype=np.float32)
        if len(logic_orders) > 1:
            vals = []
            for order_idx, order in enumerate(logic_orders):
                eta = eta_by_order[order_idx]
                score = 0.0
                for pos, step in enumerate(order, start=1):
                    v = eta.get(step)
                    if v is None or len(v) == 0:
                        continue
                    score += float(np.max(v)) / float(pos)
                vals.append(score)
            vals_arr = np.asarray(vals, dtype=np.float32)
            mn = float(np.min(vals_arr))
            mx = float(np.max(vals_arr))
            if mx - mn > 1e-12:
                order_prior = (vals_arr - mn) / (mx - mn + 1e-12)
            else:
                order_prior = np.ones_like(vals_arr, dtype=np.float32)

        order_policy: Optional[WarmStartOrderPolicy] = None
        order_replay: List[Dict[str, Any]] = []
        if len(logic_orders) > 1:
            order_policy = WarmStartOrderPolicy(
                state_dim=int(len(new_metafeatures)),
                order_dim=len(logic_orders),
                hidden_dim=max(16, dqn_cfg.hidden_dim // 2),
                lr=dqn_cfg.lr,
                gamma=dqn_cfg.gamma,
                reward_clip=dqn_cfg.reward_clip,
                target_q_clip=dqn_cfg.target_q_clip,
                grad_clip_norm=dqn_cfg.grad_clip_norm,
                use_double_dqn=dqn_cfg.use_double_dqn,
                huber_delta=dqn_cfg.huber_delta,
            )

        # Online proxy-reward training loop (same proxy signal as ACORec).
        # This lets RL optimize directly against the current dataset's proxy objective.
        eps_start = float(params.get("dqn_epsilon_start", 0.35))
        eps_end = float(params.get("dqn_epsilon_end", 0.05))
        updates_per_episode = int(params.get("dqn_updates_per_episode", params.get("dqn_epochs", 1)))
        replay_warmup = int(params.get("dqn_replay_warmup", 16))
        target_sync = max(1, int(params.get("dqn_target_update_interval", 5)))
        order_eps_start = float(params.get("dqn_order_epsilon_start", eps_start))
        order_eps_end = float(params.get("dqn_order_epsilon_end", eps_end))
        order_updates = int(params.get("dqn_order_updates_per_episode", updates_per_episode))
        order_warmup = int(params.get("dqn_order_replay_warmup", replay_warmup))

        rng = np.random.RandomState(seed)
        evaluated: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], float]] = {}
        unsorted_results: List[Tuple[Dict[str, Any], float]] = []
        history: List[Dict[str, Any]] = []
        running_best: Optional[float] = None

        max_attempts = max(sample_budget * 5, sample_budget)
        episodes = int(sample_budget)
        for i in range(max_attempts):
            if len(evaluated) >= episodes:
                break
            frac = float(len(evaluated)) / float(max(1, episodes - 1))
            eps = eps_start + (eps_end - eps_start) * frac
            order_eps = order_eps_start + (order_eps_end - order_eps_start) * frac
            if order_policy is not None:
                order_idx = order_policy.sample_order(
                    metafeatures=new_metafeatures.astype(np.float32),
                    order_context=order_prior,
                    rng=rng,
                    epsilon=max(0.0, min(1.0, order_eps)),
                )
            else:
                order_idx = 0

            step_order = logic_orders[order_idx]
            ordered_options = {step: options[step] for step in step_order}
            policy = policies_by_order[order_idx]
            eta = eta_by_order[order_idx]
            cfg = policy.sample_pipeline(
                metafeatures=new_metafeatures.astype(np.float32),
                warm_context=eta,
                rng=rng,
                epsilon=max(0.0, min(1.0, eps)),
            )
            key = (order_idx, tuple((step, cfg.get(step)) for step in step_order))
            if key in evaluated:
                continue

            cfg_with_order = dict(cfg)
            cfg_with_order["step_order"] = list(step_order)
            _best, _best_score, sorted_eval, _unsorted_eval = self._evaluate_candidates_with_simple_models(
                new_dataset,
                target_column,
                [cfg_with_order],
                proxy_settings=proxy_settings,
            )
            if not sorted_eval:
                continue
            score = float(sorted_eval[0][1])
            evaluated[key] = (cfg_with_order, score)
            unsorted_results.append((cfg_with_order, score))

            # Convert this evaluated episode into transitions.
            offsets, history_dim = build_action_offsets(ordered_options)
            episode_history = np.zeros(history_dim, dtype=np.float32)
            for step_idx, step in enumerate(step_order):
                action = ordered_options[step].index(cfg_with_order[step])
                state = np.concatenate([new_metafeatures.astype(np.float32), episode_history]).astype(np.float32)
                context = np.asarray(eta.get(step), dtype=np.float32)

                next_history = episode_history.copy()
                next_history[offsets[step] + action] = 1.0
                next_state = np.concatenate([new_metafeatures.astype(np.float32), next_history]).astype(np.float32)

                done = step_idx == (len(step_order) - 1)
                reward = score if done else 0.0
                next_context = (
                    np.asarray(eta.get(step_order[step_idx + 1]), dtype=np.float32)
                    if not done
                    else np.zeros((1,), dtype=np.float32)
                )

                replay_by_order[order_idx][step_idx].append(
                    {
                        "state": state,
                        "context": context,
                        "action": int(action),
                        "reward": float(reward),
                        "done": bool(done),
                        "next_state": next_state,
                        "next_context": next_context,
                    }
                )
                episode_history = next_history

            step_policy_stats: Dict[str, Any] = {"updates": 0, "mean_loss": None, "last_loss": None}
            if len(evaluated) >= replay_warmup:
                step_policy_stats = policy.learn_from_replay(
                    replay_by_step=replay_by_order[order_idx],
                    rng=rng,
                    n_updates=updates_per_episode,
                )
                if len(evaluated) % target_sync == 0:
                    policy.sync_target()

            order_policy_stats: Dict[str, Any] = {"updates": 0, "mean_loss": None, "last_loss": None}
            if order_policy is not None:
                state = new_metafeatures.astype(np.float32)
                context = order_prior.astype(np.float32)
                order_replay.append(
                    {
                        "state": state,
                        "context": context,
                        "action": int(order_idx),
                        "reward": float(score),
                        "done": True,
                        "next_state": state,
                        "next_context": context,
                    }
                )
                if len(order_replay) >= order_warmup:
                    order_policy_stats = order_policy.learn_from_replay(
                        replay=order_replay,
                        rng=rng,
                        n_updates=order_updates,
                        batch_size=int(params.get("dqn_order_batch_size", dqn_cfg.batch_size)),
                    )
                    if len(order_replay) % target_sync == 0:
                        order_policy.sync_target()

            if running_best is None or score > running_best:
                running_best = score
            history.append(
                {
                    "iteration": len(evaluated),
                    "episode_score": float(score),
                    "best_score": running_best,
                    "epsilon": float(max(0.0, min(1.0, eps))),
                    "order_epsilon": float(max(0.0, min(1.0, order_eps))),
                    "order_idx": int(order_idx),
                    "order": list(step_order),
                    "policy_updates": int(step_policy_stats.get("updates", 0) or 0),
                    "policy_mean_loss": step_policy_stats.get("mean_loss"),
                    "policy_last_loss": step_policy_stats.get("last_loss"),
                    "order_policy_updates": int(order_policy_stats.get("updates", 0) or 0),
                    "order_policy_mean_loss": order_policy_stats.get("mean_loss"),
                    "order_policy_last_loss": order_policy_stats.get("last_loss"),
                }
            )

        sorted_results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)

        if not sorted_results:
            return [], [], history

        return sorted_results[:n_pipelines], unsorted_results, history

    def recommend(
        self,
        new_dataset,
        target_column: Optional[str] = None,
        k: int = 5,
        eval_k: int = 3,
        use_autogluon: bool = True,
        time_limit_per_model: int = 300,
        metafeatures_func=None,
        use_aco: bool = False,
        aco_params: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, List[str]]] = None,
        search_ordering: bool = False,
        num_orders: int = 1,
        order_strategy: str = "fixed",
        order_constraints: Optional[List[Tuple[str, str]]] = None,
        optimizer: str = "aco",
        sample_budget: int = 100,
        proxy_settings: Optional[Dict[str, Any]] = None,
        final_autogluon_topk: int = 1,
    ) -> Dict[str, Any]:
        if metafeatures_func is None:
            raise ValueError("metafeatures_func must be provided")

        options = options or self.pipeline_options
        aco_params = aco_params or {}
        require_autogluon = bool(aco_params.get("require_autogluon", True))

        if use_autogluon and require_autogluon:
            ag_err = _autogluon_runtime_error()
            if ag_err is not None:
                raise RuntimeError(
                    "AutoGluon is required for evaluation but is unavailable. "
                    f"Reason: {ag_err}. "
                    "Install compatible dependencies (e.g., requirements-kaggle.txt) and retry."
                )

        new_mf = metafeatures_func(new_dataset)
        new_mf_df = pd.DataFrame([new_mf]).reindex(columns=self.metafeatures_df.columns, fill_value=0)
        new_mf_imputed = self.imputer.transform(new_mf_df)
        new_mf_scaled = self.scaler.transform(new_mf_imputed).ravel()

        if use_aco:
            base_order = list(options.keys())
            aco_seed = int(aco_params.get("seed", 42))
            autogluon_runtime_enabled = bool(AUTOGLUON_AVAILABLE)
            optimizer_name = optimizer.lower().strip()
            if optimizer_name == "":
                optimizer_name = "aco"
            if search_ordering:
                constraints = order_constraints if order_constraints is not None else DEFAULT_ORDERING_CONSTRAINTS
                constraints = [(a, b) for a, b in constraints if a in base_order and b in base_order]
                order_cfg = OrderSearchConfig(
                    steps=tuple(base_order),
                    constraints=tuple(constraints),
                    max_orders=max(1, int(num_orders)),
                    strategy=order_strategy,
                    seed=aco_seed,
                )
                candidate_orders = propose_orders(order_cfg)
            else:
                candidate_orders = [base_order]

            all_top_results: List[Tuple[Dict[str, Any], float]] = []
            all_unsorted_results: List[Tuple[Dict[str, Any], float]] = []
            all_history: List[Dict[str, Any]] = []
            order_iteration_results: List[Dict[str, Any]] = []
            global_iteration = 1
            run_quick_order_eval = bool(
                search_ordering
                and use_autogluon
                and autogluon_runtime_enabled
                and target_column is not None
            )
            quick_order_eval_time_limit = int(aco_params.get("ordering_quick_time_limit", 30))
            quick_order_eval_time_limit = max(5, quick_order_eval_time_limit)

            for order_idx, order in enumerate(candidate_orders, start=1):
                ordered_options = {step: options[step] for step in order}
                if optimizer_name == "aco":
                    aco_results, aco_unsorted_res, aco_history = self._search_pipelines_aco(
                        new_dataset,
                        target_column,
                        new_mf_scaled,
                        ordered_options,
                        proxy_settings=proxy_settings,
                        n_pipelines=k,
                        n_ants=aco_params.get("n_ants", 10),
                        n_iterations=aco_params.get("n_iterations", 10),
                        seed=aco_seed + order_idx - 1,
                        alpha=float(aco_params.get("alpha", 1.0)),
                        beta=float(aco_params.get("beta", 2.0)),
                        evaporation=float(aco_params.get("evaporation", 0.2)),
                        dataset_weighting=str(aco_params.get("dataset_weighting", "similarity")),
                        heuristic_top_k=int(aco_params.get("heuristic_top_k", k)),
                        heuristic_top_l=int(aco_params.get("heuristic_top_l", 3)),
                        heuristic_similarity_temperature=float(aco_params.get("heuristic_similarity_temperature", 1.0)),
                        heuristic_eta_floor=float(aco_params.get("heuristic_eta_floor", 0.05)),
                        heuristic_transfer_method=str(aco_params.get("heuristic_transfer_method", "weighted_topk_topl")),
                        score_direction=str(aco_params.get("score_direction", "higher_is_better")),
                        query_dataset_id=aco_params.get("query_dataset_id"),
                        time_limit_per_model=time_limit_per_model,
                        metafeatures_func=metafeatures_func,
                        step_order=order,
                    )
                elif optimizer_name == "dqn":
                    aco_results, aco_unsorted_res, aco_history = self._search_pipelines_dqn(
                        new_dataset=new_dataset,
                        target_column=target_column,
                        new_metafeatures=new_mf_scaled,
                        options=ordered_options,
                        proxy_settings=proxy_settings,
                        n_pipelines=k,
                        sample_budget=sample_budget,
                        seed=aco_seed + order_idx - 1,
                        dataset_weighting=str(aco_params.get("dataset_weighting", "similarity")),
                        heuristic_top_k=int(aco_params.get("heuristic_top_k", k)),
                        heuristic_top_l=int(aco_params.get("heuristic_top_l", 3)),
                        heuristic_similarity_temperature=float(aco_params.get("heuristic_similarity_temperature", 1.0)),
                        heuristic_eta_floor=float(aco_params.get("heuristic_eta_floor", 0.05)),
                        heuristic_transfer_method=str(aco_params.get("heuristic_transfer_method", "weighted_topk_topl")),
                        score_direction=str(aco_params.get("score_direction", "higher_is_better")),
                        query_dataset_id=aco_params.get("query_dataset_id"),
                        time_limit_per_model=time_limit_per_model,
                        metafeatures_func=metafeatures_func,
                        dqn_params=aco_params,
                        enable_internal_order_policy=not search_ordering,
                    )
                else:
                    aco_results, aco_unsorted_res, aco_history = self._search_pipelines_optimizer(
                        optimizer=optimizer_name,
                        new_dataset=new_dataset,
                        target_column=target_column,
                        options=ordered_options,
                        proxy_settings=proxy_settings,
                        n_pipelines=k,
                        sample_budget=sample_budget,
                        seed=aco_seed + order_idx - 1,
                        step_order=order,
                    )

                for cfg, score in aco_results:
                    cfg2 = dict(cfg)
                    cfg2["step_order"] = list(order)
                    all_top_results.append((cfg2, float(score)))
                for cfg, score in aco_unsorted_res:
                    cfg2 = dict(cfg)
                    cfg2["step_order"] = list(order)
                    all_unsorted_results.append((cfg2, float(score)))
                order_best_cfg: Optional[Dict[str, Any]] = None
                order_proxy_score: Optional[float] = None
                if aco_results:
                    order_best_cfg = dict(aco_results[0][0])
                    order_best_cfg["step_order"] = list(order)
                    order_proxy_score = float(aco_results[0][1])

                order_autogluon_score: Optional[float] = None
                order_autogluon_error: Optional[str] = None
                if run_quick_order_eval and order_best_cfg is not None:
                    try:
                        ag_cfg, ag_score, ag_results, _ = self._evaluate_candidates_with_autogluon(
                            new_dataset,
                            target_column,
                            [order_best_cfg],
                            time_limit_per_model=quick_order_eval_time_limit,
                        )
                        if ag_cfg is not None and ag_results and np.isfinite(ag_score):
                            order_best_cfg = dict(ag_cfg)
                            order_best_cfg["step_order"] = list(order)
                            order_autogluon_score = float(ag_score)
                            if self.verbose:
                                print(
                                    f"Ordering Iter {order_idx}/{len(candidate_orders)} "
                                    f"quick AutoGluon: {order_autogluon_score:.4f}"
                                )
                        else:
                            order_autogluon_error = "No valid quick AutoGluon result"
                    except Exception as exc:
                        order_autogluon_error = str(exc)
                        if _is_autogluon_unavailable_error(exc):
                            if require_autogluon:
                                raise RuntimeError(
                                    "AutoGluon is required but unavailable during quick ordering evaluation. "
                                    f"Reason: {exc}"
                                ) from exc
                            autogluon_runtime_enabled = False
                            run_quick_order_eval = False
                            if self.verbose:
                                print(
                                    f"Ordering Iter {order_idx}/{len(candidate_orders)} "
                                    "quick AutoGluon unavailable; continuing with proxy-only ordering."
                                )
                            else:
                                logger.info(
                                    "Quick AutoGluon unavailable for ordering iteration %s; "
                                    "continuing with proxy-only ordering.",
                                    order_idx,
                                )
                        else:
                            if self.verbose:
                                print(
                                    f"Ordering Iter {order_idx}/{len(candidate_orders)} "
                                    f"quick AutoGluon failed: {exc}"
                                )
                            else:
                                logger.warning(
                                    "Quick AutoGluon failed for ordering iteration %s: %s",
                                    order_idx,
                                    exc,
                                )

                order_selection_score = (
                    order_autogluon_score
                    if order_autogluon_score is not None
                    else order_proxy_score
                )
                order_iteration_results.append(
                    {
                        "iteration": order_idx,
                        "order_index": order_idx,
                        "step_order": list(order),
                        "pipeline_config": order_best_cfg,
                        "proxy_score": order_proxy_score,
                        "autogluon_score": order_autogluon_score,
                        "selection_score": order_selection_score,
                        "autogluon_error": order_autogluon_error,
                    }
                )

                if search_ordering:
                    all_history.append(
                        {
                            "iteration": order_idx,
                            "best_score": order_selection_score,
                            "proxy_score": order_proxy_score,
                            "autogluon_score": order_autogluon_score,
                            "order_index": order_idx,
                            "step_order": list(order),
                        }
                    )
                else:
                    for hist in aco_history:
                        if not isinstance(hist, dict):
                            continue
                        all_history.append(
                            {
                                "iteration": global_iteration,
                                "best_score": hist.get("best_score"),
                                "episode_score": hist.get("episode_score"),
                                "epsilon": hist.get("epsilon"),
                                "order_epsilon": hist.get("order_epsilon"),
                                "order_index": int(hist.get("order_idx", order_idx)),
                                "step_order": hist.get("order", list(order)),
                                "policy_updates": hist.get("policy_updates"),
                                "policy_mean_loss": hist.get("policy_mean_loss"),
                                "policy_last_loss": hist.get("policy_last_loss"),
                                "order_policy_updates": hist.get("order_policy_updates"),
                                "order_policy_mean_loss": hist.get("order_policy_mean_loss"),
                                "order_policy_last_loss": hist.get("order_policy_last_loss"),
                            }
                        )
                        global_iteration += 1

            used_order_level_scoring = False
            if search_ordering and order_iteration_results:
                ranked_orders = [
                    item for item in order_iteration_results
                    if item.get("pipeline_config") is not None and item.get("selection_score") is not None
                ]
                ranked_orders.sort(key=lambda x: float(x["selection_score"]), reverse=True)
                if ranked_orders:
                    used_order_level_scoring = True
                    aco_results = [
                        (dict(item["pipeline_config"]), float(item["selection_score"]))
                        for item in ranked_orders[:k]
                    ]
                    aco_unsorted_res = [
                        (dict(item["pipeline_config"]), float(item["selection_score"]))
                        for item in ranked_orders
                    ]
                    best_pipeline, best_score = aco_results[0]
                else:
                    aco_results = []
                    aco_unsorted_res = []
            else:
                aco_results = []
                aco_unsorted_res = []

            if not aco_results:
                dedup: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], float]] = {}
                dedup_source = all_unsorted_results if all_unsorted_results else all_top_results
                for cfg, score in dedup_source:
                    key = tuple((step, cfg.get(step)) for step in base_order) + (("step_order", tuple(cfg.get("step_order", []))),)
                    if key not in dedup or score > dedup[key][1]:
                        dedup[key] = (cfg, score)

                if not dedup:
                    raise RuntimeError("ACO search produced no valid pipeline candidates.")

                final_ranked = sorted(dedup.values(), key=lambda x: x[1], reverse=True)
                aco_results = final_ranked[:k]
                aco_unsorted_res = all_unsorted_results
                best_pipeline, best_score = aco_results[0]

            final_eval = {"method": "proxy", "score": float(best_score)}
            if use_autogluon:
                if autogluon_runtime_enabled and target_column is not None:
                    try:
                        topk = max(1, int(final_autogluon_topk))
                        if aco_results:
                            ag_candidates = [dict(cfg) for cfg, _sc in aco_results[:topk]]
                        else:
                            ag_candidates = [best_pipeline]
                        ag_best_cfg, ag_score, ag_results, _ag_unsorted = self._evaluate_candidates_with_autogluon(
                            new_dataset,
                            target_column,
                            ag_candidates,
                            time_limit_per_model=time_limit_per_model,
                        )
                        if ag_best_cfg is not None and ag_results and np.isfinite(ag_score):
                            best_pipeline = ag_best_cfg
                            final_eval = {"method": "autogluon", "score": float(ag_score)}
                        else:
                            final_eval = {
                                "method": "autogluon_failed",
                                "score": float(best_score),
                                "error": "No candidate produced valid AutoGluon evaluation results",
                            }
                    except Exception as exc:
                        if _is_autogluon_unavailable_error(exc):
                            if require_autogluon:
                                raise RuntimeError(
                                    "AutoGluon is required but unavailable during final evaluation. "
                                    f"Reason: {exc}"
                                ) from exc
                            autogluon_runtime_enabled = False
                            if self.verbose:
                                print("Final AutoGluon unavailable; falling back to simple-model final evaluation.")
                            else:
                                logger.info(
                                    "AutoGluon unavailable at final evaluation; "
                                    "falling back to simple-model final evaluation."
                                )
                            topk = max(1, int(final_autogluon_topk))
                            if aco_results:
                                fallback_candidates = [dict(cfg) for cfg, _sc in aco_results[:topk]]
                            else:
                                fallback_candidates = [best_pipeline]
                            simple_best_cfg, simple_best_score, simple_all, _simple_unsorted = (
                                self._evaluate_candidates_with_simple_models(
                                    new_dataset,
                                    target_column,
                                    fallback_candidates,
                                    proxy_settings=proxy_settings,
                                )
                            )
                            if simple_best_cfg is not None and simple_all and np.isfinite(simple_best_score):
                                best_pipeline = simple_best_cfg
                                final_eval = {"method": "simple_models_fallback", "score": float(simple_best_score)}
                            else:
                                final_eval = {"method": "autogluon_unavailable", "score": float(best_score), "error": str(exc)}
                        else:
                            logger.warning("AutoGluon final evaluation failed: %s", exc)
                            final_eval = {"method": "autogluon_failed", "score": float(best_score), "error": str(exc)}
                else:
                    final_eval = {"method": "autogluon_unavailable", "score": float(best_score)}
            return {
                "pipeline_config": best_pipeline,
                "recommended_performance": best_score,
                "final_evaluation": final_eval,
                "final_performance": float(final_eval.get("score", best_score)),
                "confidence": "high" if best_score > 0.8 else "low",
                "aco_results": aco_unsorted_res,
                "aco_history": all_history,
                "optimizer": optimizer_name,
                "ordering_search": {
                    "enabled": bool(search_ordering),
                    "strategy": order_strategy,
                    "num_orders_requested": int(num_orders),
                    "num_orders_evaluated": len(candidate_orders),
                    "orders": candidate_orders,
                    "quick_autogluon_per_iteration": bool(run_quick_order_eval),
                    "quick_autogluon_time_limit": quick_order_eval_time_limit if run_quick_order_eval else None,
                    "selection_metric": "autogluon_quick_or_proxy" if used_order_level_scoring else "proxy",
                },
                "ordering_iteration_results": order_iteration_results if search_ordering else [],
                "proxy_settings": proxy_settings or {},
                "final_autogluon_topk": max(1, int(final_autogluon_topk)),
                "aco_hyperparams": {
                    "alpha": float(aco_params.get("alpha", 1.0)),
                    "beta": float(aco_params.get("beta", 2.0)),
                    "evaporation": float(aco_params.get("evaporation", 0.2)),
                    "dataset_weighting": str(aco_params.get("dataset_weighting", "similarity")),
                    "heuristic_top_k": int(aco_params.get("heuristic_top_k", k)),
                    "heuristic_top_l": int(aco_params.get("heuristic_top_l", 3)),
                    "heuristic_similarity_temperature": float(
                        aco_params.get("heuristic_similarity_temperature", 1.0)
                    ),
                    "heuristic_eta_floor": float(aco_params.get("heuristic_eta_floor", 0.05)),
                    "heuristic_transfer_method": str(
                        aco_params.get("heuristic_transfer_method", "weighted_topk_topl")
                    ),
                    "score_direction": str(aco_params.get("score_direction", "higher_is_better")),
                    "require_autogluon": bool(aco_params.get("require_autogluon", True)),
                },
            }

        sims = self._compute_dataset_similarities(new_mf_scaled)

        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        top_datasets = [ds for ds, _ in sims[:k]]
        top_sims = np.array([s for _, s in sims[:k]], dtype=float)
        if top_sims.sum() == 0:
            top_sims = np.ones_like(top_sims)

        perf_subset = self.performance_matrix.loc[:, top_datasets].fillna(0)
        weighted_avg_perf = np.average(perf_subset.values, axis=1, weights=top_sims)
        candidate_perfs = pd.Series(weighted_avg_perf, index=self.performance_matrix.index)
        pipeline_ranking = candidate_perfs.sort_values(ascending=False).index.tolist()

        top_candidate_names = pipeline_ranking[:eval_k]
        top_candidate_configs = [cfg for cfg in self.pipeline_configs if cfg.get("name") in top_candidate_names]

        if use_autogluon and AUTOGLUON_AVAILABLE and target_column is not None and len(top_candidate_configs) > 0:
            eval_method = "autogluon"
            try:
                best_cfg, best_score, all_results, _unsorted_res = self._evaluate_candidates_with_autogluon(
                    new_dataset,
                    target_column,
                    top_candidate_configs,
                    time_limit_per_model=time_limit_per_model,
                )
            except Exception as exc:
                logger.warning("AutoGluon evaluation failed, falling back to simple models: %s", exc)
                eval_method = "simple_models"
                best_cfg, best_score, all_results, _unsorted_res = self._evaluate_candidates_with_simple_models(
                    new_dataset,
                    target_column,
                    top_candidate_configs,
                )

            if best_cfg is None or not all_results:
                top_pipeline_name = pipeline_ranking[0]
                top_pipeline_score = candidate_perfs[top_pipeline_name]
                top_pipeline_config = next(
                    (cfg for cfg in self.pipeline_configs if cfg.get("name") == top_pipeline_name),
                    None,
                )
                return {
                    "pipeline_config": top_pipeline_config,
                    "expected_performance": float(top_pipeline_score),
                    "similar_datasets": top_datasets,
                    "pipeline_ranking": pipeline_ranking[:k],
                    "top_candidates": [(cfg["name"], float(candidate_perfs[cfg["name"]])) for cfg in top_candidate_configs],
                    "confidence": "low",
                    "similarity_scores": dict(sims[:k]),
                    "model_type": self.metric_type,
                    "evaluation_method": "fallback_prediction_only",
                }

            return {
                "pipeline_config": best_cfg,
                "expected_performance": float(best_score),
                "similar_datasets": top_datasets,
                "pipeline_ranking": all_results,
                "top_candidates_evaluated": [(cfg["name"], sc) for cfg, sc in all_results],
                "confidence": "high",
                "similarity_scores": dict(sims[:k]),
                "model_type": self.metric_type,
                "evaluation_method": eval_method,
            }

        top_pipeline_name = pipeline_ranking[0]
        top_pipeline_score = candidate_perfs[top_pipeline_name]
        top_pipeline_config = next(
            (cfg for cfg in self.pipeline_configs if cfg.get("name") == top_pipeline_name),
            None,
        )
        return {
            "pipeline_config": top_pipeline_config,
            "expected_performance": float(top_pipeline_score),
            "similar_datasets": top_datasets,
            "pipeline_ranking": pipeline_ranking[:k],
            "top_candidates": [(cfg["name"], float(candidate_perfs[cfg["name"]])) for cfg in top_candidate_configs],
            "confidence": "medium",
            "similarity_scores": dict(sims[:k]),
            "model_type": self.metric_type,
            "evaluation_method": "prediction_only",
        }
