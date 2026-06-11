"""Meta-learning pipeline recommender (ported from notebook)."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import re

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
from ..search.heuristics import compute_aco_heuristic, select_top_k_neighbors
from ..search.aco import search_pipelines_aco
from ..search.optimizers import search_pipelines_with_optimizer
from ..search.evaluation import evaluate_candidates_simple, evaluate_candidates_autogluon
from ..search.ordering import OrderSearchConfig, all_topological_orders, heuristic_score_order, propose_orders
from ..utils.operator_spec import base_operator_name

logger = get_logger(__name__)

try:  # optional AutoGluon availability flag
    from autogluon.tabular import TabularPredictor  # type: ignore
    AUTOGLUON_AVAILABLE = True
except Exception:  # pragma: no cover
    AUTOGLUON_AVAILABLE = False


def _autogluon_runtime_error() -> Optional[str]:
    # Prefer a direct runtime import check. If AutoGluon imports successfully,
    # we treat the environment as usable even if metadata/version pins look odd.
    try:
        from autogluon.tabular import TabularPredictor as _TabularPredictor  # noqa: F401
        from autogluon.features.generators import IdentityFeatureGenerator as _IdentityFeatureGenerator  # noqa: F401
        return None
    except Exception as exc:
        import_error = str(exc)
    try:
        import numpy as _np
    except Exception as exc:
        return f"NumPy import failed: {exc}; AutoGluon import failed: {import_error}"
    try:
        major = int(str(_np.__version__).split(".")[0])
    except Exception:
        major = 0
    if major >= 2:
        return f"AutoGluon requires NumPy < 2.0 (found {_np.__version__}). Import error: {import_error}"
    return import_error


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
            if pd.isna(val):
                return ""
            if isinstance(val, (int, np.integer)):
                return str(int(val))
            if isinstance(val, (float, np.floating)):
                f = float(val)
                if np.isfinite(f) and abs(f - round(f)) <= 1e-9:
                    return str(int(round(f)))
                return str(val).strip()

            s = str(val).strip()
            float_like = re.fullmatch(r"([0-9]+)\.0+", s)
            if float_like:
                return float_like.group(1)

            prefixed = re.fullmatch(r"(?i)(?:d|dataset|openml)[_\-: ]*([0-9]+)", s)
            if prefixed:
                return prefixed.group(1)
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
        lambda_smooth: float = 0.0,
        early_stop_rounds: int = 0,
        min_improvement: float = 0.0,
        step_order: Optional[List[str]] = None,
        legacy_notebook_aco: bool = False,
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
            early_stop_rounds=early_stop_rounds,
            min_improvement=min_improvement,
            verbose=self.verbose,
            return_history=True,
            legacy_notebook_aco=legacy_notebook_aco,
        )
        if isinstance(result, tuple) and len(result) == 3:
            return result
        final, unsorted = result
        return final, unsorted, []

    def _per_feature_candidate_values(
        self,
        *,
        step: str,
        feature_kind: str,
        options: Dict[str, List[str]],
    ) -> List[str]:
        raw_values = [str(v) for v in options.get(step, [])]
        if not raw_values:
            return ["none"]

        def _dedup(vals: List[str]) -> List[str]:
            seen = set()
            out = []
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
            return out

        if step == "scaling":
            if feature_kind != "numeric":
                return ["none"]
            return _dedup(raw_values)

        if step == "encoding":
            if feature_kind != "categorical":
                return ["none"]
            return _dedup(raw_values)

        if step == "imputation":
            if feature_kind == "categorical":
                preferred = [v for v in raw_values if base_operator_name(v) in {"none", "most_frequent", "constant"}]
                if preferred:
                    return _dedup(preferred)
            return _dedup(raw_values)

        return ["none"]

    def _search_pipelines_aco_per_feature(
        self,
        *,
        new_dataset: Any,
        target_column: str,
        new_metafeatures: np.ndarray,
        options: Dict[str, List[str]],
        proxy_settings: Optional[Dict[str, Any]],
        n_pipelines: int,
        n_ants: int,
        n_iterations: int,
        seed: int,
        alpha: float,
        beta: float,
        evaporation: float,
        top_k_pheromone: int,
        weight_method: str,
        markov_order: int,
        lambda_smooth: float,
        use_all_iter_pipelines: bool,
        step_order: Optional[List[str]],
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
        per_feature_steps: Optional[List[str]] = None,
        per_feature_early_stop_rounds: int = 0,
        per_feature_min_improvement: float = 0.0,
        per_feature_feature_patience: int = 0,
        per_feature_feature_min_improvement: float = 0.0,
        per_feature_max_features: int = 0,
    ) -> Tuple[
        List[Tuple[Dict[str, Any], float]],
        List[Tuple[Dict[str, Any], float]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        if not isinstance(new_dataset, pd.DataFrame):
            raise ValueError("Per-feature ACO mode requires new_dataset as DataFrame")
        if target_column not in new_dataset.columns:
            raise ValueError(f"Target column {target_column!r} missing from dataset")

        X_df = new_dataset.drop(columns=[target_column]).copy()
        if X_df.empty:
            raise ValueError("Dataset has no feature columns for per-feature optimization")

        features = [str(c) for c in X_df.columns]
        kinds = {
            str(c): ("numeric" if pd.api.types.is_numeric_dtype(X_df[c]) else "categorical")
            for c in X_df.columns
        }

        def _choose_global(step: str, default: str = "none") -> str:
            vals = [str(v) for v in options.get(step, [])]
            if not vals:
                return default
            if len(vals) == 1:
                return vals[0]
            for v in vals:
                if base_operator_name(v) == base_operator_name(default):
                    return v
            for v in vals:
                if base_operator_name(v) == "none":
                    return v
            return vals[0]

        outlier_choice = _choose_global("outlier_removal", default="none")
        fs_choice = _choose_global("feature_selection", default="none")
        dr_choice = _choose_global("dimensionality_reduction", default="none")
        order = list(step_order) if isinstance(step_order, list) and step_order else list(options.keys())

        # Keep Phase-1/2 transfer active: seed per-feature local eta from
        # transferred global operator heuristic (eta_norm).
        global_eta = self._compute_aco_heuristic(
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

        imputation_map: Dict[str, str] = {}
        scaling_map: Dict[str, str] = {}
        encoding_map: Dict[str, str] = {}

        encoding_defaults = [str(v) for v in options.get("encoding", []) if base_operator_name(v) != "none"]
        default_encoding = encoding_defaults[0] if encoding_defaults else "onehot"
        for col in features:
            kind = kinds[col]
            has_missing = bool(X_df[col].isna().any())
            if kind == "numeric":
                imputation_map[col] = "mean" if has_missing else "none"
                scaling_map[col] = "none"
                encoding_map[col] = "none"
            else:
                imputation_map[col] = "most_frequent" if has_missing else "none"
                scaling_map[col] = "none"
                encoding_map[col] = default_encoding

        indep_steps = per_feature_steps or ["imputation", "scaling", "encoding"]
        indep_steps = [s for s in indep_steps if s in {"imputation", "scaling", "encoding"}]
        if not indep_steps:
            indep_steps = ["imputation", "scaling", "encoding"]

        feature_logs: List[Dict[str, Any]] = []
        all_history: List[Dict[str, Any]] = []
        all_unsorted: List[Tuple[Dict[str, Any], float]] = []
        stage_best_score: Optional[float] = None
        no_improve_features = 0
        global_iter = 1

        feature_sequence = list(features)
        if int(per_feature_max_features) > 0:
            feature_sequence = feature_sequence[: int(per_feature_max_features)]

        for feat_idx, feature in enumerate(feature_sequence, start=1):
            kind = kinds[feature]
            local_options: Dict[str, List[str]] = {}
            for step in indep_steps:
                local_options[step] = self._per_feature_candidate_values(
                    step=step,
                    feature_kind=kind,
                    options=options,
                )
            if not local_options:
                continue

            local_eta: Dict[str, np.ndarray] = {}
            local_eta_by_choice: Dict[str, Dict[str, float]] = {}
            for step, vals in local_options.items():
                global_vals = [str(v) for v in options.get(step, [])]
                step_eta = np.asarray(global_eta.get(step, np.ones(len(global_vals), dtype=float)), dtype=float)
                eta_lookup: Dict[str, float] = {}
                for idx, gval in enumerate(global_vals):
                    if idx < len(step_eta):
                        val = float(step_eta[idx])
                        if np.isfinite(val) and val > 0 and gval not in eta_lookup:
                            eta_lookup[gval] = val
                local_arr = []
                local_map: Dict[str, float] = {}
                for v in vals:
                    eta_v = float(eta_lookup.get(str(v), 1.0))
                    local_arr.append(eta_v)
                    local_map[str(v)] = eta_v
                local_eta[step] = np.asarray(local_arr, dtype=float)
                local_eta_by_choice[step] = local_map

            def _build_full_cfg(local_cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
                imp = dict(imputation_map)
                scl = dict(scaling_map)
                enc = dict(encoding_map)
                for step, val in local_cfg.items():
                    if step == "imputation":
                        imp[feature] = str(val)
                    elif step == "scaling":
                        scl[feature] = str(val)
                    elif step == "encoding":
                        enc[feature] = str(val)
                cfg: Dict[str, Any] = {
                    "imputation": imp,
                    "scaling": scl,
                    "encoding": enc,
                    "outlier_removal": outlier_choice,
                    "feature_selection": fs_choice,
                    "dimensionality_reduction": dr_choice,
                    "name": name,
                }
                if order:
                    cfg["step_order"] = list(order)
                return cfg

            def _evaluate_local(sampled_cfgs: List[Dict[str, Any]]):
                cands: List[Dict[str, Any]] = []
                for j, cfg_local in enumerate(sampled_cfgs):
                    ops = ",".join(f"{k}={v}" for k, v in cfg_local.items())
                    full_cfg = _build_full_cfg(cfg_local, f"feature={feature}|{ops}|cand={j+1}")
                    full_cfg["__local_cfg__"] = dict(cfg_local)
                    cands.append(full_cfg)
                best_cfg, best_score, eval_results, unsorted_results = self._evaluate_candidates_with_simple_models(
                    new_dataset,
                    target_column,
                    cands,
                    proxy_settings=proxy_settings,
                )
                if best_cfg is None or not eval_results:
                    return None, np.nan, [], []

                def _to_local(cfg: Dict[str, Any]) -> Dict[str, Any]:
                    local = cfg.get("__local_cfg__")
                    if isinstance(local, dict):
                        return dict(local)
                    return {k: v for k, v in cfg.items() if k in local_options}

                eval_local = [(_to_local(cfg), float(sc)) for cfg, sc in eval_results]
                unsorted_local = [(_to_local(cfg), float(sc)) for cfg, sc in unsorted_results]
                best_local = _to_local(best_cfg)
                return best_local, float(best_score), eval_local, unsorted_local

            local_final, local_unsorted, local_history = search_pipelines_aco(
                options=local_options,
                evaluate_fn=_evaluate_local,
                eta=local_eta,
                n_pipelines=1,
                n_ants=n_ants,
                n_iterations=n_iterations,
                seed=seed + feat_idx - 1,
                alpha=alpha,
                beta=beta,
                evaporation=evaporation,
                top_k_pheromone=top_k_pheromone,
                use_all_iter_pipelines=use_all_iter_pipelines,
                weight_method=weight_method,
                markov_order=markov_order,
                lambda_smooth=lambda_smooth,
                early_stop_rounds=max(0, int(per_feature_early_stop_rounds)),
                min_improvement=float(per_feature_min_improvement),
                verbose=self.verbose,
                return_history=True,
            )

            for row in local_history:
                if isinstance(row, dict):
                    all_history.append(
                        {
                            "iteration": global_iter,
                            "feature_index": feat_idx,
                            "feature": feature,
                            "feature_kind": kind,
                            "feature_local_iteration": row.get("iteration"),
                            "best_score": row.get("best_score"),
                        }
                    )
                    global_iter += 1

            if local_final:
                best_cfg, best_score = local_final[0]
                if "imputation" in best_cfg:
                    imputation_map[feature] = str(best_cfg.get("imputation"))
                if "scaling" in best_cfg:
                    scaling_map[feature] = str(best_cfg.get("scaling"))
                if "encoding" in best_cfg:
                    encoding_map[feature] = str(best_cfg.get("encoding"))
                all_unsorted.append((_build_full_cfg(best_cfg, f"feature={feature}|selected"), float(best_score)))

                feature_logs.append(
                    {
                        "feature_index": feat_idx,
                        "feature": feature,
                        "feature_kind": kind,
                        "status": "optimized",
                        "selected_ops": {
                            "imputation": imputation_map.get(feature),
                            "scaling": scaling_map.get(feature),
                            "encoding": encoding_map.get(feature),
                        },
                        "best_score_after_feature": float(best_score),
                        "local_history_points": len(local_history),
                        "local_candidates": len(local_unsorted),
                        "local_options": {k: list(v) for k, v in local_options.items()},
                        "local_eta": local_eta_by_choice,
                        "heuristic_source": "phase2_transferred_eta_norm",
                    }
                )

                if stage_best_score is None or float(best_score) > (stage_best_score + float(per_feature_feature_min_improvement)):
                    stage_best_score = float(best_score)
                    no_improve_features = 0
                else:
                    no_improve_features += 1
            else:
                no_improve_features += 1
                feature_logs.append(
                    {
                        "feature_index": feat_idx,
                        "feature": feature,
                        "feature_kind": kind,
                        "status": "no_valid_candidate",
                        "selected_ops": {
                            "imputation": imputation_map.get(feature),
                            "scaling": scaling_map.get(feature),
                            "encoding": encoding_map.get(feature),
                        },
                        "local_history_points": len(local_history),
                        "local_candidates": len(local_unsorted),
                        "local_options": {k: list(v) for k, v in local_options.items()},
                        "local_eta": local_eta_by_choice,
                        "heuristic_source": "phase2_transferred_eta_norm",
                    }
                )

            if int(per_feature_feature_patience) > 0 and no_improve_features >= int(per_feature_feature_patience):
                feature_logs.append(
                    {
                        "feature_index": feat_idx,
                        "feature": feature,
                        "feature_kind": kind,
                        "status": "early_stop_features",
                        "reason": (
                            "no_improvement_across_features "
                            f"for {no_improve_features} consecutive features"
                        ),
                    }
                )
                break

        final_cfg: Dict[str, Any] = {
            "imputation": dict(imputation_map),
            "scaling": dict(scaling_map),
            "encoding": dict(encoding_map),
            "outlier_removal": outlier_choice,
            "feature_selection": fs_choice,
            "dimensionality_reduction": dr_choice,
            "name": "per_feature_independent_pipeline",
        }
        if order:
            final_cfg["step_order"] = list(order)

        final_best_cfg, final_best_score, final_all, _final_unsorted = self._evaluate_candidates_with_simple_models(
            new_dataset,
            target_column,
            [final_cfg],
            proxy_settings=proxy_settings,
        )
        if final_best_cfg is None or not final_all or not np.isfinite(final_best_score):
            final_best_cfg = final_cfg
            final_best_score = float(stage_best_score) if stage_best_score is not None else float("nan")
        else:
            final_best_cfg = final_best_cfg
            final_best_score = float(final_best_score)

        top = [(final_best_cfg, final_best_score)]
        unsorted = list(all_unsorted)
        if not unsorted:
            unsorted = list(top)
        return top[: max(1, int(n_pipelines))], unsorted, all_history, feature_logs

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

    @staticmethod
    def _candidate_key_for_options(
        cfg: Mapping[str, Any],
        options: Mapping[str, Sequence[str]],
        step_order: Optional[Sequence[str]] = None,
    ) -> Tuple[Tuple[str, Any], ...]:
        order = list(step_order) if step_order else list(options.keys())
        return tuple((step, cfg.get(step)) for step in order)

    @staticmethod
    def _compatible_option_value(raw_value: Any, choices: Sequence[str]) -> str:
        choice_values = [str(v) for v in choices]
        if not choice_values:
            return "none"

        if raw_value is not None:
            raw_str = str(raw_value)
            if raw_str in choice_values:
                return raw_str
            raw_base = base_operator_name(raw_str)
            for choice in choice_values:
                if base_operator_name(choice) == raw_base:
                    return choice

        for choice in choice_values:
            if base_operator_name(choice) == "none":
                return choice
        return choice_values[0]

    def _coerce_pipeline_to_options(
        self,
        cfg: Mapping[str, Any],
        options: Mapping[str, Sequence[str]],
        *,
        name: Optional[str] = None,
        step_order: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        coerced: Dict[str, Any] = {}
        for step, choices in options.items():
            coerced[step] = self._compatible_option_value(cfg.get(step), choices)
        if name is not None:
            coerced["name"] = str(name)
        elif cfg.get("name") is not None:
            coerced["name"] = str(cfg.get("name"))
        if step_order:
            coerced["step_order"] = list(step_order)
        return coerced

    def _dedupe_candidate_configs(
        self,
        candidates: Sequence[Mapping[str, Any]],
        options: Mapping[str, Sequence[str]],
        *,
        step_order: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for cfg in candidates:
            key = self._candidate_key_for_options(cfg, options, step_order=step_order)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dict(cfg))
        return deduped

    def _retrieved_seed_configs(
        self,
        *,
        new_metafeatures: np.ndarray,
        options: Mapping[str, Sequence[str]],
        neighbor_k: int,
        top_l: int,
        score_direction: str,
        query_dataset_id: Optional[Any],
        step_order: Optional[Sequence[str]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[Any, float]]]:
        cfg_by_name = {
            str(cfg.get("name")): cfg
            for cfg in self.pipeline_configs
            if isinstance(cfg, dict) and cfg.get("name") is not None
        }
        sims = self._compute_dataset_similarities(new_metafeatures)
        top_neighbors = select_top_k_neighbors(
            dataset_similarities=sims,
            top_k=max(1, int(neighbor_k)),
            query_dataset_id=query_dataset_id,
        )

        seeds: List[Dict[str, Any]] = []
        seed_rows: List[Dict[str, Any]] = []
        ascending = str(score_direction).lower().strip() not in {"higher_is_better", "max", "maximize"}

        for neighbor_rank, (dataset_id, sim) in enumerate(top_neighbors, start=1):
            ds_key = str(dataset_id)
            if ds_key not in self.performance_matrix_imputed.columns:
                continue
            scores = pd.to_numeric(self.performance_matrix_imputed[ds_key], errors="coerce")
            scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
            if scores.empty:
                continue
            ranked = scores.sort_values(ascending=ascending)
            for pipeline_rank, (pipeline_name, hist_score) in enumerate(ranked.head(max(1, int(top_l))).items(), start=1):
                pipeline_key = str(pipeline_name)
                raw_cfg = cfg_by_name.get(pipeline_key)
                if raw_cfg is None:
                    continue
                cfg = self._coerce_pipeline_to_options(
                    raw_cfg,
                    options,
                    name=pipeline_key,
                    step_order=step_order,
                )
                seeds.append(cfg)
                seed_rows.append(
                    {
                        "neighbor_dataset": ds_key,
                        "neighbor_rank": int(neighbor_rank),
                        "similarity": float(sim),
                        "pipeline_name": pipeline_key,
                        "pipeline_rank": int(pipeline_rank),
                        "historical_score": float(hist_score),
                    }
                )

        seeds = self._dedupe_candidate_configs(seeds, options, step_order=step_order)
        return seeds, seed_rows, top_neighbors

    def _local_mutation_candidates(
        self,
        *,
        seed_configs: Sequence[Mapping[str, Any]],
        options: Mapping[str, Sequence[str]],
        sample_budget: int,
        radius: int,
        random_candidates: int,
        seed: int,
        step_order: Optional[Sequence[str]],
    ) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(seed)
        order = list(step_order) if step_order else list(options.keys())
        budget = max(int(sample_budget), len(seed_configs), 1)
        candidates: List[Dict[str, Any]] = [dict(cfg) for cfg in seed_configs]

        def add_candidate(cfg: Mapping[str, Any]) -> bool:
            if len(candidates) >= budget:
                return False
            before = len(self._dedupe_candidate_configs(candidates, options, step_order=order))
            candidates.append(dict(cfg))
            after = len(self._dedupe_candidate_configs(candidates, options, step_order=order))
            if after == before:
                candidates.pop()
                return True
            return len(candidates) < budget

        def mutation_name(base_name: str, changes: Sequence[Tuple[str, str]]) -> str:
            suffix = ",".join(f"{step}={value}" for step, value in changes)
            return f"{base_name}|local:{suffix}"

        for base in seed_configs:
            base_name = str(base.get("name", "retrieved"))
            for step in order:
                for value in [str(v) for v in options.get(step, [])]:
                    if value == str(base.get(step)):
                        continue
                    cfg = dict(base)
                    cfg[step] = value
                    cfg["name"] = mutation_name(base_name, [(step, value)])
                    if step_order:
                        cfg["step_order"] = list(step_order)
                    if not add_candidate(cfg):
                        return self._dedupe_candidate_configs(candidates, options, step_order=order)

        if int(radius) >= 2:
            for base in seed_configs:
                base_name = str(base.get("name", "retrieved"))
                for i, step_a in enumerate(order):
                    for step_b in order[i + 1 :]:
                        vals_a = [str(v) for v in options.get(step_a, []) if str(v) != str(base.get(step_a))]
                        vals_b = [str(v) for v in options.get(step_b, []) if str(v) != str(base.get(step_b))]
                        for value_a in vals_a:
                            for value_b in vals_b:
                                cfg = dict(base)
                                cfg[step_a] = value_a
                                cfg[step_b] = value_b
                                cfg["name"] = mutation_name(base_name, [(step_a, value_a), (step_b, value_b)])
                                if step_order:
                                    cfg["step_order"] = list(step_order)
                                if not add_candidate(cfg):
                                    return self._dedupe_candidate_configs(candidates, options, step_order=order)

        random_added = 0
        max_attempts = max(50, int(random_candidates) * 20)
        for _ in range(max_attempts):
            if len(candidates) >= budget or random_added >= max(0, int(random_candidates)):
                break
            cfg = {step: str(options[step][int(rng.randint(0, len(options[step])))]) for step in order}
            cfg["name"] = f"retrieval_local_random_{random_added + 1}"
            if step_order:
                cfg["step_order"] = list(step_order)
            before = len(self._dedupe_candidate_configs(candidates, options, step_order=order))
            candidates.append(cfg)
            after = len(self._dedupe_candidate_configs(candidates, options, step_order=order))
            if after > before:
                random_added += 1
            else:
                candidates.pop()

        return self._dedupe_candidate_configs(candidates, options, step_order=order)

    def _search_pipelines_retrieval_local(
        self,
        *,
        new_dataset: Any,
        target_column: str,
        new_metafeatures: np.ndarray,
        options: Dict[str, List[str]],
        proxy_settings: Optional[Dict[str, Any]],
        n_pipelines: int,
        sample_budget: int,
        seed: int,
        score_direction: str,
        query_dataset_id: Optional[Any],
        step_order: Optional[List[str]],
        neighbor_k: int = 1,
        top_l: int = 1,
        radius: int = 1,
        random_candidates: int = 0,
    ) -> Tuple[
        List[Tuple[Dict[str, Any], float]],
        List[Tuple[Dict[str, Any], float]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        order = list(step_order) if step_order else list(options.keys())
        seed_configs, seed_rows, neighbors = self._retrieved_seed_configs(
            new_metafeatures=new_metafeatures,
            options=options,
            neighbor_k=neighbor_k,
            top_l=top_l,
            score_direction=score_direction,
            query_dataset_id=query_dataset_id,
            step_order=order,
        )
        if not seed_configs:
            return [], [], [
                {
                    "iteration": 0,
                    "best_score": None,
                    "optimizer": "retrieval_local",
                    "event": "no_seed_pipeline",
                    "neighbor_k": int(neighbor_k),
                    "top_l": int(top_l),
                }
            ], []

        candidates = self._local_mutation_candidates(
            seed_configs=seed_configs,
            options=options,
            sample_budget=sample_budget,
            radius=radius,
            random_candidates=random_candidates,
            seed=seed,
            step_order=order,
        )
        if self.verbose:
            neighbor_desc = ", ".join(f"{ds}:{sim:.4f}" for ds, sim in neighbors[: max(1, int(neighbor_k))])
            protected_names = ", ".join(str(cfg.get("name", "unnamed")) for cfg in seed_configs)
            print(
                "Retrieval-local search: "
                f"neighbors=[{neighbor_desc}], protected=[{protected_names}], "
                f"candidates={len(candidates)}, radius={int(radius)}"
            )

        best_cfg, best_score, sorted_results, unsorted_results = self._evaluate_candidates_with_simple_models(
            new_dataset,
            target_column,
            candidates,
            proxy_settings=proxy_settings,
        )
        if best_cfg is None or not sorted_results:
            return [], [], [
                {
                    "iteration": 0,
                    "best_score": None,
                    "optimizer": "retrieval_local",
                    "event": "no_valid_proxy_candidate",
                    "neighbor_k": int(neighbor_k),
                    "top_l": int(top_l),
                    "candidate_count": int(len(candidates)),
                }
            ], seed_configs

        history: List[Dict[str, Any]] = [
            {
                "iteration": 0,
                "best_score": None,
                "optimizer": "retrieval_local",
                "event": "setup",
                "neighbor_k": int(neighbor_k),
                "top_l": int(top_l),
                "radius": int(radius),
                "sample_budget": int(sample_budget),
                "seed_count": int(len(seed_configs)),
                "candidate_count": int(len(candidates)),
                "protected_candidates": [str(cfg.get("name", "unnamed")) for cfg in seed_configs],
                "seed_rows": seed_rows,
            }
        ]
        running_best: Optional[float] = None
        for idx, (cfg, score) in enumerate(unsorted_results, start=1):
            running_best = float(score) if running_best is None else max(float(running_best), float(score))
            history.append(
                {
                    "iteration": int(idx),
                    "best_score": float(running_best),
                    "episode_score": float(score),
                    "optimizer": "retrieval_local",
                    "candidate_name": str(cfg.get("name", f"candidate_{idx}")),
                }
            )

        return sorted_results[: max(1, int(n_pipelines))], unsorted_results, history, seed_configs

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
            per_feature_mode = bool(aco_params.get("per_feature_independent_search", False))
            if per_feature_mode and optimizer_name != "aco":
                raise ValueError("per_feature_independent_search requires optimizer='aco'")
            retrieval_local_protected_candidates: List[Dict[str, Any]] = []

            if per_feature_mode:
                if target_column is None:
                    raise ValueError("target_column is required for per-feature ACO search")
                step_list = aco_params.get("per_feature_steps", ["imputation", "scaling", "encoding"])
                if isinstance(step_list, str):
                    parsed = [s.strip() for s in step_list.split(",") if s.strip()]
                    step_list = parsed if parsed else ["imputation", "scaling", "encoding"]
                elif isinstance(step_list, (tuple, list)):
                    step_list = [str(s).strip() for s in step_list if str(s).strip()]
                else:
                    step_list = ["imputation", "scaling", "encoding"]

                aco_results, aco_unsorted_res, all_history, per_feature_pipeline_log = self._search_pipelines_aco_per_feature(
                    new_dataset=new_dataset,
                    target_column=target_column,
                    new_metafeatures=new_mf_scaled,
                    options=options,
                    proxy_settings=proxy_settings,
                    n_pipelines=k,
                    n_ants=int(aco_params.get("n_ants", 10)),
                    n_iterations=int(aco_params.get("n_iterations", 10)),
                    seed=aco_seed,
                    alpha=float(aco_params.get("alpha", 1.0)),
                    beta=float(aco_params.get("beta", 2.0)),
                    evaporation=float(aco_params.get("evaporation", 0.2)),
                    top_k_pheromone=int(aco_params.get("top_k_pheromone", 3)),
                    weight_method=str(aco_params.get("weight_method", "rank")),
                    markov_order=int(aco_params.get("markov_order", 2)),
                    lambda_smooth=float(aco_params.get("lambda_smooth", 0.0)),
                    use_all_iter_pipelines=bool(aco_params.get("use_all_iter_pipelines", False)),
                    step_order=base_order,
                    dataset_weighting=str(aco_params.get("dataset_weighting", "similarity")),
                    heuristic_top_k=int(aco_params.get("heuristic_top_k", k)),
                    heuristic_top_l=int(aco_params.get("heuristic_top_l", 3)),
                    heuristic_similarity_temperature=float(
                        aco_params.get("heuristic_similarity_temperature", 1.0)
                    ),
                    heuristic_eta_floor=float(aco_params.get("heuristic_eta_floor", 0.05)),
                    heuristic_transfer_method=str(aco_params.get("heuristic_transfer_method", "weighted_topk_topl")),
                    score_direction=str(aco_params.get("score_direction", "higher_is_better")),
                    query_dataset_id=aco_params.get("query_dataset_id"),
                    time_limit_per_model=time_limit_per_model,
                    metafeatures_func=metafeatures_func,
                    per_feature_steps=step_list,
                    per_feature_early_stop_rounds=int(aco_params.get("per_feature_early_stop_rounds", 0)),
                    per_feature_min_improvement=float(aco_params.get("per_feature_min_improvement", 0.0)),
                    per_feature_feature_patience=int(aco_params.get("per_feature_feature_patience", 0)),
                    per_feature_feature_min_improvement=float(
                        aco_params.get("per_feature_feature_min_improvement", 0.0)
                    ),
                    per_feature_max_features=int(aco_params.get("per_feature_max_features", 0)),
                )
                if not aco_results:
                    raise RuntimeError("Per-feature ACO search produced no valid pipeline candidates.")

                best_pipeline, best_score = aco_results[0]
                all_top_results: List[Tuple[Dict[str, Any], float]] = []
                order_iteration_results: List[Dict[str, Any]] = []
                candidate_orders = [base_order]
                run_quick_order_eval = False
                quick_order_eval_time_limit = int(aco_params.get("ordering_quick_time_limit", 30))
                used_order_level_scoring = False
            else:
                per_feature_pipeline_log = []
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

            if not per_feature_mode:
                all_top_results = []
                all_unsorted_results = []
                all_history = []
                order_iteration_results = []
                global_iteration = 1
                run_quick_order_eval = bool(
                    search_ordering
                    and use_autogluon
                    and autogluon_runtime_enabled
                    and target_column is not None
                )
                quick_order_eval_time_limit = int(aco_params.get("ordering_quick_time_limit", 30))
                quick_order_eval_time_limit = max(5, quick_order_eval_time_limit)

            if not per_feature_mode:
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
                            top_k_pheromone=int(aco_params.get("top_k_pheromone", 3)),
                            weight_method=str(aco_params.get("weight_method", "rank")),
                            markov_order=int(aco_params.get("markov_order", 2)),
                            lambda_smooth=float(aco_params.get("lambda_smooth", 0.0)),
                            dataset_weighting=str(aco_params.get("dataset_weighting", "similarity")),
                            heuristic_top_k=int(aco_params.get("heuristic_top_k", k)),
                            heuristic_top_l=int(aco_params.get("heuristic_top_l", 3)),
                            heuristic_similarity_temperature=float(
                                aco_params.get("heuristic_similarity_temperature", 1.0)
                            ),
                            heuristic_eta_floor=float(aco_params.get("heuristic_eta_floor", 0.05)),
                            heuristic_transfer_method=str(
                                aco_params.get("heuristic_transfer_method", "weighted_topk_topl")
                            ),
                            score_direction=str(aco_params.get("score_direction", "higher_is_better")),
                            query_dataset_id=aco_params.get("query_dataset_id"),
                            time_limit_per_model=time_limit_per_model,
                            metafeatures_func=metafeatures_func,
                            step_order=order,
                            early_stop_rounds=int(aco_params.get("early_stop_rounds", 0)),
                            min_improvement=float(aco_params.get("min_improvement", 0.0)),
                            legacy_notebook_aco=bool(aco_params.get("legacy_notebook_aco", False)),
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
                            heuristic_similarity_temperature=float(
                                aco_params.get("heuristic_similarity_temperature", 1.0)
                            ),
                            heuristic_eta_floor=float(aco_params.get("heuristic_eta_floor", 0.05)),
                            heuristic_transfer_method=str(
                                aco_params.get("heuristic_transfer_method", "weighted_topk_topl")
                            ),
                            score_direction=str(aco_params.get("score_direction", "higher_is_better")),
                            query_dataset_id=aco_params.get("query_dataset_id"),
                            time_limit_per_model=time_limit_per_model,
                            metafeatures_func=metafeatures_func,
                            dqn_params=aco_params,
                            enable_internal_order_policy=not search_ordering,
                        )
                    elif optimizer_name == "retrieval_local":
                        (
                            aco_results,
                            aco_unsorted_res,
                            aco_history,
                            protected_candidates,
                        ) = self._search_pipelines_retrieval_local(
                            new_dataset=new_dataset,
                            target_column=target_column,
                            new_metafeatures=new_mf_scaled,
                            options=ordered_options,
                            proxy_settings=proxy_settings,
                            n_pipelines=k,
                            sample_budget=sample_budget,
                            seed=aco_seed + order_idx - 1,
                            score_direction=str(aco_params.get("score_direction", "higher_is_better")),
                            query_dataset_id=aco_params.get("query_dataset_id"),
                            step_order=order,
                            neighbor_k=int(
                                aco_params.get(
                                    "retrieval_local_neighbor_k",
                                    aco_params.get("heuristic_top_k", 1),
                                )
                            ),
                            top_l=int(
                                aco_params.get(
                                    "retrieval_local_top_l",
                                    aco_params.get("heuristic_top_l", 1),
                                )
                            ),
                            radius=int(aco_params.get("retrieval_local_radius", 1)),
                            random_candidates=int(aco_params.get("retrieval_local_random_candidates", 0)),
                        )
                        retrieval_local_protected_candidates.extend(protected_candidates)
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

            if not per_feature_mode:
                used_order_level_scoring = False
            if (not per_feature_mode) and search_ordering and order_iteration_results:
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
            elif not per_feature_mode:
                aco_results = []
                aco_unsorted_res = []

            if (not per_feature_mode) and not aco_results:
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
                        if optimizer_name == "retrieval_local" and retrieval_local_protected_candidates:
                            ag_candidates = self._dedupe_candidate_configs(
                                [*retrieval_local_protected_candidates, *ag_candidates],
                                options,
                                step_order=base_order,
                            )
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
                            if optimizer_name == "retrieval_local" and retrieval_local_protected_candidates:
                                fallback_candidates = self._dedupe_candidate_configs(
                                    [*retrieval_local_protected_candidates, *fallback_candidates],
                                    options,
                                    step_order=base_order,
                                )
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
                    "enabled": bool(search_ordering and not per_feature_mode),
                    "strategy": order_strategy,
                    "num_orders_requested": int(num_orders),
                    "num_orders_evaluated": len(candidate_orders),
                    "orders": candidate_orders,
                    "quick_autogluon_per_iteration": bool(run_quick_order_eval),
                    "quick_autogluon_time_limit": quick_order_eval_time_limit if run_quick_order_eval else None,
                    "selection_metric": "autogluon_quick_or_proxy" if used_order_level_scoring else "proxy",
                },
                "ordering_iteration_results": order_iteration_results if search_ordering else [],
                "per_feature_search": {
                    "enabled": bool(per_feature_mode),
                    "steps": (
                        step_list
                        if per_feature_mode
                        else []
                    ),
                    "heuristic_source": "phase2_transferred_eta_norm" if per_feature_mode else "n/a",
                    "log": per_feature_pipeline_log,
                },
                "retrieval_local_search": {
                    "enabled": optimizer_name == "retrieval_local",
                    "incumbent_protection": bool(
                        optimizer_name == "retrieval_local" and retrieval_local_protected_candidates
                    ),
                    "protected_candidates": [
                        str(cfg.get("name", "unnamed")) for cfg in retrieval_local_protected_candidates
                    ],
                    "neighbor_k": int(
                        aco_params.get("retrieval_local_neighbor_k", aco_params.get("heuristic_top_k", 1))
                    ),
                    "top_l": int(aco_params.get("retrieval_local_top_l", aco_params.get("heuristic_top_l", 1))),
                    "radius": int(aco_params.get("retrieval_local_radius", 1)),
                    "random_candidates": int(aco_params.get("retrieval_local_random_candidates", 0)),
                },
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
                    "early_stop_rounds": int(aco_params.get("early_stop_rounds", 0)),
                    "min_improvement": float(aco_params.get("min_improvement", 0.0)),
                    "per_feature_independent_search": bool(aco_params.get("per_feature_independent_search", False)),
                    "legacy_notebook_aco": bool(aco_params.get("legacy_notebook_aco", False)),
                    "retrieval_local_neighbor_k": int(
                        aco_params.get("retrieval_local_neighbor_k", aco_params.get("heuristic_top_k", 1))
                    ),
                    "retrieval_local_top_l": int(
                        aco_params.get("retrieval_local_top_l", aco_params.get("heuristic_top_l", 1))
                    ),
                    "retrieval_local_radius": int(aco_params.get("retrieval_local_radius", 1)),
                    "retrieval_local_random_candidates": int(
                        aco_params.get("retrieval_local_random_candidates", 0)
                    ),
                },
            }

        sims = self._compute_dataset_similarities(new_mf_scaled)
        query_dataset_id = aco_params.get("query_dataset_id")
        top_neighbors = select_top_k_neighbors(
            dataset_similarities=sims,
            top_k=k,
            query_dataset_id=query_dataset_id,
        )
        if not top_neighbors:
            raise RuntimeError("No eligible similar datasets found after self-query exclusion.")

        top_datasets = [ds for ds, _ in top_neighbors]
        top_sims = np.array([s for _, s in top_neighbors], dtype=float)
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
                final_eval = {
                    "method": "fallback_prediction_only",
                    "score": float(top_pipeline_score),
                    "error": "No candidate produced valid evaluation results",
                }
                return {
                    "pipeline_config": top_pipeline_config,
                    "expected_performance": float(top_pipeline_score),
                    "recommended_performance": float(top_pipeline_score),
                    "final_evaluation": final_eval,
                    "final_performance": float(final_eval["score"]),
                    "similar_datasets": top_datasets,
                    "pipeline_ranking": pipeline_ranking[:k],
                    "top_candidates": [(cfg["name"], float(candidate_perfs[cfg["name"]])) for cfg in top_candidate_configs],
                    "confidence": "low",
                    "similarity_scores": dict(top_neighbors),
                    "model_type": self.metric_type,
                    "evaluation_method": "fallback_prediction_only",
                }

            best_name = str(best_cfg.get("name", "")) if isinstance(best_cfg, dict) else ""
            expected_score = (
                float(candidate_perfs[best_name])
                if best_name in candidate_perfs.index
                else float(best_score)
            )
            final_eval = {"method": eval_method, "score": float(best_score)}
            return {
                "pipeline_config": best_cfg,
                "expected_performance": expected_score,
                "recommended_performance": expected_score,
                "final_evaluation": final_eval,
                "final_performance": float(final_eval["score"]),
                "similar_datasets": top_datasets,
                "pipeline_ranking": all_results,
                "top_candidates_evaluated": [(cfg["name"], sc) for cfg, sc in all_results],
                "confidence": "high",
                "similarity_scores": dict(top_neighbors),
                "model_type": self.metric_type,
                "evaluation_method": eval_method,
            }

        top_pipeline_name = pipeline_ranking[0]
        top_pipeline_score = candidate_perfs[top_pipeline_name]
        top_pipeline_config = next(
            (cfg for cfg in self.pipeline_configs if cfg.get("name") == top_pipeline_name),
            None,
        )
        final_eval = {"method": "prediction_only", "score": float(top_pipeline_score)}
        return {
            "pipeline_config": top_pipeline_config,
            "expected_performance": float(top_pipeline_score),
            "recommended_performance": float(top_pipeline_score),
            "final_evaluation": final_eval,
            "final_performance": float(final_eval["score"]),
            "similar_datasets": top_datasets,
            "pipeline_ranking": pipeline_ranking[:k],
            "top_candidates": [(cfg["name"], float(candidate_perfs[cfg["name"]])) for cfg in top_candidate_configs],
            "confidence": "medium",
            "similarity_scores": dict(top_neighbors),
            "model_type": self.metric_type,
            "evaluation_method": "prediction_only",
        }
