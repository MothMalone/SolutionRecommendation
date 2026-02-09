"""Meta-learning pipeline recommender (ported from notebook)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import DEFAULT_PIPELINE_OPTIONS
from ..utils.logging import get_logger
from ..metalearning.metric import train_siamese_regression_metric, save_metric as save_metric_file, load_metric as load_metric_file
from ..search.heuristics import compute_aco_heuristic
from ..search.aco import search_pipelines_aco
from ..search.evaluation import evaluate_candidates_simple, evaluate_candidates_autogluon

logger = get_logger(__name__)

try:  # optional AutoGluon availability flag
    from autogluon.tabular import TabularPredictor  # type: ignore
    AUTOGLUON_AVAILABLE = True
except Exception:  # pragma: no cover
    AUTOGLUON_AVAILABLE = False


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

        if self.performance_matrix.notna().sum().sum() == 0:
            raise ValueError(
                "Performance matrix has no scores for the aligned datasets. "
                "This usually means the metafeatures file does not match the performance matrix dataset IDs."
            )

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

    def _compute_aco_heuristic(
        self,
        new_metafeatures: np.ndarray,
        options: Dict[str, List[str]],
        dataset_weighting: str = "equality",
        top_k: int = 10,
        use_top_pipelines_from_metric: bool = True,
        recommend_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        return compute_aco_heuristic(
            performance_matrix=self.performance_matrix,
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
        )

    def _search_pipelines_aco(
        self,
        new_dataset: Any,
        target_column: str,
        new_metafeatures: np.ndarray,
        options: Dict[str, List[str]],
        n_pipelines: int = 3,
        n_ants: int = 3,
        n_iterations: int = 5,
        seed: int = 42,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.2,
        dataset_weighting: str = "equality",
        time_limit_per_model: int = 3000,
        local_search: bool = False,
        metafeatures_func=None,
        top_k_pheromone: int = 3,
        average_pheromone_update: bool = False,
        use_all_iter_pipelines: bool = False,
        weight_method: str = "rank",
        markov_order: int = 2,
        lambda_smooth: float = 0.7,
    ):
        eta = self._compute_aco_heuristic(
            new_metafeatures,
            options,
            dataset_weighting=dataset_weighting,
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
        )

        def _evaluate(sampled_configs):
            return self._evaluate_candidates_with_simple_models(new_dataset, target_column, sampled_configs)

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

    def _evaluate_candidates_with_simple_models(self, dataset, target_column, candidate_configs):
        return evaluate_candidates_simple(
            dataset=dataset,
            target_column=target_column,
            candidate_configs=candidate_configs,
            verbose=self.verbose,
        )

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
    ) -> Dict[str, Any]:
        if metafeatures_func is None:
            raise ValueError("metafeatures_func must be provided")

        options = options or self.pipeline_options
        aco_params = aco_params or {}

        new_mf = metafeatures_func(new_dataset)
        new_mf_df = pd.DataFrame([new_mf]).reindex(columns=self.metafeatures_df.columns, fill_value=0)
        new_mf_imputed = self.imputer.transform(new_mf_df)
        new_mf_scaled = self.scaler.transform(new_mf_imputed).ravel()

        if use_aco:
            aco_results, aco_unsorted_res, aco_history = self._search_pipelines_aco(
                new_dataset,
                target_column,
                new_mf_scaled,
                options,
                n_pipelines=k,
                n_ants=aco_params.get("n_ants", 10),
                n_iterations=aco_params.get("n_iterations", 10),
                time_limit_per_model=time_limit_per_model,
                metafeatures_func=metafeatures_func,
            )
            best_pipeline, best_score = aco_results[0]
            final_eval = {"method": "proxy", "score": float(best_score)}
            if use_autogluon:
                if AUTOGLUON_AVAILABLE and target_column is not None:
                    try:
                        ag_best_cfg, ag_score, _ag_results, _ag_unsorted = self._evaluate_candidates_with_autogluon(
                            new_dataset,
                            target_column,
                            [best_pipeline],
                            time_limit_per_model=time_limit_per_model,
                        )
                        if ag_best_cfg is not None:
                            best_pipeline = ag_best_cfg
                        final_eval = {"method": "autogluon", "score": float(ag_score)}
                    except Exception as exc:
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
                "aco_history": aco_history,
            }

        sims: List[Tuple[Any, float]] = []
        if self.metric_type == "regression" and self.embedder is not None:
            try:
                import torch
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("torch is required for metric similarity") from exc

            with torch.no_grad():
                known_mf_scaled = self.scaler.transform(self.imputer.transform(self.metafeatures_df))
                known_tensor = torch.tensor(known_mf_scaled, dtype=torch.float32)
                new_tensor = torch.tensor(new_mf_scaled.reshape(1, -1), dtype=torch.float32)

                emb_known = self.embedder(known_tensor)
                emb_new = self.embedder(new_tensor).squeeze(0)

                emb_known = emb_known / (emb_known.norm(dim=1, keepdim=True) + 1e-8)
                emb_new = emb_new / (emb_new.norm() + 1e-8)

                for ds_id, h_known in zip(self.metafeatures_df.index, emb_known):
                    inter = (emb_new * h_known).unsqueeze(0)
                    sim = float(self.projector(inter).item())
                    sims.append((ds_id, sim))
        else:
            known = self.metafeatures_scaled
            cosines = cosine_similarity(known, new_mf_scaled.reshape(1, -1)).ravel()
            sims = list(zip(self.metafeatures_df.index, cosines))

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
