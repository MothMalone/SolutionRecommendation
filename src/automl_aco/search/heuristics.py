"""Heuristic computation for ACO (eta values)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logging import get_logger

logger = get_logger(__name__)


def compute_aco_heuristic(
    performance_matrix: pd.DataFrame,
    metafeatures_df: pd.DataFrame,
    pipeline_configs: List[Dict[str, Any]],
    options: Mapping[str, List[str]],
    new_metafeatures: np.ndarray,
    dataset_weighting: str = "equality",
    top_k: Optional[int] = 10,
    use_top_pipelines_from_metric: bool = True,
    recommend_func: Optional[Callable[..., Dict[str, Any]]] = None,
    recommend_kwargs: Optional[Dict[str, Any]] = None,
    metafeatures_scaled: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    eps = 1e-8
    n_datasets = metafeatures_df.shape[0]

    perf_subset = performance_matrix.copy()

    if use_top_pipelines_from_metric and recommend_func is not None:
        rec_args = recommend_kwargs or {}
        rec_result = recommend_func(**rec_args)
        top_pipelines: List[str] = []
        if isinstance(rec_result, dict):
            pipeline_cfg = rec_result.get("pipeline_config")
            if isinstance(pipeline_cfg, dict):
                name = pipeline_cfg.get("name")
                if name in performance_matrix.index:
                    top_pipelines = [name]
        if top_pipelines:
            perf_subset = performance_matrix.loc[performance_matrix.index.isin(top_pipelines)]
        else:
            logger.debug("No valid pipeline from metric; using full performance matrix")

    if dataset_weighting == "similarity" and n_datasets > 0:
        try:
            if metafeatures_scaled is None:
                known = metafeatures_df.fillna(0).values
            else:
                known = metafeatures_scaled
            sims = cosine_similarity(known, new_metafeatures.reshape(1, -1)).ravel()

            if top_k is not None and top_k < len(sims):
                top_idx = np.argsort(sims)[-top_k:]
                sims_masked = np.zeros_like(sims)
                sims_masked[top_idx] = sims[top_idx]
                sims = sims_masked

            sims = sims - sims.min()
            if sims.sum() <= 0:
                sims = np.ones_like(sims)
            dataset_weights = sims / (sims.sum() + eps)
        except Exception:
            dataset_weights = np.ones(n_datasets) / max(1, n_datasets)
    else:
        dataset_weights = np.ones(n_datasets) / max(1, n_datasets)

    meta_names = set(metafeatures_df.index)
    perf_names = set(perf_subset.columns)
    common_names = sorted(meta_names & perf_names)
    if len(common_names) == 0:
        raise ValueError("No common datasets between metafeatures and performance matrix")

    perf_tbl = perf_subset.loc[:, common_names].fillna(0)

    if perf_tbl.shape[1] == 0:
        pipeline_perf_mean = pd.Series(0, index=perf_subset.index)
    else:
        pipeline_perf_mean = pd.Series(np.dot(perf_tbl.values, dataset_weights), index=perf_tbl.index)

    eta: Dict[str, np.ndarray] = {}
    cfg_map = {cfg["name"]: cfg for cfg in pipeline_configs if "name" in cfg}

    for step, values in options.items():
        arr = [eps] * len(values)
        for i, val in enumerate(values):
            matched = [
                perf_val
                for pname, perf_val in pipeline_perf_mean.items()
                if (cfg_map.get(pname) is not None and cfg_map[pname].get(step) == val)
            ]
            if matched:
                arr[i] = float(np.mean(matched)) + eps

        arr = np.array(arr, dtype=float)
        nonzero_vals = arr[arr > eps * 2]
        if len(nonzero_vals) > 0:
            min_val = nonzero_vals.min()
            arr[arr <= eps] = min_val * 0.8
        else:
            arr[:] = 1.0 / len(arr)

        arr = arr / (arr.max() + eps) if arr.sum() > 0 else np.ones_like(arr) / len(arr)
        arr = np.clip(arr, eps, None)
        eta[step] = arr

    return eta
