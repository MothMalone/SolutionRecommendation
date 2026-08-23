"""Siamese-style regression metric utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MetricModel:
    embedder: Any
    projector: Any
    params: Dict[str, Any]


SIMILARITY_TARGET_CHOICES = {
    "rank_cosine",
    "row_zscore_cosine",
    "row_minmax_cosine",
    "legacy_global_zscore_cosine",
}

SCORE_DIRECTION_CHOICES = {"higher_is_better", "lower_is_better"}
METRIC_OBJECTIVE_CHOICES = {"embedding_cosine", "projector_product"}
METRIC_LOSS_CHOICES = {"mse", "pearson", "listwise_kl"}


def _require_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for metric training and inference") from exc


def build_metric_models(input_dim: int, hidden_dim: int, embed_dim: int):
    torch = _require_torch()
    import torch.nn as nn

    embedder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, embed_dim),
        nn.ReLU(),
    )
    projector = nn.Sequential(
        nn.Linear(embed_dim, 1),
        nn.Tanh(),
    )
    return embedder, projector


def _validate_score_direction(score_direction: str) -> str:
    direction = str(score_direction).strip().lower()
    if direction not in SCORE_DIRECTION_CHOICES:
        raise ValueError(
            f"Unsupported score_direction={score_direction!r}. "
            f"Expected one of {sorted(SCORE_DIRECTION_CHOICES)}."
        )
    return direction


def _validate_metric_objective(metric_objective: str) -> str:
    objective = str(metric_objective).strip().lower()
    if objective not in METRIC_OBJECTIVE_CHOICES:
        raise ValueError(
            f"Unsupported metric_objective={metric_objective!r}. "
            f"Expected one of {sorted(METRIC_OBJECTIVE_CHOICES)}."
        )
    return objective


def _to_higher_is_better(perf_profiles: np.ndarray, score_direction: str) -> np.ndarray:
    direction = _validate_score_direction(score_direction)
    arr = np.asarray(perf_profiles, dtype=float)
    if direction == "lower_is_better":
        arr = -arr
    return arr


def _row_rank_normalize(perf_profiles: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n_rows, n_cols = perf_profiles.shape
    if n_cols == 0:
        return np.zeros_like(perf_profiles, dtype=float)
    if n_cols == 1:
        return np.ones((n_rows, 1), dtype=float)

    out = np.zeros_like(perf_profiles, dtype=float)
    denom = float(n_cols - 1)
    for i in range(n_rows):
        # Rank preserves relative preferences and suppresses absolute-scale effects.
        row = pd.Series(perf_profiles[i], copy=False)
        ranks = row.rank(method="average", ascending=True).to_numpy(dtype=float, copy=False)
        out[i, :] = (ranks - 1.0) / max(denom, eps)
    return out


def _row_zscore_normalize(perf_profiles: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_mean = np.mean(perf_profiles, axis=1, keepdims=True)
    row_std = np.std(perf_profiles, axis=1, keepdims=True)
    denom = np.where(row_std > eps, row_std, 1.0)
    normalized = (perf_profiles - row_mean) / denom
    normalized[row_std.ravel() <= eps, :] = 0.0
    return normalized


def _row_minmax_normalize(perf_profiles: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_min = np.min(perf_profiles, axis=1, keepdims=True)
    row_max = np.max(perf_profiles, axis=1, keepdims=True)
    span = row_max - row_min
    denom = np.where(span > eps, span, 1.0)
    normalized = (perf_profiles - row_min) / denom
    normalized[span.ravel() <= eps, :] = 0.5
    return normalized


def build_similarity_target_matrix(
    perf_profiles: np.ndarray,
    similarity_target: str = "rank_cosine",
    score_direction: str = "higher_is_better",
) -> np.ndarray:
    target = str(similarity_target).strip().lower()
    if target not in SIMILARITY_TARGET_CHOICES:
        raise ValueError(
            f"Unsupported similarity_target={similarity_target!r}. "
            f"Expected one of {sorted(SIMILARITY_TARGET_CHOICES)}."
        )

    oriented = _to_higher_is_better(perf_profiles, score_direction=score_direction)
    if target == "legacy_global_zscore_cosine":
        transformed = StandardScaler().fit_transform(oriented)
    elif target == "row_zscore_cosine":
        transformed = _row_zscore_normalize(oriented)
    elif target == "row_minmax_cosine":
        transformed = _row_minmax_normalize(oriented)
    else:
        transformed = _row_rank_normalize(oriented)
    return cosine_similarity(transformed)


def train_siamese_regression_metric(
    metafeatures_df: pd.DataFrame,
    performance_matrix_imputed: pd.DataFrame,
    hidden_dim: int = 64,
    embed_dim: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    similarity_target: str = "rank_cosine",
    score_direction: str = "higher_is_better",
    metafeatures_are_preprocessed: bool = False,
    metric_objective: str = "embedding_cosine",
    metric_loss: str = "mse",
    weight_decay: float = 0.0,
    target_temperature: float = 0.1,
    prediction_temperature: float = 0.1,
) -> MetricModel:
    torch = _require_torch()
    import torch.nn as nn
    import torch.optim as optim

    np.random.seed(seed)
    objective = _validate_metric_objective(metric_objective)
    loss_kind = str(metric_loss).strip().lower()
    if loss_kind not in METRIC_LOSS_CHOICES:
        raise ValueError(f"Unsupported metric_loss={metric_loss!r}. Expected one of {sorted(METRIC_LOSS_CHOICES)}.")
    if loss_kind == "listwise_kl" and objective != "embedding_cosine":
        raise ValueError("listwise_kl requires metric_objective='embedding_cosine'")
    target_temp = float(target_temperature)
    prediction_temp = float(prediction_temperature)
    if target_temp <= 0.0 or prediction_temp <= 0.0:
        raise ValueError("Listwise temperatures must be positive")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    perf_names = set(performance_matrix_imputed.columns)
    meta_names = set(metafeatures_df.index)
    common_names = sorted(perf_names & meta_names)
    if len(common_names) == 0:
        raise ValueError("No common datasets between performance_matrix and metafeatures_df")

    perf_aligned = performance_matrix_imputed[common_names]
    meta_aligned = metafeatures_df.loc[common_names]
    assert list(perf_aligned.columns) == list(meta_aligned.index)

    if metafeatures_are_preprocessed:
        mf_scaled = meta_aligned.to_numpy(dtype=np.float32, copy=True)
        preprocessing_mode = "preprocessed_input"
    else:
        mf_imputed = SimpleImputer(strategy="mean").fit_transform(meta_aligned)
        mf_scaled = MinMaxScaler().fit_transform(mf_imputed).astype(np.float32, copy=False)
        preprocessing_mode = "mean_impute_minmax"
    mf_scaled = np.nan_to_num(mf_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    perf_profiles = perf_aligned.T.values
    S_perf = build_similarity_target_matrix(
        perf_profiles=perf_profiles,
        similarity_target=similarity_target,
        score_direction=score_direction,
    )

    N, d = mf_scaled.shape
    if N < 2:
        raise ValueError("Metric training requires at least two datasets")
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    embedder, projector = build_metric_models(d, hidden_dim, embed_dim)
    optimizer = optim.Adam(
        list(embedder.parameters()) + list(projector.parameters()),
        lr=lr,
        weight_decay=float(weight_decay),
    )
    loss_fn = nn.MSELoss()

    def _pearson_loss(pred_t, target_t):
        # 1 - Pearson correlation over the batch. Scale/shift-invariant, so it optimizes
        # *ranking agreement* (what top-K retrieval uses) and cannot collapse to predicting the
        # mean of a low-variance target the way MSE does on the compressed rank_cosine target.
        p = pred_t.reshape(-1)
        t = target_t.reshape(-1)
        p = p - p.mean()
        t = t - t.mean()
        denom = (p.norm() * t.norm()) + 1e-8
        return 1.0 - (p * t).sum() / denom

    X_i = torch.tensor(np.array([mf_scaled[i] for i, _ in pairs]), dtype=torch.float32)
    X_j = torch.tensor(np.array([mf_scaled[j] for _, j in pairs]), dtype=torch.float32)
    y_pairs = torch.tensor(np.array([S_perf[i, j] for i, j in pairs]), dtype=torch.float32).unsqueeze(1)

    X_all = torch.tensor(mf_scaled, dtype=torch.float32)
    target_similarity = torch.tensor(S_perf, dtype=torch.float32)
    diagonal_mask = torch.eye(N, dtype=torch.bool)
    target_logits = (target_similarity / target_temp).masked_fill(diagonal_mask, float("-inf"))
    target_distribution = torch.softmax(target_logits, dim=1)

    for epoch in range(epochs):
        if loss_kind == "listwise_kl":
            embeddings = embedder(X_all)
            embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
            predicted_similarity = embeddings @ embeddings.T
            predicted_logits = (predicted_similarity / prediction_temp).masked_fill(
                diagonal_mask, float("-inf")
            )
            predicted_log_distribution = torch.log_softmax(predicted_logits, dim=1)
            valid_pairs = ~diagonal_mask
            cross_entropy = -(
                target_distribution[valid_pairs] * predicted_log_distribution[valid_pairs]
            ).reshape(N, N - 1).sum(dim=1)
            loss = cross_entropy.mean()
        else:
            emb_i = embedder(X_i)
            emb_j = embedder(X_j)

            emb_i = emb_i / (emb_i.norm(dim=1, keepdim=True) + 1e-8)
            emb_j = emb_j / (emb_j.norm(dim=1, keepdim=True) + 1e-8)

            if objective == "embedding_cosine":
                pred = (emb_i * emb_j).sum(dim=1, keepdim=True)
            else:
                x_pair = emb_i * emb_j
                pred = projector(x_pair)
            loss = _pearson_loss(pred, y_pairs) if loss_kind == "pearson" else loss_fn(pred, y_pairs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    params = {
        "input_dim": d,
        "hidden_dim": hidden_dim,
        "embed_dim": embed_dim,
        "similarity_target": str(similarity_target),
        "score_direction": str(score_direction),
        "metafeature_preprocessing": preprocessing_mode,
        "metric_objective": objective,
        "metric_loss": loss_kind,
        "weight_decay": float(weight_decay),
        "target_temperature": target_temp,
        "prediction_temperature": prediction_temp,
    }
    return MetricModel(embedder=embedder, projector=projector, params=params)


def save_metric(model: MetricModel, path: str) -> str:
    torch = _require_torch()
    payload = {
        "metric_type": "regression",
        "metric_params": model.params,
        "embedder_state": model.embedder.state_dict(),
        "projector_state": model.projector.state_dict(),
    }
    torch.save(payload, path)
    return path


def load_metric(path: str, map_location: str = "cpu") -> MetricModel:
    torch = _require_torch()
    payload = torch.load(path, map_location=map_location)
    if payload.get("metric_type") != "regression":
        raise ValueError("Unsupported metric type in saved model")
    params = payload.get("metric_params", {})
    d = params.get("input_dim")
    hidden_dim = params.get("hidden_dim", 64)
    embed_dim = params.get("embed_dim", 64)
    if d is None:
        raise ValueError("Saved metric missing input_dim")

    embedder, projector = build_metric_models(d, hidden_dim, embed_dim)
    embedder.load_state_dict(payload["embedder_state"])
    projector.load_state_dict(payload["projector_state"])
    return MetricModel(embedder=embedder, projector=projector, params=params)
