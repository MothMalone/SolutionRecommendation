"""Siamese-style regression metric utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MetricModel:
    embedder: Any
    projector: Any
    params: Dict[str, Any]


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


def train_siamese_regression_metric(
    metafeatures_df: pd.DataFrame,
    performance_matrix_imputed: pd.DataFrame,
    hidden_dim: int = 64,
    embed_dim: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
) -> MetricModel:
    torch = _require_torch()
    import torch.nn as nn
    import torch.optim as optim

    np.random.seed(seed)
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

    mf_scaled = pd.DataFrame(meta_aligned).fillna(0).values.astype(np.float32)
    perf_profiles = perf_aligned.T.values
    perf_profiles_std = StandardScaler().fit_transform(perf_profiles)
    S_perf = cosine_similarity(perf_profiles_std)

    N, d = mf_scaled.shape
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    embedder, projector = build_metric_models(d, hidden_dim, embed_dim)
    optimizer = optim.Adam(list(embedder.parameters()) + list(projector.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    X_i = torch.tensor(np.array([mf_scaled[i] for i, _ in pairs]), dtype=torch.float32)
    X_j = torch.tensor(np.array([mf_scaled[j] for _, j in pairs]), dtype=torch.float32)
    y_pairs = torch.tensor(np.array([S_perf[i, j] for i, j in pairs]), dtype=torch.float32).unsqueeze(1)

    for epoch in range(epochs):
        emb_i = embedder(X_i)
        emb_j = embedder(X_j)

        emb_i = emb_i / (emb_i.norm(dim=1, keepdim=True) + 1e-8)
        emb_j = emb_j / (emb_j.norm(dim=1, keepdim=True) + 1e-8)

        x_pair = emb_i * emb_j
        pred = projector(x_pair)
        loss = loss_fn(pred, y_pairs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    params = {"input_dim": d, "hidden_dim": hidden_dim, "embed_dim": embed_dim}
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
