"""Context-gated DQN policy for stage-wise pipeline construction.

This module adapts the idea of CtxPipe's gated context integration:
- a state stream (dataset + partial pipeline state),
- a context stream (warm-start priors),
- multiplicative gating between state/context representations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _require_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "optimizer='dqn' requires torch. Install optional dependency: pip install torch"
        ) from exc


@dataclass(frozen=True)
class DQNPolicyConfig:
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.95
    epochs: int = 80
    batch_size: int = 64
    target_update_interval: int = 5
    warmstart_weight: float = 0.5
    loss_fn: str = "huber"
    huber_delta: float = 1.0
    grad_clip_norm: float = 5.0
    reward_clip: float = 1.0
    target_q_clip: float = 5.0
    use_double_dqn: bool = True

    def cache_key(self) -> Tuple[Any, ...]:
        return (
            self.hidden_dim,
            self.lr,
            self.gamma,
            self.epochs,
            self.batch_size,
            self.target_update_interval,
            self.warmstart_weight,
            self.loss_fn,
            self.huber_delta,
            self.grad_clip_norm,
            self.reward_clip,
            self.target_q_clip,
            self.use_double_dqn,
        )


def build_action_offsets(
    options: Mapping[str, Sequence[str]],
) -> Tuple[Dict[str, int], int]:
    offsets: Dict[str, int] = {}
    cur = 0
    for step, values in options.items():
        offsets[step] = cur
        cur += len(values)
    return offsets, cur


def _normalize_context(vec: np.ndarray) -> np.ndarray:
    if vec.size == 0:
        return vec
    v = vec.astype(np.float32, copy=True)
    finite = np.isfinite(v)
    if not finite.any():
        return np.ones_like(v, dtype=np.float32) / float(len(v))
    v[~finite] = np.nanmin(v[finite]) if finite.any() else 0.0
    mn = float(np.min(v))
    mx = float(np.max(v))
    if mx - mn < 1e-12:
        return np.ones_like(v, dtype=np.float32) / float(len(v))
    v = (v - mn) / (mx - mn + 1e-12)
    v += 1e-6
    return v.astype(np.float32)


def _valid_pipeline_rows(
    performance_matrix: pd.DataFrame,
    pipeline_configs: Sequence[Mapping[str, Any]],
    options: Mapping[str, Sequence[str]],
) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    for cfg in pipeline_configs:
        name = cfg.get("name")
        if name is None or name not in performance_matrix.index:
            continue
        ok = True
        for step, values in options.items():
            if cfg.get(step) not in values:
                ok = False
                break
        if ok:
            rows.append(cfg)
    return rows


def build_dataset_step_context(
    performance_matrix: pd.DataFrame,
    pipeline_configs: Sequence[Mapping[str, Any]],
    options: Mapping[str, Sequence[str]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build per-dataset, per-step action priors used as contextual signal."""
    valid_rows = _valid_pipeline_rows(performance_matrix, pipeline_configs, options)
    if not valid_rows:
        raise ValueError("No valid pipeline rows for DQN warm-start context.")

    global_step_priors: Dict[str, np.ndarray] = {}
    for step, values in options.items():
        arr = []
        for val in values:
            scores = []
            for cfg in valid_rows:
                if cfg.get(step) != val:
                    continue
                name = cfg["name"]
                row_scores = performance_matrix.loc[name].astype(float).to_numpy()
                row_scores = row_scores[np.isfinite(row_scores)]
                if row_scores.size > 0:
                    scores.extend(row_scores.tolist())
            arr.append(float(np.mean(scores)) if scores else 0.0)
        global_step_priors[step] = _normalize_context(np.array(arr, dtype=np.float32))

    dataset_context: Dict[str, Dict[str, np.ndarray]] = {}
    for ds in performance_matrix.columns:
        ds_key = str(ds)
        step_ctx: Dict[str, np.ndarray] = {}
        for step, values in options.items():
            arr = []
            for val in values:
                scores = []
                for cfg in valid_rows:
                    if cfg.get(step) != val:
                        continue
                    name = cfg["name"]
                    sc = performance_matrix.at[name, ds]
                    if pd.notna(sc):
                        scores.append(float(sc))
                if scores:
                    arr.append(float(np.mean(scores)))
                else:
                    fallback = global_step_priors[step]
                    arr.append(float(fallback[len(arr)]))
            step_ctx[step] = _normalize_context(np.array(arr, dtype=np.float32))
        dataset_context[ds_key] = step_ctx
    return dataset_context


def build_offline_transitions(
    performance_matrix: pd.DataFrame,
    metafeatures_scaled: pd.DataFrame,
    pipeline_configs: Sequence[Mapping[str, Any]],
    options: Mapping[str, Sequence[str]],
    dataset_context: Mapping[str, Mapping[str, np.ndarray]],
) -> Tuple[List[List[Dict[str, Any]]], int]:
    """Create offline replay transitions from historical pipeline outcomes."""
    step_order = list(options.keys())
    offsets, history_dim = build_action_offsets(options)

    valid_rows = _valid_pipeline_rows(performance_matrix, pipeline_configs, options)
    common_ds = [str(ds) for ds in performance_matrix.columns if str(ds) in metafeatures_scaled.index]
    if not common_ds:
        raise ValueError("No common datasets between performance matrix and metafeatures for DQN.")

    transitions_by_step: List[List[Dict[str, Any]]] = [[] for _ in step_order]
    for ds in common_ds:
        mf_vec = metafeatures_scaled.loc[ds].to_numpy(dtype=np.float32)
        if ds not in dataset_context:
            continue
        ctx_ds = dataset_context[ds]

        for cfg in valid_rows:
            name = cfg["name"]
            score = performance_matrix.at[name, ds]
            if pd.isna(score):
                continue
            terminal_reward = float(score)
            history = np.zeros(history_dim, dtype=np.float32)
            for step_idx, step in enumerate(step_order):
                action = options[step].index(cfg[step])
                state = np.concatenate([mf_vec, history]).astype(np.float32)
                context = ctx_ds[step].astype(np.float32)

                next_history = history.copy()
                next_history[offsets[step] + action] = 1.0
                next_state = np.concatenate([mf_vec, next_history]).astype(np.float32)

                done = step_idx == (len(step_order) - 1)
                reward = terminal_reward if done else 0.0
                next_context = (
                    ctx_ds[step_order[step_idx + 1]].astype(np.float32)
                    if not done
                    else np.zeros((1,), dtype=np.float32)
                )

                transitions_by_step[step_idx].append(
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
                history = next_history

    state_dim = int(metafeatures_scaled.shape[1] + history_dim)
    return transitions_by_step, state_dim


class WarmStartDQNPolicy:
    """Stage-wise offline DQN policy with context gating."""

    def __init__(
        self,
        options: Mapping[str, Sequence[str]],
        state_dim: int,
        config: Optional[DQNPolicyConfig] = None,
    ):
        torch = _require_torch()
        import torch.nn as nn

        self._torch = torch
        self.options: Dict[str, List[str]] = {k: list(v) for k, v in options.items()}
        self.step_order = list(self.options.keys())
        self.offsets, self.history_dim = build_action_offsets(self.options)
        self.state_dim = state_dim
        self.config = config or DQNPolicyConfig()

        class _ContextGatedQNetwork(nn.Module):
            def __init__(self, state_dim_: int, context_dim_: int, action_dim_: int, hidden_dim_: int):
                super().__init__()
                self.state_encoder = nn.Sequential(
                    nn.Linear(state_dim_, hidden_dim_),
                    nn.ReLU(),
                    nn.Linear(hidden_dim_, hidden_dim_),
                    nn.ReLU(),
                )
                self.context_encoder = nn.Sequential(
                    nn.Linear(context_dim_, hidden_dim_),
                    nn.ReLU(),
                )
                self.gate = nn.Linear(hidden_dim_ * 2, hidden_dim_)
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim_, hidden_dim_),
                    nn.ReLU(),
                    nn.Linear(hidden_dim_, action_dim_),
                )

            def forward(self, state, context):
                # CtxPipe-style idea: learn a gate over context, then fuse multiplicatively.
                state_h = self.state_encoder(state)
                ctx_h = self.context_encoder(context)
                gate = torch.sigmoid(self.gate(torch.cat([state_h, ctx_h], dim=1)))
                fused = state_h * (1.0 + gate * ctx_h)
                return self.head(fused)

        self.nets = []
        self.target_nets = []
        self.optimizers = []
        for step in self.step_order:
            action_dim = len(self.options[step])
            net = _ContextGatedQNetwork(
                state_dim_=self.state_dim,
                context_dim_=action_dim,
                action_dim_=action_dim,
                hidden_dim_=self.config.hidden_dim,
            )
            tgt = _ContextGatedQNetwork(
                state_dim_=self.state_dim,
                context_dim_=action_dim,
                action_dim_=action_dim,
                hidden_dim_=self.config.hidden_dim,
            )
            tgt.load_state_dict(net.state_dict())
            self.nets.append(net)
            self.target_nets.append(tgt)
            self.optimizers.append(torch.optim.Adam(net.parameters(), lr=self.config.lr))

        self.training_summary: Dict[str, Any] = {}

    def _train_batch(self, step_idx: int, batch: Sequence[Mapping[str, Any]]) -> float:
        torch = self._torch
        if self.config.loss_fn == "mse":
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.SmoothL1Loss(beta=float(self.config.huber_delta))

        state_arr = np.nan_to_num(np.stack([x["state"] for x in batch]), nan=0.0, posinf=1e6, neginf=-1e6)
        context_arr = np.nan_to_num(np.stack([x["context"] for x in batch]), nan=0.0, posinf=1e6, neginf=-1e6)
        action_arr = np.asarray([x["action"] for x in batch], dtype=np.int64)
        action_arr = np.clip(action_arr, 0, len(self.options[self.step_order[step_idx]]) - 1)
        reward_arr = np.asarray([x["reward"] for x in batch], dtype=np.float32)

        state = torch.tensor(state_arr, dtype=torch.float32)
        context = torch.tensor(context_arr, dtype=torch.float32)
        action = torch.tensor(action_arr, dtype=torch.long)
        reward = torch.tensor(reward_arr, dtype=torch.float32)
        done = torch.tensor([1.0 if x["done"] else 0.0 for x in batch], dtype=torch.float32)
        if float(self.config.reward_clip) > 0:
            reward = torch.clamp(reward, -float(self.config.reward_clip), float(self.config.reward_clip))

        q_all = self.nets[step_idx](state, context)
        q_sa = q_all.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target = reward.clone()
            if step_idx < len(self.step_order) - 1:
                next_state_arr = np.nan_to_num(
                    np.stack([x["next_state"] for x in batch]),
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6,
                )
                next_context_arr = np.nan_to_num(
                    np.stack([x["next_context"] for x in batch]),
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6,
                )
                next_state = torch.tensor(next_state_arr, dtype=torch.float32)
                next_context = torch.tensor(next_context_arr, dtype=torch.float32)
                if self.config.use_double_dqn:
                    next_action = self.nets[step_idx + 1](next_state, next_context).argmax(dim=1, keepdim=True)
                    next_q = self.target_nets[step_idx + 1](next_state, next_context).gather(1, next_action).squeeze(1)
                else:
                    next_q = self.target_nets[step_idx + 1](next_state, next_context).max(dim=1).values
                target = reward + self.config.gamma * (1.0 - done) * next_q
            if float(self.config.target_q_clip) > 0:
                target = torch.clamp(target, -float(self.config.target_q_clip), float(self.config.target_q_clip))

        loss = loss_fn(q_sa, target)
        self.optimizers[step_idx].zero_grad()
        loss.backward()
        if float(self.config.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(self.nets[step_idx].parameters(), float(self.config.grad_clip_norm))
        self.optimizers[step_idx].step()
        return float(loss.item())

    def sync_target(self) -> None:
        for idx in range(len(self.nets)):
            self.target_nets[idx].load_state_dict(self.nets[idx].state_dict())

    def fit(self, transitions_by_step: Sequence[Sequence[Mapping[str, Any]]], seed: int = 42) -> Dict[str, Any]:
        rng = np.random.RandomState(seed)
        self._torch.manual_seed(seed)

        per_epoch_loss: List[float] = []
        for epoch in range(self.config.epochs):
            epoch_losses: List[float] = []
            for step_idx in reversed(range(len(self.step_order))):
                transitions = transitions_by_step[step_idx]
                if not transitions:
                    continue
                order = rng.permutation(len(transitions))
                for start in range(0, len(order), self.config.batch_size):
                    batch_ids = order[start : start + self.config.batch_size]
                    batch = [transitions[int(i)] for i in batch_ids]
                    loss = self._train_batch(step_idx=step_idx, batch=batch)
                    epoch_losses.append(loss)

            if (epoch + 1) % self.config.target_update_interval == 0:
                self.sync_target()

            per_epoch_loss.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

        self.training_summary = {
            "epochs": self.config.epochs,
            "num_steps": len(self.step_order),
            "num_transitions": int(sum(len(t) for t in transitions_by_step)),
            "mean_loss": float(np.mean(per_epoch_loss)) if per_epoch_loss else None,
            "last_loss": float(per_epoch_loss[-1]) if per_epoch_loss else None,
            "loss_curve": per_epoch_loss,
        }
        return dict(self.training_summary)

    def learn_from_replay(
        self,
        replay_by_step: Sequence[Sequence[Mapping[str, Any]]],
        rng: np.random.RandomState,
        n_updates: int = 1,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        batch_size = int(batch_size or self.config.batch_size)
        losses: List[float] = []
        updates = 0
        for _ in range(max(1, int(n_updates))):
            for step_idx in reversed(range(len(self.step_order))):
                transitions = replay_by_step[step_idx]
                if not transitions:
                    continue
                n = min(batch_size, len(transitions))
                ids = rng.choice(len(transitions), size=n, replace=False)
                batch = [transitions[int(i)] for i in ids]
                loss = self._train_batch(step_idx=step_idx, batch=batch)
                losses.append(loss)
                updates += 1
        return {
            "updates": updates,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "last_loss": float(losses[-1]) if losses else None,
        }

    def sample_pipeline(
        self,
        metafeatures: np.ndarray,
        warm_context: Mapping[str, np.ndarray],
        rng: np.random.RandomState,
        epsilon: float = 0.1,
    ) -> Dict[str, Any]:
        torch = self._torch
        cfg: Dict[str, Any] = {}
        history = np.zeros(self.history_dim, dtype=np.float32)
        mf = metafeatures.astype(np.float32)

        for step_idx, step in enumerate(self.step_order):
            state = np.concatenate([mf, history]).astype(np.float32)
            context = warm_context.get(step)
            if context is None:
                context = np.ones(len(self.options[step]), dtype=np.float32) / float(len(self.options[step]))
            context = _normalize_context(np.asarray(context, dtype=np.float32))

            state_t = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
            context_t = torch.tensor(context.reshape(1, -1), dtype=torch.float32)
            with torch.no_grad():
                q_values = self.nets[step_idx](state_t, context_t).cpu().numpy().ravel()
            q_values = np.nan_to_num(q_values, nan=0.0, posinf=1e6, neginf=-1e6)

            if self.config.warmstart_weight > 0:
                q_values = q_values + float(self.config.warmstart_weight) * context

            if rng.rand() < epsilon:
                action = int(rng.randint(0, len(self.options[step])))
            else:
                best = np.flatnonzero(q_values == np.max(q_values))
                action = int(rng.choice(best))

            cfg[step] = self.options[step][action]
            history[self.offsets[step] + action] = 1.0
        return cfg


class WarmStartOrderPolicy:
    """Context-gated DQN policy for selecting logical pipeline order."""

    def __init__(
        self,
        state_dim: int,
        order_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.95,
        reward_clip: float = 1.0,
        target_q_clip: float = 5.0,
        grad_clip_norm: float = 5.0,
        use_double_dqn: bool = True,
        huber_delta: float = 1.0,
    ):
        torch = _require_torch()
        import torch.nn as nn

        self._torch = torch
        self.order_dim = int(order_dim)
        self.gamma = float(gamma)
        self.reward_clip = float(reward_clip)
        self.target_q_clip = float(target_q_clip)
        self.grad_clip_norm = float(grad_clip_norm)
        self.use_double_dqn = bool(use_double_dqn)
        self.huber_delta = float(huber_delta)

        class _OrderQNetwork(nn.Module):
            def __init__(self, state_dim_: int, context_dim_: int, out_dim_: int, hidden_dim_: int):
                super().__init__()
                self.state_encoder = nn.Sequential(
                    nn.Linear(state_dim_, hidden_dim_),
                    nn.ReLU(),
                    nn.Linear(hidden_dim_, hidden_dim_),
                    nn.ReLU(),
                )
                self.context_encoder = nn.Sequential(
                    nn.Linear(context_dim_, hidden_dim_),
                    nn.ReLU(),
                )
                self.gate = nn.Linear(hidden_dim_ * 2, hidden_dim_)
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim_, hidden_dim_),
                    nn.ReLU(),
                    nn.Linear(hidden_dim_, out_dim_),
                )

            def forward(self, state, context):
                state_h = self.state_encoder(state)
                ctx_h = self.context_encoder(context)
                gate = torch.sigmoid(self.gate(torch.cat([state_h, ctx_h], dim=1)))
                fused = state_h * (1.0 + gate * ctx_h)
                return self.head(fused)

        self.net = _OrderQNetwork(
            state_dim_=int(state_dim),
            context_dim_=self.order_dim,
            out_dim_=self.order_dim,
            hidden_dim_=int(hidden_dim),
        )
        self.target_net = _OrderQNetwork(
            state_dim_=int(state_dim),
            context_dim_=self.order_dim,
            out_dim_=self.order_dim,
            hidden_dim_=int(hidden_dim),
        )
        self.target_net.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(lr))

    def sample_order(
        self,
        metafeatures: np.ndarray,
        order_context: np.ndarray,
        rng: np.random.RandomState,
        epsilon: float = 0.1,
    ) -> int:
        torch = self._torch
        state = metafeatures.astype(np.float32).reshape(1, -1)
        context = _normalize_context(order_context).astype(np.float32).reshape(1, -1)
        with torch.no_grad():
            q_values = self.net(
                torch.tensor(state, dtype=torch.float32),
                torch.tensor(context, dtype=torch.float32),
            ).cpu().numpy().ravel()
        q_values = np.nan_to_num(q_values, nan=0.0, posinf=1e6, neginf=-1e6)

        if rng.rand() < float(max(0.0, min(1.0, epsilon))):
            return int(rng.randint(0, self.order_dim))
        best = np.flatnonzero(q_values == np.max(q_values))
        return int(rng.choice(best))

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.net.state_dict())

    def learn_from_replay(
        self,
        replay: Sequence[Mapping[str, Any]],
        rng: np.random.RandomState,
        n_updates: int = 1,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        torch = self._torch
        if not replay:
            return {"updates": 0, "mean_loss": None, "last_loss": None}

        loss_fn = torch.nn.SmoothL1Loss(beta=self.huber_delta)
        losses: List[float] = []
        updates = 0
        n = int(max(1, min(batch_size, len(replay))))
        for _ in range(max(1, int(n_updates))):
            ids = rng.choice(len(replay), size=n, replace=False)
            batch = [replay[int(i)] for i in ids]
            state_arr = np.nan_to_num(np.stack([x["state"] for x in batch]), nan=0.0, posinf=1e6, neginf=-1e6)
            context_arr = np.nan_to_num(np.stack([x["context"] for x in batch]), nan=0.0, posinf=1e6, neginf=-1e6)
            next_state_arr = np.nan_to_num(
                np.stack([x["next_state"] for x in batch]),
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )
            next_context_arr = np.nan_to_num(
                np.stack([x["next_context"] for x in batch]),
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )
            state = torch.tensor(state_arr, dtype=torch.float32)
            context = torch.tensor(context_arr, dtype=torch.float32)
            action = torch.tensor(np.clip([int(x["action"]) for x in batch], 0, self.order_dim - 1), dtype=torch.long)
            reward = torch.tensor([float(x["reward"]) for x in batch], dtype=torch.float32)
            done = torch.tensor([1.0 if x.get("done", True) else 0.0 for x in batch], dtype=torch.float32)
            if self.reward_clip > 0:
                reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)
            next_state = torch.tensor(next_state_arr, dtype=torch.float32)
            next_context = torch.tensor(next_context_arr, dtype=torch.float32)

            q_all = self.net(state, context)
            q_sa = q_all.gather(1, action.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if self.use_double_dqn:
                    next_action = self.net(next_state, next_context).argmax(dim=1, keepdim=True)
                    next_q = self.target_net(next_state, next_context).gather(1, next_action).squeeze(1)
                else:
                    next_q = self.target_net(next_state, next_context).max(dim=1).values
                target = reward + self.gamma * (1.0 - done) * next_q
                if self.target_q_clip > 0:
                    target = torch.clamp(target, -self.target_q_clip, self.target_q_clip)

            loss = loss_fn(q_sa, target)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            losses.append(float(loss.item()))
            updates += 1

        return {
            "updates": updates,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "last_loss": float(losses[-1]) if losses else None,
        }
