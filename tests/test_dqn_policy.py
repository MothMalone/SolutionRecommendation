import numpy as np
import pandas as pd
import pytest

from automl_aco.metalearning.dqn_policy import (
    DQNPolicyConfig,
    WarmStartDQNPolicy,
    WarmStartOrderPolicy,
    build_dataset_step_context,
    build_offline_transitions,
)


def _toy_data():
    performance_matrix = pd.DataFrame(
        [
            [0.80, 0.70],
            [0.60, 0.90],
            [0.75, 0.65],
        ],
        index=["p1", "p2", "p3"],
        columns=["1", "2"],
    )
    metafeatures_scaled = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=["1", "2"],
        columns=["mf1", "mf2"],
    )
    pipeline_configs = [
        {"name": "p1", "imputation": "none", "scaling": "none"},
        {"name": "p2", "imputation": "mean", "scaling": "standard"},
        {"name": "p3", "imputation": "none", "scaling": "standard"},
    ]
    options = {"imputation": ["none", "mean"], "scaling": ["none", "standard"]}
    return performance_matrix, metafeatures_scaled, pipeline_configs, options


def test_build_dataset_step_context_shapes():
    perf, _mf, cfgs, options = _toy_data()
    ctx = build_dataset_step_context(perf, cfgs, options)
    assert set(ctx.keys()) == {"1", "2"}
    for ds in ctx:
        assert ctx[ds]["imputation"].shape == (2,)
        assert ctx[ds]["scaling"].shape == (2,)


def test_build_offline_transitions_returns_stepwise_buffers():
    perf, mf, cfgs, options = _toy_data()
    ctx = build_dataset_step_context(perf, cfgs, options)
    transitions, state_dim = build_offline_transitions(
        performance_matrix=perf,
        metafeatures_scaled=mf,
        pipeline_configs=cfgs,
        options=options,
        dataset_context=ctx,
    )
    assert len(transitions) == 2  # two stages
    assert state_dim == 6  # 2 metafeatures + 4 history slots
    assert len(transitions[0]) > 0
    assert len(transitions[1]) > 0


def test_warmstart_dqn_policy_smoke_train_and_sample():
    pytest.importorskip("torch")
    perf, mf, cfgs, options = _toy_data()
    ctx = build_dataset_step_context(perf, cfgs, options)
    transitions, state_dim = build_offline_transitions(
        performance_matrix=perf,
        metafeatures_scaled=mf,
        pipeline_configs=cfgs,
        options=options,
        dataset_context=ctx,
    )

    policy = WarmStartDQNPolicy(
        options=options,
        state_dim=state_dim,
        config=DQNPolicyConfig(epochs=2, batch_size=8, hidden_dim=16),
    )
    summary = policy.fit(transitions_by_step=transitions, seed=0)
    assert summary["num_transitions"] > 0

    warm_context = {"imputation": np.array([0.9, 0.1]), "scaling": np.array([0.1, 0.9])}
    cfg = policy.sample_pipeline(
        metafeatures=mf.loc["1"].to_numpy(dtype=np.float32),
        warm_context=warm_context,
        rng=np.random.RandomState(0),
        epsilon=0.0,
    )
    assert set(cfg.keys()) == set(options.keys())
    assert cfg["imputation"] in options["imputation"]
    assert cfg["scaling"] in options["scaling"]


def test_warmstart_order_policy_smoke():
    pytest.importorskip("torch")
    rng = np.random.RandomState(0)
    policy = WarmStartOrderPolicy(state_dim=3, order_dim=4, hidden_dim=8, lr=1e-3, gamma=0.95)
    mf = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    order_ctx = np.array([0.2, 0.8, 0.4, 0.1], dtype=np.float32)
    action = policy.sample_order(metafeatures=mf, order_context=order_ctx, rng=rng, epsilon=0.0)
    assert 0 <= action < 4

    replay = [
        {
            "state": mf,
            "context": order_ctx,
            "action": int(action),
            "reward": 0.7,
            "done": True,
            "next_state": mf,
            "next_context": order_ctx,
        }
        for _ in range(8)
    ]
    out = policy.learn_from_replay(replay=replay, rng=rng, n_updates=2, batch_size=4)
    assert out["updates"] > 0
