"""Tests for ``pcrc.algorithms.sac_continuous``.

Actor / Q output shapes, deterministic action, update produces finite
metrics + advances target nets, save/load round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pcrc.algorithms.sac_continuous import (
    SACActor,
    SACAgent,
    SACContinuousConfig,
    SACQNetwork,
)


def _tiny_cfg(**overrides) -> SACContinuousConfig:
    cfg = SACContinuousConfig(
        state_dim=8,
        action_dim=4,
        hidden_sizes=(16, 16),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_actor_sample_shapes() -> None:
    cfg = _tiny_cfg()
    actor = SACActor(cfg)
    s = torch.zeros(5, cfg.state_dim)
    a, logp, tanh_mean = actor.sample(s)
    assert a.shape == (5, cfg.action_dim)
    assert logp.shape == (5,)
    assert tanh_mean.shape == (5, cfg.action_dim)


def test_q_network_output_shape() -> None:
    cfg = _tiny_cfg()
    q = SACQNetwork(cfg)
    s = torch.zeros(3, cfg.state_dim)
    a = torch.zeros(3, cfg.action_dim)
    out = q(s, a)
    assert out.shape == (3,)


def test_agent_act_returns_action_within_bounds() -> None:
    cfg = _tiny_cfg(action_low=-2.0, action_high=2.0)
    agent = SACAgent(cfg)
    state = np.zeros(cfg.state_dim, dtype=np.float32)
    a = agent.act(state, deterministic=True)
    assert a.shape == (cfg.action_dim,)
    assert (a >= -2.0).all() and (a <= 2.0).all()


def test_agent_update_returns_finite_metrics() -> None:
    cfg = _tiny_cfg()
    agent = SACAgent(cfg)
    B = 16
    s = np.random.default_rng(0).normal(size=(B, cfg.state_dim)).astype(np.float32)
    a = np.random.default_rng(1).normal(size=(B, cfg.action_dim)).astype(np.float32)
    r = np.random.default_rng(2).normal(size=B).astype(np.float32)
    s_next = np.random.default_rng(3).normal(size=(B, cfg.state_dim)).astype(np.float32)
    d = (np.random.default_rng(4).random(B) > 0.9).astype(np.float32)
    metrics = agent.update(s, a, r, s_next, d)
    for k, v in metrics.items():
        assert np.isfinite(v), k


def _rng_batch(cfg: SACContinuousConfig, b: int):
    rng = np.random.default_rng(0)
    return (
        rng.normal(size=(b, cfg.state_dim)).astype(np.float32),
        rng.normal(size=(b, cfg.action_dim)).astype(np.float32),
        rng.normal(size=b).astype(np.float32),
        rng.normal(size=(b, cfg.state_dim)).astype(np.float32),
        np.zeros(b, dtype=np.float32),
    )


def test_bc_lambda_zero_is_identical_to_vanilla() -> None:
    """bc_lambda=0 must be bit-for-bit identical to the no-demo update.

    Guards the Nair-2018 demos+BC lever's default-OFF contract: a demo
    minibatch may be supplied but with bc_lambda=0 it must not perturb the
    actor params at all (and emit no bc_* metrics).
    """
    cfg = _tiny_cfg()
    s, a, r, s_next, d = _rng_batch(cfg, 16)
    ds = np.random.default_rng(7).normal(size=(8, cfg.state_dim)).astype(np.float32)
    da = np.random.default_rng(8).normal(size=(8, cfg.action_dim)).astype(np.float32)

    # Seed BEFORE constructing each agent so both start from identical weights
    # and consume the in-update RNG (actor.sample) identically.
    torch.manual_seed(0)
    a1 = SACAgent(_tiny_cfg())
    a1.obs_rms.update(torch.as_tensor(s))
    torch.manual_seed(123)
    m_plain = a1.update(s, a, r, s_next, d)
    plain_params = [p.detach().clone() for p in a1.actor.parameters()]

    torch.manual_seed(0)
    a2 = SACAgent(_tiny_cfg())
    a2.obs_rms.update(torch.as_tensor(s))
    torch.manual_seed(123)
    m_demo = a2.update(s, a, r, s_next, d, demo_states=ds, demo_actions=da, bc_lambda=0.0)
    demo_params = [p.detach().clone() for p in a2.actor.parameters()]

    assert "bc_loss" not in m_plain and "bc_loss" not in m_demo
    for p, q in zip(plain_params, demo_params, strict=True):
        assert torch.allclose(p, q), "bc_lambda=0 changed the actor parameters"


def test_bc_loss_q_filter_metrics() -> None:
    """With bc_lambda>0 the update returns finite bc_loss + a [0,1] Q-filter frac."""
    cfg = _tiny_cfg()
    agent = SACAgent(cfg)
    s, a, r, s_next, d = _rng_batch(cfg, 16)
    agent.obs_rms.update(torch.as_tensor(s))
    ds = np.random.default_rng(7).normal(size=(8, cfg.state_dim)).astype(np.float32)
    da = np.random.default_rng(8).uniform(-1, 1, size=(8, cfg.action_dim)).astype(np.float32)
    m = agent.update(s, a, r, s_next, d, demo_states=ds, demo_actions=da, bc_lambda=1.0)
    assert np.isfinite(m["bc_loss"]) and m["bc_loss"] >= 0.0
    assert 0.0 <= m["bc_frac"] <= 1.0
    # No-filter clones every demo (frac == 1).
    m2 = agent.update(
        s, a, r, s_next, d, demo_states=ds, demo_actions=da, bc_lambda=1.0, use_q_filter=False
    )
    assert m2["bc_frac"] == 1.0


def test_agent_update_advances_target_nets() -> None:
    cfg = _tiny_cfg()
    agent = SACAgent(cfg)
    initial = {k: v.clone() for k, v in agent.target_q1.state_dict().items()}
    B = 8
    s = np.random.default_rng(0).normal(size=(B, cfg.state_dim)).astype(np.float32)
    a = np.random.default_rng(1).normal(size=(B, cfg.action_dim)).astype(np.float32)
    r = np.zeros(B, dtype=np.float32)
    s_next = np.random.default_rng(2).normal(size=(B, cfg.state_dim)).astype(np.float32)
    d = np.zeros(B, dtype=np.float32)
    # Run enough updates that the EMA visibly moves.
    for _ in range(20):
        agent.update(s, a, r, s_next, d)
    final = agent.target_q1.state_dict()
    diff = sum((final[k] - initial[k]).abs().sum().item() for k in initial)
    assert diff > 0


def test_save_load_roundtrip(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    a_agent = SACAgent(cfg)
    # One update so state-dicts differ from init.
    B = 8
    s = np.random.default_rng(0).normal(size=(B, cfg.state_dim)).astype(np.float32)
    a = np.random.default_rng(1).normal(size=(B, cfg.action_dim)).astype(np.float32)
    r = np.zeros(B, dtype=np.float32)
    s_next = np.random.default_rng(2).normal(size=(B, cfg.state_dim)).astype(np.float32)
    d = np.zeros(B, dtype=np.float32)
    a_agent.update(s, a, r, s_next, d)

    p = tmp_path / "sac.pt"
    a_agent.save(p)
    b_agent = SACAgent(cfg)
    b_agent.load(p)
    for k in a_agent.actor.state_dict():
        torch.testing.assert_close(a_agent.actor.state_dict()[k], b_agent.actor.state_dict()[k])
    # Alpha restored.
    assert torch.isclose(a_agent.alpha, b_agent.alpha)


# ----------------------------------------------------------- PCGrad (interference)


def test_pcgrad_project_orthogonal_is_sum() -> None:
    # Orthogonal task grads don't conflict (dot >= 0) -> projection leaves them, sum = elementwise sum.
    from pcrc.algorithms.sac_continuous import _pcgrad_project

    g = [[torch.tensor([1.0, 0.0])], [torch.tensor([0.0, 1.0])]]
    out = _pcgrad_project(g)
    assert torch.allclose(out[0], torch.tensor([1.0, 1.0]))


def test_pcgrad_project_opposing_cancels() -> None:
    # Exactly opposing grads (dot < 0) project each onto the other's normal -> cancel to ~0.
    from pcrc.algorithms.sac_continuous import _pcgrad_project

    g = [[torch.tensor([1.0, -1.0])], [torch.tensor([-1.0, 1.0])]]
    out = _pcgrad_project(g)
    assert torch.allclose(out[0], torch.zeros(2), atol=1e-6)


def test_pcgrad_update_runs_and_is_finite() -> None:
    # update() with pcgrad on + group_ids splits the minibatch by group and steps without error.
    cfg = _tiny_cfg(pcgrad=True)
    agent = SACAgent(cfg)
    rng = np.random.default_rng(0)
    B = 16
    states = rng.normal(size=(B, cfg.state_dim)).astype(np.float32)
    nstates = rng.normal(size=(B, cfg.state_dim)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(B, cfg.action_dim)).astype(np.float32)
    rewards = rng.uniform(-1, 0, size=B).astype(np.float32)
    dones = np.zeros(B, dtype=np.float32)
    gids = (rng.uniform(size=B) > 0.5).astype(np.int64)  # 2 groups
    m = agent.update(states, actions, rewards, nstates, dones, group_ids=gids)
    assert all(np.isfinite(v) for v in m.values())
