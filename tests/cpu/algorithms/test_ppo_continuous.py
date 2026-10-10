"""Tests for :mod:`pcrc.algorithms.ppo_continuous`.

Asserts the algorithmic building blocks are correct: actor / critic
output shapes, GAE matches hand-computed values, the clipped surrogate
returns expected sign + magnitude, save/load round-trip restores
state-dict bit-for-bit, and a tiny end-to-end ``PPOAgent.update`` runs
without NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pcrc.algorithms.ppo_continuous import (
    PPOActor,
    PPOAgent,
    PPOContinuousConfig,
    PPOCritic,
    compute_gae,
    ppo_clipped_loss,
)


def _tiny_cfg(**overrides) -> PPOContinuousConfig:
    cfg = PPOContinuousConfig(
        observation_dim=4,
        action_dim=2,
        hidden_sizes=(8, 8),
        minibatch_size=4,
        n_epochs=2,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Actor / Critic
# ---------------------------------------------------------------------------


def test_actor_output_shapes() -> None:
    cfg = _tiny_cfg()
    actor = PPOActor(cfg)
    obs = torch.zeros(5, cfg.observation_dim)
    action, logp, ent = actor.act(obs)
    assert action.shape == (5, cfg.action_dim)
    assert logp.shape == (5,)
    assert ent.shape == (5,)


def test_actor_deterministic_action_is_mean() -> None:
    cfg = _tiny_cfg()
    actor = PPOActor(cfg)
    obs = torch.zeros(3, cfg.observation_dim)
    action, _, _ = actor.act(obs, deterministic=True)
    dist = actor.distribution(obs)
    torch.testing.assert_close(action, dist.mean)


def test_critic_output_shape() -> None:
    cfg = _tiny_cfg()
    critic = PPOCritic(cfg)
    obs = torch.zeros(7, cfg.observation_dim)
    v = critic(obs)
    assert v.shape == (7,)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def test_gae_no_done_matches_truncated_horizon() -> None:
    # Hand-computed GAE for T=3, gamma=0.9, lambda=1.0, dones=0:
    # delta_t = r_t + gamma * V_{t+1} - V_t
    # A_t = sum_{k=t}^{T-1} (gamma*lambda)^{k-t} delta_k
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.5, 0.5, 0.5])
    dones = np.zeros(3)
    next_value = 0.0
    adv, ret = compute_gae(rewards, values, dones, next_value, gamma=0.9, gae_lambda=1.0)
    # Hand calc:
    # delta_2 = 1 + 0.9*0 - 0.5 = 0.5; A_2 = 0.5
    # delta_1 = 1 + 0.9*0.5 - 0.5 = 0.95; A_1 = 0.95 + 0.9*0.5 = 1.4
    # delta_0 = 1 + 0.9*0.5 - 0.5 = 0.95; A_0 = 0.95 + 0.9*1.4 = 2.21
    np.testing.assert_allclose(adv, [2.21, 1.4, 0.5], atol=1e-9)
    np.testing.assert_allclose(ret, adv + values, atol=1e-9)


def test_gae_terminal_done_zeros_bootstrap() -> None:
    rewards = np.array([1.0, 1.0])
    values = np.array([0.0, 0.0])
    dones = np.array([0.0, 1.0])  # terminal at t=1
    next_value = 100.0  # would explode if not gated by dones[1]=1
    adv, _ = compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=1.0)
    # delta_1 = 1 + 0.99 * 100 * (1-1) - 0 = 1; A_1 = 1.
    assert np.isclose(adv[1], 1.0)


def test_gae_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_gae(np.zeros(4), np.zeros(3), np.zeros(4), 0.0, gamma=0.99, gae_lambda=0.95)


# ---------------------------------------------------------------------------
# PPO clipped surrogate
# ---------------------------------------------------------------------------


def test_clipped_surrogate_at_unit_ratio_returns_minus_mean_advantage() -> None:
    new_logp = torch.zeros(4)
    old_logp = torch.zeros(4)
    adv = torch.tensor([1.0, 2.0, -1.0, 0.5])
    loss, clip_frac = ppo_clipped_loss(new_logp, old_logp, adv, clip_eps=0.2)
    # ratio = 1 → surrogate = adv → mean = 0.625 → loss = -0.625.
    assert np.isclose(loss.item(), -0.625, atol=1e-9)
    # No clipping when ratio == 1.
    assert np.isclose(clip_frac.item(), 0.0, atol=1e-9)


def test_clipped_surrogate_clips_at_eps_for_positive_advantage() -> None:
    # log ratio = 5 → ratio = 148, A = 1. min(148, 1.2) = 1.2 → loss = -1.2.
    new_logp = torch.tensor([5.0])
    old_logp = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    loss, clip_frac = ppo_clipped_loss(new_logp, old_logp, adv, clip_eps=0.2)
    assert np.isclose(loss.item(), -1.2, atol=1e-6)
    assert np.isclose(clip_frac.item(), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def test_agent_act_returns_correct_shapes() -> None:
    cfg = _tiny_cfg()
    agent = PPOAgent(cfg)
    obs = np.zeros(cfg.observation_dim, dtype=np.float32)
    action, logp, value = agent.act(obs)
    assert action.shape == (cfg.action_dim,)
    assert isinstance(logp, float)
    assert isinstance(value, float)


def test_agent_act_deterministic_is_reproducible() -> None:
    cfg = _tiny_cfg()
    agent = PPOAgent(cfg)
    obs = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    a, _, _ = agent.act(obs, deterministic=True)
    b, _, _ = agent.act(obs, deterministic=True)
    np.testing.assert_array_equal(a, b)


def test_agent_update_runs_without_nan() -> None:
    cfg = _tiny_cfg()
    agent = PPOAgent(cfg)
    n = 16
    obs = np.random.default_rng(0).normal(size=(n, cfg.observation_dim)).astype(np.float32)
    actions = np.random.default_rng(1).normal(size=(n, cfg.action_dim)).astype(np.float32)
    old_logp = np.zeros(n, dtype=np.float32)
    advantages = np.random.default_rng(2).normal(size=n).astype(np.float32)
    returns = advantages + np.random.default_rng(3).normal(size=n).astype(np.float32)
    metrics = agent.update(obs, actions, old_logp, advantages, returns)
    for k, v in metrics.items():
        assert not np.isnan(v), k


def test_agent_save_load_roundtrip(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    agent_a = PPOAgent(cfg)
    # Force a real update so state-dicts differ from init.
    n = 8
    obs = np.random.default_rng(0).normal(size=(n, cfg.observation_dim)).astype(np.float32)
    actions = np.random.default_rng(1).normal(size=(n, cfg.action_dim)).astype(np.float32)
    old_logp = np.zeros(n, dtype=np.float32)
    advantages = np.random.default_rng(2).normal(size=n).astype(np.float32)
    returns = advantages + 0.1
    agent_a.update(obs, actions, old_logp, advantages, returns)
    p = tmp_path / "agent.pt"
    agent_a.save(p)

    agent_b = PPOAgent(cfg)
    agent_b.load(p)
    # Identical actor state-dicts after load.
    sa = agent_a.actor.state_dict()
    sb = agent_b.actor.state_dict()
    for k in sa:
        torch.testing.assert_close(sa[k], sb[k])
    # Identical critic state-dicts.
    sa = agent_a.critic.state_dict()
    sb = agent_b.critic.state_dict()
    for k in sa:
        torch.testing.assert_close(sa[k], sb[k])
