"""CPU tests for the CRL (contrastive) agent — B2/M4."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pcrc.algorithms.crl_continuous import CRLAgent, CRLContinuousConfig
from pcrc.algorithms.sac_continuous import SACContinuousConfig


def _agent(obs_dim=6, goal_dim=2, action_dim=2, repr_dim=16):
    sac = SACContinuousConfig(
        state_dim=obs_dim + goal_dim, action_dim=action_dim, hidden_sizes=(32, 32),
        action_low=-1.0, action_high=1.0,
    )
    return CRLAgent(CRLContinuousConfig(sac=sac, goal_dim=goal_dim, repr_dim=repr_dim))


def test_invalid_config_raises():
    sac = SACContinuousConfig(state_dim=8, action_dim=2, action_low=-1.0, action_high=1.0)
    with pytest.raises(ValueError):
        CRLContinuousConfig(sac=sac, goal_dim=8)  # goal_dim >= state_dim
    with pytest.raises(ValueError):
        CRLContinuousConfig(sac=sac, goal_dim=2, repr_dim=0)


def test_act_shape_and_bounds():
    a = _agent()
    action = a.act(np.zeros(8, dtype=np.float32), deterministic=True)
    assert action.shape == (2,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_update_crl_runs_and_learns_contrast():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    a = _agent()
    obs = rng.normal(size=(64, 6)).astype(np.float32)
    act = rng.uniform(-1, 1, size=(64, 2)).astype(np.float32)
    # goals correlated with obs -> contrastive accuracy should rise above chance
    goals = (obs[:, :2] + 0.1 * rng.normal(size=(64, 2))).astype(np.float32)
    first = a.update_crl(obs, act, goals)
    for _ in range(150):
        last = a.update_crl(obs, act, goals)
    assert last["contrastive_acc"] > max(first["contrastive_acc"], 1.5 / 64)
    assert np.isfinite(last["critic_loss"]) and np.isfinite(last["actor_loss"])


def test_update_raises_on_td_path():
    a = _agent()
    assert a.supports_td is False
    with pytest.raises(RuntimeError):
        a.update(None, None, None, None, None)


def test_actor_state_dict_compatible_with_sac_actor():
    # Worker pool loads CRL actor weights into a SACAgent-built actor — keys must match.
    from pcrc.algorithms.sac_continuous import SACActor

    a = _agent()
    sac_actor = SACActor(a.cfg.sac)
    sac_actor.load_state_dict(a.actor.state_dict())  # must not raise
