"""Tests for ``pcrc.algorithms.pnp_demos`` (scripted demo source, no env).

Covers the controller's action bounds + phase progression and the
save/load round-trip of the demos ``.npz`` (the BC + replay seed file).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pcrc.algorithms.her import GoalConditionedTransition
from pcrc.algorithms.pnp_demos import (
    ScriptedPnPController,
    load_demos,
    save_demos,
)


def test_controller_action_in_bounds() -> None:
    ctrl = ScriptedPnPController()
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.normal(size=25).astype(np.float32)
        goal = rng.normal(size=3).astype(np.float32)
        a = ctrl.act(obs, goal)
        assert a.shape == (4,)
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_controller_phase_advances_when_aligned() -> None:
    """Phase 0 -> 1 once the gripper hovers above the object (xy aligned, z at hover)."""
    ctrl = ScriptedPnPController()
    obj = np.array([1.3, 0.7, 0.42])
    obs = np.zeros(25, dtype=np.float32)
    obs[3:6] = obj
    obs[:3] = obj + np.array([0.0, 0.0, 0.03])  # exactly at hover pose
    assert ctrl._phase == 0
    ctrl.act(obs, np.array([1.2, 0.8, 0.6]))
    assert ctrl._phase == 1


def _mk_transition(cid: int, ep: int) -> GoalConditionedTransition:
    obs = np.arange(25, dtype=np.float32)
    return GoalConditionedTransition(
        observation=obs,
        action=np.ones(4, dtype=np.float32),
        next_observation=obs + 1,
        achieved_goal=obs[3:6].astype(np.float64),
        next_achieved_goal=(obs[3:6] + 1).astype(np.float64),
        reward=-1.0,
        terminated=False,
        truncated=False,
        source_context_id=cid,
        original_desired_goal=np.array([0.1, 0.2, 0.3]),
        training_desired_goal=np.array([0.1, 0.2, 0.3]),
        relabel_strategy="none",
        relabel_source_step=None,
        episode_id=ep,
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    episodes = [[_mk_transition(3, 0), _mk_transition(3, 0)], [_mk_transition(5, 1)]]
    out = tmp_path / "demos.npz"
    result = save_demos(episodes, out)
    assert result.n_transitions == 3
    data = load_demos(out)
    assert data["state"].shape == (3, 28)  # 25 obs + 3 goal
    assert data["action"].shape == (3, 4)
    assert set(data["source_context_id"].tolist()) == {3, 5}
    # State is [observation, original_desired_goal].
    assert np.allclose(data["state"][0, 25:], [0.1, 0.2, 0.3])
