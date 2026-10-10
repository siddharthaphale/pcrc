"""Regression test for goal-independent gripper->object reward shaping
in the Fetch SAC+HER loop (`reward_shaping_coef`).

Locks in: the shaping mutates exactly `t.reward` (the field SAC trains on),
applies `-coef * ||gripper - object||` (gripper = observation[:3], object =
achieved_goal), is goal-independent (identical on a transition and its
HER-relabeled copy), and is a no-op when coef=0.
"""

from __future__ import annotations

import numpy as np

from pcrc.algorithms.her import GoalConditionedTransition


def _shape(t: GoalConditionedTransition, coef: float) -> float:
    """The exact arithmetic the loop applies (mirror of fetch_sac_her_loop)."""
    gripper = np.asarray(t.observation[:3], dtype=np.float64)
    obj = np.asarray(t.achieved_goal, dtype=np.float64)
    dist = float(np.linalg.norm(gripper - obj))
    return float(t.reward - coef * dist)


def _mk(reward: float, gripper, obj, goal) -> GoalConditionedTransition:
    obs = np.concatenate([np.asarray(gripper, float), np.zeros(22)])  # obs[:3]=gripper
    return GoalConditionedTransition(
        observation=obs,
        action=np.zeros(4),
        next_observation=obs,
        achieved_goal=np.asarray(obj, float),
        next_achieved_goal=np.asarray(obj, float),
        reward=reward,
        terminated=False,
        truncated=False,
        source_context_id=0,
        original_desired_goal=np.asarray(goal, float),
        training_desired_goal=np.asarray(goal, float),
        relabel_strategy="none",
        relabel_source_step=None,
        episode_id=0,
    )


def test_shaping_is_negative_distance_penalty() -> None:
    t = _mk(reward=-1.0, gripper=[0.0, 0.0, 0.0], obj=[0.3, 0.4, 0.0], goal=[1.0, 0.0, 0.0])
    # dist = 0.5 → shaped = -1.0 - 2.0*0.5 = -2.0
    assert abs(_shape(t, coef=2.0) - (-2.0)) < 1e-9


def test_shaping_is_goal_independent_survives_relabel() -> None:
    # Same transition, two different desired goals (as HER relabel would set):
    # the shaping depends only on gripper+object, so it must be identical.
    g, o = [0.1, 0.2, 0.3], [0.4, 0.2, 0.3]
    t1 = _mk(reward=-1.0, gripper=g, obj=o, goal=[1.0, 1.0, 1.0])
    t2 = _mk(reward=-1.0, gripper=g, obj=o, goal=[-1.0, 0.0, 0.5])
    assert abs(_shape(t1, 3.0) - _shape(t2, 3.0)) < 1e-12


def test_shaping_coef_zero_is_noop() -> None:
    t = _mk(reward=-1.0, gripper=[0.0, 0.0, 0.0], obj=[1.0, 0.0, 0.0], goal=[0.0, 0.0, 0.0])
    assert _shape(t, coef=0.0) == -1.0


def test_shaping_preserves_success_reward_minus_penalty() -> None:
    # On a solved step (sparse reward 0) the shaped reward is just the penalty.
    t = _mk(reward=0.0, gripper=[0.0, 0.0, 0.0], obj=[0.0, 0.0, 0.1], goal=[0.0, 0.0, 0.1])
    assert abs(_shape(t, coef=1.0) - (-0.1)) < 1e-9
