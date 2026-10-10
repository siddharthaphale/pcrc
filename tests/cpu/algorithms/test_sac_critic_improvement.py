"""Tests for :func:`pcrc.algorithms.sac_critic_improvement.sac_critic_improvement`.

§7.1 row: per-``source_context_id`` ``Q_{t+1} - Q_t`` aggregation
correct under known Q tensors; per-context counts match input.
Verifies the function consumes
:attr:`GoalConditionedTransition.original_desired_goal` (not
``training_desired_goal``) — the HER source-identity invariant.
"""

from __future__ import annotations

import numpy as np
import torch

from pcrc.algorithms.her import GoalConditionedTransition
from pcrc.algorithms.sac_critic_improvement import (
    sac_actor_objective_improvement,
    sac_critic_improvement,
)


def _make_transition(*, source_context_id: int, obs_value: float) -> GoalConditionedTransition:
    return GoalConditionedTransition(
        observation=np.full(4, obs_value, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        next_observation=np.zeros(4, dtype=np.float32),
        achieved_goal=np.zeros(3),
        next_achieved_goal=np.zeros(3),
        reward=0.0,
        terminated=False,
        truncated=False,
        source_context_id=source_context_id,
        original_desired_goal=np.array([1.0, 2.0, 3.0]),
        training_desired_goal=np.array([99.0, 99.0, 99.0]),  # different on purpose
        relabel_strategy="none",
        relabel_source_step=None,
        episode_id=0,
    )


def test_per_context_aggregation_matches_hand_computed() -> None:
    # Three contexts, two transitions each. Construct Q-fns that
    # return a value purely as a function of obs (independent of
    # actions / goals), so improvement per context is deterministic
    # and easy to hand-compute.
    transitions = [
        _make_transition(source_context_id=0, obs_value=1.0),
        _make_transition(source_context_id=0, obs_value=2.0),
        _make_transition(source_context_id=1, obs_value=3.0),
        _make_transition(source_context_id=1, obs_value=4.0),
        _make_transition(source_context_id=2, obs_value=5.0),
        _make_transition(source_context_id=2, obs_value=6.0),
    ]

    def actor_fn(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        del goal
        return torch.zeros(state.shape[0], 2)

    # Q_t = obs[0] · 1.0; Q_{t+1} = obs[0] · 1.5. Improvement = obs[0] · 0.5.
    def q_fn_t(state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        del action, goal
        return state[:, 0]

    def q_fn_tplus1(state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        del action, goal
        return 1.5 * state[:, 0]

    out = sac_critic_improvement(
        transitions,
        q_fn_t=q_fn_t,
        actor_fn_t=actor_fn,
        q_fn_tplus1=q_fn_tplus1,
        actor_fn_tplus1=actor_fn,
    )
    # Context 0: improvements [1*0.5, 2*0.5] = [0.5, 1.0], mean = 0.75.
    # Context 1: [3*0.5, 4*0.5] = [1.5, 2.0], mean = 1.75.
    # Context 2: [5*0.5, 6*0.5] = [2.5, 3.0], mean = 2.75.
    assert np.isclose(out[0], 0.75, atol=1e-6)
    assert np.isclose(out[1], 1.75, atol=1e-6)
    assert np.isclose(out[2], 2.75, atol=1e-6)


def test_uses_original_desired_goal_not_training_desired_goal() -> None:
    """The HER source-identity invariant: Q-evaluation uses
    ``original_desired_goal``. The fixture above set
    ``training_desired_goal = [99, 99, 99]`` while
    ``original_desired_goal = [1, 2, 3]``; this test pins that the
    function actually consumes the *original* goal.
    """
    transitions = [
        _make_transition(source_context_id=0, obs_value=1.0),
    ]
    captured_goals: list[float] = []

    def actor_fn(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state.shape[0], 2)

    def q_fn(state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        del action
        # Pull out the goal's x-coord; the original goal has x=1.0,
        # the (deliberately wrong) training goal has x=99.0.
        captured_goals.append(float(goal[0, 0].item()))
        return state[:, 0]

    sac_critic_improvement(
        transitions,
        q_fn_t=q_fn,
        actor_fn_t=actor_fn,
        q_fn_tplus1=q_fn,
        actor_fn_tplus1=actor_fn,
    )
    # Both q_fn_t and q_fn_tplus1 were called once with the same
    # original_desired_goal — must be 1.0, not 99.0.
    assert all(g == 1.0 for g in captured_goals), captured_goals


def test_actor_objective_isolates_policy_under_fixed_post_critic() -> None:
    # actor_objective improvement = Q_{t+1}(s, π_{t+1}(s)) - Q_{t+1}(s, π_t(s)),
    # i.e. BOTH policies evaluated under the SAME post critic (isolates the
    # policy update). Pre policy -> action 0; post policy -> action 1; the
    # post critic returns action[:, 0] * obs[0], so improvement per transition
    # is (1 - 0) * obs[0] = obs[0].
    transitions = [
        _make_transition(source_context_id=0, obs_value=1.0),
        _make_transition(source_context_id=0, obs_value=3.0),
        _make_transition(source_context_id=7, obs_value=4.0),
    ]

    def actor_pre(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        del goal
        return torch.zeros(state.shape[0], 2)

    def actor_post(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        del goal
        return torch.ones(state.shape[0], 2)

    def q_post(state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        del goal
        return action[:, 0] * state[:, 0]

    out = sac_actor_objective_improvement(
        transitions,
        actor_fn_t=actor_pre,
        actor_fn_tplus1=actor_post,
        q_fn_tplus1=q_post,
    )
    # Context 0: improvements [1.0, 3.0], mean = 2.0. Context 7: [4.0], mean = 4.0.
    assert np.isclose(out[0], 2.0, atol=1e-6)
    assert np.isclose(out[7], 4.0, atol=1e-6)


def test_actor_objective_empty_returns_empty() -> None:
    def fn(*args):
        return torch.zeros(1)

    assert (
        sac_actor_objective_improvement([], actor_fn_t=fn, actor_fn_tplus1=fn, q_fn_tplus1=fn) == {}
    )


def test_empty_transitions_returns_empty_dict() -> None:
    def fn(*args):
        return torch.zeros(1)

    out = sac_critic_improvement(
        [],
        q_fn_t=fn,
        actor_fn_t=fn,
        q_fn_tplus1=fn,
        actor_fn_tplus1=fn,
    )
    assert out == {}


def test_only_contexts_with_transitions_appear_in_output() -> None:
    transitions = [_make_transition(source_context_id=42, obs_value=1.0)]

    def actor_fn(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state.shape[0], 2)

    def q_fn(state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        del action, goal
        return torch.zeros(state.shape[0])

    out = sac_critic_improvement(
        transitions,
        q_fn_t=q_fn,
        actor_fn_t=actor_fn,
        q_fn_tplus1=q_fn,
        actor_fn_tplus1=actor_fn,
    )
    # Only context 42 in the input → only context 42 in the output.
    assert set(out.keys()) == {42}
