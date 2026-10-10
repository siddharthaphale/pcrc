"""Tests for HER reward correctness.

§7.1 row: for ``final`` strategy, the last transition's relabeled
reward indicates success (the achieved goal IS the new training goal);
for ``future``, the source step's ``next_achieved_goal`` becomes the
new goal and reward is consistent with the supplied ``reward_fn``.
"""

from __future__ import annotations

import numpy as np

from pcrc.algorithms.her import (
    GoalConditionedTransition,
    relabel,
    sparse_fetch_reward,
)


def _make_episode(
    length: int, achieved_trajectory: np.ndarray, desired_goal: np.ndarray
) -> list[GoalConditionedTransition]:
    """Build a synthetic episode with a controlled achieved-goal trajectory.

    ``achieved_trajectory`` has shape ``(length + 1, 3)`` — index 0 is
    the starting achieved goal, indices 1..length are the
    ``next_achieved_goal`` after each step.
    """
    out: list[GoalConditionedTransition] = []
    for t in range(length):
        out.append(
            GoalConditionedTransition(
                observation=np.zeros(4, dtype=np.float32),
                action=np.zeros(4, dtype=np.float32),
                next_observation=np.zeros(4, dtype=np.float32),
                achieved_goal=achieved_trajectory[t].copy(),
                next_achieved_goal=achieved_trajectory[t + 1].copy(),
                reward=float(sparse_fetch_reward(achieved_trajectory[t + 1], desired_goal, {})),
                terminated=(t == length - 1),
                truncated=False,
                source_context_id=0,
                original_desired_goal=desired_goal.copy(),
                training_desired_goal=desired_goal.copy(),
                relabel_strategy="none",
                relabel_source_step=None,
                episode_id=0,
            )
        )
    return out


def test_final_relabel_last_transition_is_success() -> None:
    # Build an episode where the original desired_goal is far from any
    # achieved goal. Under "final" relabel, every transition's new
    # training_desired_goal equals the LAST next_achieved_goal in the
    # episode, so:
    #   - The relabel of the LAST transition has achieved == goal →
    #     sparse reward 0 (success).
    #   - Relabels of earlier transitions need not be success since
    #     their next_achieved_goal differs from the final.
    desired_goal = np.array([10.0, 10.0, 10.0])
    achieved = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ]
    )
    ep = _make_episode(length=3, achieved_trajectory=achieved, desired_goal=desired_goal)
    out = relabel(ep, strategy="final", rng=np.random.default_rng(0))
    relabeled_last = [
        t for t in out if t.relabel_strategy == "final" and t.next_achieved_goal[0] == 0.3
    ]
    assert len(relabeled_last) == 1
    assert relabeled_last[0].reward == 0.0  # success under sparse_fetch_reward


def test_final_relabel_training_goal_equals_episode_final_achieved() -> None:
    desired_goal = np.array([10.0, 10.0, 10.0])
    achieved = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ]
    )
    ep = _make_episode(length=3, achieved_trajectory=achieved, desired_goal=desired_goal)
    out = relabel(ep, strategy="final", rng=np.random.default_rng(0))
    final_achieved = achieved[-1]
    for t in out:
        if t.relabel_strategy == "final":
            np.testing.assert_array_equal(t.training_desired_goal, final_achieved)


def test_future_relabel_uses_source_step_next_achieved_goal() -> None:
    desired_goal = np.array([10.0, 10.0, 10.0])
    achieved = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ]
    )
    ep = _make_episode(length=3, achieved_trajectory=achieved, desired_goal=desired_goal)
    out = relabel(ep, strategy="future", n_sampled_goal=10, rng=np.random.default_rng(0))
    for t in out:
        if t.relabel_strategy == "future":
            src_step = t.relabel_source_step
            assert src_step is not None
            # training_desired_goal must equal the source step's
            # next_achieved_goal (which is achieved[src_step + 1] in the
            # original trajectory, since transition at index src_step
            # has next_achieved_goal = achieved[src_step + 1]).
            np.testing.assert_array_equal(t.training_desired_goal, achieved[src_step + 1])


def test_custom_reward_fn_is_honored() -> None:
    # Pass a custom reward that returns +1 if the achieved equals the
    # goal in the X dim, else -1. Verify the relabeled rewards follow
    # this rule.
    def custom(ach: np.ndarray, des: np.ndarray, info: dict) -> float:
        del info
        return 1.0 if float(ach[0]) == float(des[0]) else -1.0

    desired_goal = np.array([10.0, 10.0, 10.0])
    achieved = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ]
    )
    ep = _make_episode(length=3, achieved_trajectory=achieved, desired_goal=desired_goal)
    out = relabel(
        ep,
        strategy="final",
        reward_fn=custom,
        rng=np.random.default_rng(0),
    )
    for t in out:
        if t.relabel_strategy == "final":
            # For the LAST transition next_achieved_goal[0] = 0.3 ==
            # training_desired_goal[0] → reward = +1.
            # For earlier transitions next_achieved_goal[0] differs →
            # reward = -1.
            expected = (
                1.0 if float(t.next_achieved_goal[0]) == float(t.training_desired_goal[0]) else -1.0
            )
            assert t.reward == expected


def test_originals_keep_their_reward() -> None:
    # The originals returned from relabel(...) should still carry the
    # rewards they were constructed with — not the recomputed ones.
    desired_goal = np.array([10.0, 10.0, 10.0])
    achieved = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ]
    )
    ep = _make_episode(length=3, achieved_trajectory=achieved, desired_goal=desired_goal)
    out = relabel(ep, strategy="final", rng=np.random.default_rng(0))
    originals = [t for t in out if t.relabel_strategy == "none"]
    for orig, t in zip(ep, originals, strict=True):
        assert orig.reward == t.reward
