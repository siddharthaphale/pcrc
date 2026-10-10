"""Tests for the HER context-identity invariant.

§7.1 row + §3.4 schema + §9 risk #10: after
``relabel(transitions, "future", n_sampled_goal=4)``: every relabeled
transition has unchanged ``source_context_id``, unchanged
``original_desired_goal``, and (if relabeled) ``training_desired_goal
!= original_desired_goal``; ``relabel_strategy`` and
``relabel_source_step`` are set; reward is recomputed against
``training_desired_goal``.
"""

from __future__ import annotations

import numpy as np

from pcrc.algorithms.her import (
    GoalConditionedTransition,
    relabel,
    sparse_fetch_reward,
)


def _make_episode(
    episode_id: int,
    source_context_id: int,
    length: int,
    desired_goal: np.ndarray,
    achieved_goals: np.ndarray,
) -> list[GoalConditionedTransition]:
    """Build a synthetic episode with explicit achieved-goal trajectory."""
    assert achieved_goals.shape == (length + 1, 3)
    out: list[GoalConditionedTransition] = []
    for t in range(length):
        out.append(
            GoalConditionedTransition(
                observation=np.full(4, t, dtype=np.float32),
                action=np.zeros(4, dtype=np.float32),
                next_observation=np.full(4, t + 1, dtype=np.float32),
                achieved_goal=achieved_goals[t].copy(),
                next_achieved_goal=achieved_goals[t + 1].copy(),
                reward=float(sparse_fetch_reward(achieved_goals[t + 1], desired_goal, {})),
                terminated=(t == length - 1),
                truncated=False,
                source_context_id=source_context_id,
                original_desired_goal=desired_goal.copy(),
                training_desired_goal=desired_goal.copy(),
                relabel_strategy="none",
                relabel_source_step=None,
                episode_id=episode_id,
            )
        )
    return out


def test_relabel_preserves_source_context_id() -> None:
    desired_goal = np.array([1.5, 0.8, 0.5])
    ag = np.linspace(0.0, 1.0, 6).reshape(6, 1).repeat(3, axis=1)
    ep = _make_episode(
        0, source_context_id=42, length=5, desired_goal=desired_goal, achieved_goals=ag
    )
    out = relabel(ep, strategy="future", n_sampled_goal=2, rng=np.random.default_rng(0))
    for t in out:
        assert t.source_context_id == 42


def test_relabel_preserves_original_desired_goal() -> None:
    desired_goal = np.array([1.5, 0.8, 0.5])
    ag = np.linspace(0.0, 1.0, 6).reshape(6, 1).repeat(3, axis=1)
    ep = _make_episode(
        0, source_context_id=7, length=5, desired_goal=desired_goal, achieved_goals=ag
    )
    out = relabel(ep, strategy="future", n_sampled_goal=3, rng=np.random.default_rng(0))
    for t in out:
        np.testing.assert_array_equal(t.original_desired_goal, desired_goal)


def test_relabeled_transitions_have_different_training_goal() -> None:
    desired_goal = np.array([10.0, 10.0, 10.0])  # very far so no accidental match
    ag = np.linspace(0.0, 1.0, 6).reshape(6, 1).repeat(3, axis=1)
    ep = _make_episode(
        0, source_context_id=3, length=5, desired_goal=desired_goal, achieved_goals=ag
    )
    out = relabel(ep, strategy="future", n_sampled_goal=2, rng=np.random.default_rng(0))
    relabeled = [t for t in out if t.relabel_strategy != "none"]
    assert len(relabeled) > 0
    for t in relabeled:
        assert not np.array_equal(t.training_desired_goal, t.original_desired_goal)


def test_relabel_records_provenance() -> None:
    desired_goal = np.array([10.0, 10.0, 10.0])
    ag = np.linspace(0.0, 1.0, 6).reshape(6, 1).repeat(3, axis=1)
    ep = _make_episode(
        0, source_context_id=3, length=5, desired_goal=desired_goal, achieved_goals=ag
    )
    out = relabel(ep, strategy="future", n_sampled_goal=2, rng=np.random.default_rng(0))
    for t in out:
        if t.relabel_strategy == "none":
            assert t.relabel_source_step is None
        else:
            assert t.relabel_strategy in ("future", "final")
            assert isinstance(t.relabel_source_step, int)


def test_final_strategy_one_relabel_per_transition() -> None:
    desired_goal = np.array([10.0, 10.0, 10.0])
    ag = np.linspace(0.0, 1.0, 6).reshape(6, 1).repeat(3, axis=1)
    ep = _make_episode(
        0, source_context_id=3, length=5, desired_goal=desired_goal, achieved_goals=ag
    )
    out = relabel(ep, strategy="final", rng=np.random.default_rng(0))
    # Originals (5) plus one relabel each (5) = 10.
    assert len(out) == 2 * len(ep)


def test_future_strategy_emits_no_relabels_for_final_transition() -> None:
    # The last transition in an episode has no later steps → no
    # "future" relabel is possible.
    desired_goal = np.array([10.0, 10.0, 10.0])
    ag = np.linspace(0.0, 1.0, 4).reshape(4, 1).repeat(3, axis=1)
    ep = _make_episode(
        0, source_context_id=3, length=3, desired_goal=desired_goal, achieved_goals=ag
    )
    out = relabel(ep, strategy="future", n_sampled_goal=5, rng=np.random.default_rng(0))
    # Last step (index 2) cannot be relabeled.
    relabeled_for_last = [t for t in out if t.relabel_strategy != "none" and t.observation[0] == 2]
    assert len(relabeled_for_last) == 0


def test_originals_preserved_in_output() -> None:
    desired_goal = np.array([10.0, 10.0, 10.0])
    ag = np.linspace(0.0, 1.0, 4).reshape(4, 1).repeat(3, axis=1)
    ep = _make_episode(
        0, source_context_id=3, length=3, desired_goal=desired_goal, achieved_goals=ag
    )
    out = relabel(ep, strategy="future", n_sampled_goal=2, rng=np.random.default_rng(0))
    originals = [t for t in out if t.relabel_strategy == "none"]
    assert len(originals) == len(ep)
    for orig, t in zip(ep, originals, strict=True):
        assert orig.observation is t.observation
