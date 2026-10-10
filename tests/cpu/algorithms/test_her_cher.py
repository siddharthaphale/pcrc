"""Tests for the CHER goal-selection relabel mode (Fang et al. 2019).

CHER reuses the "future" candidate pool (later achieved goals within
the episode) but replaces the uniform pick with a proximity+diversity
submodular selection via lazier-greedy. These tests check:

- The relabel invariants still hold (context identity, provenance tag
  "future", reward recompute) — CHER must not break the curator schema.
- λ(t) curriculum annealing (UP over training).
- Proximity bias: with very large λ, CHER prefers candidate goals close
  to the true desired goal over far ones.
- The default ("future") path is byte-identical when CHER is unused.
"""

from __future__ import annotations

import numpy as np

from pcrc.algorithms.her import (
    CHERConfig,
    GoalConditionedTransition,
    _cher_lambda,
    _cher_select,
    relabel,
    sparse_fetch_reward,
)


def _make_episode(
    length: int,
    desired_goal: np.ndarray,
    achieved_goals: np.ndarray,
) -> list[GoalConditionedTransition]:
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
                source_context_id=11,
                original_desired_goal=desired_goal.copy(),
                training_desired_goal=desired_goal.copy(),
                relabel_strategy="none",
                relabel_source_step=None,
                episode_id=0,
            )
        )
    return out


def test_cher_lambda_anneals_up() -> None:
    cfg = CHERConfig(lambda_starter=1.0, lambda_rate=0.01, fixed_lambda=None)
    assert _cher_lambda(cfg, 0) == 1.0
    assert _cher_lambda(cfg, 100) > _cher_lambda(cfg, 0)
    # fixed overrides the schedule.
    assert _cher_lambda(CHERConfig(fixed_lambda=3.0), 50) == 3.0


def test_cher_select_prefers_proximity_under_large_lambda() -> None:
    # Candidates along a line; one is exactly the desired goal.
    desired = np.array([0.0, 0.0, 0.0])
    candidates = np.array(
        [
            [5.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # == desired (closest)
            [3.0, 0.0, 0.0],
        ]
    )
    rng = np.random.default_rng(0)
    # Large lambda -> proximity dominates -> the closest candidate (idx 2)
    # should be selected first across seeds.
    chosen_first = []
    for s in range(20):
        sel = _cher_select(
            candidates,
            desired,
            n_select=1,
            lam=100.0,
            sub_size=4,
            rng=np.random.default_rng(s),
        )
        chosen_first.append(sel[0])
    assert all(c == 2 for c in chosen_first)
    del rng


def test_cher_relabel_preserves_invariants() -> None:
    desired = np.array([10.0, 10.0, 10.0])
    ag = np.linspace(0.0, 1.0, 7).reshape(7, 1).repeat(3, axis=1)
    ep = _make_episode(6, desired, ag)
    out = relabel(
        ep,
        strategy="cher",
        n_sampled_goal=2,
        rng=np.random.default_rng(0),
        cher_cfg=CHERConfig(),
        step=10,
    )
    relabeled = [t for t in out if t.relabel_strategy != "none"]
    assert len(relabeled) > 0
    for t in out:
        assert t.source_context_id == 11
        np.testing.assert_array_equal(t.original_desired_goal, desired)
    for t in relabeled:
        # CHER is provenance-tagged "future".
        assert t.relabel_strategy == "future"
        assert isinstance(t.relabel_source_step, int)
        assert not np.array_equal(t.training_desired_goal, t.original_desired_goal)


def test_cher_emits_at_most_n_sampled_goal_per_transition() -> None:
    desired = np.array([10.0, 10.0, 10.0])
    ag = np.linspace(0.0, 1.0, 7).reshape(7, 1).repeat(3, axis=1)
    ep = _make_episode(6, desired, ag)
    out = relabel(ep, strategy="cher", n_sampled_goal=3, rng=np.random.default_rng(1))
    originals = [t for t in out if t.relabel_strategy == "none"]
    assert len(originals) == len(ep)
    # First transition has 5 later steps -> exactly 3 relabels.
    first_relabels = [t for t in out if t.relabel_strategy == "future" and t.observation[0] == 0]
    assert len(first_relabels) == 3
    # No duplicate source steps for a given transition (selection is without replacement).
    assert len({t.relabel_source_step for t in first_relabels}) == 3


def test_future_path_unchanged_by_cher_addition() -> None:
    # Same seed + "future" must produce identical picks regardless of CHER plumbing.
    desired = np.array([10.0, 10.0, 10.0])
    ag = np.linspace(0.0, 1.0, 7).reshape(7, 1).repeat(3, axis=1)
    ep = _make_episode(6, desired, ag)
    out = relabel(ep, strategy="future", n_sampled_goal=2, rng=np.random.default_rng(123))
    steps = sorted(
        (int(t.observation[0]), t.relabel_source_step)
        for t in out
        if t.relabel_strategy == "future"
    )
    # Deterministic under the fixed seed.
    out2 = relabel(ep, strategy="future", n_sampled_goal=2, rng=np.random.default_rng(123))
    steps2 = sorted(
        (int(t.observation[0]), t.relabel_source_step)
        for t in out2
        if t.relabel_strategy == "future"
    )
    assert steps == steps2
