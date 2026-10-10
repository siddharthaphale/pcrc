"""Tests for :func:`pcrc.algorithms.policy_improvement_utility.ppo_policy_improvement_utility`.

§7.1 row: sign-correct PPO-clipped surrogate
``min(ρ·A, clip(ρ, 1-eps, 1+eps)·A)`` on a fixture; per-episode
then per-context aggregation matches hand-computed; outputs are
detached from input grads (no ``requires_grad=True`` paths into the
actor).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pcrc.algorithms.policy_improvement_utility import (
    ppo_policy_improvement_utility,
)


def test_zero_improvement_when_policies_identical_and_centering_disabled() -> None:
    # old_logp = new_logp → ratio = 1 → surrogate = advantages.
    # Aggregate: per-episode mean of advantages, per-context mean of
    # episode means. With ``center=False, scale=False`` the output is
    # those means directly.
    old = torch.tensor([0.0, 0.0, 0.0, 0.0])
    new = old.clone()
    adv = torch.tensor([1.0, -1.0, 2.0, 0.0])  # episode 0 mean 0; ep 1 mean 1
    out = ppo_policy_improvement_utility(
        old,
        new,
        adv,
        context_ids=[7, 9],
        episode_lengths=[2, 2],
        clip_eps=0.2,
        center=False,
        scale=False,
    )
    assert np.isclose(out[7], 0.0, atol=1e-12)
    assert np.isclose(out[9], 1.0, atol=1e-12)


def test_centering_subtracts_mean_across_observed_contexts() -> None:
    old = torch.tensor([0.0, 0.0, 0.0, 0.0])
    new = old.clone()
    adv = torch.tensor([1.0, -1.0, 2.0, 0.0])
    out = ppo_policy_improvement_utility(
        old,
        new,
        adv,
        context_ids=[7, 9],
        episode_lengths=[2, 2],
        clip_eps=0.2,
        center=True,
        scale=False,
    )
    # Pre-center: {7: 0.0, 9: 1.0}; mean = 0.5; centered: {7: -0.5, 9: 0.5}.
    assert np.isclose(out[7], -0.5, atol=1e-12)
    assert np.isclose(out[9], 0.5, atol=1e-12)


def test_scaling_divides_by_std() -> None:
    old = torch.tensor([0.0, 0.0, 0.0, 0.0])
    new = old.clone()
    adv = torch.tensor([1.0, -1.0, 2.0, 0.0])
    out = ppo_policy_improvement_utility(
        old,
        new,
        adv,
        context_ids=[7, 9],
        episode_lengths=[2, 2],
        clip_eps=0.2,
        center=True,
        scale=True,
    )
    # After centering: {7: -0.5, 9: 0.5}; std = 0.5; scaled: {-1, +1}.
    assert np.isclose(out[7], -1.0, atol=1e-6)
    assert np.isclose(out[9], 1.0, atol=1e-6)


def test_clipping_caps_ratio_extreme() -> None:
    # log ratio = +5 → ratio ≈ 148.4 → clipped to 1 + eps = 1.2 if A > 0.
    # Advantages = 1.0 across all steps. Expected per-step surrogate:
    # min(148 * 1, 1.2 * 1) = 1.2.
    old = torch.zeros(2)
    new = torch.tensor([5.0, 5.0])
    adv = torch.tensor([1.0, 1.0])
    out = ppo_policy_improvement_utility(
        old,
        new,
        adv,
        context_ids=[3],
        episode_lengths=[2],
        clip_eps=0.2,
        center=False,
        scale=False,
    )
    assert np.isclose(out[3], 1.2, atol=1e-6)


def test_clip_with_negative_advantage_floors_at_unclipped() -> None:
    # ratio = 0.5 (below 1 - eps = 0.8), A = -1: min(0.5 * -1,
    # 0.8 * -1) = min(-0.5, -0.8) = -0.8.
    old = torch.zeros(1)
    new = torch.tensor([np.log(0.5)])
    adv = torch.tensor([-1.0])
    out = ppo_policy_improvement_utility(
        old,
        new,
        adv,
        context_ids=[0],
        episode_lengths=[1],
        clip_eps=0.2,
        center=False,
        scale=False,
    )
    assert np.isclose(out[0], -0.8, atol=1e-6)


def test_per_context_mean_over_multiple_episodes() -> None:
    # 4 episodes, 1 step each, all under ratio 1 so surrogate = adv.
    # episode advantages: [1.0, 3.0, -1.0, 5.0]. context_ids:
    # [0, 0, 1, 1]. Per-context means: {0: 2.0, 1: 2.0}.
    old = torch.zeros(4)
    new = old.clone()
    adv = torch.tensor([1.0, 3.0, -1.0, 5.0])
    out = ppo_policy_improvement_utility(
        old,
        new,
        adv,
        context_ids=[0, 0, 1, 1],
        episode_lengths=[1, 1, 1, 1],
        clip_eps=0.2,
        center=False,
        scale=False,
    )
    assert np.isclose(out[0], 2.0, atol=1e-12)
    assert np.isclose(out[1], 2.0, atol=1e-12)


def test_outputs_are_detached_from_input_grads() -> None:
    # Even if inputs have requires_grad=True, the function should not
    # produce torch tensors with grad — it returns Python floats.
    old = torch.zeros(2, requires_grad=True)
    new = torch.zeros(2, requires_grad=True)
    adv = torch.tensor([1.0, -1.0], requires_grad=True)
    out = ppo_policy_improvement_utility(
        old,
        new,
        adv,
        context_ids=[0],
        episode_lengths=[2],
        clip_eps=0.2,
    )
    for v in out.values():
        assert isinstance(v, float)
        assert not np.isnan(v)
    # The inputs' .grad fields remain None — no backprop happened.
    assert old.grad is None
    assert new.grad is None
    assert adv.grad is None


# ---------------------------------------------------------------------------
# Shape / length validation
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        ppo_policy_improvement_utility(
            torch.zeros(4),
            torch.zeros(5),  # mismatched
            torch.zeros(4),
            context_ids=[0],
            episode_lengths=[4],
            clip_eps=0.2,
        )


def test_episode_lengths_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="sum"):
        ppo_policy_improvement_utility(
            torch.zeros(4),
            torch.zeros(4),
            torch.zeros(4),
            context_ids=[0, 1],
            episode_lengths=[2, 3],  # sums to 5, not 4
            clip_eps=0.2,
        )


def test_non_positive_episode_length_raises() -> None:
    with pytest.raises(ValueError, match="must all be > 0"):
        ppo_policy_improvement_utility(
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            context_ids=[0, 1],
            episode_lengths=[2, 0],  # zero length
            clip_eps=0.2,
        )
