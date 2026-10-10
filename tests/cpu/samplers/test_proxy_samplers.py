"""Tests for the three proxy samplers:
:class:`RewardGreedySampler`, :class:`HeadroomSampler`,
:class:`LearningProgressSampler`.

§7.1 row: reward-greedy concentrates on high-success contexts;
headroom peaks at ``ŝ ≈ 0.5``; learning-progress responds to
``Δŝ``; all three obey the observed-mask gating.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.samplers.base import SamplerStats
from pcrc.samplers.headroom import HeadroomSampler
from pcrc.samplers.learning_progress import LearningProgressSampler
from pcrc.samplers.reward_greedy import RewardGreedySampler


def _stats_with_success_rates(rates: np.ndarray, samples_per_arm: int = 8) -> SamplerStats:
    """All arms observed; success_sum_this_step = rate · samples per arm."""
    n = rates.shape[0]
    return SamplerStats(
        observed_mask=np.ones(n, dtype=np.bool_),
        samples_this_step=np.full(n, samples_per_arm, dtype=np.int64),
        success_sum_this_step=(rates * samples_per_arm).astype(np.float64),
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=np.zeros(n),
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=np.full(n, samples_per_arm, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Reward-greedy
# ---------------------------------------------------------------------------


def test_reward_greedy_concentrates_on_high_success_arm() -> None:
    n = 5
    sampler = RewardGreedySampler(n, alpha=10.0, ema_alpha=1.0)  # ema=1 → no smoothing
    rates = np.array([0.1, 0.2, 0.9, 0.2, 0.1])
    sampler.update(_stats_with_success_rates(rates), step=0)
    dist = sampler.distribution(step=1)
    # Arm 2 (highest rate) should dominate after the softmax.
    assert int(np.argmax(dist.final)) == 2
    assert dist.final[2] > 0.5


def test_reward_greedy_unobserved_keeps_ema() -> None:
    n = 4
    sampler = RewardGreedySampler(n, alpha=5.0, ema_alpha=1.0, initial_success_prior=0.3)
    obs = np.zeros(n, dtype=np.bool_)
    obs[0] = True
    samples = np.zeros(n, dtype=np.int64)
    samples[0] = 4
    succ = np.zeros(n)
    succ[0] = 4.0  # arm 0 saturated this round
    stats = SamplerStats(
        observed_mask=obs,
        samples_this_step=samples,
        success_sum_this_step=succ,
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=np.zeros(n),
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=np.zeros(n, dtype=np.int64),
    )
    sampler.update(stats, step=0)
    # Unobserved arms (1, 2, 3) keep prior 0.3; arm 0 jumps to 1.0.
    assert np.isclose(sampler._success_ema[1], 0.3)
    assert np.isclose(sampler._success_ema[0], 1.0)


def test_reward_greedy_alpha_zero_returns_uniform() -> None:
    n = 6
    sampler = RewardGreedySampler(n, alpha=0.0, ema_alpha=1.0)
    sampler.update(_stats_with_success_rates(np.linspace(0.0, 1.0, n)), step=0)
    dist = sampler.distribution(step=1)
    np.testing.assert_allclose(dist.final, np.full(n, 1.0 / n), atol=1e-12)


# ---------------------------------------------------------------------------
# Headroom
# ---------------------------------------------------------------------------


def test_headroom_peaks_at_half() -> None:
    n = 5
    sampler = HeadroomSampler(n, alpha=20.0, ema_alpha=1.0)
    # Rates clustered around 0, 0.25, 0.5, 0.75, 1.0
    rates = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    sampler.update(_stats_with_success_rates(rates), step=0)
    dist = sampler.distribution(step=1)
    # arm 2 (rate 0.5 → headroom 0.25) is the maximum-headroom arm.
    assert int(np.argmax(dist.final)) == 2


def test_headroom_unobserved_keeps_ema() -> None:
    n = 3
    sampler = HeadroomSampler(n, alpha=5.0, ema_alpha=1.0, initial_success_prior=0.5)
    obs = np.zeros(n, dtype=np.bool_)
    obs[2] = True
    samples = np.zeros(n, dtype=np.int64)
    samples[2] = 4
    succ = np.zeros(n)
    succ[2] = 4.0
    stats = SamplerStats(
        observed_mask=obs,
        samples_this_step=samples,
        success_sum_this_step=succ,
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=np.zeros(n),
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=np.zeros(n, dtype=np.int64),
    )
    sampler.update(stats, step=0)
    # Arms 0 and 1 keep prior 0.5; arm 2 jumps to 1.0 (saturated; headroom → 0).
    assert np.isclose(sampler._success_ema[0], 0.5)
    assert np.isclose(sampler._success_ema[2], 1.0)
    dist = sampler.distribution(step=1)
    # Saturated arm 2 is the LEAST preferred (headroom = 0).
    assert int(np.argmin(dist.final)) == 2


# ---------------------------------------------------------------------------
# Learning progress
# ---------------------------------------------------------------------------


def test_learning_progress_initial_distribution_is_uniform() -> None:
    n = 6
    sampler = LearningProgressSampler(n, alpha=10.0)
    dist = sampler.distribution(step=0)
    # Fast and slow EMAs initialized equal → Δ = 0 → softmax uniform.
    np.testing.assert_allclose(dist.final, np.full(n, 1.0 / n), atol=1e-12)


def test_learning_progress_responds_to_improving_arm() -> None:
    n = 4
    sampler = LearningProgressSampler(
        n,
        alpha=20.0,
        fast_ema_alpha=1.0,
        slow_ema_alpha=0.05,
        initial_success_prior=0.0,
    )
    # Round 1: arm 0 succeeds 100%; others 0%.
    rates_round_1 = np.array([1.0, 0.0, 0.0, 0.0])
    sampler.update(_stats_with_success_rates(rates_round_1), step=0)
    dist = sampler.distribution(step=1)
    # Fast EMA: arm 0 → 1.0, others → 0.0.
    # Slow EMA: arm 0 → 0.05, others → 0.0.
    # Δ: arm 0 → 0.95, others → 0.0.
    # Softmax with α=20 → arm 0 dominates.
    assert int(np.argmax(dist.final)) == 0
    assert dist.final[0] > 0.9


def test_learning_progress_rejects_inverted_ema_alphas() -> None:
    with pytest.raises(ValueError, match="must be <"):
        LearningProgressSampler(4, alpha=1.0, fast_ema_alpha=0.05, slow_ema_alpha=0.3)


def test_learning_progress_unobserved_keeps_emas() -> None:
    n = 4
    sampler = LearningProgressSampler(
        n,
        alpha=5.0,
        fast_ema_alpha=1.0,
        slow_ema_alpha=0.5,
        initial_success_prior=0.3,
    )
    obs = np.zeros(n, dtype=np.bool_)
    obs[1] = True
    samples = np.zeros(n, dtype=np.int64)
    samples[1] = 4
    succ = np.zeros(n)
    succ[1] = 4.0
    stats = SamplerStats(
        observed_mask=obs,
        samples_this_step=samples,
        success_sum_this_step=succ,
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=np.zeros(n),
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=np.zeros(n, dtype=np.int64),
    )
    sampler.update(stats, step=0)
    # Arm 0 (unobserved): fast=slow=0.3 → Δ=0
    assert np.isclose(sampler._fast_ema[0], 0.3)
    assert np.isclose(sampler._slow_ema[0], 0.3)
    # Arm 1 (observed all-success): fast → 1.0 (since fast_ema_alpha=1.0),
    # slow → 0.5*0.3 + 0.5*1.0 = 0.65; Δ = 0.35.
    assert np.isclose(sampler._fast_ema[1], 1.0)
    assert np.isclose(sampler._slow_ema[1], 0.65)


# ---------------------------------------------------------------------------
# Shape-mismatch guards (shared across all three)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sampler_factory",
    [
        lambda n: RewardGreedySampler(n, alpha=1.0),
        lambda n: HeadroomSampler(n, alpha=1.0),
        lambda n: LearningProgressSampler(n, alpha=1.0),
    ],
)
def test_shape_mismatch_in_stats_raises(sampler_factory) -> None:
    sampler = sampler_factory(5)
    bad = SamplerStats.zeros(3)  # wrong size
    with pytest.raises(ValueError, match="shape mismatch"):
        sampler.update(bad, step=0)
