"""Tests for :class:`pcrc.samplers.headroom_actor.HeadroomActorSampler`.

Covers the two signals it combines — the headroom backbone ``α·s(1-s)``
and the actor tilt ``β·û`` — plus the observed-mask gating, the
``β=0`` reduction to pure headroom, and input validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.samplers.base import SamplerStats
from pcrc.samplers.headroom import HeadroomSampler
from pcrc.samplers.headroom_actor import HeadroomActorSampler


def _stats(rates: np.ndarray, utils: np.ndarray, samples_per_arm: int = 8) -> SamplerStats:
    """All arms observed; success/utility sums = value · samples per arm."""
    n = rates.shape[0]
    return SamplerStats(
        observed_mask=np.ones(n, dtype=np.bool_),
        samples_this_step=np.full(n, samples_per_arm, dtype=np.int64),
        success_sum_this_step=(rates * samples_per_arm).astype(np.float64),
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=(utils * samples_per_arm).astype(np.float64),
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=np.full(n, samples_per_arm, dtype=np.int64),
    )


def test_tilt_breaks_ties_between_equal_difficulty_contexts() -> None:
    """Two frontier contexts (s=0.5) split by the actor utility."""
    n = 5
    sampler = HeadroomActorSampler(n, alpha=20.0, beta=3.0, ema_alpha=1.0, util_ema_alpha=1.0)
    rates = np.array([0.0, 0.0, 0.5, 0.5, 0.0])  # arms 2,3 on the frontier
    utils = np.array([0.0, 0.0, 2.0, -1.0, 0.0])  # tilt toward 2, away from 3
    sampler.update(_stats(rates, utils), step=0)
    dist = sampler.distribution(step=1).final
    assert dist[2] > dist[3]  # same difficulty, higher utility wins
    assert int(np.argmax(dist)) == 2


def test_beta_zero_reduces_to_headroom() -> None:
    """β=0 must match HeadroomSampler exactly (same α, no cap/floor)."""
    n = 6
    rates = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 0.5])
    utils = np.array([5.0, -3.0, 2.0, 1.0, -1.0, 4.0])  # arbitrary; must be ignored
    ha = HeadroomActorSampler(n, alpha=20.0, beta=0.0, ema_alpha=1.0, util_ema_alpha=1.0)
    hd = HeadroomSampler(n, alpha=20.0, ema_alpha=1.0)
    ha.update(_stats(rates, utils), step=0)
    hd.update(_stats(rates, utils), step=0)
    np.testing.assert_allclose(ha.distribution(1).final, hd.distribution(1).final, atol=1e-12)


def test_unobserved_keeps_emas() -> None:
    """Unobserved arms retain prior success (0.5) and utility (0.0)."""
    n = 4
    sampler = HeadroomActorSampler(n, alpha=20.0, beta=3.0, ema_alpha=1.0, util_ema_alpha=1.0)
    obs = np.zeros(n, dtype=np.bool_)
    obs[0] = True
    samples = np.zeros(n, dtype=np.int64)
    samples[0] = 4
    stats = SamplerStats(
        observed_mask=obs,
        samples_this_step=samples,
        success_sum_this_step=np.array([2.0, 0.0, 0.0, 0.0]),  # arm0 → s=0.5
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=np.array([8.0, 0.0, 0.0, 0.0]),  # arm0 → u=2.0
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=samples,
    )
    sampler.update(stats, step=0)
    dist = sampler.distribution(step=1).final
    # arm0: frontier (s=0.5) + positive tilt → strictly largest mass.
    assert int(np.argmax(dist)) == 0
    # arms 1-3 unchanged from the prior → equal to each other.
    np.testing.assert_allclose(dist[1:], dist[1] * np.ones(3), atol=1e-12)


@pytest.mark.parametrize("bad", [{"alpha": -1.0}, {"beta": -0.5}, {"util_ema_alpha": 0.0}])
def test_invalid_hyperparameters_raise(bad: dict) -> None:
    kwargs = {"alpha": 20.0, "beta": 1.0, **bad}
    with pytest.raises(ValueError):
        HeadroomActorSampler(4, **kwargs)
