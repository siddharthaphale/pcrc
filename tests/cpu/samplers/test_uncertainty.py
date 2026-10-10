"""Tests for :class:`pcrc.samplers.uncertainty.UncertaintySampler`."""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.samplers.base import SamplerStats
from pcrc.samplers.headroom_actor import HeadroomActorSampler
from pcrc.samplers.uncertainty import UncertaintySampler


def _stats(utils: np.ndarray, samples_per_arm: int = 8) -> SamplerStats:
    n = utils.shape[0]
    return SamplerStats(
        observed_mask=np.ones(n, dtype=np.bool_),
        samples_this_step=np.full(n, samples_per_arm, dtype=np.int64),
        success_sum_this_step=np.zeros(n),
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=(utils * samples_per_arm).astype(np.float64),
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=np.full(n, samples_per_arm, dtype=np.int64),
    )


def test_concentrates_on_high_uncertainty_arm() -> None:
    n = 5
    s = UncertaintySampler(n, alpha=10.0, ema_alpha=1.0)
    u = np.array([0.1, 0.1, 0.9, 0.1, 0.1])  # arm 2 highest uncertainty
    s.update(_stats(u), step=0)
    d = s.distribution(step=1).final
    assert int(np.argmax(d)) == 2
    assert d[2] > 0.5


def test_matches_headroom_actor_alpha0() -> None:
    # UncertaintySampler(alpha=β) ≡ HeadroomActorSampler(alpha=0, beta=β) — the R5 stand-in.
    n = 6
    u = np.array([0.2, -0.1, 0.5, 0.3, -0.4, 0.1])
    unc = UncertaintySampler(n, alpha=3.0, ema_alpha=1.0)
    ha = HeadroomActorSampler(n, alpha=0.0, beta=3.0, ema_alpha=1.0, util_ema_alpha=1.0)
    unc.update(_stats(u), step=0)
    ha.update(_stats(u), step=0)
    np.testing.assert_allclose(unc.distribution(1).final, ha.distribution(1).final, atol=1e-12)


def test_unobserved_keeps_ema() -> None:
    n = 4
    s = UncertaintySampler(n, alpha=5.0, ema_alpha=1.0)
    obs = np.zeros(n, dtype=np.bool_)
    obs[0] = True
    samples = np.zeros(n, dtype=np.int64)
    samples[0] = 4
    stats = SamplerStats(
        observed_mask=obs,
        samples_this_step=samples,
        success_sum_this_step=np.zeros(n),
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=np.array([4.0, 0.0, 0.0, 0.0]),  # arm0 u=1.0
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=samples,
    )
    s.update(stats, step=0)
    d = s.distribution(step=1).final
    assert int(np.argmax(d)) == 0
    np.testing.assert_allclose(d[1:], d[1] * np.ones(3), atol=1e-12)


@pytest.mark.parametrize("bad", [{"alpha": -1.0}, {"ema_alpha": 0.0}, {"ema_alpha": 1.5}])
def test_invalid_hyperparameters_raise(bad: dict) -> None:
    with pytest.raises(ValueError):
        UncertaintySampler(4, alpha=bad.get("alpha", 3.0), ema_alpha=bad.get("ema_alpha", 0.2))
