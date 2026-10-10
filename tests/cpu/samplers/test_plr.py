"""Tests for :class:`pcrc.samplers.plr.PLRSampler`."""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.samplers.base import SamplerStats
from pcrc.samplers.plr import PLRSampler


def _stats(
    utils: np.ndarray, observed: np.ndarray | None = None, samples_per_arm: int = 8
) -> SamplerStats:
    n = utils.shape[0]
    obs = np.ones(n, dtype=np.bool_) if observed is None else observed
    samples = np.where(obs, samples_per_arm, 0).astype(np.int64)
    return SamplerStats(
        observed_mask=obs,
        samples_this_step=samples,
        success_sum_this_step=np.zeros(n),
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=(utils * samples).astype(np.float64),
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=samples,
    )


def test_concentrates_on_high_score_arm() -> None:
    # rank-1 (highest |TD-error|) should get the most mass at low rho.
    n = 5
    s = PLRSampler(n, beta=0.3, rho=0.0, score_ema_alpha=1.0)
    u = np.array([0.1, 0.1, 0.9, 0.1, 0.1])  # arm 2 highest score
    s.update(_stats(u), step=1)
    d = s.distribution(step=2).final
    assert int(np.argmax(d)) == 2


def test_rank_based_not_magnitude_based() -> None:
    # PLR prioritizes by RANK, not raw magnitude: a huge outlier and a slight edge produce
    # the same rank-1 share. Distinguishes PLR from exp(alpha*u) softmax.
    n = 4
    s1 = PLRSampler(n, beta=0.5, rho=0.0, score_ema_alpha=1.0)
    s2 = PLRSampler(n, beta=0.5, rho=0.0, score_ema_alpha=1.0)
    s1.update(_stats(np.array([0.40, 0.30, 0.20, 0.10])), step=1)
    s2.update(_stats(np.array([99.0, 0.30, 0.20, 0.10])), step=1)  # same ranks, wild magnitude
    np.testing.assert_allclose(s1.distribution(2).final, s2.distribution(2).final, atol=1e-12)


def test_staleness_revives_unsampled_contexts() -> None:
    # With rho>0, a context never sampled (max staleness) gets non-trivial mass even with score 0.
    n = 4
    s = PLRSampler(n, beta=0.5, rho=0.5, score_ema_alpha=1.0)
    obs = np.array([True, True, True, False])  # arm 3 never seen
    s.update(_stats(np.array([0.9, 0.5, 0.3, 0.0]), observed=obs), step=5)
    d = s.distribution(step=6).final
    # unseen arm 3 should not be starved (staleness term lifts it above a pure-rank 0-score arm)
    assert d[3] > 0.0
    assert d[3] > d.min() - 1e-9


def test_unobserved_keeps_score() -> None:
    n = 4
    s = PLRSampler(n, beta=0.5, rho=0.0, score_ema_alpha=1.0)
    obs = np.array([True, False, False, False])
    s.update(_stats(np.array([0.8, 0.0, 0.0, 0.0]), observed=obs), step=1)
    # arm 0 seen with top score → rank 1 → highest mass
    d = s.distribution(step=2).final
    assert int(np.argmax(d)) == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"beta": 0.0},
        {"beta": -1.0},
        {"rho": -0.1},
        {"rho": 1.1},
        {"score_ema_alpha": 0.0},
        {"score_ema_alpha": 1.5},
    ],
)
def test_invalid_hyperparameters_raise(bad: dict) -> None:
    with pytest.raises(ValueError):
        PLRSampler(
            4,
            beta=bad.get("beta", 0.5),
            rho=bad.get("rho", 0.1),
            score_ema_alpha=bad.get("score_ema_alpha", 0.3),
        )
