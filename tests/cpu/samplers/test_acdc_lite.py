"""Tests for :class:`pcrc.samplers.acdc_lite.ACDCLiteSampler`.

§7.1 row: quality kernel monotonic in distance, σ matches config; λ
schedule monotone in s̄; score_t formula; diversity depends on
recent rollouts (not static features alone).
"""

from __future__ import annotations

import numpy as np

from pcrc.samplers.acdc_lite import ACDCLiteSampler, grid_coverage_diversity
from pcrc.samplers.base import SamplerStats


def _stats_with_dist_and_success(
    n: int, observed: list[int], distances: list[float], successes: list[float]
) -> SamplerStats:
    obs = np.zeros(n, dtype=np.bool_)
    samples = np.zeros(n, dtype=np.int64)
    success_sum = np.zeros(n, dtype=np.float64)
    return_sum = np.zeros(n, dtype=np.float64)
    for cid, d, s in zip(observed, distances, successes, strict=True):
        obs[cid] = True
        samples[cid] = 4
        success_sum[cid] = s * 4
        # The sampler reads -distance from return_sum; encode the
        # negative-distance convention here.
        return_sum[cid] = -d * 4
    return SamplerStats(
        observed_mask=obs,
        samples_this_step=samples,
        success_sum_this_step=success_sum,
        return_sum_this_step=return_sum,
        utility_sum_this_step=np.zeros(n),
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=np.zeros(n, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Quality kernel
# ---------------------------------------------------------------------------


def test_quality_at_zero_distance_is_one() -> None:
    sampler = ACDCLiteSampler(4, sigma_quality=0.2, initial_distance_prior=0.0)
    # At distance 0, exp(-0/(2 σ²)) = 1.
    quality = sampler._quality_kernel()
    np.testing.assert_allclose(quality, np.ones(4))


def test_quality_monotonic_decay_with_distance() -> None:
    sampler = ACDCLiteSampler(3, sigma_quality=0.2, ema_alpha=1.0, initial_distance_prior=0.0)
    sampler.update(
        _stats_with_dist_and_success(3, [0, 1, 2], [0.0, 0.2, 0.5], [0.0, 0.0, 0.0]),
        step=0,
    )
    q = sampler._quality_kernel()
    # Monotone decreasing in distance.
    assert q[0] > q[1] > q[2]


# ---------------------------------------------------------------------------
# λ schedule
# ---------------------------------------------------------------------------


def test_lambda_at_low_s_bar_close_to_lambda_min() -> None:
    sampler = ACDCLiteSampler(
        4,
        lambda_min=0.1,
        lambda_max=0.9,
        lambda_center=0.5,
        lambda_scale=50.0,
    )
    # s_bar = 0 → far below center → λ near λ_min.
    sampler._s_bar = 0.0
    assert sampler._lambda_t() < 0.2


def test_lambda_at_high_s_bar_close_to_lambda_max() -> None:
    sampler = ACDCLiteSampler(
        4,
        lambda_min=0.1,
        lambda_max=0.9,
        lambda_center=0.5,
        lambda_scale=50.0,
    )
    sampler._s_bar = 1.0
    assert sampler._lambda_t() > 0.8


def test_lambda_monotone_in_s_bar() -> None:
    sampler = ACDCLiteSampler(4, lambda_min=0.1, lambda_max=0.9)
    lambdas = []
    for s_bar in np.linspace(0.0, 1.0, 11):
        sampler._s_bar = float(s_bar)
        lambdas.append(sampler._lambda_t())
    assert all(lambdas[i] <= lambdas[i + 1] + 1e-9 for i in range(len(lambdas) - 1))


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def test_diversity_uses_recent_rollouts_not_static_features() -> None:
    """Two samplers with identical bank state but different rollout
    achieved-goal histories must produce different diversity values.
    """
    a = ACDCLiteSampler(3, sigma_quality=0.2)
    b = ACDCLiteSampler(3, sigma_quality=0.2)
    a.record_achieved_goals(0, np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]))
    b.record_achieved_goals(0, np.array([[0.1, 0.1, 0.1], [1.9, 1.9, 0.9], [1.0, 0.5, 0.5]]))
    da = a._diversity_per_context()
    db = b._diversity_per_context()
    assert not np.allclose(da, db)


def test_grid_coverage_diversity_counts_unique_cells() -> None:
    # Two points in distinct cells → 2; both in same cell → 1.
    ags = np.array([[0.0, 0.0, 0.0], [1.99, 1.99, 0.99]])
    assert grid_coverage_diversity(ags, grid_resolution=2) == 2
    ags = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]])
    assert grid_coverage_diversity(ags, grid_resolution=2) == 1


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def test_distribution_sum_to_one() -> None:
    sampler = ACDCLiteSampler(5, sigma_quality=0.3)
    dist = sampler.distribution(step=0)
    assert np.isclose(dist.final.sum(), 1.0, atol=1e-6)
