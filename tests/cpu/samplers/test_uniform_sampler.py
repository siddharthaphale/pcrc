"""Tests for :class:`pcrc.samplers.uniform.UniformSampler`.

§7.1 row: distribution is exactly ``1/|X|`` on each step;
``sample_flat`` / ``sample_grouped`` produce empirical uniform within
Monte-Carlo tolerance.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from pcrc.samplers.base import SamplerStats
from pcrc.samplers.uniform import UniformSampler


def test_distribution_is_exactly_uniform() -> None:
    n = 17
    sampler = UniformSampler(n)
    dist = sampler.distribution(step=0)
    expected = np.full(n, 1.0 / n)
    np.testing.assert_allclose(dist.final, expected, atol=1e-12)
    np.testing.assert_allclose(dist.raw, expected, atol=1e-12)
    assert np.isclose(dist.entropy, float(np.log(n)), atol=1e-12)
    assert np.isclose(dist.max_probability, 1.0 / n, atol=1e-12)


def test_update_is_noop() -> None:
    sampler = UniformSampler(5)
    stats = SamplerStats.zeros(5)
    sampler.update(stats, step=0)
    dist = sampler.distribution(step=1)
    np.testing.assert_allclose(dist.final, np.full(5, 1.0 / 5))


def test_distribution_independent_of_step() -> None:
    sampler = UniformSampler(8)
    a = sampler.distribution(step=0)
    b = sampler.distribution(step=10000)
    np.testing.assert_array_equal(a.final, b.final)


def test_sample_flat_empirical_uniform() -> None:
    n = 5
    sampler = UniformSampler(n)
    rng = np.random.default_rng(0)
    batch = sampler.sample_flat(n=100_000, step=0, rng=rng)
    counts = Counter(batch.sampled_context_ids)
    empirical = np.array([counts.get(i, 0) for i in range(n)], dtype=np.float64) / 100_000
    # 3σ tolerance under multinomial: σ = sqrt(p(1-p)/N) ≈ sqrt(0.8 / 100k) ≈ 2.8e-3
    np.testing.assert_allclose(empirical, np.full(n, 1.0 / n), atol=0.01)


def test_sample_grouped_distinct_under_uniform() -> None:
    n = 20
    sampler = UniformSampler(n)
    rng = np.random.default_rng(0)
    batch = sampler.sample_grouped(num_groups=5, rollouts_per_group=4, step=0, rng=rng)
    assert len(batch.unique_context_ids) == 5
    assert len(set(batch.sampled_context_ids)) == 5
    assert len(batch.sampled_context_ids) == 20
