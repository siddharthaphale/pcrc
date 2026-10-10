"""Tests for ``Sampler.sample_flat`` / ``Sampler.sample_grouped`` and the
``SampledBatch`` shape contract.

§7.1 row: flat sampling preserves the full sequence with duplicates;
grouped sampling produces ``num_groups`` distinct contexts each
repeated ``rollouts_per_group`` times; ``rollout_counts ==
Counter(sampled_context_ids)``; ``unique_context_ids ==
sorted(set(sampled_context_ids))``.

Uses :class:`UniformSampler` as the test driver since it has no state
and produces an exactly-uniform distribution.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from pcrc.samplers.base import SampledBatch, SamplerDistribution
from pcrc.samplers.uniform import UniformSampler


def _flat_uniform_dist(n: int) -> SamplerDistribution:
    p = np.full(n, 1.0 / n, dtype=np.float64)
    return SamplerDistribution(
        raw=p,
        floored=p,
        capped=p,
        final=p,
        entropy=float(np.log(n)),
        max_probability=1.0 / n,
    )


# ---------------------------------------------------------------------------
# SampledBatch coherence (frozen dataclass __post_init__ checks)
# ---------------------------------------------------------------------------


def test_sampled_batch_rejects_mismatched_unique() -> None:
    dist = _flat_uniform_dist(4)
    with pytest.raises(ValueError, match="unique_context_ids"):
        SampledBatch(
            sampled_context_ids=[0, 1, 2, 2],
            unique_context_ids=[0, 1, 2, 3],  # 3 not in sampled
            rollout_counts={0: 1, 1: 1, 2: 2},
            distribution=dist,
        )


def test_sampled_batch_rejects_mismatched_counts() -> None:
    dist = _flat_uniform_dist(4)
    with pytest.raises(ValueError, match="rollout_counts"):
        SampledBatch(
            sampled_context_ids=[0, 1, 2, 2],
            unique_context_ids=[0, 1, 2],
            rollout_counts={0: 1, 1: 1, 2: 1},  # under-counts the duplicate
            distribution=dist,
        )


# ---------------------------------------------------------------------------
# sample_flat
# ---------------------------------------------------------------------------


def test_flat_sampling_returns_n_ids_with_replacement() -> None:
    n = 5
    sampler = UniformSampler(n)
    rng = np.random.default_rng(0)
    batch = sampler.sample_flat(n=20, step=0, rng=rng)
    assert len(batch.sampled_context_ids) == 20
    # With n=5 and 20 draws under uniform, duplicates are nearly certain.
    assert len(batch.unique_context_ids) <= n
    assert batch.unique_context_ids == sorted(set(batch.sampled_context_ids))
    assert batch.rollout_counts == dict(Counter(batch.sampled_context_ids))


def test_flat_sampling_distribution_attached() -> None:
    sampler = UniformSampler(8)
    rng = np.random.default_rng(0)
    batch = sampler.sample_flat(n=4, step=0, rng=rng)
    assert isinstance(batch.distribution, SamplerDistribution)
    assert np.allclose(batch.distribution.final, np.full(8, 1.0 / 8))


def test_flat_sampling_rejects_non_positive_n() -> None:
    sampler = UniformSampler(4)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n must be > 0"):
        sampler.sample_flat(n=0, step=0, rng=rng)


# ---------------------------------------------------------------------------
# sample_grouped
# ---------------------------------------------------------------------------


def test_grouped_sampling_produces_distinct_contexts() -> None:
    sampler = UniformSampler(10)
    rng = np.random.default_rng(0)
    batch = sampler.sample_grouped(num_groups=4, rollouts_per_group=3, step=0, rng=rng)
    assert len(batch.unique_context_ids) == 4
    assert len(set(batch.unique_context_ids)) == 4
    assert len(batch.sampled_context_ids) == 4 * 3
    for cid, count in batch.rollout_counts.items():
        assert count == 3, cid


def test_grouped_sampling_rejects_more_groups_than_contexts() -> None:
    sampler = UniformSampler(4)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exceeds n_contexts"):
        sampler.sample_grouped(num_groups=5, rollouts_per_group=2, step=0, rng=rng)


def test_grouped_sampling_seed_determinism() -> None:
    sampler = UniformSampler(10)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = sampler.sample_grouped(num_groups=3, rollouts_per_group=2, step=0, rng=rng_a)
    b = sampler.sample_grouped(num_groups=3, rollouts_per_group=2, step=0, rng=rng_b)
    assert a.sampled_context_ids == b.sampled_context_ids


def test_grouped_sampling_rejects_when_distribution_too_sparse() -> None:
    """If `distribution.final` has fewer non-zero entries than num_groups,
    we cannot draw that many distinct contexts without replacement.

    Simulated by stuffing a zero-weight distribution into a custom
    sampler — the uniform sampler always has full support, so we test
    the path with a thin subclass.
    """
    from pcrc.samplers.base import Sampler  # noqa: PLC0415

    class _ThinSampler(Sampler):
        name = "thin"

        def update(self, stats, step):
            del stats, step

        def distribution(self, step):
            del step
            n = self.n_contexts
            p = np.zeros(n, dtype=np.float64)
            p[0] = 0.6
            p[1] = 0.4
            return SamplerDistribution(
                raw=p,
                floored=p,
                capped=p,
                final=p,
                entropy=0.0,
                max_probability=0.6,
            )

    rng = np.random.default_rng(0)
    sampler = _ThinSampler(5)
    with pytest.raises(ValueError, match="non-zero entries"):
        sampler.sample_grouped(num_groups=3, rollouts_per_group=1, step=0, rng=rng)
