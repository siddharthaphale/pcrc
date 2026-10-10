"""Tests for :class:`pcrc.samplers.tabular_actor_curator.TabularActorCuratorSampler`.

§7.1 row: wraps :class:`OSMDUpdater` correctly. Full-vector update:
the utility vector passed to ``osmd.step`` always has shape ``(|X|,)``;
unobserved entries are exactly ``0.0``. Hand-computed match: on a
4-arm fixture with a known observed / unobserved split, the resulting
weights match the explicit log-space update
``log w' = log w + η · u`` followed by LSE renormalization, where
``u[unobserved] = 0``. Unobserved-no-shift: running the same update
twice with different observed-arm utilities while keeping unobserved
arm utility at 0 leaves the unobserved arm's *relative* log-weight
unchanged across the two updates.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.samplers.base import SamplerStats
from pcrc.samplers.tabular_actor_curator import TabularActorCuratorSampler


def _stats_with_utility(
    n: int,
    observed: list[int],
    utility_per_rollout: list[float],
    samples_per_arm: int = 4,
) -> SamplerStats:
    """Construct a SamplerStats with the given utility on observed arms."""
    obs = np.zeros(n, dtype=np.bool_)
    samples = np.zeros(n, dtype=np.int64)
    util_sum = np.zeros(n, dtype=np.float64)
    for cid, u in zip(observed, utility_per_rollout, strict=True):
        obs[cid] = True
        samples[cid] = samples_per_arm
        util_sum[cid] = u * samples_per_arm
    return SamplerStats(
        observed_mask=obs,
        samples_this_step=samples,
        success_sum_this_step=np.zeros(n),
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=util_sum,
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=np.zeros(n, dtype=np.int64),
    )


def test_initial_weights_are_uniform() -> None:
    sampler = TabularActorCuratorSampler(8, eta=1.0)
    np.testing.assert_allclose(sampler.get_osmd_weights(), np.full(8, 1.0 / 8), atol=1e-12)


def test_update_with_no_observations_is_noop() -> None:
    sampler = TabularActorCuratorSampler(5, eta=2.0)
    before = sampler.get_osmd_weights().copy()
    sampler.update(SamplerStats.zeros(5), step=0)
    np.testing.assert_allclose(sampler.get_osmd_weights(), before)


def test_full_vector_hand_computed_match() -> None:
    """4-arm fixture with arms 0,2 observed, arms 1,3 unobserved.

    Expected post-step weights, computed by hand:
        eta = 1.0
        u_full = [0.3, 0.0, -0.2, 0.0]  (mean per rollout for observed,
                                          0 for unobserved)
        log w' = log(1/4) + 1.0 · u_full
               = [-1.386 + 0.3, -1.386 + 0.0, -1.386 - 0.2, -1.386 + 0.0]
        post-LSE: softmax([0.3, 0.0, -0.2, 0.0])

    The exact closed-form check pins the algebraic identity that the
    sampler runs exactly OSMD under this construction.
    """
    n = 4
    sampler = TabularActorCuratorSampler(n, eta=1.0)
    stats = _stats_with_utility(n, observed=[0, 2], utility_per_rollout=[0.3, -0.2])
    sampler.update(stats, step=0)

    raw_logits = np.array([0.3, 0.0, -0.2, 0.0], dtype=np.float64)
    raw_logits -= raw_logits.max()
    expected = np.exp(raw_logits)
    expected = expected / expected.sum()
    np.testing.assert_allclose(sampler.get_osmd_weights(), expected, atol=1e-12)


def test_unobserved_no_shift_relative_to_self() -> None:
    """Run the same update twice with different observed-arm utilities;
    the unobserved arm's log-weight relative to its own initial value
    is unchanged across runs (modulo the global LSE normalizer).

    Concretely: with two updates each adding η · u to observed arms
    and 0 to unobserved arms, the unobserved arm's relative
    log-probability ratio to the uniform-shifted reference is the
    same in both runs.
    """
    n = 5
    sampler_a = TabularActorCuratorSampler(n, eta=0.5)
    sampler_b = TabularActorCuratorSampler(n, eta=0.5)

    sampler_a.update(
        _stats_with_utility(n, observed=[0, 1, 2], utility_per_rollout=[0.1, 0.2, 0.3]),
        step=0,
    )
    sampler_b.update(
        _stats_with_utility(n, observed=[0, 1, 2], utility_per_rollout=[1.5, -0.7, 0.9]),
        step=0,
    )
    # Unobserved arms 3 and 4: their RATIO should be identical in both
    # samplers — they receive the same zero shift, so w[3]/w[4] = 1
    # before and after (it's exactly 1 here because uniform start).
    wa = sampler_a.get_osmd_weights()
    wb = sampler_b.get_osmd_weights()
    assert np.isclose(wa[3] / wa[4], 1.0, atol=1e-12)
    assert np.isclose(wb[3] / wb[4], 1.0, atol=1e-12)
    # And the unobserved-vs-unobserved ratio is preserved under any
    # observed-side utility vector.
    assert np.isclose(wa[3] / wa[4], wb[3] / wb[4], atol=1e-12)


def test_eta_zero_leaves_weights_unchanged() -> None:
    n = 4
    sampler = TabularActorCuratorSampler(n, eta=0.0)
    sampler.update(
        _stats_with_utility(n, observed=[0, 1, 2, 3], utility_per_rollout=[1, 2, 3, 4]),
        step=0,
    )
    np.testing.assert_allclose(sampler.get_osmd_weights(), np.full(n, 1.0 / n), atol=1e-12)


def test_distribution_applies_floor_and_cap() -> None:
    """After a strong update concentrates OSMD weights, the
    distribution bundle's floored / capped layers should reflect the
    exploration floor.
    """
    n = 5
    sampler = TabularActorCuratorSampler(n, eta=10.0, epsilon=0.2, q_max=0.5)
    sampler.update(
        _stats_with_utility(n, observed=[0], utility_per_rollout=[5.0]),
        step=0,
    )
    dist = sampler.distribution(step=1)
    # Floor applied: every arm has at least 0.2 / 5 = 0.04 floor mass.
    assert dist.floored.min() >= 0.2 / n - 1e-9
    # Cap applied: no arm exceeds 0.5.
    assert dist.capped.max() <= 0.5 + 1e-9
    # Raw is the OSMD weights (no floor); should concentrate on arm 0.
    assert int(np.argmax(dist.raw)) == 0


def test_shape_mismatch_in_stats_raises() -> None:
    sampler = TabularActorCuratorSampler(5, eta=1.0)
    bad = SamplerStats.zeros(3)
    with pytest.raises(ValueError, match="shape mismatch"):
        sampler.update(bad, step=0)


def test_negative_eta_rejected() -> None:
    with pytest.raises(ValueError, match="eta must be >= 0"):
        TabularActorCuratorSampler(4, eta=-1.0)
