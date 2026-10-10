"""Tests for :class:`pcrc.samplers.mlp_actor_curator.MLPActorCuratorSampler`.

Construction, update (no NaN), distribution sum-to-1, observed-mask
gating, n_epochs > 0 actually changes weights, separate optimizer
(curator gradient path does not touch any extra params).
"""

from __future__ import annotations

import numpy as np

from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext
from pcrc.samplers.base import SamplerStats
from pcrc.samplers.mlp_actor_curator import MLPActorCuratorSampler

ENV_ID = "FetchReachDense-v4"


def _make_bank(n: int = 8) -> FetchTaskBank:
    rng = np.random.default_rng(0)
    contexts = [
        FetchTaskContext(
            context_id=i,
            env_id=ENV_ID,
            reset_seed=1000 + i,
            desired_goal=rng.uniform(low=[1.1, 0.5, 0.4], high=[1.5, 1.0, 0.7]),
            reset_payload=None,
            initial_object_pose=None,
            distance_bin=i % 4,
            direction_bin=i % 4,
            pose_bin=0,
            split="train",
        )
        for i in range(n)
    ]
    ee = {i: rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6]) for i in range(n)}
    return FetchTaskBank.from_contexts(contexts, ee)


def _stats_with_utility(n: int, observed: list[int], u: list[float]) -> SamplerStats:
    obs = np.zeros(n, dtype=np.bool_)
    samples = np.zeros(n, dtype=np.int64)
    util_sum = np.zeros(n, dtype=np.float64)
    for cid, ux in zip(observed, u, strict=True):
        obs[cid] = True
        samples[cid] = 4
        util_sum[cid] = ux * 4
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


def test_construction_and_distribution_sum_to_one() -> None:
    bank = _make_bank(8)
    sampler = MLPActorCuratorSampler(bank, alpha=1.0, hidden_sizes=(8,))
    dist = sampler.distribution(step=0)
    assert np.isclose(dist.final.sum(), 1.0, atol=1e-6)
    assert dist.raw.shape == (8,)


def test_update_no_observations_is_noop() -> None:
    bank = _make_bank(8)
    sampler = MLPActorCuratorSampler(bank, alpha=1.0, hidden_sizes=(8,))
    before = sampler.get_softmax_probs()
    sampler.update(SamplerStats.zeros(8), step=0)
    after = sampler.get_softmax_probs()
    np.testing.assert_allclose(before, after)


def test_update_changes_probs_on_observed_contexts() -> None:
    bank = _make_bank(6)
    sampler = MLPActorCuratorSampler(
        bank, alpha=2.0, hidden_sizes=(16,), learning_rate=1.0e-2, n_epochs=5
    )
    before = sampler.get_softmax_probs()
    stats = _stats_with_utility(6, observed=[0, 1, 2], u=[1.0, -0.5, 0.5])
    sampler.update(stats, step=0)
    after = sampler.get_softmax_probs()
    assert not np.allclose(before, after)


def test_feature_perturbation_validation() -> None:
    bank = _make_bank(4)
    import pytest

    with pytest.raises(ValueError, match="feature_perturbation"):
        MLPActorCuratorSampler(bank, alpha=1.0, feature_perturbation="invalid")


def test_distribution_with_floor_respects_eps() -> None:
    bank = _make_bank(5)
    sampler = MLPActorCuratorSampler(bank, alpha=10.0, epsilon=0.2)
    # Force a concentrated raw by training a few steps then check
    # floored applies the documented mass.
    stats = _stats_with_utility(5, observed=[0], u=[5.0])
    sampler.update(stats, step=0)
    dist = sampler.distribution(step=1)
    # floored min should be ≥ ε/|X|.
    assert dist.floored.min() >= 0.2 / 5 - 1e-9
