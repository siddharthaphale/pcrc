"""Tests for the §4 Priority 2 hard gate on MLP AC controls.

§7.1 row: shuffled-feature permutation is seed-stable across updates;
not re-shuffled per-update; main path never sees ``context_id`` or any
unique ID as a feature.
"""

from __future__ import annotations

import numpy as np

from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext
from pcrc.samplers.mlp_actor_curator import MLPActorCuratorSampler

ENV_ID = "FetchReachDense-v4"


def _make_bank(n: int = 10) -> FetchTaskBank:
    rng = np.random.default_rng(0)
    contexts = [
        FetchTaskContext(
            context_id=i,
            env_id=ENV_ID,
            reset_seed=1000 + i,
            desired_goal=rng.uniform(low=[1.1, 0.5, 0.4], high=[1.5, 1.0, 0.7]),
            reset_payload=None,
            initial_object_pose=None,
            distance_bin=i % 5,
            direction_bin=i % 5,
            pose_bin=0,
            split="train",
        )
        for i in range(n)
    ]
    ee = {i: rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6]) for i in range(n)}
    return FetchTaskBank.from_contexts(contexts, ee)


def test_shuffled_permutation_is_seed_stable() -> None:
    """Same seed → same shuffled feature matrix; different seed → different."""
    bank = _make_bank(10)
    a = MLPActorCuratorSampler(
        bank,
        alpha=1.0,
        hidden_sizes=(4,),
        feature_perturbation="shuffled",
        feature_perturbation_seed=42,
    )
    b = MLPActorCuratorSampler(
        bank,
        alpha=1.0,
        hidden_sizes=(4,),
        feature_perturbation="shuffled",
        feature_perturbation_seed=42,
    )
    np.testing.assert_array_equal(a._features.numpy(), b._features.numpy())

    c = MLPActorCuratorSampler(
        bank,
        alpha=1.0,
        hidden_sizes=(4,),
        feature_perturbation="shuffled",
        feature_perturbation_seed=99,
    )
    assert not np.allclose(a._features.numpy(), c._features.numpy())


def test_shuffled_is_not_reshuffled_per_update() -> None:
    """The feature matrix is the same before / after an update step."""
    bank = _make_bank(10)
    sampler = MLPActorCuratorSampler(
        bank,
        alpha=1.0,
        hidden_sizes=(4,),
        feature_perturbation="shuffled",
        feature_perturbation_seed=0,
    )
    before = sampler._features.numpy().copy()
    # Stats with an arbitrary observation pattern.
    obs = np.zeros(10, dtype=np.bool_)
    obs[:3] = True
    samples = np.zeros(10, dtype=np.int64)
    samples[:3] = 4
    util_sum = np.zeros(10, dtype=np.float64)
    util_sum[:3] = [4.0, -4.0, 0.0]
    from pcrc.samplers.base import SamplerStats  # noqa: PLC0415

    stats = SamplerStats(
        observed_mask=obs,
        samples_this_step=samples,
        success_sum_this_step=np.zeros(10),
        return_sum_this_step=np.zeros(10),
        utility_sum_this_step=util_sum,
        utility_sq_sum_this_step=np.zeros(10),
        success_ema=np.zeros(10),
        return_ema=np.zeros(10),
        utility_ema=np.zeros(10),
        n_visits_total=np.zeros(10, dtype=np.int64),
    )
    sampler.update(stats, step=0)
    after = sampler._features.numpy()
    np.testing.assert_array_equal(before, after)


def test_main_path_features_never_contain_context_id() -> None:
    """The MLP curator's feature matrix never contains a literal
    ``context_id`` value as a feature entry.

    The test directly inspects the curator-internal feature matrix
    against ``bank.features(cid)`` — we assert the curator does NOT
    pass anything beyond what the bank's documented feature schema
    provides. The :class:`FetchTaskBank` feature schema itself is
    guarded by ``test_fetch_task_bank_features.py`` and asserts
    against unique-per-context hashes; the curator-side test just
    pins that the curator faithfully echoes the bank's row.
    """
    bank = _make_bank(8)
    sampler = MLPActorCuratorSampler(bank, alpha=1.0, hidden_sizes=(4,))
    feats = sampler._features.numpy()
    for cid in range(8):
        bank_row = bank.features(cid).astype(np.float32)
        np.testing.assert_allclose(feats[cid], bank_row, atol=1e-6)


def test_random_perturbation_replaces_features() -> None:
    """``feature_perturbation='random'`` makes the feature matrix
    different from the bank's.
    """
    bank = _make_bank(8)
    sampler_clean = MLPActorCuratorSampler(
        bank, alpha=1.0, hidden_sizes=(4,), feature_perturbation="none"
    )
    sampler_rand = MLPActorCuratorSampler(
        bank,
        alpha=1.0,
        hidden_sizes=(4,),
        feature_perturbation="random",
        feature_perturbation_seed=0,
    )
    assert not np.allclose(sampler_clean._features.numpy(), sampler_rand._features.numpy())
