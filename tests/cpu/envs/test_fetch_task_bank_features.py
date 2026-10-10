"""Tests for ``FetchTaskBank.features`` — MLP curator featurization.

§7.1 row "MLP Actor-Curator featurization": the feature vector returned
must contain the documented schema, never contain ``context_id`` /
``reset_seed`` / unique-per-context hashes, normalize continuous
features with train-bank statistics, and produce identical output for
identical ``context_id``.
"""

from __future__ import annotations

import numpy as np

from pcrc.envs.fetch_task_bank import (
    CONTINUOUS_FEATURE_DIM,
    FetchTaskBank,
    FetchTaskContext,
)

ENV_ID_REACH = "FetchReachDense-v4"
ENV_ID_PUSH = "FetchPushDense-v4"


def _make_reach_context(
    cid: int, dist_bin: int = 0, dir_bin: int = 0, split: str = "train"
) -> FetchTaskContext:
    rng = np.random.default_rng(1000 + cid)
    return FetchTaskContext(
        context_id=cid,
        env_id=ENV_ID_REACH,
        reset_seed=1000 + cid,
        desired_goal=rng.uniform(low=[1.1, 0.5, 0.4], high=[1.5, 1.0, 0.7]),
        reset_payload=None,
        initial_object_pose=None,
        distance_bin=dist_bin,
        direction_bin=dir_bin,
        pose_bin=0,
        split=split,
    )


def _make_push_context(cid: int, pose_bin: int = 0) -> FetchTaskContext:
    rng = np.random.default_rng(2000 + cid)
    obj_pose = rng.uniform(low=[1.1, 0.5, 0.42], high=[1.5, 1.0, 0.42])
    return FetchTaskContext(
        context_id=cid,
        env_id=ENV_ID_PUSH,
        reset_seed=2000 + cid,
        desired_goal=rng.uniform(low=[1.1, 0.5, 0.42], high=[1.5, 1.0, 0.42]),
        reset_payload={"object_pose": obj_pose},
        initial_object_pose=obj_pose,
        distance_bin=0,
        direction_bin=0,
        pose_bin=pose_bin,
        split="train",
    )


def _make_reach_bank(n: int = 16) -> FetchTaskBank:
    rng = np.random.default_rng(0)
    contexts = [_make_reach_context(i, dist_bin=i % 5, dir_bin=(i + 1) % 5) for i in range(n)]
    ee = {i: rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6]) for i in range(n)}
    return FetchTaskBank.from_contexts(contexts, ee)


def _make_push_bank(n: int = 12) -> FetchTaskBank:
    rng = np.random.default_rng(1)
    contexts = [_make_push_context(i, pose_bin=i % 4) for i in range(n)]
    ee = {i: rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6]) for i in range(n)}
    return FetchTaskBank.from_contexts(contexts, ee)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_continuous_block_width_matches_documented_dim() -> None:
    # 3 + 3 + 3 + 1 + 3 + 3 + 3 = 19.
    assert CONTINUOUS_FEATURE_DIM == 19


def test_reach_feature_dim_is_continuous_plus_one_hots() -> None:
    bank = _make_reach_bank(n=10)  # uses dist_bin 0..4, dir_bin 0..4, pose_bin 0
    # Expected: 19 + 5 + 5 + 1 + 0 (no env one-hot on single-env bank).
    assert bank.feature_dim == 19 + 5 + 5 + 1
    vec = bank.features(bank.train_ids()[0])
    assert vec.shape == (bank.feature_dim,)


def test_push_feature_dim_uses_pose_bin_width() -> None:
    bank = _make_push_bank(n=12)
    # dist_bin / dir_bin all 0 → width 1 each; pose_bin 0..3 → width 4.
    assert bank.feature_dim == 19 + 1 + 1 + 4


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_features_identical_for_same_context_id() -> None:
    bank = _make_reach_bank()
    cid = bank.train_ids()[0]
    v1 = bank.features(cid)
    v2 = bank.features(cid)
    assert np.array_equal(v1, v2)


def test_features_differ_across_contexts() -> None:
    bank = _make_reach_bank()
    a, b = bank.train_ids()[:2]
    assert not np.allclose(bank.features(a), bank.features(b))


# ---------------------------------------------------------------------------
# Train-bank statistics
# ---------------------------------------------------------------------------


def test_train_split_continuous_block_is_z_scored() -> None:
    bank = _make_reach_bank(n=40)
    train_blocks = np.stack(
        [bank.features(cid)[:CONTINUOUS_FEATURE_DIM] for cid in bank.train_ids()]
    )
    # Mean ~ 0, std ~ 1 across the train split (post-normalization).
    assert np.allclose(train_blocks.mean(axis=0), 0.0, atol=1e-10)
    # std == 1 wherever the raw feature was non-degenerate. Direction
    # unit-vector entries can be near-constant (all goals in one octant)
    # and saturate the 1e-6 std floor → very large normalized std.
    # Assert the typical case: most features are well-scaled.
    stds = train_blocks.std(axis=0)
    # At least 12 of the 19 continuous features should normalize to std ≈ 1.
    near_one = np.isclose(stds, 1.0, atol=1e-6)
    assert int(near_one.sum()) >= 12


def test_held_out_features_use_train_normalization() -> None:
    # The held-out split is not allowed to contribute to the stats. If it
    # were, swapping a held-out context's geometry would shift the mean
    # and change the train-split feature values. Verify by reconstructing
    # the bank with a different held-out split and asserting the train
    # feature vectors are unchanged.
    rng = np.random.default_rng(7)
    train_ctx = [_make_reach_context(i, dist_bin=i % 3, dir_bin=i % 3) for i in range(10)]
    ee = {i: rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6]) for i in range(20)}

    held_a = [
        _make_reach_context(10 + i, dist_bin=i % 3, dir_bin=i % 3, split="held_out")
        for i in range(10)
    ]
    held_b = [
        _make_reach_context(10 + i, dist_bin=i % 3, dir_bin=i % 3, split="held_out")
        for i in range(10)
    ]
    # Perturb held_b's desired_goals so its stats would differ if used.
    held_b = [
        FetchTaskContext(
            context_id=c.context_id,
            env_id=c.env_id,
            reset_seed=c.reset_seed,
            desired_goal=c.desired_goal + 5.0,  # large shift
            reset_payload=c.reset_payload,
            initial_object_pose=c.initial_object_pose,
            distance_bin=c.distance_bin,
            direction_bin=c.direction_bin,
            pose_bin=c.pose_bin,
            split="held_out",
        )
        for c in held_b
    ]
    bank_a = FetchTaskBank.from_contexts(train_ctx + held_a, ee)
    bank_b = FetchTaskBank.from_contexts(train_ctx + held_b, ee)
    for cid in bank_a.train_ids():
        assert np.array_equal(bank_a.features(cid), bank_b.features(cid)), cid


# ---------------------------------------------------------------------------
# Memorization safety
# ---------------------------------------------------------------------------


def test_features_never_contain_context_id() -> None:
    # Pick a context_id that is large and unique to make accidental
    # inclusion easy to spot. Featurization must NOT echo the raw int.
    rng = np.random.default_rng(13)
    n_train = 8
    contexts = [
        _make_reach_context(99000 + i, dist_bin=i % 4, dir_bin=i % 4) for i in range(n_train)
    ]
    ee = {99000 + i: rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6]) for i in range(n_train)}
    bank = FetchTaskBank.from_contexts(contexts, ee)
    for cid in bank.train_ids():
        vec = bank.features(cid)
        # No entry equals the context_id as a literal float.
        assert not np.isclose(vec, float(cid)).any(), cid
        # No entry equals the reset_seed.
        assert not np.isclose(vec, float(bank[cid].reset_seed)).any(), cid


def test_features_never_contain_reset_seed_via_distinct_seeds() -> None:
    # Two contexts that differ only in reset_seed (same desired_goal,
    # same initial achieved goal, same bins) must produce the same
    # feature vector — i.e. reset_seed is not in the feature path.
    base_dg = np.array([1.3, 0.7, 0.5])
    base_ag = np.array([1.2, 0.6, 0.5])
    c0 = FetchTaskContext(
        context_id=0,
        env_id=ENV_ID_REACH,
        reset_seed=42,
        desired_goal=base_dg,
        reset_payload=None,
        initial_object_pose=None,
        distance_bin=0,
        direction_bin=0,
        pose_bin=0,
        split="train",
    )
    c1 = FetchTaskContext(
        context_id=1,
        env_id=ENV_ID_REACH,
        reset_seed=99999,
        desired_goal=base_dg,
        reset_payload=None,
        initial_object_pose=None,
        distance_bin=0,
        direction_bin=0,
        pose_bin=0,
        split="train",
    )
    ee = {0: base_ag.copy(), 1: base_ag.copy()}
    bank = FetchTaskBank.from_contexts([c0, c1], ee)
    assert np.array_equal(bank.features(0), bank.features(1))


# ---------------------------------------------------------------------------
# Mixed-env bank — env one-hot
# ---------------------------------------------------------------------------


def test_mixed_env_bank_appends_env_one_hot() -> None:
    rng = np.random.default_rng(5)
    reach = [_make_reach_context(i) for i in range(4)]
    push = [_make_push_context(10 + i) for i in range(4)]
    ee = {
        c.context_id: rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6]) for c in reach + push
    }
    bank = FetchTaskBank.from_contexts(reach + push, ee)
    # env_id should be "mixed", and env_id_onehot_keys should be sorted
    # by env id string.
    assert bank.env_id == "mixed"
    assert bank.env_id_onehot_keys == tuple(sorted({ENV_ID_REACH, ENV_ID_PUSH}))
    # Feature vec ends with a 2-wide env one-hot.
    vec = bank.features(reach[0].context_id)
    assert vec.shape == (bank.feature_dim,)
    expected_dim = (
        CONTINUOUS_FEATURE_DIM
        + 1  # all dist_bin = 0 → width 1
        + 1  # all dir_bin = 0 → width 1
        + 1  # all pose_bin = 0 → width 1
        + 2  # env one-hot
    )
    assert bank.feature_dim == expected_dim
