"""Tests for ``pcrc.envs.fetch_task_bank``.

Covers the §7.1 hard gates on the bank itself: same ``context_id`` →
identical ``desired_goal`` and identical stored ``reset_payload`` entries;
stratified split sizes; semantic JSON round-trip
(``np.array_equal`` for arrays, ``==`` for scalars). Does NOT touch a
live gymnasium-robotics env — that is covered by
``test_fetch_environment_reset.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pcrc.envs.fetch_task_bank import (
    KNOWN_RESET_PAYLOAD_KEYS,
    SCHEMA_VERSION,
    FetchTaskBank,
    FetchTaskContext,
)

ENV_ID_REACH = "FetchReachDense-v4"
ENV_ID_PUSH = "FetchPushDense-v4"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_reach_context(cid: int, split: str = "train", seed_base: int = 1000) -> FetchTaskContext:
    rng = np.random.default_rng(seed_base + cid)
    return FetchTaskContext(
        context_id=cid,
        env_id=ENV_ID_REACH,
        reset_seed=seed_base + cid,
        desired_goal=rng.uniform(low=[1.1, 0.5, 0.4], high=[1.5, 1.0, 0.7]),
        reset_payload=None,
        initial_object_pose=None,
        distance_bin=cid % 5,
        direction_bin=(cid + 1) % 5,
        pose_bin=0,
        split=split,
    )


def _make_push_context(cid: int, split: str = "train", seed_base: int = 2000) -> FetchTaskContext:
    rng = np.random.default_rng(seed_base + cid)
    init_qpos = rng.uniform(-0.1, 0.1, size=22)
    init_qvel = np.zeros(21)
    obj_pose = rng.uniform(low=[1.1, 0.5, 0.42], high=[1.5, 1.0, 0.42])
    return FetchTaskContext(
        context_id=cid,
        env_id=ENV_ID_PUSH,
        reset_seed=seed_base + cid,
        desired_goal=rng.uniform(low=[1.1, 0.5, 0.42], high=[1.5, 1.0, 0.42]),
        reset_payload={
            "init_qpos": init_qpos,
            "init_qvel": init_qvel,
            "object_pose": obj_pose,
        },
        initial_object_pose=obj_pose,
        distance_bin=cid % 5,
        direction_bin=(cid + 2) % 5,
        pose_bin=cid % 4,
        split=split,
    )


def _make_reach_bank(num_train: int = 12, num_held_out: int = 4) -> FetchTaskBank:
    contexts: list[FetchTaskContext] = []
    ee: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for i in range(num_train):
        contexts.append(_make_reach_context(i, split="train"))
        ee[i] = rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6])
    for j in range(num_held_out):
        cid = num_train + j
        contexts.append(_make_reach_context(cid, split="held_out"))
        ee[cid] = rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6])
    return FetchTaskBank.from_contexts(contexts, ee)


def _make_push_bank(num_train: int = 10, num_held_out: int = 3) -> FetchTaskBank:
    contexts: list[FetchTaskContext] = []
    ee: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(1)
    for i in range(num_train):
        contexts.append(_make_push_context(i, split="train"))
        ee[i] = rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6])
    for j in range(num_held_out):
        cid = num_train + j
        contexts.append(_make_push_context(cid, split="held_out"))
        ee[cid] = rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6])
    return FetchTaskBank.from_contexts(contexts, ee)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_context_id_returns_identical_desired_goal() -> None:
    bank = _make_reach_bank()
    cid = bank.train_ids()[0]
    first = bank[cid]
    second = bank[cid]
    assert first is second  # dataclass frozen + same dict lookup
    assert np.array_equal(first.desired_goal, second.desired_goal)


def test_same_context_id_returns_identical_reset_payload() -> None:
    bank = _make_push_bank()
    cid = bank.train_ids()[0]
    first = bank[cid]
    second = bank[cid]
    assert first.reset_payload is not None
    assert second.reset_payload is not None
    assert set(first.reset_payload.keys()) == set(second.reset_payload.keys())
    for k in first.reset_payload:
        assert np.array_equal(first.reset_payload[k], second.reset_payload[k])


def test_stratified_split_sizes_match_request() -> None:
    bank = _make_reach_bank(num_train=20, num_held_out=7)
    assert len(bank.train_ids()) == 20
    assert len(bank.held_out_ids()) == 7
    assert len(bank) == 27
    # Disjoint splits.
    assert not (set(bank.train_ids()) & set(bank.held_out_ids()))


def test_contains_and_getitem() -> None:
    bank = _make_reach_bank()
    cid = bank.train_ids()[0]
    assert cid in bank
    assert 99999 not in bank
    with pytest.raises(KeyError):
        _ = bank[99999]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_empty_contexts_rejected() -> None:
    with pytest.raises(ValueError, match="contexts is empty"):
        FetchTaskBank.from_contexts([], {})


def test_duplicate_context_id_rejected() -> None:
    c0 = _make_reach_context(0)
    c0_dup = _make_reach_context(0, seed_base=99)  # different content, same id
    ee = {0: np.array([1.3, 0.7, 0.5])}
    with pytest.raises(ValueError, match="duplicate context_id"):
        FetchTaskBank.from_contexts([c0, c0_dup], ee)


def test_missing_initial_achieved_goal_rejected() -> None:
    c0 = _make_reach_context(0)
    c1 = _make_reach_context(1)
    ee = {0: np.array([1.3, 0.7, 0.5])}  # missing 1
    with pytest.raises(ValueError, match="missing context_ids"):
        FetchTaskBank.from_contexts([c0, c1], ee)


def test_empty_train_split_rejected() -> None:
    contexts = [_make_reach_context(i, split="held_out") for i in range(3)]
    ee = {i: np.array([1.3, 0.7, 0.5]) for i in range(3)}
    with pytest.raises(ValueError, match="train split is empty"):
        FetchTaskBank.from_contexts(contexts, ee)


def test_unknown_reset_payload_key_rejected_on_serialize() -> None:
    rng = np.random.default_rng(0)
    bad = FetchTaskContext(
        context_id=0,
        env_id=ENV_ID_PUSH,
        reset_seed=0,
        desired_goal=rng.uniform(size=3),
        reset_payload={"unknown_key": np.zeros(3)},
        initial_object_pose=np.zeros(3),
        distance_bin=0,
        direction_bin=0,
        pose_bin=0,
        split="train",
    )
    bank = FetchTaskBank.from_contexts([bad], {0: np.zeros(3)})
    with pytest.raises(ValueError, match="unknown reset_payload key"):
        bank.save(Path("/tmp/_should_not_write.json"))


def test_known_reset_payload_keys_frozen_set() -> None:
    # Guard against accidental mutation; the wrapper relies on this set.
    assert "init_qpos" in KNOWN_RESET_PAYLOAD_KEYS
    assert "object_pose" in KNOWN_RESET_PAYLOAD_KEYS
    assert isinstance(KNOWN_RESET_PAYLOAD_KEYS, frozenset)


# ---------------------------------------------------------------------------
# JSON round-trip (semantic equality, not byte-for-byte)
# ---------------------------------------------------------------------------


def _assert_contexts_equal(a: FetchTaskContext, b: FetchTaskContext) -> None:
    assert a.context_id == b.context_id
    assert a.env_id == b.env_id
    assert a.reset_seed == b.reset_seed
    assert np.array_equal(a.desired_goal, b.desired_goal)
    if a.reset_payload is None:
        assert b.reset_payload is None
    else:
        assert b.reset_payload is not None
        assert set(a.reset_payload) == set(b.reset_payload)
        for k in a.reset_payload:
            assert np.array_equal(a.reset_payload[k], b.reset_payload[k]), k
    if a.initial_object_pose is None:
        assert b.initial_object_pose is None
    else:
        assert np.array_equal(a.initial_object_pose, b.initial_object_pose)
    assert a.distance_bin == b.distance_bin
    assert a.direction_bin == b.direction_bin
    assert a.pose_bin == b.pose_bin
    assert a.split == b.split


def test_reach_bank_save_load_semantic_roundtrip(tmp_path: Path) -> None:
    bank = _make_reach_bank()
    p = tmp_path / "reach_bank.json"
    bank.save(p)
    loaded = FetchTaskBank.load(p)
    assert loaded.env_id == bank.env_id
    assert loaded.env_id_onehot_keys == bank.env_id_onehot_keys
    assert sorted(loaded.contexts) == sorted(bank.contexts)
    for cid in bank.contexts:
        _assert_contexts_equal(loaded[cid], bank[cid])
        assert np.array_equal(loaded.initial_achieved_goals[cid], bank.initial_achieved_goals[cid])
    assert np.array_equal(loaded.feature_stats.mean, bank.feature_stats.mean)
    assert np.array_equal(loaded.feature_stats.std, bank.feature_stats.std)


def test_push_bank_save_load_preserves_reset_payload(tmp_path: Path) -> None:
    bank = _make_push_bank()
    p = tmp_path / "push_bank.json"
    bank.save(p)
    loaded = FetchTaskBank.load(p)
    for cid in bank.train_ids() + bank.held_out_ids():
        _assert_contexts_equal(loaded[cid], bank[cid])
        # Specific reset_payload sub-keys preserved exactly.
        for k in ("init_qpos", "init_qvel", "object_pose"):
            assert np.array_equal(
                loaded[cid].reset_payload[k],
                bank[cid].reset_payload[k],
            ), f"context {cid} key {k}"


def test_load_refuses_stale_schema_version(tmp_path: Path) -> None:
    bank = _make_reach_bank()
    p = tmp_path / "stale.json"
    bank.save(p)
    payload = json.loads(p.read_text())
    payload["schema_version"] = SCHEMA_VERSION + 99
    p.write_text(json.dumps(payload, sort_keys=True, indent=2))
    with pytest.raises(ValueError, match="schema_version"):
        FetchTaskBank.load(p)


def test_save_is_deterministic_for_identical_banks(tmp_path: Path) -> None:
    bank_a = _make_reach_bank()
    bank_b = _make_reach_bank()
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    bank_a.save(pa)
    bank_b.save(pb)
    # Same construction (same fixtures, same seed) → same JSON bytes
    # under sort_keys=True. This is a stronger guarantee than the
    # general save / load round-trip and is OK to assert because both
    # banks come from the deterministic _make_reach_bank fixture.
    assert pa.read_bytes() == pb.read_bytes()
