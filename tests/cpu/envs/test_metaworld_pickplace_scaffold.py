"""Meta-World PickPlace PCRC scaffold: bank save/load + hash, and reset-level invariants.

Pins the Phase B contracts the dev/final runs rely on:
  * a deterministic bank (same seed -> same SHA-256), persisted and round-tripped;
  * disjoint train / held-out splits;
  * level 0 (start_progress=0) is the true ungrasped start;
  * higher scaffold levels move the object toward the goal (and lift it for in-air goals).
"""

from __future__ import annotations

import numpy as np
import pytest

metaworld = pytest.importorskip("metaworld")

from pcrc.envs.metaworld_env import (  # noqa: E402
    MetaWorldTaskBank,
    MetaWorldTaskEnvironment,
    parse_metaworld_env_id,
)

ENV_ID = "metaworld:pick-place-v3:handnear:graspcur"
TASK = "pick-place-v3"


@pytest.fixture(scope="module")
def bank() -> MetaWorldTaskBank:
    return MetaWorldTaskBank.build(TASK, n_train=4, n_held=2, seed=0)


def test_bank_build_is_deterministic_by_seed(bank: MetaWorldTaskBank) -> None:
    again = MetaWorldTaskBank.build(TASK, n_train=4, n_held=2, seed=0)
    assert bank.sha256() == again.sha256()
    assert len(bank.sha256()) == 64  # hex digest


def test_bank_save_load_roundtrip(bank: MetaWorldTaskBank, tmp_path) -> None:
    p = bank.save(tmp_path / "mw_pp_bank.json")
    assert p.exists()
    loaded = MetaWorldTaskBank.load(p)
    # semantic round-trip: ids, splits, and rand_vecs preserved; hash stable
    assert loaded.train_ids() == bank.train_ids()
    assert loaded.held_out_ids() == bank.held_out_ids()
    for cid in bank.contexts:
        assert np.allclose(loaded[cid].rand_vec, bank[cid].rand_vec)
        assert loaded[cid].split == bank[cid].split
    assert loaded.sha256() == bank.sha256()
    # file bytes hash to sha256()
    import hashlib

    assert hashlib.sha256(p.read_bytes()).hexdigest() == bank.sha256()


def test_train_heldout_disjoint(bank: MetaWorldTaskBank) -> None:
    assert set(bank.train_ids()).isdisjoint(set(bank.held_out_ids()))
    assert len(bank.train_ids()) == 4
    assert len(bank.held_out_ids()) == 2


def test_level0_is_true_start_and_levels_move_object(bank: MetaWorldTaskBank) -> None:
    task, opts = parse_metaworld_env_id(ENV_ID)
    env = MetaWorldTaskEnvironment(bank, task, **opts)
    try:
        for cid in bank.train_ids():
            ctx = bank[cid]
            obj0 = np.asarray(ctx.obj_init, dtype=np.float64)
            goal = np.asarray(ctx.rand_vec[3:6], dtype=np.float64)
            # level 0 == true start: achieved goal equals the captured object init
            o0 = env.reset(cid, start_progress=0.0)
            assert np.allclose(o0["achieved_goal"], obj0, atol=1e-3)
            d0 = float(np.linalg.norm(o0["achieved_goal"] - goal))
            # a higher level moves the object closer to the goal than the true start
            o9 = env.reset(cid, start_progress=0.9)
            d9 = float(np.linalg.norm(o9["achieved_goal"] - goal))
            assert d9 < d0 + 1e-6
            # in-air goals must lift the object off the table at the high level
            if float(goal[2]) > 0.10:
                assert float(o9["achieved_goal"][2]) > 0.08
    finally:
        env.close()
