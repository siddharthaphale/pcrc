"""MiniGrid DoorKey PCRC scaffold: bank save/load + hash, and stage reset invariants.

Pins the Phase C contracts the dev/final runs rely on: a deterministic bank persisted and round-tripped;
disjoint train / held-out; level 0 is the unmodified env; higher stages give the key, open the door, and
move the agent toward the goal.
"""

from __future__ import annotations

import pytest

pytest.importorskip("minigrid")

import gymnasium as gym  # noqa: E402

from pcrc.envs.minigrid_env import (  # noqa: E402
    MiniGridTaskBank,
    MiniGridTaskEnvironment,
)

ENV_ID = "MiniGrid-DoorKey-8x8-v0"


@pytest.fixture(scope="module")
def bank() -> MiniGridTaskBank:
    return MiniGridTaskBank.build(ENV_ID, n_train=4, n_held=2, seed=0)


def test_bank_deterministic_and_roundtrip(bank: MiniGridTaskBank, tmp_path) -> None:
    again = MiniGridTaskBank.build(ENV_ID, n_train=4, n_held=2, seed=0)
    assert bank.sha256() == again.sha256() and len(bank.sha256()) == 64
    p = bank.save(tmp_path / "mg_bank.json")
    loaded = MiniGridTaskBank.load(p)
    assert loaded.train_ids() == bank.train_ids()
    assert loaded.held_out_ids() == bank.held_out_ids()
    assert loaded.sha256() == bank.sha256()
    import hashlib

    assert hashlib.sha256(p.read_bytes()).hexdigest() == bank.sha256()


def test_train_heldout_disjoint(bank: MiniGridTaskBank) -> None:
    assert set(bank.train_ids()).isdisjoint(set(bank.held_out_ids()))
    assert len(bank.train_ids()) == 4 and len(bank.held_out_ids()) == 2


def test_stage_reset_invariants(bank: MiniGridTaskBank) -> None:
    env = MiniGridTaskEnvironment(bank, ENV_ID, max_steps=80)
    try:
        for cid in bank.train_ids():
            # reference unmodified reset for the level-0 check
            ref = gym.make(ENV_ID)
            ref.reset(seed=bank[cid].seed)
            ref_pos = tuple(int(v) for v in ref.unwrapped.agent_pos)
            ref.close()

            o0 = env.reset(cid, start_progress=0.0)
            assert o0.shape == (env.obs_dim,)
            u = env._u
            assert tuple(int(v) for v in u.agent_pos) == ref_pos  # level 0 == unmodified env
            assert u.carrying is None

            env.reset(cid, start_progress=0.25)  # stage 1: carrying the key
            assert env._u.carrying is not None

            env.reset(cid, start_progress=0.75)  # stage 3: door open
            door_open = any(
                type(env._u.grid.get(x, y)).__name__ == "Door" and env._u.grid.get(x, y).is_open
                for y in range(env._u.height)
                for x in range(env._u.width)
            )
            assert door_open
            # agent stands on a free cell at every stage
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                env.reset(cid, start_progress=frac)
                pos = tuple(int(v) for v in env._u.agent_pos)
                c = env._u.grid.get(*pos)
                assert c is None or c.can_overlap()
    finally:
        env.close()


MULTIROOM_ID = "MiniGrid-MultiRoom-N4-S5-v0"


def test_multiroom_route_scaffold() -> None:
    """MultiRoom route-progress scaffold: level 0 == unmodified env; every stage lands on a free cell and
    moves the agent closer to the goal (later room)."""

    bank = MiniGridTaskBank.build(MULTIROOM_ID, n_train=3, n_held=1, seed=0)
    assert set(bank.train_ids()).isdisjoint(set(bank.held_out_ids()))
    env = MiniGridTaskEnvironment(bank, MULTIROOM_ID, max_steps=120)
    try:
        for cid in bank.train_ids():
            ref = gym.make(MULTIROOM_ID)
            ref.reset(seed=bank[cid].seed)
            ref_pos = tuple(int(v) for v in ref.unwrapped.agent_pos)
            ref.close()

            o0 = env.reset(cid, start_progress=0.0)
            assert o0.shape == (env.obs_dim,)
            assert tuple(int(v) for v in env._u.agent_pos) == ref_pos  # level 0 == unmodified

            def goal_dist():
                from minigrid.core.world_object import Goal

                u = env._u
                g = next(
                    (
                        (x, y)
                        for y in range(u.height)
                        for x in range(u.width)
                        if isinstance(u.grid.get(x, y), Goal)
                    ),
                    None,
                )
                return abs(int(u.agent_pos[0]) - g[0]) + abs(int(u.agent_pos[1]) - g[1])

            d0 = goal_dist()
            for frac in (0.25, 0.5, 0.75, 1.0):
                env.reset(cid, start_progress=frac)
                pos = tuple(int(v) for v in env._u.agent_pos)
                c = env._u.grid.get(*pos)
                assert c is None or c.can_overlap()
            assert goal_dist() < d0  # stage 4 is closer to the goal than the true start
    finally:
        env.close()
