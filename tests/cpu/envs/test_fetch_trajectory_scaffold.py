"""Stage-0 contract for the Fetch in-air MuJoCo trajectory-scaffold codec + reset branch.

The hard physical case: a grasped in-air state. We build the reference "trajectory" frames from the
manual reverse-curriculum's own synthesized grasps, serialize the full MuJoCo state (qpos/qvel/ctrl/
mocap), and require a trajectory-source reset to (a) reproduce the observation and (b) HOLD the grasp
over free simulator steps — the property that justifies serializing ``ctrl`` (the finger targets).
Needs the robotics extra; skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.registry import ScaffoldSource

pytest.importorskip("gymnasium_robotics")
pytest.importorskip("mujoco")

from pcrc.envs.fetch_environment import (  # noqa: E402
    FetchStateCodec,
    FetchTaskEnvironment,
)
from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext  # noqa: E402
from pcrc.envs.trajectory_scaffold import (  # noqa: E402
    ReferenceTrajectoryScaffoldBuilder,
    ScaffoldLadder,
    TrajectoryFrame,
)

ENV_ID = "FetchPickAndPlace-v4"
PNP_FRACS = (0.0, 0.25, 0.5, 0.75, 0.9)  # continuous suite level_fractions
_TABLE_Z = 0.42  # block rests near here; an in-air grasp keeps it well above


def _inair_bank() -> FetchTaskBank:
    """Two contexts (1 train, 1 held-out) with IN-AIR goals (z > 0.45 triggers grasp synthesis)."""
    contexts, ee = [], {}
    for i, split in ((0, "train"), (1, "held_out")):
        contexts.append(
            FetchTaskContext(
                context_id=i,
                env_id=ENV_ID,
                reset_seed=3000 + i,
                desired_goal=np.array([1.30, 0.75, 0.55]),  # in air
                reset_payload=None,
                initial_object_pose=None,
                distance_bin=0,
                direction_bin=0,
                pose_bin=0,
                split=split,
            )
        )
        ee[i] = np.array([1.30, 0.75, 0.42])
    return FetchTaskBank.from_contexts(contexts, ee)


@pytest.fixture(scope="module")
def bank() -> FetchTaskBank:
    return _inair_bank()


def _build_ladder_and_obs(bank: FetchTaskBank):
    """Ladder over train contexts from manual in-air grasp states + the obs each manual reset gave."""
    builder = ReferenceTrajectoryScaffoldBuilder(PNP_FRACS)
    codec = FetchStateCodec()
    src = FetchTaskEnvironment(bank, ENV_ID)  # manual source
    ladder = ScaffoldLadder(env_id=ENV_ID, quantiles=PNP_FRACS)
    captured: dict[tuple[int, int], dict] = {}
    try:
        for cid in bank.train_ids():
            frames = []
            for level, frac in enumerate(PNP_FRACS):
                obs = src.reset(cid, frac)
                captured[(cid, level)] = {
                    k: np.asarray(v) for k, v in obs.items() if k != "source_context_id"
                }
                frames.append(
                    TrajectoryFrame(
                        codec.serialize(src._env.unwrapped), cumulative_progress=float(level)
                    )
                )
            ladder.add_context(builder.build_context(cid, frames))
    finally:
        src.close()
    return ladder, captured


def test_codec_state_roundtrip(bank: FetchTaskBank):
    """serialize(restore(serialize(s))) preserves qpos/qvel/ctrl (pure codec fidelity)."""
    codec = FetchStateCodec()
    env = FetchTaskEnvironment(bank, ENV_ID)
    try:
        cid = bank.train_ids()[0]
        env.reset(cid, 0.9)  # in-air grasp
        s = codec.serialize(env._env.unwrapped)
        codec.restore(env._env.unwrapped, s)
        s2 = codec.serialize(env._env.unwrapped)
        for k in ("qpos", "qvel", "ctrl"):
            assert np.allclose(s[k], s2[k]), k
    finally:
        env.close()


def test_trajectory_reset_matches_manual_snapshot(bank: FetchTaskBank):
    ladder, captured = _build_ladder_and_obs(bank)
    traj = FetchTaskEnvironment(bank, ENV_ID, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=ladder)
    try:
        for cid in bank.train_ids():
            for level, frac in enumerate(PNP_FRACS):
                if level == 0:
                    continue
                out = traj.reset(cid, frac)
                exp = captured[(cid, level)]
                assert np.allclose(out["observation"], exp["observation"], atol=1e-5)
                assert np.allclose(out["achieved_goal"], exp["achieved_goal"], atol=1e-5)
    finally:
        traj.close()


def test_grasp_held_after_restore(bank: FetchTaskBank):
    """The restored in-air grasp must not drop the block — the reason ctrl is serialized."""
    import mujoco  # noqa: PLC0415

    ladder, _ = _build_ladder_and_obs(bank)
    traj = FetchTaskEnvironment(bank, ENV_ID, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=ladder)
    try:
        cid = bank.train_ids()[0]
        out = traj.reset(cid, 0.9)  # most-assisted, in air
        assert out["achieved_goal"][2] > _TABLE_Z + 0.03  # grasped above the table
        u = traj._env.unwrapped
        for _ in range(10):  # free steps: ctrl + mocap hold the grasp
            mujoco.mj_step(u.model, u.data)
        z_after = float(u._get_obs()["achieved_goal"][2])
        assert z_after > _TABLE_Z + 0.03, f"block dropped to z={z_after}"
    finally:
        traj.close()


def test_trajectory_reset_determinism_and_save_load(bank: FetchTaskBank, tmp_path):
    ladder, _ = _build_ladder_and_obs(bank)
    reloaded = ScaffoldLadder.load(ladder.save(tmp_path / "pnp_ladder.json"))
    a = FetchTaskEnvironment(bank, ENV_ID, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=ladder)
    b = FetchTaskEnvironment(bank, ENV_ID, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=reloaded)
    try:
        for cid in bank.train_ids():
            for frac in (0.25, 0.5, 0.75, 0.9):
                oa, oa2 = a.reset(cid, frac), a.reset(cid, frac)
                assert np.allclose(oa["observation"], oa2["observation"])  # determinism
                ob = b.reset(cid, frac)
                assert np.allclose(oa["observation"], ob["observation"])  # survives save/load
    finally:
        a.close()
        b.close()


def test_ladder_only_covers_train_contexts(bank: FetchTaskBank):
    ladder, _ = _build_ladder_and_obs(bank)
    assert ladder.context_ids() == bank.train_ids()
    assert set(ladder.context_ids()).isdisjoint(bank.held_out_ids())


def test_trajectory_source_requires_ladder(bank: FetchTaskBank):
    with pytest.raises(ValueError):
        FetchTaskEnvironment(bank, ENV_ID, scaffold_source=ScaffoldSource.TRAJECTORY)
    with pytest.raises(ValueError):
        FetchTaskEnvironment(bank, ENV_ID, scaffold_source="bogus")
