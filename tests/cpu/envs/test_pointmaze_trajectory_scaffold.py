"""Stage-0 contract for the PointMaze MuJoCo trajectory-scaffold codec + reset branch.

Uses the manual geodesic scaffold's own states as the reference "trajectory" frames (deterministic,
CPU-cheap MuJoCo), so a trajectory-source reset must restore the exact snapshot — the MuJoCo state
round-trips and the rebuilt observation (PointMaze obs == [qpos, qvel]) matches the captured one.
Needs the robotics extra (mujoco + gymnasium-robotics); skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.registry import ScaffoldSource

pytest.importorskip("gymnasium_robotics")
pytest.importorskip("mujoco")

from pcrc.envs.pointmaze_env import (  # noqa: E402
    PointMazeStateCodec,
    PointMazeTaskBank,
    PointMazeTaskEnvironment,
)
from pcrc.envs.trajectory_scaffold import (  # noqa: E402
    ReferenceTrajectoryScaffoldBuilder,
    ScaffoldLadder,
    TrajectoryFrame,
)

ENV_ID = "PointMaze_Medium-v3"
PM_FRACS = (0.0, 0.25, 0.5, 0.75, 0.9)  # PointMaze level_fractions (continuous suite)


@pytest.fixture(scope="module")
def bank() -> PointMazeTaskBank:
    return PointMazeTaskBank.build(env_id=ENV_ID, n_train=3, n_held=2, seed=0)


def _build_ladder_and_obs(bank: PointMazeTaskBank):
    """Ladder over train contexts from the manual scaffold states + the obs each manual reset gave."""
    builder = ReferenceTrajectoryScaffoldBuilder(PM_FRACS)
    codec = PointMazeStateCodec()
    src = PointMazeTaskEnvironment(bank, ENV_ID)  # manual source
    ladder = ScaffoldLadder(env_id=ENV_ID, quantiles=PM_FRACS)
    captured: dict[tuple[int, int], dict] = {}
    try:
        for cid in bank.train_ids():
            frames = []
            for level, frac in enumerate(PM_FRACS):
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


def test_codec_state_roundtrip(bank: PointMazeTaskBank):
    """serialize(restore(serialize(s))) preserves qpos/qvel (pure codec fidelity)."""
    codec = PointMazeStateCodec()
    env = PointMazeTaskEnvironment(bank, ENV_ID)
    try:
        cid = bank.train_ids()[0]
        env.reset(cid, 0.5)
        s = codec.serialize(env._env.unwrapped)
        codec.restore(env._env.unwrapped, s)
        s2 = codec.serialize(env._env.unwrapped)
        assert np.allclose(s["qpos"], s2["qpos"]) and np.allclose(s["qvel"], s2["qvel"])
    finally:
        env.close()


def test_trajectory_reset_matches_manual_snapshot(bank: PointMazeTaskBank):
    """Restoring an assisted-level snapshot reproduces the captured manual observation."""
    ladder, captured = _build_ladder_and_obs(bank)
    traj = PointMazeTaskEnvironment(
        bank, ENV_ID, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=ladder
    )
    try:
        for cid in bank.train_ids():
            for level, frac in enumerate(PM_FRACS):
                if level == 0:
                    continue  # level 0 is a stochastic true-start reset (re-randomized), checked separately
                out = traj.reset(cid, frac)
                exp = captured[(cid, level)]
                # The codec restores the agent's qpos/qvel exactly, so observation == [qpos, qvel]
                # and achieved_goal match the captured snapshot bit-for-bit.
                assert np.allclose(out["observation"], exp["observation"], atol=1e-6)
                assert np.allclose(out["achieved_goal"], exp["achieved_goal"], atol=1e-6)
                # PointMaze re-randomizes the goal WITHIN the goal cell each reset (it is not part of
                # qpos), so desired_goal is not bit-identical — only guaranteed to stay in the same
                # cell. The manual scaffold has the same property.
                assert np.max(np.abs(out["desired_goal"] - exp["desired_goal"])) < 1.0
    finally:
        traj.close()


def test_trajectory_reset_determinism_and_level0(bank: PointMazeTaskBank):
    ladder, _ = _build_ladder_and_obs(bank)
    traj = PointMazeTaskEnvironment(
        bank, ENV_ID, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=ladder
    )
    try:
        for cid in bank.train_ids():
            a = traj.reset(cid, 0.75)
            b = traj.reset(cid, 0.75)
            assert np.allclose(a["observation"], b["observation"])  # restore is deterministic
            o0 = traj.reset(cid, 0.0)  # level 0: valid finite true-start obs
            assert np.all(np.isfinite(o0["observation"]))
            assert o0["observation"].shape == a["observation"].shape
    finally:
        traj.close()


def test_trajectory_reset_survives_ladder_save_load(bank: PointMazeTaskBank, tmp_path):
    ladder, _ = _build_ladder_and_obs(bank)
    reloaded = ScaffoldLadder.load(ladder.save(tmp_path / "pm_ladder.json"))
    a = PointMazeTaskEnvironment(bank, ENV_ID, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=ladder)
    b = PointMazeTaskEnvironment(
        bank, ENV_ID, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=reloaded
    )
    try:
        for cid in bank.train_ids():
            for frac in (0.25, 0.5, 0.75, 0.9):
                assert np.allclose(
                    a.reset(cid, frac)["observation"], b.reset(cid, frac)["observation"]
                )
    finally:
        a.close()
        b.close()


def test_ladder_only_covers_train_contexts(bank: PointMazeTaskBank):
    ladder, _ = _build_ladder_and_obs(bank)
    assert ladder.context_ids() == bank.train_ids()
    assert set(ladder.context_ids()).isdisjoint(bank.held_out_ids())


def test_trajectory_source_requires_ladder(bank: PointMazeTaskBank):
    with pytest.raises(ValueError):
        PointMazeTaskEnvironment(bank, ENV_ID, scaffold_source=ScaffoldSource.TRAJECTORY)
    with pytest.raises(ValueError):
        PointMazeTaskEnvironment(bank, ENV_ID, scaffold_source="bogus")
