"""Stage-0 contracts for automatic trajectory-derived scaffolds.

Two layers:
  * the generic, sim-free builder/ladder (quantile selection, level-0 reset sentinel, save/load hash);
  * the MiniGrid DoorKey slice (GridStateCodec serialize/restore + the trajectory reset branch).

For the MiniGrid slice we use the *manual* stage operator's states as the reference "trajectory"
frames — deterministic and CPU-cheap — so a trajectory-source reset must reproduce the manual
scaffold's state exactly (same observation), while exercising the full serialize -> ladder ->
save/load -> restore path. Mirrors the Stage-0 unit-test list in
docs/Automatic_Trajectory_Scaffold_Execution_Plan.md §3.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.envs.trajectory_scaffold import (
    ReferenceTrajectoryScaffoldBuilder,
    ScaffoldLadder,
    TrajectoryFrame,
)
from pcrc.registry import ScaffoldSource

# --------------------------------------------------------------------------- generic builder/ladder


def _frames(progresses, states=None):
    states = states or [{"i": i} for i in range(len(progresses))]
    return [TrajectoryFrame(s, float(p)) for s, p in zip(states, progresses, strict=True)]


def test_builder_selects_earliest_index_per_quantile():
    b = ReferenceTrajectoryScaffoldBuilder((0.0, 0.25, 0.5, 0.75, 0.9))
    cells = b.build_context(7, _frames(range(11)))  # cumulative 0..10 -> normalized 0,0.1,...,1.0
    assert [c.level for c in cells] == [0, 1, 2, 3, 4]
    assert [c.progress_quantile for c in cells] == [0.0, 0.25, 0.5, 0.75, 0.9]
    # earliest index i with i/10 >= q
    assert [c.trajectory_index for c in cells] == [0, 3, 5, 8, 9]
    # indices are non-decreasing across levels (ordering invariant #1)
    idxs = [c.trajectory_index for c in cells]
    assert idxs == sorted(idxs)


def test_level0_is_reset_sentinel():
    cells = ReferenceTrajectoryScaffoldBuilder((0.0, 0.5, 1.0)).build_context(0, _frames(range(5)))
    assert cells[0].level == 0
    assert cells[0].serialized_state is None  # original reset, not a snapshot
    assert cells[0].state_hash == "reset"
    assert cells[0].trajectory_index == 0
    assert all(c.serialized_state is not None for c in cells[1:])


def test_degenerate_flat_trajectory_collapses_to_start():
    cells = ReferenceTrajectoryScaffoldBuilder((0.0, 0.5, 1.0)).build_context(0, _frames([0, 0, 0]))
    # no progress to distinguish levels -> every assisted level uses the start state (index 0)
    assert [c.trajectory_index for c in cells] == [0, 0, 0]
    assert cells[1].serialized_state == {"i": 0} and cells[2].serialized_state == {"i": 0}


def test_builder_rejects_bad_quantiles():
    with pytest.raises(ValueError):
        ReferenceTrajectoryScaffoldBuilder((0.25, 0.5))  # must start at 0.0
    with pytest.raises(ValueError):
        ReferenceTrajectoryScaffoldBuilder((0.0, 0.5, 0.25))  # non-decreasing
    with pytest.raises(ValueError):
        ReferenceTrajectoryScaffoldBuilder((0.0, 1.5))  # out of [0,1]


def test_builder_rejects_non_monotonic_progress():
    b = ReferenceTrajectoryScaffoldBuilder((0.0, 0.5, 1.0))
    with pytest.raises(ValueError):
        b.build_context(0, _frames([0.0, 1.0, 0.5]))


def test_ladder_save_load_roundtrip_and_hash(tmp_path):
    b = ReferenceTrajectoryScaffoldBuilder((0.0, 0.5, 1.0))
    lad = ScaffoldLadder(env_id="probe-v0", quantiles=(0.0, 0.5, 1.0))
    for cid in (0, 1, 2):
        lad.add_context(b.build_context(cid, _frames(range(6))))
    p = lad.save(tmp_path / "ladder.json")
    loaded = ScaffoldLadder.load(p)
    assert loaded.sha256() == lad.sha256() and len(lad.sha256()) == 64
    assert loaded.context_ids() == lad.context_ids() == [0, 1, 2]
    import hashlib

    assert (
        hashlib.sha256(p.read_bytes()).hexdigest() == lad.sha256()
    )  # file bytes == canonical payload
    # cell hashes are stable across the round-trip
    for key, cell in lad.cells.items():
        assert loaded.cells[key].state_hash == cell.state_hash


# ------------------------------------------------------------------------------- MiniGrid slice

pytest.importorskip("minigrid")

from pcrc.envs.minigrid_env import (  # noqa: E402
    GridStateCodec,
    MiniGridTaskBank,
    MiniGridTaskEnvironment,
)

ENV_ID = "MiniGrid-DoorKey-8x8-v0"
MG_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)  # MiniGrid level_fractions


@pytest.fixture(scope="module")
def bank() -> MiniGridTaskBank:
    return MiniGridTaskBank.build(ENV_ID, n_train=4, n_held=2, seed=0)


def _manual_ladder(bank: MiniGridTaskBank) -> ScaffoldLadder:
    """Build a ladder over train contexts using the manual stage states as reference frames."""
    builder = ReferenceTrajectoryScaffoldBuilder(MG_QUANTILES)
    codec = GridStateCodec()
    src = MiniGridTaskEnvironment(bank, ENV_ID, max_steps=80)  # manual source
    ladder = ScaffoldLadder(env_id=ENV_ID, quantiles=MG_QUANTILES)
    try:
        for cid in bank.train_ids():
            frames = []
            for level in range(len(MG_QUANTILES)):
                src.reset(cid, level / (len(MG_QUANTILES) - 1))  # manual stage == level
                frames.append(
                    TrajectoryFrame(codec.serialize(src._u), cumulative_progress=float(level))
                )
            ladder.add_context(builder.build_context(cid, frames))
    finally:
        src.close()
    return ladder


def test_ladder_only_covers_train_contexts(bank: MiniGridTaskBank):
    lad = _manual_ladder(bank)
    assert lad.context_ids() == bank.train_ids()
    assert set(lad.context_ids()).isdisjoint(bank.held_out_ids())  # no train/held-out leakage


def test_trajectory_reset_matches_manual_scaffold(bank: MiniGridTaskBank):
    """Restoring a trajectory snapshot reproduces the manual scaffold's state bit-for-bit (obs)."""
    lad = _manual_ladder(bank)
    manual = MiniGridTaskEnvironment(bank, ENV_ID, max_steps=80)
    traj = MiniGridTaskEnvironment(
        bank, ENV_ID, max_steps=80, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=lad
    )
    try:
        for cid in bank.train_ids():
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                om = manual.reset(cid, frac)
                ot = traj.reset(cid, frac)
                assert np.array_equal(om, ot), f"cid={cid} frac={frac}: traj != manual obs"
    finally:
        manual.close()
        traj.close()


def test_trajectory_reset_stage_invariants_and_determinism(bank: MiniGridTaskBank):
    lad = _manual_ladder(bank)
    import gymnasium as gym

    traj = MiniGridTaskEnvironment(
        bank, ENV_ID, max_steps=80, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=lad
    )
    try:
        for cid in bank.train_ids():
            ref = gym.make(ENV_ID)
            ref.reset(seed=bank[cid].seed)
            ref_pos = tuple(int(v) for v in ref.unwrapped.agent_pos)
            ref.close()

            o0a = traj.reset(cid, 0.0)
            assert tuple(int(v) for v in traj._u.agent_pos) == ref_pos  # level 0 == unmodified env
            assert traj._u.carrying is None
            o0b = traj.reset(cid, 0.0)
            assert np.array_equal(o0a, o0b)  # determinism: same reset twice -> identical obs

            traj.reset(cid, 0.25)  # stage 1: carrying the key
            assert traj._u.carrying is not None

            traj.reset(cid, 0.75)  # stage 3: door open
            assert any(
                type(traj._u.grid.get(x, y)).__name__ == "Door" and traj._u.grid.get(x, y).is_open
                for y in range(traj._u.height)
                for x in range(traj._u.width)
            )
            for frac in MG_QUANTILES:  # every level lands on a free cell
                traj.reset(cid, frac)
                c = traj._u.grid.get(*tuple(int(v) for v in traj._u.agent_pos))
                assert c is None or c.can_overlap()
    finally:
        traj.close()


def test_trajectory_reset_survives_ladder_save_load(bank: MiniGridTaskBank, tmp_path):
    lad = _manual_ladder(bank)
    reloaded = ScaffoldLadder.load(lad.save(tmp_path / "mg_ladder.json"))
    a = MiniGridTaskEnvironment(
        bank, ENV_ID, max_steps=80, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=lad
    )
    b = MiniGridTaskEnvironment(
        bank, ENV_ID, max_steps=80, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=reloaded
    )
    try:
        for cid in bank.train_ids():
            for frac in MG_QUANTILES:
                assert np.array_equal(a.reset(cid, frac), b.reset(cid, frac))
    finally:
        a.close()
        b.close()


def test_trajectory_source_requires_ladder(bank: MiniGridTaskBank):
    with pytest.raises(ValueError):
        MiniGridTaskEnvironment(bank, ENV_ID, scaffold_source=ScaffoldSource.TRAJECTORY)
    with pytest.raises(ValueError):
        MiniGridTaskEnvironment(bank, ENV_ID, scaffold_source="bogus")


# --------------------------------------------------------------- Stage-4 multi-trajectory ablation


def test_build_context_multi_merges_candidates():
    """N trajectories merge into one cell per level: serialized_states holds all, primary == traj 0."""
    builder = ReferenceTrajectoryScaffoldBuilder((0.0, 0.5, 1.0))
    # three distinct trajectories, each 3 frames with the same progress profile but different states
    fl = [[TrajectoryFrame({"t": k, "i": i}, float(i)) for i in range(3)] for k in range(3)]
    cells = builder.build_context_multi(7, fl)
    assert [c.level for c in cells] == [0, 1, 2]
    assert cells[0].serialized_state is None and cells[0].serialized_states is None  # level-0 reset
    for cell in cells[1:]:
        assert cell.serialized_states is not None and len(cell.serialized_states) == 3
        # primary mirrors the candidate list head (single-traj readers stay correct)
        assert cell.serialized_state == cell.serialized_states[0]
        # the three candidates come from the three distinct trajectories
        assert sorted(s["t"] for s in cell.serialized_states) == [0, 1, 2]


def test_sample_state_uniform_over_candidates_and_single_traj_deterministic():
    """Multi-cell: sample_state covers all candidates; single-traj cell: returns the lone state."""
    builder = ReferenceTrajectoryScaffoldBuilder((0.0, 1.0))
    fl = [[TrajectoryFrame({"t": k}, float(i)) for i in range(2)] for k in range(3)]
    lad = ScaffoldLadder(env_id="demo", quantiles=(0.0, 1.0))
    lad.add_context(builder.build_context_multi(0, fl))
    rng = np.random.default_rng(0)
    seen = {lad.sample_state(0, 1, rng)["t"] for _ in range(200)}
    assert seen == {0, 1, 2}  # uniform sampling reaches every candidate

    # round-trip through save/load preserves the candidate list
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        lad.save(f.name)
        rl = ScaffoldLadder.load(f.name)
    assert rl.get(0, 1).serialized_states == lad.get(0, 1).serialized_states

    # single-trajectory cell: rng is irrelevant, always the one serialized_state
    single = ScaffoldLadder(env_id="demo", quantiles=(0.0, 1.0))
    single.add_context(builder.build_context(0, fl[0]))
    assert single.get(0, 1).serialized_states is None
    assert (
        single.sample_state(0, 1, rng)
        == single.sample_state(0, 1, None)
        == fl[0][1].serialized_state
    )
