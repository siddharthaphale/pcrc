"""C6 offline validation harness for trajectory-derived scaffold ladders (Gate 0).

Restores every assisted ``(context, level)`` cell of a ladder, runs a few settling simulator steps,
and checks the restored state is valid: finite observations, in-bounds objects, and — for
manipulation — a retained grasp. Reports the invalid-reset rate per task; **Gate 0 requires it
below 5%** (Automatic_Trajectory_Scaffold_Execution_Plan.md §3 / §10).

By default it builds a small self-contained ladder per task from the *manual* scaffold states (the
reference-trajectory stand-in) so the codec/restore path can be gated before any RL or any C4 builder
exists. The same :func:`validate_ladder` core validates a real ladder built from planner/expert
trajectories — point it at a built ladder + matching bank once those exist.

Usage::

    uv run --extra robotics python scripts/validate_traj_scaffold.py            # all three tasks
    uv run --extra robotics python scripts/validate_traj_scaffold.py --task fetch --settle 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from pcrc.envs.trajectory_scaffold import (
    ReferenceTrajectoryScaffoldBuilder,
    ScaffoldLadder,
    TrajectoryFrame,
)
from pcrc.registry import ScaffoldSource

# Fetch workspace bounds (FetchPickAndPlace table frame) and table height.
_FETCH_XY = ((1.0, 1.6), (0.4, 1.1))
_TABLE_Z = 0.42
_FETCH_Z = (_TABLE_Z, 0.9)


@dataclass
class CellResult:
    context_id: int
    level: int
    ok: bool
    reason: str = ""


@dataclass
class ValidationReport:
    task: str
    cells: list[CellResult] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_invalid(self) -> int:
        return sum(1 for c in self.cells if not c.ok)

    @property
    def invalid_rate(self) -> float:
        return self.n_invalid / self.n_cells if self.n_cells else 0.0

    def passed(self, threshold: float = 0.05) -> bool:
        return self.invalid_rate < threshold

    def summary(self) -> str:
        head = (
            f"[{self.task}] cells={self.n_cells} invalid={self.n_invalid} "
            f"rate={self.invalid_rate:.3f} -> {'PASS' if self.passed() else 'FAIL'} (Gate 0 < 0.05)"
        )
        bad = [f"    (cid={c.context_id}, L={c.level}): {c.reason}" for c in self.cells if not c.ok]
        return "\n".join([head, *bad])


def _finite(*arrs) -> bool:
    return all(np.all(np.isfinite(np.asarray(a))) for a in arrs)


def validate_ladder(
    env, ladder: ScaffoldLadder, train_ids, kind: str, settle: int
) -> ValidationReport:
    """Restore every assisted cell, settle, and record per-cell validity. ``env`` is trajectory-source."""
    report = ValidationReport(task=kind)
    fracs = ladder.quantiles
    for cid in train_ids:
        for level in range(1, ladder.n_levels):
            ok, reason = True, ""
            try:
                obs = env.reset(cid, fracs[level])
                vec = obs["observation"] if isinstance(obs, dict) else obs
                if not _finite(vec):
                    ok, reason = False, "non-finite observation"
                else:
                    ok, reason = _settle_and_check(env, kind, settle)
            except Exception as e:  # noqa: BLE001 -- any restore failure is an invalid reset
                ok, reason = False, f"{type(e).__name__}: {e}"
            report.cells.append(CellResult(cid, level, ok, reason))
    return report


def _settle_and_check(env, kind: str, settle: int) -> tuple[bool, str]:
    """Run ``settle`` free simulator steps (ctrl/mocap held) and validate the resulting state."""
    if kind == "doorkey":
        u = env._u
        c = u.grid.get(*tuple(int(v) for v in u.agent_pos))
        if not (c is None or c.can_overlap()):
            return False, "agent on a non-overlappable cell"
        return True, ""

    import mujoco  # noqa: PLC0415

    u = env._env.unwrapped
    model, data = (
        (u.point_env.model, u.point_env.data) if kind == "pointmaze" else (u.model, u.data)
    )
    # z of the object at the restored state (before settling), to measure drop under stabilization.
    z_before = float(np.asarray(u._get_obs()["achieved_goal"])[2]) if kind == "fetch" else 0.0
    for _ in range(settle):
        mujoco.mj_step(model, data)
    if not _finite(data.qpos, data.qvel):
        return False, "non-finite qpos/qvel after settling"
    if kind == "fetch":
        obj = np.asarray(u._get_obs()["achieved_goal"], dtype=np.float64)
        (xl, xh), (yl, yh), (zl, zh) = _FETCH_XY[0], _FETCH_XY[1], _FETCH_Z
        if not (xl <= obj[0] <= xh and yl <= obj[1] <= yh and zl <= obj[2] <= zh):
            return False, f"object out of workspace bounds: {obj.round(3).tolist()}"
        # Stability, not absolute height: a valid restored state stays where it was placed. Low levels
        # legitimately sit near the table (the reverse-curriculum target there is near the start); a
        # FAILED grasp instead lets the object fall out of the gripper (large drop under settling).
        if z_before - obj[2] > 0.08:
            return (
                False,
                f"unstable: object fell {z_before - obj[2]:.3f} under settling (z={obj[2]:.3f})",
            )
    return True, ""


# --------------------------------------------------------------------------- per-task demo ladders


def _build_manual_ladder(env_factory, train_ids, fracs, serialize):
    """Build a ladder from a manual-source env's states (the reference-trajectory stand-in)."""
    builder = ReferenceTrajectoryScaffoldBuilder(fracs)
    src = env_factory()
    ladder = ScaffoldLadder(env_id=getattr(src, "env_id", "demo"), quantiles=tuple(fracs))
    try:
        for cid in train_ids:
            frames = [
                TrajectoryFrame(serialize(src, cid, frac), cumulative_progress=float(level))
                for level, frac in enumerate(fracs)
            ]
            ladder.add_context(builder.build_context(cid, frames))
    finally:
        src.close()
    return ladder


def _validate_pointmaze(settle: int, n_train: int) -> ValidationReport:
    from pcrc.envs.pointmaze_env import (
        PointMazeStateCodec,
        PointMazeTaskBank,
        PointMazeTaskEnvironment,
    )

    env_id, fracs = "PointMaze_Medium-v3", (0.0, 0.25, 0.5, 0.75, 0.9)
    bank = PointMazeTaskBank.build(env_id=env_id, n_train=n_train, n_held=2, seed=0)
    codec = PointMazeStateCodec()

    def serialize(src, cid, frac):
        src.reset(cid, frac)
        return codec.serialize(src._env.unwrapped)

    ladder = _build_manual_ladder(
        lambda: PointMazeTaskEnvironment(bank, env_id), bank.train_ids(), fracs, serialize
    )
    env = PointMazeTaskEnvironment(
        bank, env_id, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=ladder
    )
    try:
        return validate_ladder(env, ladder, bank.train_ids(), "pointmaze", settle)
    finally:
        env.close()


def _validate_doorkey(settle: int, n_train: int) -> ValidationReport:
    from pcrc.envs.minigrid_env import GridStateCodec, MiniGridTaskBank, MiniGridTaskEnvironment

    env_id, fracs = "MiniGrid-DoorKey-8x8-v0", (0.0, 0.25, 0.5, 0.75, 1.0)
    bank = MiniGridTaskBank.build(env_id, n_train=n_train, n_held=2, seed=0)
    codec = GridStateCodec()

    def serialize(src, cid, frac):
        src.reset(cid, frac)
        return codec.serialize(src._u)

    ladder = _build_manual_ladder(
        lambda: MiniGridTaskEnvironment(bank, env_id, max_steps=80),
        bank.train_ids(),
        fracs,
        serialize,
    )
    env = MiniGridTaskEnvironment(
        bank, env_id, max_steps=80, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=ladder
    )
    try:
        return validate_ladder(env, ladder, bank.train_ids(), "doorkey", settle)
    finally:
        env.close()


def _validate_fetch(settle: int, n_train: int) -> ValidationReport:
    from pcrc.envs.fetch_environment import FetchStateCodec, FetchTaskEnvironment
    from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext

    env_id, fracs = "FetchPickAndPlace-v4", (0.0, 0.25, 0.5, 0.75, 0.9)
    contexts, ee = [], {}
    for i in range(n_train + 1):
        contexts.append(
            FetchTaskContext(
                context_id=i,
                env_id=env_id,
                reset_seed=3000 + i,
                desired_goal=np.array([1.30, 0.75, 0.55]),
                reset_payload=None,
                initial_object_pose=None,
                distance_bin=0,
                direction_bin=0,
                pose_bin=0,
                split="train" if i < n_train else "held_out",
            )
        )
        ee[i] = np.array([1.30, 0.75, 0.42])
    bank = FetchTaskBank.from_contexts(contexts, ee)
    codec = FetchStateCodec()

    def serialize(src, cid, frac):
        src.reset(cid, frac)
        return codec.serialize(src._env.unwrapped)

    ladder = _build_manual_ladder(
        lambda: FetchTaskEnvironment(bank, env_id), bank.train_ids(), fracs, serialize
    )
    env = FetchTaskEnvironment(bank, env_id, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=ladder)
    try:
        return validate_ladder(env, ladder, bank.train_ids(), "fetch", settle)
    finally:
        env.close()


_TASKS = {"pointmaze": _validate_pointmaze, "doorkey": _validate_doorkey, "fetch": _validate_fetch}


def _validate_loaded(task: str, ladder_path, bank_path, settle: int) -> ValidationReport:
    """Validate a REAL ladder + matching bank built by a C4 builder (full-bank Gate 0)."""
    from pathlib import Path

    ladder = ScaffoldLadder.load(Path(ladder_path))
    if task == "pointmaze":
        from pcrc.envs.pointmaze_env import PointMazeTaskBank, PointMazeTaskEnvironment

        bank = PointMazeTaskBank.load(Path(bank_path))
        env = PointMazeTaskEnvironment(
            bank, bank.env_id, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=ladder
        )
    elif task == "doorkey":
        from pcrc.envs.minigrid_env import MiniGridTaskBank, MiniGridTaskEnvironment

        bank = MiniGridTaskBank.load(Path(bank_path))
        env = MiniGridTaskEnvironment(
            bank,
            bank.env_id,
            max_steps=160,
            scaffold_source=ScaffoldSource.TRAJECTORY,
            scaffold_ladder=ladder,
        )
    elif task == "fetch":
        from pcrc.envs.fetch_environment import FetchTaskEnvironment
        from pcrc.envs.fetch_task_bank import FetchTaskBank

        bank = FetchTaskBank.load(Path(bank_path))
        env = FetchTaskEnvironment(
            bank, bank.env_id, scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=ladder
        )
    else:
        raise NotImplementedError(f"--ladder validation not wired for task {task!r} yet")
    try:
        return validate_ladder(env, ladder, bank.train_ids(), task, settle)
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="validate_traj_scaffold")
    p.add_argument("--task", choices=[*_TASKS, "all"], default="all")
    p.add_argument("--settle", type=int, default=10, help="free simulator steps after each restore")
    p.add_argument("--n-train", type=int, default=4, help="train contexts to validate (demo banks)")
    p.add_argument(
        "--ladder", type=str, default=None, help="validate a REAL ladder JSON (+ --bank)"
    )
    p.add_argument("--bank", type=str, default=None, help="bank matching --ladder")
    a = p.parse_args(argv)

    if a.ladder is not None:
        if a.bank is None or a.task == "all":
            p.error("--ladder requires --bank and a specific --task")
        reports = [_validate_loaded(a.task, a.ladder, a.bank, a.settle)]
    else:
        tasks = list(_TASKS) if a.task == "all" else [a.task]
        reports = [_TASKS[t](a.settle, a.n_train) for t in tasks]
    print("\n".join(r.summary() for r in reports))
    all_pass = all(r.passed() for r in reports)
    print(f"\nGate 0: {'PASS' if all_pass else 'FAIL'} (all tasks invalid-reset rate < 5%)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
