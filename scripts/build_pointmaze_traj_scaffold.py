"""Build a trajectory-derived scaffold ladder for PointMaze (Stage 1, C4).

Reference trajectory = the shortest-path planner from each train context's ``reset_cell`` to its
``goal_cell`` (the same planner the manual geodesic scaffold uses). We walk the agent along the path
cells, serialize the full MuJoCo state at each, and select states at cumulative-arc-length quantiles
{0, 0.25, 0.5, 0.75, 0.9} via :class:`ReferenceTrajectoryScaffoldBuilder`. The result is a
:class:`ScaffoldLadder` JSON (sibling to ``results/banks/``), restorable by
``PointMazeTaskEnvironment(scaffold_source="trajectory")`` and runnable via the ``trajectory_*`` conditions.

Stage-1 expectation: automatic ~ manual geodesic scaffold. Validate the output with
``scripts/validate_traj_scaffold.py --task pointmaze --ladder <out> --bank <bank>``.

Usage::

    uv run --extra robotics python scripts/build_pointmaze_traj_scaffold.py \
        --bank results/banks/pointmaze_large_geodesic_v2.json \
        --out results/banks/scaffold_ladders/pointmaze_large_s0.json
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from pcrc.envs.pointmaze_env import (
    PointMazeStateCodec,
    PointMazeTaskBank,
    PointMazeTaskEnvironment,
    _shortest_path,
)
from pcrc.envs.trajectory_scaffold import (
    ReferenceTrajectoryScaffoldBuilder,
    ScaffoldLadder,
    TrajectoryFrame,
)

QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9)


def build_ladder(
    bank: PointMazeTaskBank, env_id: str, quantiles=QUANTILES
) -> tuple[ScaffoldLadder, int]:
    """Build a ladder over ``bank``'s train contexts from shortest-path reference trajectories."""
    builder = ReferenceTrajectoryScaffoldBuilder(quantiles)
    codec = PointMazeStateCodec()
    src = PointMazeTaskEnvironment(bank, env_id)
    ladder = ScaffoldLadder(env_id=env_id, quantiles=tuple(quantiles))
    n_short = 0
    try:
        maze_map = src._maze_map
        for cid in bank.train_ids():
            ctx = bank.contexts[cid]
            path = _shortest_path(maze_map, tuple(ctx.reset_cell), tuple(ctx.goal_cell))
            if path is None or len(path) < 2:
                n_short += 1
                path = [tuple(ctx.reset_cell)]
            frames: list[TrajectoryFrame] = []
            cum, prev = 0.0, None
            for cell in path:
                src._env.reset(
                    options={"goal_cell": np.array(ctx.goal_cell), "reset_cell": np.array(cell)}
                )
                state = codec.serialize(src._env.unwrapped)
                xy = np.asarray(src._env.unwrapped.point_env.data.qpos[:2], dtype=np.float64).copy()
                if prev is not None:
                    cum += float(np.linalg.norm(xy - prev))
                prev = xy
                frames.append(TrajectoryFrame(state, cum))
            ladder.add_context(builder.build_context(cid, frames))
    finally:
        src.close()
    return ladder, n_short


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_pointmaze_traj_scaffold")
    p.add_argument("--bank", type=Path, required=True, help="saved PointMaze bank (geodesic)")
    p.add_argument("--out", type=Path, required=True, help="output ScaffoldLadder JSON")
    a = p.parse_args(argv)

    bank = PointMazeTaskBank.load(a.bank)
    ladder, n_short = build_ladder(bank, bank.env_id)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ladder.save(a.out)
    sha = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(
        f"[build_pointmaze_traj_scaffold] env={bank.env_id} train={len(bank.train_ids())} "
        f"levels={ladder.n_levels} degenerate_paths={n_short}",
        flush=True,
    )
    print(f"  -> {a.out}  sha256={sha[:16]}…", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
