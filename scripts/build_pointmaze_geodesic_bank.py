"""Build a STANDARD geodesic PointMaze bank in a challenging-but-learnable distance band.

Course-project Stage C (provenance: docs/EXPERIMENTS.md): a navigation
breadth task using the same geodesic scaffold + controller as PointMaze Medium, but a harder layout
(PointMaze_Large-v3) and a longer geodesic-distance band so true-start uniform does not trivially
saturate. NOT the retired off-path/bypass geometry — this is the plain shortest-path reset scaffold.

Usage:
    uv run python scripts/build_pointmaze_geodesic_bank.py \
        --env-id PointMaze_Large-v3 --out <path>.json \
        --min-distance 9 --max-distance 13 --seed 20260622
Prints the bank path, distance histogram, and SHA-256 (record it in RUN_LOG.md).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pcrc.envs.pointmaze_env import PointMazeTaskBank


def main(argv: list[str] | None = None) -> int:
    """Build, save, and hash a geodesic-banded PointMaze bank."""
    p = argparse.ArgumentParser(prog="build_pointmaze_geodesic_bank")
    p.add_argument("--env-id", default="PointMaze_Large-v3")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-held", type=int, default=50)
    p.add_argument("--min-distance", type=int, default=9)
    p.add_argument("--max-distance", type=int, default=13)
    p.add_argument("--pool-candidates", type=int, default=12000)
    p.add_argument("--seed", type=int, default=20260622)
    a = p.parse_args(argv)

    bank = PointMazeTaskBank.build(
        env_id=a.env_id,
        n_train=a.n_train,
        n_held=a.n_held,
        seed=a.seed,
        min_distance=a.min_distance,
        max_distance=a.max_distance,
        pool_candidates=a.pool_candidates,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bank.save(a.out)
    sha = hashlib.sha256(a.out.read_bytes()).hexdigest()

    # distance histogram over the saved contexts (verify the band + no leakage)
    from collections import Counter

    from pcrc.envs.pointmaze_env import _ensure_gym, _maze_distance

    gym = _ensure_gym()
    probe = gym.make(a.env_id)
    mm = probe.unwrapped.maze.maze_map
    probe.close()
    dists = Counter()
    train_ids = set(bank.train_ids())
    held_ids = set(bank.held_out_ids())
    for cid, c in bank.contexts.items():
        dists[_maze_distance(mm, c.reset_cell, c.goal_cell)] += 1
    leakage = {(c.reset_cell, c.goal_cell) for i, c in bank.contexts.items() if i in train_ids} & {
        (c.reset_cell, c.goal_cell) for i, c in bank.contexts.items() if i in held_ids
    }
    print(
        f"[build-pm-geo] {a.env_id} -> {a.out}\n"
        f"  contexts: {len(bank)} ({len(train_ids)} train / {len(held_ids)} held), "
        f"train_ids contiguous: {bank.train_ids() == list(range(a.n_train))}\n"
        f"  distance band [{a.min_distance},{a.max_distance}] histogram: "
        f"{dict(sorted(dists.items()))}\n"
        f"  start/goal leakage train∩held: {len(leakage)} (must be 0)\n"
        f"  SHA-256: {sha}"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
