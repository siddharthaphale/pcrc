"""Tests for PointMaze geodesic scaffold support.

The pure path/index helpers run on CPU (no sim). The env-instantiation tests
need the gymnasium-robotics + mujoco stack, which is **not** in the CPU `test`
extra, so they ``importorskip`` and run only on a sim box (plan §3, Stage 1 log).
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.envs.pointmaze_env import (
    _maze_distance,
    _scaffold_path_index,
    _shortest_path,
)

# 4x5 maze: border walls ("1"), open interior ("0"). A clear corridor exists.
MAZE = [
    ["1", "1", "1", "1", "1"],
    ["1", "0", "0", "0", "1"],
    ["1", "0", "1", "0", "1"],
    ["1", "1", "1", "1", "1"],
]


# ----------------------------------------------------------------- _shortest_path


def test_shortest_path_endpoints_and_contiguity() -> None:
    a, b = (1, 1), (2, 3)
    path = _shortest_path(MAZE, a, b)
    assert path is not None
    assert path[0] == a and path[-1] == b
    # length matches the distance metric (+1 for inclusive endpoints).
    assert len(path) == _maze_distance(MAZE, a, b) + 1
    # every step is a single 4-neighbour move onto an open cell.
    for (r0, c0), (r1, c1) in zip(path, path[1:]):  # noqa: B905 — pairwise, unequal length
        assert abs(r1 - r0) + abs(c1 - c0) == 1
        assert MAZE[r1][c1] == "0"


def test_shortest_path_same_cell() -> None:
    assert _shortest_path(MAZE, (1, 1), (1, 1)) == [(1, 1)]


def test_shortest_path_unreachable_is_none() -> None:
    walled = [
        ["1", "1", "1", "1"],
        ["1", "0", "1", "0"],  # (1,1) and (1,3) separated by walls
        ["1", "1", "1", "1"],
    ]
    assert _shortest_path(walled, (1, 1), (1, 3)) is None


# ------------------------------------------------------------- _scaffold_path_index


def test_scaffold_index_true_start_is_zero() -> None:
    assert _scaffold_path_index(5, 0.0) == 0
    assert _scaffold_path_index(5, -0.3) == 0


def test_scaffold_index_never_lands_on_goal() -> None:
    # Goal is index len-1; the easiest scaffold must clamp to len-2.
    for path_len in range(2, 12):
        assert _scaffold_path_index(path_len, 1.0) == path_len - 2
        assert _scaffold_path_index(path_len, 0.9) <= path_len - 2


def test_scaffold_index_short_path_collapses_to_start() -> None:
    # distance-1 path (len 2): no room for a non-trivial scaffold → index 0.
    assert _scaffold_path_index(2, 0.9) == 0
    assert _scaffold_path_index(1, 0.9) == 0


def test_scaffold_index_monotone_in_alpha() -> None:
    idxs = [_scaffold_path_index(11, a) for a in (0.0, 0.25, 0.5, 0.75, 0.9)]
    assert idxs == sorted(idxs)
    assert idxs[0] == 0  # α=0 → true start


# ----------------------------------------------- env-instantiation (needs sim stack)


def test_pointmaze_reset_scaffold_on_sim_box() -> None:
    pytest.importorskip("gymnasium_robotics")
    pytest.importorskip("mujoco")
    from pcrc.envs.pointmaze_env import PointMazeTaskBank, PointMazeTaskEnvironment

    bank = PointMazeTaskBank.build("PointMaze_Medium-v3", n_train=20, n_held=4, seed=0)
    env = PointMazeTaskEnvironment(bank, "PointMaze_Medium-v3")
    try:
        # Pick the train context with the longest path so a scaffold is non-trivial.
        cid, best = bank.train_ids()[0], -1
        for c in bank.train_ids():
            ctx = bank[c]
            p = _shortest_path(env._maze_map, tuple(ctx.reset_cell), tuple(ctx.goal_cell))
            if p is not None and len(p) > best:
                best, cid = len(p), c
        assert best >= 4  # need room for a real scaffold

        def start_goal_dist(obs: dict) -> float:
            return float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))

        # PointMaze adds per-reset noise to BOTH start and goal positions, so exact
        # equality across resets is wrong — assert the geometric invariants instead.
        d0 = start_goal_dist(env.reset(cid, start_progress=0.0))  # true start: far from goal
        d0b = start_goal_dist(env.reset(cid))  # default arg behaves like α=0
        d_sc = start_goal_dist(env.reset(cid, start_progress=0.9))  # scaffold: near the goal
        assert abs(d0 - d0b) < 1.5  # default reset ≈ α=0 (modulo per-reset noise)
        assert d_sc < 0.5 * d0  # scaffold resets land much closer to the goal
    finally:
        env.close()
