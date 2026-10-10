"""Geodesic-banded PointMaze bank (course-project Stage C) — scaffold invariants.

The PointMaze Large breadth task uses the standard geodesic reset scaffold on a longer distance band.
These tests pin §C2: ℓ=0 reproduces the true-start cell, the scaffold reset moves monotonically toward
the goal as ℓ grows, all scaffold cells are open, train ids are contiguous, and there is no train/held
start–goal leakage. (Continuous within-cell reset jitter is the env's standard stochasticity and is not
asserted — only the cell-level scaffold geometry is deterministic.)
"""

from __future__ import annotations

import pytest

pytest.importorskip("gymnasium_robotics")

from pcrc.envs.pointmaze_env import (  # noqa: E402
    PointMazeTaskBank,
    _ensure_gym,
    _maze_distance,
    _scaffold_path_index,
    _shortest_path,
)

ENV_ID = "PointMaze_Large-v3"
BAND = (9, 13)
FRACS = [0.0, 0.25, 0.5, 0.75, 0.9]


@pytest.fixture(scope="module")
def band_bank() -> PointMazeTaskBank:
    return PointMazeTaskBank.build(
        env_id=ENV_ID,
        n_train=20,
        n_held=10,
        seed=20260622,
        min_distance=BAND[0],
        max_distance=BAND[1],
        pool_candidates=6000,
    )


def test_band_and_split(band_bank: PointMazeTaskBank) -> None:
    gym = _ensure_gym()
    probe = gym.make(ENV_ID)
    mm = probe.unwrapped.maze.maze_map
    probe.close()
    # every context's geodesic distance is inside the requested band
    for c in band_bank.contexts.values():
        d = _maze_distance(mm, c.reset_cell, c.goal_cell)
        assert BAND[0] <= d <= BAND[1]
    # loop invariant: train ids are a contiguous 0..n_train-1 block
    assert band_bank.train_ids() == list(range(20))
    # no train/held start–goal leakage
    tr = {(band_bank[i].reset_cell, band_bank[i].goal_cell) for i in band_bank.train_ids()}
    he = {(band_bank[i].reset_cell, band_bank[i].goal_cell) for i in band_bank.held_out_ids()}
    assert tr.isdisjoint(he)


def test_scaffold_geometry_invariants(band_bank: PointMazeTaskBank) -> None:
    gym = _ensure_gym()
    probe = gym.make(ENV_ID)
    mm = probe.unwrapped.maze.maze_map
    probe.close()
    for cid in list(band_bank.contexts)[:20]:
        c = band_bank[cid]
        path = _shortest_path(mm, c.reset_cell, c.goal_cell)
        assert path is not None and path[0] == c.reset_cell and path[-1] == c.goal_cell
        cells = [path[_scaffold_path_index(len(path), a)] for a in FRACS]
        # ℓ=0 is the true-start cell
        assert cells[0] == c.reset_cell
        # scaffold reset moves monotonically toward the goal (distance non-increasing in ℓ)
        dists = [_maze_distance(mm, cell, c.goal_cell) for cell in cells]
        assert all(dists[i] >= dists[i + 1] for i in range(len(dists) - 1))
        # a scaffold reset never lands on the goal cell, and all cells are open
        assert all(cell != c.goal_cell for cell in cells)
        assert all(str(mm[r][col]) == "0" for (r, col) in cells)
