"""Tests for the Nav-bypass scaffold geometry (autonomous plan Gate D / breadth).

Covers the pure helpers `_detour_ratio` / `_offpath_scaffold_cell` (off-path resets that
bypass a bottleneck for high-detour contexts) and `PointMazeTaskBank` save/load with
`route_group`. Uses constructed maze maps — no live MuJoCo env needed.
"""

from __future__ import annotations

import pytest

from pcrc.envs.pointmaze_env import (
    PointMazeContext,
    PointMazeTaskBank,
    _detour_ratio,
    _offpath_scaffold_cell,
)

# U-shaped maze: start (1,1) and goal (1,3) are 2 apart in a straight line but the only
# path runs down the left arm, across the bottom, and up the right arm (geodesic 6).
U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]
# Straight open corridor (no detour).
STRAIGHT = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]


def test_detour_ratio_separates_bypass_from_direct() -> None:
    assert _detour_ratio(U_MAZE, (1, 1), (1, 3)) >= 2.0  # windy U → bypass
    assert _detour_ratio(STRAIGHT, (1, 1), (1, 5)) == pytest.approx(1.0)  # straight → direct


def test_offpath_bypasses_the_bottleneck() -> None:
    # High-α off-path reset lands on the GOAL side (right column, col 3), skipping the U detour.
    cell = _offpath_scaffold_cell(U_MAZE, (1, 1), (1, 3), 0.9)
    assert cell[1] == 3 and cell != (1, 1)
    # α=0 is the true start; the goal cell is never a reset.
    assert _offpath_scaffold_cell(U_MAZE, (1, 1), (1, 3), 0.0) == (1, 1)
    assert _offpath_scaffold_cell(U_MAZE, (1, 1), (1, 3), 0.99) != (1, 3)


def test_offpath_direct_tracks_the_straight_line() -> None:
    # On a straight corridor the off-path reset is just the geometric interpolation.
    assert _offpath_scaffold_cell(STRAIGHT, (1, 1), (1, 5), 0.5) == (1, 3)
    assert _offpath_scaffold_cell(STRAIGHT, (1, 1), (1, 5), 0.99) != (1, 5)  # never the goal


def test_offpath_low_alpha_stays_start_side_in_bypass() -> None:
    # The off-path reset JUMPS start-side → goal-side as α crosses the wall (it does not cover the
    # detour cells the geodesic would) — the mechanism behind the bypass group's decreasing transfer.
    start_side = _offpath_scaffold_cell(U_MAZE, (1, 1), (1, 3), 0.3)
    assert start_side[1] == 1  # left (start) column


def test_bank_save_load_roundtrip_preserves_route_group(tmp_path) -> None:
    ctxs = [
        PointMazeContext(0, (1, 1), (1, 3), 0, "train", route_group=1),
        PointMazeContext(1, (2, 1), (2, 3), 0, "held_out", route_group=0),
    ]
    bank = PointMazeTaskBank(ctxs, "PointMaze_Large-v3")
    p = tmp_path / "bank.json"
    bank.save(p)
    b2 = PointMazeTaskBank.load(p)
    assert b2.env_id == "PointMaze_Large-v3"
    assert b2[0].route_group == 1 and b2[1].route_group == 0
    assert b2[0].reset_cell == (1, 1) and b2[0].goal_cell == (1, 3)
    assert b2.train_ids() == [0] and b2.held_out_ids() == [1]
