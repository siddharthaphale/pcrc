"""Build a trajectory-derived scaffold ladder for MiniGrid DoorKey (Stage 2).

Reference trajectory = the **shortest action solution** of each train context's DoorKey layout,
found by breadth-first search. The BFS is *faithful*: it expands successors by ``restore + step`` on
the real MiniGrid env (via :class:`~pcrc.envs.minigrid_env.GridStateCodec`), so it models no MiniGrid
dynamics itself and the serialized states it records are byte-for-byte what training restores. A node
is ``(x, y, dir, has_key, door_open)`` — a complete state descriptor given the fixed layout. The search
terminates at the first state from which ``forward`` reaches the goal, and **drops that final
goal-reaching step** so the most-assisted level (quantile 1.0) sits *adjacent* to the goal rather than
on it (restoring onto the goal would terminate immediately and teach nothing).

The recorded states are then selected at **uniform step-count quantiles** {0, 0.25, 0.5, 0.75, 1.0}
(no stage labels used during generation — the whole point of Stage 2) via
:class:`ReferenceTrajectoryScaffoldBuilder`. Level 0 (quantile 0.0) is the original reset.

Stage-2 expectation: the uniform quantiles recover the key -> door -> goal stage structure that the
manual ``_apply_doorkey_scaffold`` hand-codes. Validate with
``scripts/validate_traj_scaffold.py --task doorkey --ladder <out> ...`` and inspect the printed
stage-composition summary (Gate 2 wants automatic states to span >= 3 distinct behavioral stages).

Usage::

    uv run --no-sync python scripts/build_doorkey_traj_scaffold.py \
        --out results/banks/scaffold_ladders/doorkey_8x8_s0.json
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  -- registers MiniGrid-* envs
import numpy as np

from pcrc.envs.minigrid_env import GridStateCodec, MiniGridTaskBank
from pcrc.envs.trajectory_scaffold import (
    ReferenceTrajectoryScaffoldBuilder,
    ScaffoldLadder,
    TrajectoryFrame,
)

QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)  # matches config/minigrid_doorkey.yaml level_fractions

# MiniGrid encodings (constants, not magic numbers): see minigrid.core.constants.
_DOOR, _KEY, _GOAL = 4, 5, 8  # OBJECT_TO_IDX
_DOOR_OPEN = 0  # STATE_TO_IDX["open"]
# Actions that can change a DoorKey state (drop/done are never useful in a solution).
_LEFT, _RIGHT, _FORWARD, _PICKUP, _TOGGLE = 0, 1, 2, 3, 5
_BFS_ACTIONS = (_LEFT, _RIGHT, _FORWARD, _PICKUP, _TOGGLE)


def _node_key(u) -> tuple[int, int, int, bool, bool]:
    """Complete state descriptor given the fixed layout: agent pose + key/door progress flags."""
    from minigrid.core.world_object import Door, Key  # noqa: PLC0415

    has_key = isinstance(u.carrying, Key)
    door_open = False
    for y in range(u.height):
        for x in range(u.width):
            c = u.grid.get(x, y)
            if isinstance(c, Door):
                door_open = bool(c.is_open)
    return (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir), has_key, door_open)


def solve_context(env, u, codec: GridStateCodec, seed: int, actions=_BFS_ACTIONS) -> list[dict] | None:
    """BFS shortest action solution; return the list of serialized states start..goal-adjacent.

    Returns ``None`` if no solution is found (defensive — solvable DoorKey layouts always have one).
    The returned path *excludes* the terminal on-goal state (final forward step dropped). ``actions``
    is the order BFS expands successors in; permuting it yields a *different* shortest solution where
    the layout admits more than one (Stage-4 multi-trajectory ablation).
    """
    env.reset(seed=seed)
    start_state = codec.serialize(u)
    start_key = _node_key(u)
    # key -> (parent_key | None, serialized_state)
    visited: dict[tuple, tuple] = {start_key: (None, start_state)}
    queue = deque([start_key])
    goal_node: tuple | None = None

    while queue and goal_node is None:
        k = queue.popleft()
        _parent, state = visited[k]
        for a in actions:
            codec.restore(u, state)
            u.step_count = 0  # keep truncation out of the way; codec doesn't track the counter
            _o, r, term, _trunc, _info = env.step(a)
            if term and r > 0:  # this action reaches the goal -> k is goal-adjacent; drop the step
                goal_node = k
                break
            if term:  # terminated without success (no such case in DoorKey, but be safe)
                continue
            nk = _node_key(u)
            if nk not in visited:
                visited[nk] = (k, codec.serialize(u))
                queue.append(nk)
    if goal_node is None:
        return None

    path: list[dict] = []
    k = goal_node
    while k is not None:
        parent, state = visited[k]
        path.append(state)
        k = parent
    path.reverse()
    return path


def stage_of(state: dict) -> tuple[bool, bool, bool]:
    """Behavioral stage flags for a serialized grid state: (carrying_key, door_open, beyond_door)."""
    grid = np.asarray(state["grid"], dtype=np.uint8)  # (W, H, 3): [x, y] = (type, color, st)
    carrying = state["carrying"]
    has_key = carrying is not None and int(carrying[0]) == _KEY
    door_open = False
    door_x = None
    types = grid[:, :, 0]
    st = grid[:, :, 2]
    xs, ys = np.where(types == _DOOR)
    if len(xs):
        door_x = int(xs[0])
        door_open = int(st[xs[0], ys[0]]) == _DOOR_OPEN
    beyond_door = door_x is not None and int(state["agent_pos"][0]) > door_x
    return (has_key, door_open, beyond_door)


def _frames_for(path, codec, u):
    """Frame sequence (state, step-index progress) for a BFS path; degenerate paths collapse to s_0."""
    if path is None or len(path) < 2:
        return [TrajectoryFrame(codec.serialize(u), 0.0)] if path is None else None
    return [TrajectoryFrame(s, float(i)) for i, s in enumerate(path)]


def build_ladder(bank: MiniGridTaskBank, max_steps: int, quantiles=QUANTILES, n_traj: int = 1, seed: int = 0):
    """Build a ladder over ``bank``'s train contexts from BFS shortest-solution trajectories.

    ``n_traj``>1 (Stage-4 ablation): per context, run BFS ``n_traj`` times with permuted action
    orders to recover distinct shortest solutions, and merge their quantile states into one
    multi-trajectory cell (the env samples 1-of-N per reset). Distinct solutions per context are
    de-duplicated, so a layout with a unique solution simply yields one candidate there.
    """
    builder = ReferenceTrajectoryScaffoldBuilder(quantiles)
    codec = GridStateCodec()
    env = gym.make(bank.env_id, max_steps=int(max_steps))
    u = env.unwrapped
    ladder = ScaffoldLadder(env_id=bank.env_id, quantiles=tuple(quantiles))
    rng = np.random.default_rng(seed)
    n_unsolved, n_degenerate = 0, 0
    path_lens: list[int] = []
    spans: list[int] = []  # distinct behavioral stages spanned by the selected (level>0) states
    distinct_per_ctx: list[int] = []  # # distinct solutions found (multi-traj diagnostic)
    try:
        for cid in bank.train_ids():
            seed_c = bank.contexts[cid].seed
            orders = [_BFS_ACTIONS] if n_traj == 1 else [
                tuple(int(x) for x in rng.permutation(_BFS_ACTIONS)) for _ in range(n_traj)
            ]
            paths, seen = [], set()
            for order in orders:
                path = solve_context(env, u, codec, seed_c, actions=order)
                if path is None or len(path) < 2:
                    continue
                # cheap signature to de-dup identical solutions (agent x-trace + length)
                sig = len(path), tuple(p["agent_pos"][0] for p in path)
                if sig not in seen:
                    seen.add(sig)
                    paths.append(path)
            if not paths:
                n_unsolved += 1
                n_degenerate += 1
                paths = [[codec.serialize(u)]]  # degenerate: collapse to s_0
            distinct_per_ctx.append(len(paths))
            path_lens.append(len(paths[0]))
            frames_list = [_frames_for(p, codec, u) for p in paths]
            frames_list = [f for f in frames_list if f is not None] or [
                [TrajectoryFrame(codec.serialize(u), 0.0)]
            ]
            cells = (
                builder.build_context(cid, frames_list[0])
                if n_traj == 1
                else builder.build_context_multi(cid, frames_list)
            )
            ladder.add_context(cells)
            # Level 0 is always the unassisted start: no key, door shut, before the door.
            stage_set = {(False, False, False)}
            stage_set |= {stage_of(c.serialized_state) for c in cells if c.serialized_state is not None}
            spans.append(len(stage_set))
    finally:
        env.close()
    stats = {
        "n_unsolved": n_unsolved,
        "n_degenerate": n_degenerate,
        "path_len_min": int(min(path_lens)) if path_lens else 0,
        "path_len_median": int(np.median(path_lens)) if path_lens else 0,
        "path_len_max": int(max(path_lens)) if path_lens else 0,
        "stage_span_min": int(min(spans)) if spans else 0,
        "stage_span_median": float(np.median(spans)) if spans else 0.0,
        "frac_ctx_span_ge3": float(np.mean([s >= 3 for s in spans])) if spans else 0.0,
        "distinct_traj_median": float(np.median(distinct_per_ctx)) if distinct_per_ctx else 0.0,
        "distinct_traj_max": int(max(distinct_per_ctx)) if distinct_per_ctx else 0,
    }
    return ladder, stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_doorkey_traj_scaffold")
    p.add_argument("--out", type=Path, required=True, help="output ScaffoldLadder JSON")
    p.add_argument(
        "--bank-out",
        type=Path,
        default=None,
        help="also save the (deterministic) bank JSON here for provenance + Gate-0 validation",
    )
    p.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-held", type=int, default=100)
    p.add_argument("--bank-seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=160, help="env max_steps (config loop.max_steps)")
    p.add_argument(
        "--n-levels",
        type=int,
        default=None,
        help="scaffold level count (Stage-B long-horizon sweep). N -> evenly-spaced quantiles "
        "[0, 1/(N-1), ..., 1.0]. Default None keeps the 5-level {0,.25,.5,.75,1.0}.",
    )
    p.add_argument(
        "--n-traj",
        type=int,
        default=1,
        help="reference trajectories per context (Stage-4 ablation: 1 vs 3, env samples 1-of-N)",
    )
    p.add_argument("--traj-seed", type=int, default=0, help="seed for BFS action-order permutations")
    a = p.parse_args(argv)

    quantiles = (
        tuple(i / (a.n_levels - 1) for i in range(a.n_levels)) if a.n_levels else QUANTILES
    )

    bank = MiniGridTaskBank.build(
        a.env_id, n_train=a.n_train, n_held=a.n_held, seed=a.bank_seed
    )
    print(
        f"[build_doorkey_traj_scaffold] env={a.env_id} bank_sha={bank.sha256()[:12]} "
        f"train={len(bank.train_ids())} held={len(bank.held_out_ids())} n_traj={a.n_traj}",
        flush=True,
    )
    ladder, stats = build_ladder(
        bank, a.max_steps, quantiles=quantiles, n_traj=a.n_traj, seed=a.traj_seed
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ladder.save(a.out)
    if a.bank_out is not None:
        bank.save(a.bank_out)
        print(f"  bank -> {a.bank_out}", flush=True)
    sha = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(
        f"  levels={ladder.n_levels} contexts={len(ladder.context_ids())} "
        f"unsolved={stats['n_unsolved']} degenerate={stats['n_degenerate']}",
        flush=True,
    )
    print(
        f"  solution length: min={stats['path_len_min']} median={stats['path_len_median']} "
        f"max={stats['path_len_max']}",
        flush=True,
    )
    print(
        f"  stage span (distinct (key,door,beyond) over levels>0): min={stats['stage_span_min']} "
        f"median={stats['stage_span_median']} frac_ctx>=3stages={stats['frac_ctx_span_ge3']:.2f}",
        flush=True,
    )
    print(
        f"  distinct solutions/context: median={stats['distinct_traj_median']} "
        f"max={stats['distinct_traj_max']} (n_traj requested={a.n_traj})",
        flush=True,
    )
    print(f"  -> {a.out}  sha256={sha[:16]}…", flush=True)
    # Gate-2 stage-structure precondition: most contexts should span >= 3 behavioral stages.
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
