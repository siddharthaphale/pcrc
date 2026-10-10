"""DoorKey natural pacing-conflict candidate pool (Test Campaign 4 Stage C, C.2.1).

For N candidate DoorKey layouts (one per seed), BFS-solve and store geometry/solution features used to
define natural easy/hard pacing groups BEFORE any RL (campaign C.2.2). Reuses the env-stepping BFS from
build_doorkey_traj_scaffold.py. Per-context features:

  T          solution length (steps to goal-adjacent)
  t_key      first step carrying the key
  t_door     first step the door is open
  t_goal     = T (goal-reaching step dropped, so T is the goal-adjacent index)
  d_start_key/d_key_door/d_door_goal   grid (wall-aware) shortest distances
  stage_span distinct (key,door,beyond) stages spanned by the quantile ladder

Also prints the feature distribution + a quick separability read (do solution-length / t_key/T quantiles
admit distinct easy/hard groups?), the cheap viability signal for Stage C.

Usage: uv run python scripts/build_doorkey_conflict_pool.py --n-pool 2000 --out results/banks/doorkey_conflict_pool.json
"""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from pcrc.envs.minigrid_env import GridStateCodec

# Load the sibling builder script for its env-stepping BFS (solve_context) + stage_of.
_SPEC = importlib.util.spec_from_file_location(
    "_dk_builder", Path(__file__).with_name("build_doorkey_traj_scaffold.py")
)
_DK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DK)
solve_context, stage_of = _DK.solve_context, _DK.stage_of

_WALL = 2


def _grid_dist(grid_types: np.ndarray, src: tuple[int, int], dst: tuple[int, int]) -> int:
    """Wall-aware grid BFS distance (4-connected); doors/keys treated as passable for geometry."""
    W, H = grid_types.shape
    if src == dst:
        return 0
    seen = {src}
    q = deque([(src, 0)])
    while q:
        (x, y), d = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nx < W and 0 <= ny < H) or (nx, ny) in seen:
                continue
            if grid_types[nx, ny] == _WALL:
                continue
            if (nx, ny) == dst:
                return d + 1
            seen.add((nx, ny))
            q.append(((nx, ny), d + 1))
    return -1  # unreachable (shouldn't happen for solvable layouts)


def _locate(grid_types, kind):
    xs, ys = np.where(grid_types == kind)
    return (int(xs[0]), int(ys[0])) if len(xs) else None


def features_for_path(path, start_pos) -> dict:
    """Extract solution features from a BFS path of serialized states."""
    T = len(path) - 1  # path[0] is the reset; index T is goal-adjacent
    stages = [stage_of(s) for s in path]  # (has_key, door_open, beyond_door)
    t_key = next((i for i, st in enumerate(stages) if st[0]), T)
    t_door = next((i for i, st in enumerate(stages) if st[1]), T)
    grid = np.asarray(path[0]["grid"], dtype=np.uint8)
    gt = grid[:, :, 0]
    kpos, dpos, gpos = _locate(gt, 5), _locate(gt, 4), _locate(gt, 8)  # key, door, goal
    d_sk = _grid_dist(gt, tuple(start_pos), kpos) if kpos else -1
    d_kd = _grid_dist(gt, kpos, dpos) if (kpos and dpos) else -1
    d_dg = _grid_dist(gt, dpos, gpos) if (dpos and gpos) else -1
    span = len({(False, False, False), *[s for s in stages]})
    return {
        "T": T, "t_key": t_key, "t_door": t_door, "t_goal": T,
        "d_start_key": d_sk, "d_key_door": d_kd, "d_door_goal": d_dg,
        "stage_span": span,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_doorkey_conflict_pool")
    p.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    p.add_argument("--n-pool", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    seeds = rng.choice(10_000_000, size=a.n_pool, replace=False).tolist()
    env = gym.make(a.env_id, max_steps=int(a.max_steps))
    u = env.unwrapped
    codec = GridStateCodec()
    pool, n_unsolved = [], 0
    try:
        for i, s in enumerate(seeds):
            path = solve_context(env, u, codec, int(s))
            if path is None or len(path) < 2:
                n_unsolved += 1
                continue
            env.reset(seed=int(s))
            start_pos = (int(u.agent_pos[0]), int(u.agent_pos[1]))
            feats = features_for_path(path, start_pos)
            feats.update({"context_seed": int(s)})
            pool.append(feats)
    finally:
        env.close()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"env_id": a.env_id, "seed": a.seed, "contexts": pool}, indent=2))
    print(f"[pool] env={a.env_id} n_pool={a.n_pool} solved={len(pool)} unsolved={n_unsolved} -> {a.out}")

    T = np.array([c["T"] for c in pool])
    tkr = np.array([c["t_key"] / max(1, c["T"]) for c in pool])
    tdr = np.array([c["t_door"] / max(1, c["T"]) for c in pool])
    span = np.array([c["stage_span"] for c in pool])
    print("\nfeature distribution (solved contexts):")
    for name, arr in [("T", T), ("t_key/T", tkr), ("t_door/T", tdr), ("stage_span", span)]:
        q = np.percentile(arr, [10, 35, 50, 65, 90])
        print(f"  {name:11s} min={arr.min():.2f} p10={q[0]:.2f} p35={q[1]:.2f} med={q[2]:.2f} "
              f"p65={q[3]:.2f} p90={q[4]:.2f} max={arr.max():.2f}")

    # cheap separability read: campaign easy (T low, t_key/T<=0.30) vs hard (T high, t_key/T>=0.35)
    T35, T65 = np.percentile(T, [35, 65])
    easy = (T <= T35) & (tkr <= 0.30)
    hard = (T >= T65) & ((tkr >= 0.35) | (tdr >= 0.70))
    print(f"\nproto-group sizes (pre-RL geometry only): easy~{int(easy.sum())}  hard~{int(hard.sum())} "
          f"(of {len(pool)}); need >=75 each (fallback) / >=150 each (target).")
    print(f"  easy median T={np.median(T[easy]):.1f} t_key/T={np.median(tkr[easy]):.2f}; "
          f"hard median T={np.median(T[hard]):.1f} t_key/T={np.median(tkr[hard]):.2f}"
          if easy.any() and hard.any() else "  (one proto-group empty)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
