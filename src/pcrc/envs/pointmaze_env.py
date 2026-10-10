"""PointMaze task bank + environment — a NON-Fetch sparse goal-reaching env.

Duck-typed to the FetchTaskBank / FetchTaskEnvironment interfaces so the existing
SAC+HER loop (and the value-uncertainty curriculum) run on it unchanged.

PointMaze (gymnasium-robotics) is goal-conditioned exactly like Fetch:
``observation`` (x,y,vx,vy), ``achieved_goal`` (x,y), ``desired_goal`` (x,y),
sparse reward, continuous 2-D action. A "context" is a fixed (reset_cell,
goal_cell) pair in the maze; difficulty = maze distance start→goal, which gives
a Push-like heterogeneous low-success band (cross-domain test of the uncertainty
curriculum). Success threshold is 0.45 (half a maze cell) — pass
``reward_distance_threshold=0.45`` to the loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from pcrc.envs.trajectory_scaffold import ScaffoldLadder
from pcrc.registry import ScaffoldSource


def _ensure_gym():
    """Lazily import gymnasium + register gymnasium-robotics envs.

    Kept out of module import so the pure maze-geometry helpers (``_maze_distance``,
    ``_shortest_path``, ``_scaffold_path_index``) are importable without the sim
    stack, which is absent in the CPU ``test`` extra.
    """
    import gymnasium as gym  # noqa: PLC0415
    import gymnasium_robotics  # noqa: PLC0415

    gym.register_envs(gymnasium_robotics)
    return gym


class PointMazeStateCodec:
    """Full MuJoCo state serialize/restore for PointMaze (a
    :class:`~pcrc.envs.trajectory_scaffold.StateCodec`).

    The point has 2 DOF, so the simulator state is just ``qpos`` (x, y) and ``qvel`` (vx, vy).
    ``restore`` writes them back and runs ``mj_forward`` so derived quantities (and the rebuilt
    observation) are consistent. Operates on the unwrapped ``PointMazeEnv``; ``.point_env`` holds
    the MuJoCo ``model``/``data``.
    """

    def serialize(self, u) -> dict:
        d = u.point_env.data
        return {"qpos": [float(v) for v in d.qpos], "qvel": [float(v) for v in d.qvel]}

    def restore(self, u, state) -> None:
        import mujoco  # noqa: PLC0415

        pe = u.point_env
        pe.data.qpos[:] = np.asarray(state["qpos"], dtype=np.float64)
        pe.data.qvel[:] = np.asarray(state["qvel"], dtype=np.float64)
        mujoco.mj_forward(pe.model, pe.data)


@dataclass
class PointMazeContext:
    """One PointMaze task: fixed start + goal cells (+ a difficulty bin)."""

    context_id: int
    reset_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    distance_bin: int  # maze-distance bucket (duck-types FetchTaskContext.distance_bin)
    split: str
    # Fetch has distance/direction/pose difficulty bins; PointMaze difficulty is
    # 1-D (maze distance), so direction_bin/pose_bin are constant 0 — present only
    # to satisfy evaluate_fetch's per-bin reporting (duck-typing).
    direction_bin: int = 0
    pose_bin: int = 0
    # Route group for the Nav-bypass heterogeneity bank (Gate D / breadth): 0 = direct
    # (low detour ratio, off-path scaffold preserves the route), 1 = bypass (high detour
    # ratio, off-path scaffold skips a bottleneck → decreasing transfer). 0 otherwise.
    route_group: int = 0


def _valid_cells(maze_map: list) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r, row in enumerate(maze_map)
        for c, v in enumerate(row)
        if str(v) == "0"  # 0 = open, 1 = wall, "g"/"r" reset/goal markers
    ]


def _maze_distance(maze_map: list, a: tuple[int, int], b: tuple[int, int]) -> int:
    """BFS shortest path length (in cells) between two open cells; -1 if unreachable."""
    rows, cols = len(maze_map), len(maze_map[0])
    seen = {a}
    q = deque([(a, 0)])
    while q:
        (r, c), d = q.popleft()
        if (r, c) == b:
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            in_bounds = 0 <= nr < rows and 0 <= nc < cols
            if in_bounds and str(maze_map[nr][nc]) == "0" and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return -1


def _detour_ratio(maze_map: list, a: tuple[int, int], b: tuple[int, int]) -> float:
    """``geodesic(a,b) / euclidean(a,b)`` — the route's "windiness".

    ≈1 ⇒ a straight corridor (off-path scaffold ≈ geodesic, *preserves* the route);
    ≫1 ⇒ the maze forces a big detour around a bottleneck, so a straight-line
    (off-path) scaffold reset lands *past* the bottleneck and *bypasses* it. This is
    the route-group label for the Nav-bypass heterogeneity bank (Gate D / breadth).
    """
    g = _maze_distance(maze_map, a, b)
    if g < 0:
        return float("inf")
    eu = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    return g / max(eu, 1e-9)


def _offpath_scaffold_cell(
    maze_map: list,
    a: tuple[int, int],
    b: tuple[int, int],
    alpha: float,
    cells: list[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Reset cell a fraction ``alpha`` along the STRAIGHT LINE a→b, snapped to the
    nearest open cell (≠ goal). Unlike the geodesic reset this ignores walls, so for a
    high-detour ("bypass") context a high-``alpha`` reset lands past the bottleneck on
    the goal side — the scaffolded policy never learns the detour the true start needs
    (a genuine *decreasing* transfer profile). ``alpha<=0`` ⇒ the true start.
    """
    if alpha <= 0.0:
        return tuple(a)
    cells = cells if cells is not None else _valid_cells(maze_map)
    pr = a[0] + alpha * (b[0] - a[0])
    pc = a[1] + alpha * (b[1] - a[1])
    best, best_d = None, float("inf")
    for r, c in cells:
        if (r, c) == tuple(b):  # never reset on the goal cell (clamp like the geodesic path)
            continue
        d = (r - pr) ** 2 + (c - pc) ** 2
        if d < best_d:
            best_d, best = d, (r, c)
    return best if best is not None else tuple(a)


def _shortest_path(
    maze_map: list, a: tuple[int, int], b: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """BFS shortest path of cells from ``a`` to ``b`` inclusive; ``None`` if unreachable.

    PCRC scaffold levels reset the agent a fraction ``α`` of the way along this
    path (geodesic reset distance, framing doc §15.1). ``path[0] == a`` (true
    start) and ``path[-1] == b`` (goal); length is ``_maze_distance(a, b) + 1``.
    """
    if a == b:
        return [a]
    rows, cols = len(maze_map), len(maze_map[0])
    prev: dict[tuple[int, int], tuple[int, int]] = {a: a}
    q = deque([a])
    while q:
        cur = q.popleft()
        if cur == b:
            path = [b]
            while path[-1] != a:
                path.append(prev[path[-1]])
            path.reverse()
            return path
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            in_bounds = 0 <= nr < rows and 0 <= nc < cols
            if in_bounds and str(maze_map[nr][nc]) == "0" and (nr, nc) not in prev:
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    return None


def _scaffold_path_index(path_len: int, start_progress: float) -> int:
    """Index into a length-``path_len`` shortest path for scaffold fraction ``α``.

    ``α<=0`` → index 0 (true start). Otherwise the geodesic fraction, **clamped to
    ``path_len-2``** so a scaffold reset never lands on the goal cell (the last
    path index) — short paths collapse toward the start instead (plan §3.2).
    """
    if path_len < 2 or start_progress <= 0.0:
        return 0
    return min(int(round(start_progress * (path_len - 1))), path_len - 2)


class PointMazeTaskBank:
    """Duck-typed FetchTaskBank for PointMaze: fixed (start, goal) contexts."""

    def __init__(self, contexts: list[PointMazeContext], env_id: str):
        """Build the bank from explicit contexts (use ``build`` to sample them)."""
        self.env_id = env_id
        self.contexts = {c.context_id: c for c in contexts}
        self._train = sorted(c.context_id for c in contexts if c.split == "train")
        self._held = sorted(c.context_id for c in contexts if c.split == "held_out")

    def __len__(self) -> int:
        """Number of contexts (train + held_out)."""
        return len(self.contexts)

    def __getitem__(self, context_id: int) -> PointMazeContext:
        """Context lookup by id (duck-types FetchTaskBank.__getitem__)."""
        return self.contexts[context_id]

    def train_ids(self) -> list[int]:
        """Sorted training context ids."""
        return list(self._train)

    def held_out_ids(self) -> list[int]:
        """Sorted held-out context ids."""
        return list(self._held)

    def save(self, path: Path) -> None:
        """Serialize contexts (+ route_group) to JSON so a strata bank is reusable + analyzable."""
        data = {
            "env_id": self.env_id,
            "schema": "pointmaze-1",
            "contexts": [
                {
                    "context_id": c.context_id,
                    "reset_cell": list(c.reset_cell),
                    "goal_cell": list(c.goal_cell),
                    "distance_bin": c.distance_bin,
                    "split": c.split,
                    "direction_bin": c.direction_bin,
                    "pose_bin": c.pose_bin,
                    "route_group": c.route_group,
                }
                for c in self.contexts.values()
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> PointMazeTaskBank:
        """Load a bank saved by :meth:`save`."""
        data = json.loads(Path(path).read_text())
        ctxs = [
            PointMazeContext(
                context_id=int(c["context_id"]),
                reset_cell=tuple(c["reset_cell"]),
                goal_cell=tuple(c["goal_cell"]),
                distance_bin=int(c["distance_bin"]),
                split=c["split"],
                direction_bin=int(c.get("direction_bin", 0)),
                pose_bin=int(c.get("pose_bin", 0)),
                route_group=int(c.get("route_group", 0)),
            )
            for c in data["contexts"]
        ]
        return cls(ctxs, data["env_id"])

    @classmethod
    def build_strata(
        cls,
        env_id: str = "PointMaze_Large-v3",
        *,
        n_per_group: int = 75,
        n_held_per_group: int = 25,
        direct_max: float = 1.3,
        bypass_min: float = 2.0,
        min_distance: int = 2,
        direct_min_distance: int = 1,
        bypass_max_distance: int | None = None,
        bypass_sharp_alpha: float | None = None,
        bypass_sharp_max_frac: float = 0.45,
        seed: int = 0,
    ) -> PointMazeTaskBank:
        """Bimodal route-group bank for the Nav-bypass conflict experiment (Gate D / breadth).

        Enumerates all (start, goal) pairs, labels **direct** (detour ratio < ``direct_max``) vs
        **bypass** (ratio ≥ ``bypass_min``), drops the middle, and takes ``n_per_group`` of each
        (``n_held_per_group`` held). The per-group distance bounds (``direct_min_distance``,
        ``bypass_max_distance``) make direct LONGER (needs scaffold → increasing transfer) and bypass
        SHORTER (learnable at the true start, but hurt by the off-path scaffold that skips its detour
        → decreasing transfer) so the two genuinely conflict. Train ids are 0..N_train-1 (loop
        invariant); group recoverable via ``route_group`` (0=direct, 1=bypass). Use with
        ``scaffold_geometry="offpath"``.
        """
        gym = _ensure_gym()
        probe = gym.make(env_id)
        maze_map = probe.unwrapped.maze.maze_map
        probe.close()
        cells = _valid_cells(maze_map)

        def _bypassed_at(a, b, alpha, d):
            # off-path reset at `alpha` is geodesically near the goal (it short-circuited the
            # detour) ⇒ training there never teaches the true-start route → catastrophic.
            oc = _offpath_scaffold_cell(maze_map, a, b, alpha, cells)
            return _maze_distance(maze_map, oc, b) <= bypass_sharp_max_frac * d

        direct, bypass = [], []
        for a in cells:
            for b in cells:
                if a == b:
                    continue
                d = _maze_distance(maze_map, a, b)
                if d < min_distance:
                    continue
                ratio = _detour_ratio(maze_map, a, b)
                if ratio < direct_max and d >= direct_min_distance:
                    direct.append((a, b, d))
                elif ratio >= bypass_min and (
                    bypass_max_distance is None or d <= bypass_max_distance
                ):
                    # optional sharpness gate: bypassed already at the lowest non-zero scaffold,
                    # so NO scaffold level is a workable compromise (Gate D sharper conflict).
                    if bypass_sharp_alpha is not None and not _bypassed_at(
                        a, b, bypass_sharp_alpha, d
                    ):
                        continue
                    bypass.append((a, b, d))
        if len(direct) < n_per_group or len(bypass) < n_per_group:
            raise ValueError(
                f"insufficient pairs: direct {len(direct)}, bypass {len(bypass)}; "
                f"need {n_per_group} each (relax cuts or use a larger maze)"
            )
        rng = np.random.default_rng(seed)
        rng.shuffle(direct)
        rng.shuffle(bypass)
        n_train_g = n_per_group - n_held_per_group
        tagged = []  # (a, b, split, route_group)
        for group_id, pairs in ((0, direct[:n_per_group]), (1, bypass[:n_per_group])):
            for k, (a, b, _d) in enumerate(pairs):
                split = "train" if k < n_train_g else "held_out"
                tagged.append((a, b, split, group_id))
        # train block (ids 0..) then held block, preserving the loop invariant
        ordered = [t for t in tagged if t[2] == "train"] + [t for t in tagged if t[2] == "held_out"]
        ctxs = [
            PointMazeContext(
                context_id=k,
                reset_cell=tuple(a),
                goal_cell=tuple(b),
                distance_bin=0,
                split=split,
                route_group=group_id,
            )
            for k, (a, b, split, group_id) in enumerate(ordered)
        ]
        return cls(ctxs, env_id)

    @classmethod
    def build(
        cls,
        env_id: str = "PointMaze_Medium-v3",
        *,
        n_train: int = 100,
        n_held: int = 50,
        seed: int = 0,
        min_distance: int | None = None,
        max_distance: int | None = None,
        pool_candidates: int | None = None,
    ) -> PointMazeTaskBank:
        """Sample (reset, goal) pairs across the maze, binned by start→goal distance.

        ``min_distance``/``max_distance`` restrict to a geodesic-distance BAND (challenging-but-
        learnable breadth task, e.g. PointMaze Large's 60–90th-percentile band); ``None`` = no bound
        (the original unbanded behavior is preserved bit-for-bit when both are ``None``).
        ``pool_candidates`` sets how many raw pairs to sample before dedup/band-filter (default
        ``20·need``; raise it so a narrow band still yields ``need`` unique pairs).
        """
        gym = _ensure_gym()
        probe = gym.make(env_id)
        maze_map = probe.unwrapped.maze.maze_map
        probe.close()
        cells = _valid_cells(maze_map)
        rng = np.random.default_rng(seed)
        need = n_train + n_held
        banded = min_distance is not None or max_distance is not None
        n_pool = pool_candidates if pool_candidates is not None else 20 * need
        lo = min_distance if min_distance is not None else 1
        hi = max_distance if max_distance is not None else 10**9
        # candidate (start, goal) pairs with a real path of length >= 1 (and within the band if set)
        pairs = []
        for _ in range(n_pool):
            i, j = rng.integers(0, len(cells)), rng.integers(0, len(cells))
            if i == j:
                continue
            a, b = cells[i], cells[j]
            d = _maze_distance(maze_map, a, b)
            if d >= max(1, lo) and d <= hi:
                pairs.append((a, b, d))
        # dedup, sort by distance for stratified bins
        uniq = {(a, b): d for a, b, d in pairs}
        pairs = [(a, b, d) for (a, b), d in uniq.items()]
        rng.shuffle(pairs)
        if len(pairs) < need:
            raise ValueError(
                f"only {len(pairs)} unique in-band pairs for {env_id} band [{min_distance},"
                f"{max_distance}] from {n_pool} candidates; need {need} — widen band or raise pool"
            )
        pairs = pairs[:need]
        dists = sorted({d for _, _, d in pairs})
        bin_of = {d: min(4, i * 5 // max(1, len(dists))) for i, d in enumerate(dists)}
        ctxs = []
        for k, (a, b, d) in enumerate(pairs):
            split = "train" if k < n_train else "held_out"
            ctxs.append(
                PointMazeContext(
                    context_id=k,
                    reset_cell=tuple(a),
                    goal_cell=tuple(b),
                    distance_bin=bin_of[d],
                    split=split,
                )
            )
        # reindex so train ids are 0..n_train-1 (loop invariant) — already true by construction
        return cls(ctxs, env_id)


class PointMazeTaskEnvironment:
    """Duck-typed FetchTaskEnvironment for PointMaze."""

    def __init__(
        self,
        bank: PointMazeTaskBank,
        env_id: str,
        max_steps: int | None = None,
        scaffold_geometry: str = "geodesic",
        *,
        scaffold_source: str = ScaffoldSource.MANUAL,
        scaffold_ladder: ScaffoldLadder | None = None,
    ):
        """Wrap a gymnasium PointMaze env bound to ``bank``'s contexts.

        ``scaffold_geometry``: ``"geodesic"`` (default — reset α along the shortest
        path, preserves the route) or ``"offpath"`` (reset α along the straight line,
        snapped to the nearest open cell — bypasses bottlenecks for high-detour
        contexts; the Nav-bypass heterogeneity setting, Gate D / breadth).
        """
        if scaffold_geometry not in ("geodesic", "offpath"):
            raise ValueError(
                f"scaffold_geometry must be geodesic|offpath; got {scaffold_geometry!r}"
            )
        self._bank = bank
        self._scaffold_geometry = scaffold_geometry
        gym = _ensure_gym()
        self._env = gym.make(
            env_id, continuing_task=False, reset_target=False, max_episode_steps=max_steps or 300
        )
        self._ctx: PointMazeContext | None = None
        # Maze layout + per-context shortest paths (for PCRC geodesic scaffold
        # resets). Cached lazily on first scaffolded reset of each context.
        self._maze_map = self._env.unwrapped.maze.maze_map
        self._cells = _valid_cells(self._maze_map)  # for off-path snapping
        self._path_cache: dict[int, list[tuple[int, int]] | None] = {}
        # Scaffold state source: "manual" runs the geodesic reset-cell operator; "trajectory"
        # restores a serialized MuJoCo snapshot per (context, level) from an extracted ladder
        # (Automatic_Trajectory_Scaffold_Execution_Plan.md). The controller is identical for both.
        if scaffold_source not in (ScaffoldSource.MANUAL, ScaffoldSource.TRAJECTORY):
            raise ValueError(
                f"scaffold_source must be 'manual' or 'trajectory'; got {scaffold_source!r}"
            )
        if scaffold_source == ScaffoldSource.TRAJECTORY and scaffold_ladder is None:
            raise ValueError("scaffold_source='trajectory' requires scaffold_ladder")
        self._scaffold_source = scaffold_source
        self._ladder = scaffold_ladder
        self._codec = PointMazeStateCodec()

    @property
    def action_space(self) -> Any:
        """Underlying gymnasium action space (2-D continuous)."""
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        """Underlying gymnasium goal-conditioned Dict observation space."""
        return self._env.observation_space

    def reset(self, context_id: int, start_progress: float = 0.0) -> dict[str, Any]:
        """Reset to ``context_id``'s (reset, goal) cells; return tagged obs dict.

        ``start_progress`` is the PCRC scaffold fraction ``α``: ``0`` is the true
        start (bit-identical to the unscaffolded reset), ``α>0`` resets the agent
        a fraction ``α`` of the way along the shortest path toward the goal. The
        goal cell is never changed; eval always passes ``α=0``. Mirrors the Fetch
        ``reset(context_id, start_progress=…)`` interface so the SAC+HER loop is
        env-agnostic.
        """
        ctx = self._bank.contexts[context_id]
        self._ctx = ctx
        if self._scaffold_source == ScaffoldSource.TRAJECTORY:
            obs = self._reset_from_trajectory(context_id, ctx, float(start_progress))
        else:
            reset_cell = self._scaffold_reset_cell(context_id, ctx, float(start_progress))
            obs, _ = self._env.reset(
                options={
                    "goal_cell": np.array(ctx.goal_cell),
                    "reset_cell": np.array(reset_cell),
                }
            )
        out = {k: np.asarray(v, dtype=np.float64) for k, v in obs.items()}
        out["source_context_id"] = context_id
        return out

    def _reset_from_trajectory(
        self, context_id: int, ctx: PointMazeContext, start_progress: float
    ) -> dict[str, Any]:
        """True-start reset (fixes the goal), then restore the ``(context, level)`` snapshot.

        Level 0 returns the unscaffolded reset. For level > 0 we restore the serialized MuJoCo
        state and rebuild ``observation`` / ``achieved_goal`` from the new pose (PointMaze
        observation == ``[qpos, qvel]``); ``desired_goal`` is unchanged.
        """
        obs, _ = self._env.reset(
            options={
                "goal_cell": np.array(ctx.goal_cell),
                "reset_cell": np.array(ctx.reset_cell),
            }
        )
        level = self._recover_level(start_progress)
        if level == 0:
            return obs
        cell = self._ladder.get(context_id, level)
        self._codec.restore(self._env.unwrapped, cell.serialized_state)
        data = self._env.unwrapped.point_env.data
        obs = dict(obs)
        obs["observation"] = np.concatenate([np.asarray(data.qpos), np.asarray(data.qvel)])
        obs["achieved_goal"] = np.asarray(data.qpos[:2])
        return obs

    def _recover_level(self, start_progress: float) -> int:
        """Nearest scaffold level to the emitted fraction, over the ladder's own quantiles."""
        q = np.asarray(self._ladder.quantiles, dtype=np.float64)
        return int(np.argmin(np.abs(q - float(start_progress))))

    def _scaffold_reset_cell(
        self, context_id: int, ctx: PointMazeContext, start_progress: float
    ) -> tuple[int, int]:
        """Cell a fraction ``start_progress`` along the start→goal shortest path.

        ``start_progress<=0`` returns the true start cell unchanged. The chosen
        index is **clamped to never land on the goal cell** (``len(path)-2``), so
        short-path contexts (distance 1) collapse to the true start rather than a
        trivially-already-success reset (plan §3.2).
        """
        if start_progress <= 0.0:
            return ctx.reset_cell
        if self._scaffold_geometry == "offpath":
            return _offpath_scaffold_cell(
                self._maze_map,
                tuple(ctx.reset_cell),
                tuple(ctx.goal_cell),
                start_progress,
                cells=self._cells,
            )
        if context_id not in self._path_cache:
            self._path_cache[context_id] = _shortest_path(
                self._maze_map, tuple(ctx.reset_cell), tuple(ctx.goal_cell)
            )
        path = self._path_cache[context_id]
        if path is None or len(path) < 2:
            return ctx.reset_cell
        return path[_scaffold_path_index(len(path), start_progress)]

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step the env; return obs with reward mapped to the Fetch 0/-1 convention."""
        act = np.asarray(action, dtype=np.float32)
        obs, _reward, terminated, truncated, info = self._env.step(act)
        out = {k: np.asarray(v, dtype=np.float64) for k, v in obs.items()}
        # Convert to Fetch sparse convention (0 / -1 at the 0.45 cell threshold) so the
        # env reward is consistent with the loop's HER relabel reward_fn. info["is_success"]
        # is set for downstream success accounting.
        d = float(np.linalg.norm(out["achieved_goal"] - out["desired_goal"]))
        success = bool(info.get("success", d < 0.45))
        info = {**info, "is_success": success}
        return out, (0.0 if success else -1.0), bool(terminated), bool(truncated), info

    def close(self) -> None:
        """Close the underlying gymnasium env."""
        self._env.close()
