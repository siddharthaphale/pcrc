"""MiniGrid task bank + environment for PCRC: discrete, partially observed, stage-based scaffold.

Duck-typed to the PCRC controller interface used by the training loop:
``reset(context_id, start_progress) -> flat obs vector``, ``step(action)``, ``action_space``, ``obs_dim``.

A *context* is a layout seed (DoorKey randomizes wall column, door row, key, and start per seed). The
scaffold maps ``start_progress`` to a DoorKey *stage* by manipulating grid state after reset:

* stage 0: original task (no key, locked door, original start) -- evaluation always uses this;
* stage 1: original start, already carrying the key;
* stage 2: carrying the key, standing in front of the (locked) door on the start side;
* stage 3: door open, standing just past the doorway on the goal side;
* stage 4: standing next to the goal.

Level 0 equals the unmodified environment. Reward and success are unchanged (success = reaching the goal).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  -- registers the MiniGrid-* envs with gymnasium on import
import numpy as np

from pcrc.envs.trajectory_scaffold import ScaffoldLadder
from pcrc.registry import ScaffoldSource

N_STAGES = 5


class GridStateCodec:
    """Full MiniGrid state serialize/restore (a :class:`~pcrc.envs.trajectory_scaffold.StateCodec`).

    MiniGrid state is fully captured by ``Grid.encode()`` — one ``(type, color, state)`` triple per
    cell, which preserves walls, the goal, the key, and the door's open/locked ``state`` — plus the
    agent pose and any carried object (held off-grid). ``restore`` rebuilds the grid via
    ``Grid.decode`` and re-attaches the carried object, so a state snapshotted at a trajectory
    quantile is reproduced exactly (the determinism invariant in the Stage-0 tests).
    """

    def serialize(self, u) -> dict:
        carrying = None if u.carrying is None else [int(v) for v in u.carrying.encode()]
        return {
            "grid": u.grid.encode().astype(np.uint8).tolist(),
            "agent_pos": [int(u.agent_pos[0]), int(u.agent_pos[1])],
            "agent_dir": int(u.agent_dir),
            "carrying": carrying,
        }

    def restore(self, u, state) -> None:
        from minigrid.core.grid import Grid  # noqa: PLC0415
        from minigrid.core.world_object import WorldObj  # noqa: PLC0415

        grid, _vis = Grid.decode(np.asarray(state["grid"], dtype=np.uint8))
        u.grid = grid
        u.agent_pos = (int(state["agent_pos"][0]), int(state["agent_pos"][1]))
        u.agent_dir = int(state["agent_dir"])
        if state["carrying"] is None:
            u.carrying = None
        else:
            obj = WorldObj.decode(*(int(v) for v in state["carrying"]))
            if obj is not None:
                obj.cur_pos = np.array([-1, -1])
            u.carrying = obj


@dataclass
class MiniGridContext:
    """One layout: a reset seed fixes the DoorKey grid; duck-types the other task contexts."""

    context_id: int
    seed: int
    split: str


class MiniGridTaskBank:
    """Deterministic bank of DoorKey layout seeds with disjoint train / held-out splits."""

    def __init__(self, contexts: list[MiniGridContext], env_id: str):
        self.env_id = env_id
        self.contexts = {c.context_id: c for c in contexts}
        self._train = sorted(c.context_id for c in contexts if c.split == "train")
        self._held = sorted(c.context_id for c in contexts if c.split == "held_out")

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, cid: int) -> MiniGridContext:
        return self.contexts[cid]

    def train_ids(self) -> list[int]:
        return list(self._train)

    def held_out_ids(self) -> list[int]:
        return list(self._held)

    @classmethod
    def build(
        cls,
        env_id: str = "MiniGrid-DoorKey-8x8-v0",
        *,
        n_train: int = 200,
        n_held: int = 100,
        seed: int = 0,
    ) -> MiniGridTaskBank:
        """Sample distinct layout seeds; the first ``n_train`` are train, the rest held-out (disjoint)."""
        rng = np.random.default_rng(seed)
        seeds = rng.choice(10_000_000, size=n_train + n_held, replace=False).tolist()
        ctxs = [
            MiniGridContext(
                context_id=i, seed=int(s), split=("train" if i < n_train else "held_out")
            )
            for i, s in enumerate(seeds)
        ]
        return cls(ctxs, env_id)

    # ----- (de)serialization + hashing -----
    def _to_dict(self) -> dict:
        return {
            "env_id": self.env_id,
            "schema": "minigrid-1",
            "contexts": [
                {"context_id": c.context_id, "seed": c.seed, "split": c.split}
                for c in sorted(self.contexts.values(), key=lambda c: c.context_id)
            ],
        }

    def _payload(self) -> str:
        return json.dumps(self._to_dict(), sort_keys=True, indent=2)

    def sha256(self) -> str:
        return hashlib.sha256(self._payload().encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self._payload())
        return p

    @classmethod
    def load(cls, path: str | Path) -> MiniGridTaskBank:
        d = json.loads(Path(path).read_text())
        ctxs = [
            MiniGridContext(int(c["context_id"]), int(c["seed"]), str(c["split"]))
            for c in d["contexts"]
        ]
        return cls(ctxs, str(d["env_id"]))


class MiniGridTaskEnvironment:
    """DoorKey wrapper exposing a flat obs vector + the stage scaffold, bound to a bank's contexts."""

    def __init__(
        self,
        bank: MiniGridTaskBank,
        env_id: str,
        max_steps: int | None = None,
        *,
        scaffold_source: str = ScaffoldSource.MANUAL,
        scaffold_ladder: ScaffoldLadder | None = None,
    ):
        self._bank = bank
        self._make_kwargs = {} if max_steps is None else {"max_steps": int(max_steps)}
        self._env = gym.make(env_id, **self._make_kwargs)
        self._u = self._env.unwrapped
        o, _ = self._env.reset(seed=0)
        self.img_shape = tuple(int(v) for v in np.asarray(o["image"]).shape)  # (7, 7, 3) H,W,C
        self.extra_dim = 4 + 1  # direction one-hot + carrying flag
        self.obs_dim = int(np.prod(self.img_shape)) + self.extra_dim  # flat: image + extras
        self.action_space = self._env.action_space  # Discrete(7)
        self.n_actions = int(self.action_space.n)
        self._task = "multiroom" if "MultiRoom" in env_id else "doorkey"
        self._ctx: MiniGridContext | None = None
        # Scaffold state source: "manual" runs the hand-written stage operator (_apply_scaffold);
        # "trajectory" restores a serialized snapshot per (context, level) from an extracted ladder
        # (Automatic_Trajectory_Scaffold_Execution_Plan.md). The controller is identical for both.
        if scaffold_source not in (ScaffoldSource.MANUAL, ScaffoldSource.TRAJECTORY):
            raise ValueError(
                f"scaffold_source must be 'manual' or 'trajectory'; got {scaffold_source!r}"
            )
        if scaffold_source == ScaffoldSource.TRAJECTORY and scaffold_ladder is None:
            raise ValueError("scaffold_source='trajectory' requires scaffold_ladder")
        self._scaffold_source = scaffold_source
        self._ladder = scaffold_ladder
        self._codec = GridStateCodec()
        # Per-env RNG for the Stage-4 multi-trajectory ablation (sample 1-of-N states per reset).
        # Unused for single-trajectory ladders, so the default scaffold path stays deterministic.
        self._traj_rng = np.random.default_rng(0) if scaffold_source == ScaffoldSource.TRAJECTORY else None

    # ----- world-object helpers -----
    def _locate(self):
        from minigrid.core.world_object import Door, Goal, Key  # noqa: PLC0415

        key = door = goal = None
        kpos = dpos = gpos = None
        for y in range(self._u.height):
            for x in range(self._u.width):
                c = self._u.grid.get(x, y)
                if isinstance(c, Key):
                    key, kpos = c, (x, y)
                elif isinstance(c, Door):
                    door, dpos = c, (x, y)
                elif isinstance(c, Goal):
                    goal, gpos = c, (x, y)
        return key, kpos, door, dpos, goal, gpos

    def _free(self, x: int, y: int) -> bool:
        if not (0 <= x < self._u.width and 0 <= y < self._u.height):
            return False
        c = self._u.grid.get(x, y)
        return c is None or c.can_overlap()

    def _place(self, x: int, y: int, direction: int) -> bool:
        if self._free(x, y):
            self._u.agent_pos = (int(x), int(y))
            self._u.agent_dir = int(direction)
            return True
        return False

    def _apply_scaffold(self, stage: int) -> None:
        """Dispatch to the task-specific scaffold (only called for stage > 0; stage 0 is the true start)."""
        if self._task == "multiroom":
            self._apply_multiroom_scaffold(stage)
        else:
            self._apply_doorkey_scaffold(stage)

    def _apply_multiroom_scaffold(self, stage: int) -> None:
        """Route progress: place the agent at a later room along the room sequence (stage 4 = next to the
        goal). Doors ahead stay closed (unlocked) so the agent still navigates and opens them.
        """
        from minigrid.core.world_object import Goal  # noqa: PLC0415

        u = self._u
        rooms = getattr(u, "rooms", None)
        if not rooms:
            return
        gpos = next(
            (
                (x, y)
                for y in range(u.height)
                for x in range(u.width)
                if isinstance(u.grid.get(x, y), Goal)
            ),
            None,
        )
        if stage >= N_STAGES - 1 and gpos is not None:
            for ax, ay, ad in (
                (gpos[0] - 1, gpos[1], 0),
                (gpos[0], gpos[1] - 1, 1),
                (gpos[0] + 1, gpos[1], 2),
                (gpos[0], gpos[1] + 1, 3),
            ):
                if self._place(ax, ay, ad):
                    return
        target = max(0, min(len(rooms) - 1, round(stage / (N_STAGES - 1) * (len(rooms) - 1))))
        room = rooms[target]
        tx, ty, sw, sh = int(room.top[0]), int(room.top[1]), int(room.size[0]), int(room.size[1])
        cx, cy = tx + sw // 2, ty + sh // 2
        ad = 0
        ex = getattr(room, "exitDoorPos", None)
        if ex is not None:
            dx, dy = int(ex[0]) - cx, int(ex[1]) - cy
            ad = 0 if dx > 0 else 2 if dx < 0 else (1 if dy > 0 else 3)
        if self._place(cx, cy, ad):
            return
        for yy in range(ty + 1, ty + sh - 1):  # fallback: any free interior cell
            for xx in range(tx + 1, tx + sw - 1):
                if self._place(xx, yy, ad):
                    return

    def _apply_doorkey_scaffold(self, stage: int) -> None:
        """Manipulate post-reset grid state to realize the requested DoorKey stage (start side x<door_x)."""
        key, kpos, door, dpos, goal, gpos = self._locate()
        if stage >= 1 and key is not None and kpos is not None:
            self._u.grid.set(kpos[0], kpos[1], None)
            self._u.carrying = key
            key.cur_pos = np.array([-1, -1])
        if stage >= 2 and dpos is not None:
            self._place(dpos[0] - 1, dpos[1], 0)  # in front of the door, facing it (east)
        if stage >= 3 and door is not None and dpos is not None:
            door.is_locked = False
            door.is_open = True
            self._place(dpos[0] + 1, dpos[1], 0)  # just past the doorway on the goal side
        if stage >= 4 and gpos is not None:
            for ax, ay, ad in (
                (gpos[0] - 1, gpos[1], 0),
                (gpos[0], gpos[1] - 1, 1),
                (gpos[0] + 1, gpos[1], 2),
                (gpos[0], gpos[1] + 1, 3),
            ):
                if self._place(ax, ay, ad):
                    break

    def _obs(self) -> np.ndarray:
        """Flat obs = normalized 7x7x3 image + direction one-hot + carrying flag (so the policy can see
        it holds the key). A CNN policy reshapes the leading image block; an MLP uses the flat vector.
        """
        o = self._u.gen_obs()
        img = np.asarray(o["image"], dtype=np.float32).flatten() / 10.0
        d = int(o["direction"])
        dir1h = np.zeros(4, dtype=np.float32)
        dir1h[d] = 1.0
        carry = np.array([1.0 if self._u.carrying is not None else 0.0], dtype=np.float32)
        return np.concatenate([img, dir1h, carry])

    def reset(self, context_id: int, start_progress: float = 0.0) -> np.ndarray:
        """Reset to the context's layout, then apply a scaffold stage.

        ``start_progress`` is the controller fraction in [0, 1]; it maps to a discrete level (==
        DoorKey stage) as ``round(start_progress * (N_STAGES - 1))`` (0 -> true start, 1 -> next to
        the goal). Level 0 is the unmodified reset for *both* sources. For ``scaffold_source="manual"``
        levels >0 run the hand-written stage operator; for ``"trajectory"`` they restore the
        serialized snapshot for ``(context_id, level)`` from the extracted ladder.
        """
        ctx = self._bank[context_id]
        self._ctx = ctx
        # Evenly spaced quantiles -> round(alpha * (N_STAGES-1)) is the robust nearest-index recovery.
        level = int(round(float(start_progress) * (N_STAGES - 1)))
        level = max(0, min(N_STAGES - 1, level))
        self._env.reset(seed=ctx.seed)
        if level > 0:
            if self._scaffold_source == ScaffoldSource.TRAJECTORY:
                self._codec.restore(
                    self._u, self._ladder.sample_state(context_id, level, self._traj_rng)
                )
            else:
                self._apply_scaffold(level)
        return self._obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Episodic step; success = reaching the goal (terminated with positive reward)."""
        _o, r, term, trunc, info = self._env.step(int(action))
        success = bool(term and r > 0)
        return self._obs(), float(r), bool(term), bool(trunc), {**info, "is_success": success}

    def close(self) -> None:
        self._env.close()
