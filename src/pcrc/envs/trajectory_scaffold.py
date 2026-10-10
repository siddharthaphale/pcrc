"""Automatic trajectory-derived scaffold construction (Stage 0).

Replaces hand-written per-env scaffold operators with a generic extractor: given one successful
reference trajectory per training context, select full simulator states at fixed quantiles of
cumulative trajectory progress and use them as ordered training resets. The PCRC controller is
unchanged — it still emits ``(context, level, fraction)`` and the env maps ``fraction -> level``
and restores ``serialized_state`` instead of running the manual operator.

Plan: docs/Automatic_Trajectory_Scaffold_Execution_Plan.md (§1 architecture, §2.1 artifact, §3 Stage 0).

This module is deliberately **env-agnostic and sim-free** (CPU-testable): it operates on already
serialized frames. Per-env serialization/restore lives behind the :class:`StateCodec` protocol
(implemented in each ``*_env.py``); per-env progress metrics live in the builder scripts. Level 0
(quantile 0.0) is *always* the original reset — its ``serialized_state`` is ``None``, which signals
the env to use its normal unscaffolded reset (bit-identical to true start).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_RESET_HASH = "reset"  # state_hash sentinel for level 0 (original reset, serialized_state is None)


@runtime_checkable
class StateCodec(Protocol):
    """Per-env full-state serialize/restore. Implemented in each ``*_env.py``.

    ``serialize`` reads the *current* simulator state into a JSON-able dict; ``restore`` writes a
    previously serialized dict back onto a freshly-reset env. ``restore(u, serialize(u))`` must be a
    no-op on observations (determinism test), and both operate on the *unwrapped* gym env.
    """

    def serialize(self, env_unwrapped: Any) -> dict: ...

    def restore(self, env_unwrapped: Any, state: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class TrajectoryFrame:
    """One step of a reference trajectory: a restorable state + its cumulative progress.

    ``cumulative_progress`` is monotonically non-decreasing along the trajectory. It may be raw
    (e.g. cumulative arc length / event count); the builder normalizes by the final frame so the
    last frame sits at 1.0. ``frames[0]`` is the trajectory start ``s_0`` (progress 0).
    """

    serialized_state: dict
    cumulative_progress: float


@dataclass(frozen=True)
class ScaffoldCell:
    """One ``(context, level)`` rung of a scaffold ladder (artifact schema, execution-plan §2.1).

    ``serialized_state`` is the single (or primary) restorable state for this rung. The optional
    ``serialized_states`` carries N candidate states from N independent reference trajectories
    (Stage-4 trajectory-count ablation); when present, the env samples one uniformly per reset via
    :meth:`ScaffoldLadder.sample_state`. ``serialized_state`` mirrors ``serialized_states[0]`` so the
    single-trajectory artifact format (and all readers of ``.serialized_state``) is unchanged.
    """

    context_id: int
    level: int
    progress_quantile: float
    trajectory_index: int
    serialized_state: dict | None  # None == use the env's original reset (level 0)
    state_hash: str
    validation_status: str = "unvalidated"  # set by the offline validation harness (C6)
    serialized_states: list[dict] | None = None  # N>1 candidates (Stage-4 ablation); else None

    def to_dict(self) -> dict:
        d = {
            "context_id": self.context_id,
            "level": self.level,
            "progress_quantile": self.progress_quantile,
            "trajectory_index": self.trajectory_index,
            "serialized_state": self.serialized_state,
            "state_hash": self.state_hash,
            "validation_status": self.validation_status,
        }
        if self.serialized_states is not None:  # keep single-traj artifacts byte-identical
            d["serialized_states"] = self.serialized_states
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ScaffoldCell:
        return cls(
            context_id=int(d["context_id"]),
            level=int(d["level"]),
            progress_quantile=float(d["progress_quantile"]),
            trajectory_index=int(d["trajectory_index"]),
            serialized_state=d["serialized_state"],
            state_hash=str(d["state_hash"]),
            validation_status=str(d.get("validation_status", "unvalidated")),
            serialized_states=d.get("serialized_states"),
        )


def _hash_state(state: Mapping[str, Any] | None) -> str:
    """Stable sha256 over a canonicalized serialized state (``_RESET_HASH`` for the level-0 reset)."""
    if state is None:
        return _RESET_HASH
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ScaffoldLadder:
    """Bank-like artifact: serialized reset states at fixed progress quantiles, per train context.

    Indexed by ``(context_id, level)`` where ``level`` is the quantile index (0 == true start).
    Mirrors the existing JSON banks under ``results/banks/`` — save/load + sha256 for provenance.
    """

    env_id: str
    quantiles: tuple[float, ...]
    cells: dict[tuple[int, int], ScaffoldCell] = field(default_factory=dict)

    @property
    def n_levels(self) -> int:
        return len(self.quantiles)

    def context_ids(self) -> list[int]:
        return sorted({cid for (cid, _lvl) in self.cells})

    def get(self, context_id: int, level: int) -> ScaffoldCell:
        try:
            return self.cells[(int(context_id), int(level))]
        except KeyError as e:
            raise KeyError(
                f"no scaffold cell for (context={context_id}, level={level}); "
                f"ladder has contexts {self.context_ids()[:8]}… levels 0..{self.n_levels - 1}"
            ) from e

    def sample_state(self, context_id: int, level: int, rng=None) -> dict | None:
        """Return the restorable state for ``(context, level)``.

        Single-trajectory cell → its ``serialized_state`` (``rng`` unused, so the single-trajectory
        path stays deterministic and byte-identical). Multi-trajectory cell (Stage-4 ablation) →
        one of the N candidate ``serialized_states`` chosen uniformly with ``rng`` (a numpy
        Generator; falls back to the primary state if ``rng`` is None).
        """
        cell = self.get(context_id, level)
        states = cell.serialized_states
        if not states:
            return cell.serialized_state
        if rng is None:
            return states[0]
        return states[int(rng.integers(len(states)))]

    def add_context(self, cells: Sequence[ScaffoldCell]) -> None:
        for c in cells:
            self.cells[(c.context_id, c.level)] = c

    # ----- (de)serialization + hashing (mirrors the existing bank artifacts) -----
    def to_dict(self) -> dict:
        return {
            "schema": "scaffold-ladder-1",
            "env_id": self.env_id,
            "quantiles": list(self.quantiles),
            "cells": [
                c.to_dict()
                for c in sorted(self.cells.values(), key=lambda c: (c.context_id, c.level))
            ],
        }

    def _payload(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def sha256(self) -> str:
        return hashlib.sha256(self._payload().encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self._payload())
        return p

    @classmethod
    def load(cls, path: str | Path) -> ScaffoldLadder:
        d = json.loads(Path(path).read_text())
        lad = cls(env_id=str(d["env_id"]), quantiles=tuple(float(q) for q in d["quantiles"]))
        lad.add_context([ScaffoldCell.from_dict(c) for c in d["cells"]])
        return lad


class ReferenceTrajectoryScaffoldBuilder:
    """Build a :class:`ScaffoldLadder` by selecting trajectory states at fixed progress quantiles.

    For each context, ``build_context`` takes the reference trajectory (a sequence of
    :class:`TrajectoryFrame`) and, for each quantile ``q``, picks the *earliest* frame whose
    normalized cumulative progress reaches ``q``. Level 0 (``q == 0.0``) is the original reset
    (``serialized_state = None``); larger levels select later — more assisted — states.
    """

    def __init__(self, quantiles: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 0.90)):
        q = tuple(float(x) for x in quantiles)
        if len(q) < 1:
            raise ValueError("quantiles must be non-empty")
        if q[0] != 0.0:
            raise ValueError(f"quantiles[0] must be 0.0 (true start); got {q[0]}")
        if any(b < a for a, b in zip(q, q[1:], strict=False)):
            raise ValueError(f"quantiles must be non-decreasing; got {q}")
        if any(not (0.0 <= x <= 1.0) for x in q):
            raise ValueError(f"quantiles must lie in [0, 1]; got {q}")
        self.quantiles = q

    def build_context(
        self, context_id: int, frames: Sequence[TrajectoryFrame]
    ) -> list[ScaffoldCell]:
        """Select one :class:`ScaffoldCell` per quantile from ``context_id``'s reference trajectory."""
        if len(frames) < 1:
            raise ValueError(f"context {context_id}: trajectory must have >= 1 frame")
        prog = [float(f.cumulative_progress) for f in frames]
        if any(b < a for a, b in zip(prog, prog[1:], strict=False)):
            raise ValueError(f"context {context_id}: cumulative_progress must be non-decreasing")
        total = prog[-1]
        # Normalize to [0, 1]. A degenerate (flat, zero-total) trajectory has no progress to
        # distinguish levels, so every assisted level collapses to the start state s_0 (index 0).
        norm = [p / total for p in prog] if total > 0 else None

        cells: list[ScaffoldCell] = []
        for level, q in enumerate(self.quantiles):
            if level == 0 or q == 0.0:
                # Unassisted level: always the original reset (signaled by serialized_state=None).
                cells.append(ScaffoldCell(context_id, level, q, 0, None, _RESET_HASH))
                continue
            idx = 0 if norm is None else self._earliest_index(norm, q)
            state = frames[idx].serialized_state
            cells.append(ScaffoldCell(context_id, level, q, idx, state, _hash_state(state)))
        return cells

    def build_context_multi(
        self, context_id: int, frames_list: Sequence[Sequence[TrajectoryFrame]]
    ) -> list[ScaffoldCell]:
        """Select cells from N independent reference trajectories for ``context_id`` (Stage-4 ablation).

        Runs :meth:`build_context` per trajectory and merges level-by-level: each assisted level's
        cell carries all N candidate states in ``serialized_states`` (primary == trajectory 0's), so
        the env can sample one uniformly per reset. Level 0 stays the original reset (no candidates).
        Trajectories that produced no usable state at a level are simply absent from that level's list.
        """
        if len(frames_list) < 1:
            raise ValueError(f"context {context_id}: need >= 1 reference trajectory")
        per_traj = [self.build_context(context_id, frames) for frames in frames_list]
        merged: list[ScaffoldCell] = []
        for level in range(len(self.quantiles)):
            base = per_traj[0][level]
            if level == 0 or base.serialized_state is None:
                merged.append(base)  # unassisted level: original reset, identical across trajectories
                continue
            states = [pt[level].serialized_state for pt in per_traj if pt[level].serialized_state]
            merged.append(
                ScaffoldCell(
                    context_id=context_id,
                    level=level,
                    progress_quantile=base.progress_quantile,
                    trajectory_index=base.trajectory_index,
                    serialized_state=states[0],
                    state_hash=base.state_hash,
                    serialized_states=states,
                )
            )
        return merged

    @staticmethod
    def _earliest_index(norm_progress: Sequence[float], q: float) -> int:
        """Earliest trajectory index whose normalized cumulative progress reaches ``q``."""
        for i, p in enumerate(norm_progress):
            if p >= q:
                return i
        return len(norm_progress) - 1  # q never reached (numerical) -> last frame
