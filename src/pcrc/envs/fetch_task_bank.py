"""Deterministic Fetch task-context bank for pre-VLA Actor-Curator runs.

A task context captures every piece of state needed to reproduce the
initial sim state and the reward computation deterministically:
``context_id`` is a stable integer index. The bank serializes to a JSON
sidecar with array-as-list encoding plus per-field ``{dtype, shape}``
metadata; ``save`` / ``load`` round-trip yields semantic equality of
every ``FetchTaskContext`` field (not byte-for-byte JSON equality).

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§3.2 (`FetchTaskContext` / `FetchTaskBank` API sketch), §3.3 (env wrapper
reset sequence consumes the bank's ``reset_payload``), §9 risk #9 (the
deterministic-reset caveat lives in the env wrapper, not in the bank).

The bank does NOT touch a live env at construction. It only authors the
deterministic plan a worker will execute when it later instantiates a
``FetchTaskEnvironment``. This keeps the bank cheap to build on CPU and
makes ``save`` / ``load`` self-contained.

Used by:
    - ``pcrc.envs.fetch_environment.FetchTaskEnvironment`` (consumes
      bank rows at ``env.reset(context_id)``)
    - ``pcrc.envs.fetch_calibration`` (rolls a starter policy on every
      train-bank context to classify into easy/medium/hard bins)
    - the MLP curator (``features(context_id)`` is the feature vector)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pcrc.utils.atomic_io import atomic_write_bytes

# Schema-version tag stored in the JSON sidecar. Bump together with any
# breaking change to ``FetchTaskContext`` so ``load`` can refuse a stale
# sidecar instead of silently mis-mapping fields.
SCHEMA_VERSION = 1

# Known keys for ``FetchTaskContext.reset_payload``. The wrapper applies
# each present key during ``env.reset(context_id)`` (see §3.3 reset
# sequence). Unknown keys raise so future env variants must extend the
# wrapper explicitly rather than silently dropping state.
KNOWN_RESET_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "init_qpos",
        "init_qvel",
        "mocap_pos",
        "mocap_quat",
        "object_pose",
        "object_vel",
        "goal",
    }
)


@dataclass(frozen=True)
class FetchTaskContext:
    """One deterministic Fetch task context.

    The fields cleanly separate (a) what selects the context
    (``context_id``), (b) what makes the reset reproducible
    (``env_id``, ``reset_seed``, ``reset_payload``), (c) the
    reward-defining target (``desired_goal``), (d) convenience
    metadata for diagnostics and the MLP curator (``initial_object_pose``,
    bins), and (e) the bank-split tag (``split``).

    Attributes:
        context_id: Stable integer index. Unique per bank.
        env_id: The registered gymnasium env id (e.g.
            ``"FetchReachDense-v4"``). NEVER hardcoded into modules —
            resolved from ``config/fetch_priority_*.yaml`` at bank
            construction (Stage A1 audit (2/3)).
        reset_seed: Seed passed to ``env.reset(seed=...)``. Same
            ``(env_id, reset_seed)`` yields the same starting MuJoCo
            qpos / qvel and the env's default goal (which the wrapper
            then overrides with ``desired_goal`` if it differs).
        desired_goal: Shape ``(3,)``. The reward target; overrides the
            env's freshly-sampled goal during ``env.reset(context_id)``.
        reset_payload: Optional dict carrying env-specific state needed
            for deterministic reset beyond the seed. Known keys:
            ``init_qpos``, ``init_qvel``, ``mocap_pos``, ``mocap_quat``,
            ``object_pose``, ``object_vel``, ``goal``. Unknown keys
            raise during JSON round-trip and at env-reset time. ``None``
            when ``env.reset(seed=reset_seed)`` is sufficient (Reach
            with no goal override beyond the bank's stored
            ``desired_goal`` falls into this case, but the wrapper still
            re-sets the goal post-reset). Extensible without schema
            migration — add keys to ``KNOWN_RESET_PAYLOAD_KEYS`` and the
            wrapper's apply path.
        initial_object_pose: Shape ``(3,)`` xyz of the object pose at
            reset, when applicable (Push / PickAndPlace). ``None`` for
            Reach. Derived from ``reset_payload`` for convenience —
            not the source of truth at reset time. Retained as a typed
            convenience for bin metadata and MLP features.
        distance_bin: 0..(num_distance_bins-1). Bin index from
            calibration; 0 = easiest.
        direction_bin: 0..(num_direction_bins-1).
        pose_bin: 0..(num_pose_bins-1). Push / PickAndPlace only;
            ignored on Reach (set to 0 by convention there).
        split: ``"train"`` or ``"held_out"``. The held-out split is
            never touched during training; eval reads it.
    """

    context_id: int
    env_id: str
    reset_seed: int
    desired_goal: np.ndarray
    reset_payload: dict[str, Any] | None
    initial_object_pose: np.ndarray | None
    distance_bin: int
    direction_bin: int
    pose_bin: int
    split: Literal["train", "held_out"]


# ---------------------------------------------------------------------------
# JSON (de)serialization helpers
# ---------------------------------------------------------------------------


def _ndarray_to_json(arr: np.ndarray) -> dict[str, Any]:
    """Encode a numpy array as ``{dtype, shape, data}``.

    Lossless for the dtypes we use here (float64 desired_goal, float64
    qpos / qvel, float64 object pose). Reconstructed via
    ``_ndarray_from_json``.
    """
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": arr.tolist(),
    }


def _ndarray_from_json(obj: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`_ndarray_to_json`. Validates shape + dtype."""
    arr = np.asarray(obj["data"], dtype=np.dtype(obj["dtype"]))
    expected_shape = tuple(obj["shape"])
    if arr.shape != expected_shape:
        raise ValueError(
            f"FetchTaskBank JSON load: array shape mismatch — "
            f"got {arr.shape}, expected {expected_shape}"
        )
    return arr


def _reset_payload_to_json(
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Encode a reset payload. Validates each key against the known set."""
    if payload is None:
        return None
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if k not in KNOWN_RESET_PAYLOAD_KEYS:
            raise ValueError(
                f"FetchTaskBank: unknown reset_payload key {k!r}. "
                f"Add it to KNOWN_RESET_PAYLOAD_KEYS and extend the env "
                f"wrapper's reset-apply path before serializing it."
            )
        if isinstance(v, np.ndarray):
            out[k] = _ndarray_to_json(v)
        else:
            raise TypeError(
                f"FetchTaskBank: reset_payload[{k!r}] must be a numpy "
                f"ndarray, got {type(v).__name__}"
            )
    return out


def _reset_payload_from_json(
    obj: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Inverse of :func:`_reset_payload_to_json`."""
    if obj is None:
        return None
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if k not in KNOWN_RESET_PAYLOAD_KEYS:
            raise ValueError(f"FetchTaskBank JSON load: unknown reset_payload key {k!r}")
        out[k] = _ndarray_from_json(v)
    return out


def _context_to_json(ctx: FetchTaskContext) -> dict[str, Any]:
    return {
        "context_id": int(ctx.context_id),
        "env_id": ctx.env_id,
        "reset_seed": int(ctx.reset_seed),
        "desired_goal": _ndarray_to_json(ctx.desired_goal),
        "reset_payload": _reset_payload_to_json(ctx.reset_payload),
        "initial_object_pose": (
            None if ctx.initial_object_pose is None else _ndarray_to_json(ctx.initial_object_pose)
        ),
        "distance_bin": int(ctx.distance_bin),
        "direction_bin": int(ctx.direction_bin),
        "pose_bin": int(ctx.pose_bin),
        "split": ctx.split,
    }


def _context_from_json(obj: dict[str, Any]) -> FetchTaskContext:
    split = obj["split"]
    if split not in ("train", "held_out"):
        raise ValueError(f"FetchTaskBank JSON load: invalid split {split!r}")
    return FetchTaskContext(
        context_id=int(obj["context_id"]),
        env_id=str(obj["env_id"]),
        reset_seed=int(obj["reset_seed"]),
        desired_goal=_ndarray_from_json(obj["desired_goal"]),
        reset_payload=_reset_payload_from_json(obj["reset_payload"]),
        initial_object_pose=(
            None
            if obj["initial_object_pose"] is None
            else _ndarray_from_json(obj["initial_object_pose"])
        ),
        distance_bin=int(obj["distance_bin"]),
        direction_bin=int(obj["direction_bin"]),
        pose_bin=int(obj["pose_bin"]),
        split=split,
    )


# ---------------------------------------------------------------------------
# Feature normalization stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FeatureStats:
    """Train-bank mean / std for the MLP curator's continuous features.

    Computed at bank construction over the train split only (never the
    held-out split, else features leak). Stored alongside the bank in
    the JSON sidecar so a reloaded bank produces identical feature
    vectors.

    The stats cover the seven continuous geometry blocks documented in
    :meth:`FetchTaskBank.features` (each is xyz=3 except the scalar
    distance and the unit direction). All seven are concatenated and
    then z-scored elementwise with ``std`` floored at 1e-6 so a
    degenerate single-context bank does not divide by zero.
    """

    mean: np.ndarray  # shape (continuous_feature_dim,)
    std: np.ndarray  # shape (continuous_feature_dim,)


def _continuous_feature_block(
    desired_goal: np.ndarray,
    initial_achieved_goal: np.ndarray,
    initial_object_pose: np.ndarray | None,
) -> np.ndarray:
    """Build the raw (un-normalized) continuous feature block.

    Concatenation order (must match :meth:`FetchTaskBank.features`
    docstring exactly):

        - desired_goal_xyz (3)
        - initial_achieved_goal_xyz (3)
        - desired_goal_xyz - initial_achieved_goal_xyz (3)
        - goal_distance_scalar (1)
        - goal_direction_unit_vector (3)
        - initial_object_pose_xyz (3, zeros for Reach)
        - desired_goal_xyz - initial_object_pose_xyz (3, zeros for Reach)

    Total continuous-feature dim = 19.
    """
    dg = np.asarray(desired_goal, dtype=np.float64).reshape(3)
    ag = np.asarray(initial_achieved_goal, dtype=np.float64).reshape(3)
    diff_dg_ag = dg - ag
    dist = float(np.linalg.norm(diff_dg_ag))
    # Unit direction; degenerate (dist == 0) maps to the zero vector
    # rather than NaN — the bin one-hot still distinguishes degenerate
    # contexts so the curator is not blinded.
    direction = diff_dg_ag / dist if dist > 1e-12 else np.zeros(3, dtype=np.float64)
    if initial_object_pose is None:
        obj_xyz = np.zeros(3, dtype=np.float64)
    else:
        obj_xyz = np.asarray(initial_object_pose, dtype=np.float64).reshape(3)
    diff_dg_obj = dg - obj_xyz
    return np.concatenate(
        [
            dg,
            ag,
            diff_dg_ag,
            np.array([dist], dtype=np.float64),
            direction,
            obj_xyz,
            diff_dg_obj,
        ]
    )


CONTINUOUS_FEATURE_DIM: int = 19  # see _continuous_feature_block


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------


@dataclass
class FetchTaskBank:
    """Deterministic, JSON-serializable bank of Fetch task contexts.

    Construction is offline: the caller supplies a list of
    ``FetchTaskContext`` objects (typically from
    ``fetch_calibration``-driven sampling), plus the train-bank
    ``initial_achieved_goal`` per context for feature normalization.
    The bank then computes and freezes the feature-normalization stats
    over the train split.

    Direct construction via ``FetchTaskBank.from_contexts(...)`` is the
    canonical entry point. The legacy ``__init__(env_id, num_train,
    num_held_out, seed)`` signature is intentionally NOT implemented —
    that would couple the bank to a live env at construction time and
    duplicate the calibration runner's responsibility (which is to
    decide which sampled contexts actually qualify as
    easy/medium/hard).

    Attributes:
        env_id: The dominant env id across the bank. Mixed-env banks
            (Reach + Push + PickAndPlace) carry the per-context
            ``env_id`` on each ``FetchTaskContext`` and set
            ``self.env_id`` to ``"mixed"``.
        contexts: dict ``context_id -> FetchTaskContext``.
        feature_stats: train-bank z-score stats applied to continuous
            features in :meth:`features`.
        initial_achieved_goals: per-context initial-EE-position xyz,
            stored alongside the bank for reproducible feature
            generation. Same key set as ``contexts``.
    """

    env_id: str
    contexts: dict[int, FetchTaskContext]
    feature_stats: _FeatureStats
    initial_achieved_goals: dict[int, np.ndarray]
    # Set of env ids one-hot-encoded in :meth:`features`. Empty for a
    # single-env bank (no env one-hot appended); populated for mixed
    # banks. Stored explicitly so a loaded bank reproduces the same
    # one-hot encoding the original used.
    env_id_onehot_keys: tuple[str, ...] = field(default_factory=tuple)

    # ----- construction -----

    @classmethod
    def from_contexts(
        cls,
        contexts: list[FetchTaskContext],
        initial_achieved_goals: dict[int, np.ndarray],
    ) -> FetchTaskBank:
        """Construct from a list of contexts plus per-context initial EE poses.

        Validates: unique ``context_id``, ``initial_achieved_goals``
        covers every context, splits are exactly ``"train"`` /
        ``"held_out"``, train split is non-empty (else feature stats
        are degenerate).

        Computes feature stats over the train split only.
        """
        if not contexts:
            raise ValueError("FetchTaskBank.from_contexts: contexts is empty")
        ids = [c.context_id for c in contexts]
        if len(set(ids)) != len(ids):
            raise ValueError("FetchTaskBank.from_contexts: duplicate context_id in contexts")
        ctx_by_id = {c.context_id: c for c in contexts}
        missing = set(ctx_by_id) - set(initial_achieved_goals)
        if missing:
            raise ValueError(
                f"FetchTaskBank.from_contexts: initial_achieved_goals "
                f"missing context_ids {sorted(missing)}"
            )

        train_ids = [c.context_id for c in contexts if c.split == "train"]
        if not train_ids:
            raise ValueError(
                "FetchTaskBank.from_contexts: train split is empty — "
                "cannot compute feature normalization stats"
            )

        # Decide single-env vs mixed-env. Mixed banks one-hot the env id.
        env_ids = sorted({c.env_id for c in contexts})
        if len(env_ids) == 1:
            bank_env_id = env_ids[0]
            onehot_keys: tuple[str, ...] = ()
        else:
            bank_env_id = "mixed"
            onehot_keys = tuple(env_ids)

        # Compute train-split feature stats.
        train_blocks = np.stack(
            [
                _continuous_feature_block(
                    desired_goal=ctx_by_id[cid].desired_goal,
                    initial_achieved_goal=initial_achieved_goals[cid],
                    initial_object_pose=ctx_by_id[cid].initial_object_pose,
                )
                for cid in train_ids
            ]
        )
        mean = train_blocks.mean(axis=0)
        std = train_blocks.std(axis=0)
        # Floor std so degenerate single-context or near-constant
        # features do not divide by zero. 1e-6 is well below MuJoCo
        # geometry scale (~1m workspace).
        std = np.maximum(std, 1e-6)
        feature_stats = _FeatureStats(mean=mean, std=std)

        # Copy initial_achieved_goals into a fresh dict keyed by int
        # for symmetry with ``contexts``.
        ee = {
            int(k): np.asarray(v, dtype=np.float64).reshape(3)
            for k, v in initial_achieved_goals.items()
        }

        return cls(
            env_id=bank_env_id,
            contexts=ctx_by_id,
            feature_stats=feature_stats,
            initial_achieved_goals=ee,
            env_id_onehot_keys=onehot_keys,
        )

    # ----- container interface -----

    def __len__(self) -> int:
        """Return total number of contexts (train + held_out)."""
        return len(self.contexts)

    def __getitem__(self, context_id: int) -> FetchTaskContext:
        """Look up a context by id; raises ``KeyError`` if absent."""
        return self.contexts[context_id]

    def __contains__(self, context_id: object) -> bool:
        """Membership test on context_id."""
        return context_id in self.contexts

    def train_ids(self) -> list[int]:
        """Sorted list of train-split context ids."""
        return sorted(cid for cid, c in self.contexts.items() if c.split == "train")

    def held_out_ids(self) -> list[int]:
        """Sorted list of held-out-split context ids."""
        return sorted(cid for cid, c in self.contexts.items() if c.split == "held_out")

    # ----- featurization -----

    @property
    def feature_dim(self) -> int:
        """Total feature-vector length (continuous + bin one-hots + env one-hot).

        Bin one-hot widths are inferred from the maximum bin index in
        the bank, plus 1. Mixed-env banks add ``len(env_id_onehot_keys)``.
        """
        max_dist = max(c.distance_bin for c in self.contexts.values()) + 1
        max_dir = max(c.direction_bin for c in self.contexts.values()) + 1
        max_pose = max(c.pose_bin for c in self.contexts.values()) + 1
        return CONTINUOUS_FEATURE_DIM + max_dist + max_dir + max_pose + len(self.env_id_onehot_keys)

    def features(self, context_id: int) -> np.ndarray:
        """Feature vector for the MLP curator at ``context_id``.

        Concatenation order (resolved schema — see plan §3.2):

            - desired_goal_xyz (3)
            - initial_achieved_goal_xyz (3)
            - desired_goal_xyz - initial_achieved_goal_xyz (3)
            - goal_distance_scalar (1)
            - goal_direction_unit_vector (3)
            - initial_object_pose_xyz (3, zeros for Reach)
            - desired_goal_xyz - initial_object_pose_xyz (3, zeros for Reach)
            - distance_bin one-hot
            - direction_bin one-hot
            - pose_bin one-hot
            - env_id one-hot (only on mixed-env banks)

        The continuous block (first 19 entries) is z-scored using the
        train-bank mean / std frozen at construction. Bin one-hot widths
        are taken from the bank's maximum bin index per axis.

        NEVER returns ``context_id``, ``reset_seed``, or any
        unique-per-context hash — those would let the MLP curator
        memorize. Tested explicitly in
        ``tests/cpu/envs/test_fetch_task_bank_features.py``.

        Raises:
            KeyError: ``context_id`` is not in the bank.
        """
        ctx = self.contexts[context_id]
        ag = self.initial_achieved_goals[context_id]
        raw = _continuous_feature_block(
            desired_goal=ctx.desired_goal,
            initial_achieved_goal=ag,
            initial_object_pose=ctx.initial_object_pose,
        )
        cont = (raw - self.feature_stats.mean) / self.feature_stats.std

        max_dist = max(c.distance_bin for c in self.contexts.values()) + 1
        max_dir = max(c.direction_bin for c in self.contexts.values()) + 1
        max_pose = max(c.pose_bin for c in self.contexts.values()) + 1
        dist_oh = np.zeros(max_dist, dtype=np.float64)
        dist_oh[ctx.distance_bin] = 1.0
        dir_oh = np.zeros(max_dir, dtype=np.float64)
        dir_oh[ctx.direction_bin] = 1.0
        pose_oh = np.zeros(max_pose, dtype=np.float64)
        pose_oh[ctx.pose_bin] = 1.0

        blocks = [cont, dist_oh, dir_oh, pose_oh]
        if self.env_id_onehot_keys:
            env_oh = np.zeros(len(self.env_id_onehot_keys), dtype=np.float64)
            env_oh[self.env_id_onehot_keys.index(ctx.env_id)] = 1.0
            blocks.append(env_oh)
        return np.concatenate(blocks)

    # ----- (de)serialization -----

    def save(self, path: Path) -> Path:
        """Atomic JSON write of the bank to ``path``.

        Schema (top-level keys):
            - ``schema_version``: int, matches ``SCHEMA_VERSION`` at
              author time.
            - ``env_id``: str.
            - ``env_id_onehot_keys``: list[str].
            - ``feature_stats``: ``{mean, std}`` each as
              ``{dtype, shape, data}``.
            - ``initial_achieved_goals``: dict[str(context_id),
              ndarray-json] — keys stringified for JSON compatibility.
            - ``contexts``: list[ndarray-json-context] sorted by
              ``context_id``.

        Uses ``sort_keys=True`` so dict iteration order is stable. The
        ``contexts`` array is explicitly pre-sorted. ``np.ndarray``
        fields are encoded via :func:`_ndarray_to_json`. Tests assert
        **semantic** round-trip equality (np.array_equal for arrays,
        == for scalars) — byte-for-byte JSON equality is not asserted
        because dict iteration order through ``json.dumps`` can differ
        across Python implementations absent the ``sort_keys`` guarantee
        we already pass.
        """
        path = Path(path)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "env_id": self.env_id,
            "env_id_onehot_keys": list(self.env_id_onehot_keys),
            "feature_stats": {
                "mean": _ndarray_to_json(self.feature_stats.mean),
                "std": _ndarray_to_json(self.feature_stats.std),
            },
            "initial_achieved_goals": {
                str(cid): _ndarray_to_json(self.initial_achieved_goals[cid])
                for cid in sorted(self.initial_achieved_goals)
            },
            "contexts": [_context_to_json(self.contexts[cid]) for cid in sorted(self.contexts)],
        }
        data = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
        return atomic_write_bytes(path, data)

    @classmethod
    def load(cls, path: Path) -> FetchTaskBank:
        """Load a bank from its JSON sidecar at ``path``.

        Refuses to load a sidecar whose ``schema_version`` differs from
        :data:`SCHEMA_VERSION` — bump the constant if you change the
        schema so stale sidecars fail loudly.
        """
        path = Path(path)
        with path.open("r") as f:
            payload = json.load(f)
        sv = int(payload.get("schema_version", -1))
        if sv != SCHEMA_VERSION:
            raise ValueError(
                f"FetchTaskBank.load({path}): schema_version={sv} "
                f"does not match SCHEMA_VERSION={SCHEMA_VERSION}; "
                f"regenerate the bank under the current schema."
            )
        feature_stats = _FeatureStats(
            mean=_ndarray_from_json(payload["feature_stats"]["mean"]),
            std=_ndarray_from_json(payload["feature_stats"]["std"]),
        )
        initial_achieved_goals = {
            int(cid_str): _ndarray_from_json(arr_obj)
            for cid_str, arr_obj in payload["initial_achieved_goals"].items()
        }
        contexts = [_context_from_json(c) for c in payload["contexts"]]
        ctx_by_id = {c.context_id: c for c in contexts}
        return cls(
            env_id=str(payload["env_id"]),
            contexts=ctx_by_id,
            feature_stats=feature_stats,
            initial_achieved_goals=initial_achieved_goals,
            env_id_onehot_keys=tuple(payload.get("env_id_onehot_keys", [])),
        )
