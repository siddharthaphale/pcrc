"""One-shot Fetch task-bank calibration.

Generates a pool of candidate Fetch task contexts, rolls a starter
policy on each, classifies into easy/medium/hard via the starter
success rate (with a dense-return fallback when binary success is too
sparse), drops saturated and dead contexts, assigns metadata bins, and
emits a serialized :class:`FetchTaskBank` plus a calibration summary
table.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§Priority 1a, §9 risk #1 (starter actor protocol — short uniform PPO
warmup; one shared starter checkpoint consumed by every Stage D
condition; not counted in the curriculum-vs-uniform budget),
§9 risk #15 (saturated / dead contexts kill curriculum signal — do
not skip calibration), §10 Stage A1 mid-run (Priority 1a hard gate).

The runner is intentionally policy-agnostic — the caller passes a
``policy_fn(obs) -> action`` callable. For Stage A1 unit tests the
caller passes a deterministic stub policy (e.g. zero action). For
Stage A1 mid-run on GPU the caller passes the loaded PPO warmup
checkpoint's forward path. This keeps the runner cheap to test on CPU
without depending on Stage A2's PPO module.

Outputs (written via :class:`pcrc.utils.RunLogger`):
    - ``calibration/per_context.jsonl`` — one row per candidate context
      with starter success rate, mean dense return, mean final goal
      distance, assigned bins, kept / dropped flag, drop reason.
    - ``calibration/summary.json`` — aggregate statistics + bank
      construction parameters + the final
      ``<bank_path>`` on disk.
    - ``calibration/bank.json`` — the serialized :class:`FetchTaskBank`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext

# Type alias for a starter policy: maps an observation dict to an action.
# The runner does NOT enforce greedy vs sampling — the caller decides
# (typically greedy for calibration to remove policy noise from the
# success-rate measurement).
PolicyFn = Callable[[dict[str, np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class CalibrationResult:
    """Per-context aggregate from rolling the starter policy.

    Attributes:
        context_id: Stable integer index assigned at sampling time.
        env_id: The registered env id for this context.
        reset_seed: Seed used during the candidate-pool probe rollout.
        desired_goal: Shape ``(3,)``; the bank's reward target.
        initial_achieved_goal: Shape ``(3,)``; EE / object pose at
            reset under this seed.
        initial_object_pose: Shape ``(3,)`` or ``None``; populated for
            Push / PickAndPlace.
        reset_payload: Dict with at minimum ``init_qpos`` /
            ``init_qvel`` so the bank can reproduce this seed's
            simulator state without depending on future
            gymnasium-robotics versions producing the same mapping.
            ``None`` only for Reach envs where seed-only is sufficient.
        success_rate: starter success rate over
            ``episodes_per_context`` trials.
        mean_dense_return: mean undiscounted return over the same
            trials.
        mean_final_goal_distance: mean L2(achieved_goal_final,
            desired_goal) across the trials.
    """

    context_id: int
    env_id: str
    reset_seed: int
    desired_goal: np.ndarray
    initial_achieved_goal: np.ndarray
    initial_object_pose: np.ndarray | None
    reset_payload: dict[str, Any] | None
    success_rate: float
    mean_dense_return: float
    mean_final_goal_distance: float


# ---------------------------------------------------------------------------
# Candidate pool generation
# ---------------------------------------------------------------------------


def _is_push_or_pick(env_id: str) -> bool:
    """Whether the env id has a Push / PickAndPlace object in scene."""
    return ("Push" in env_id) or ("PickAndPlace" in env_id)


def _probe_pool(env_id: str, n_pool: int, base_seed: int) -> list[CalibrationResult]:
    """Probe the env once per candidate seed to capture initial state.

    Does NOT roll any policy — just one reset per seed to record the
    initial achieved goal, default goal, optional initial object pose,
    and the full ``init_qpos`` / ``init_qvel`` so the bank can
    reproduce the seed-induced state under a future gymnasium-robotics
    version that may sample differently for the same seed.

    Returns a list of partially-populated :class:`CalibrationResult`
    records — the ``success_rate``, ``mean_dense_return``, and
    ``mean_final_goal_distance`` fields are placeholders set to NaN and
    filled in by :func:`_roll_starter`.

    Args:
        env_id: Registered gymnasium env id.
        n_pool: Number of candidate seeds to probe.
        base_seed: Starting seed for ``range(base_seed, base_seed+n_pool)``.
    """
    import gymnasium as gym  # noqa: PLC0415
    import gymnasium_robotics  # noqa: PLC0415

    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id)
    has_object = _is_push_or_pick(env_id)
    out: list[CalibrationResult] = []
    try:
        for i in range(n_pool):
            seed = int(base_seed + i)
            obs, _ = env.reset(seed=seed)
            init_qpos = env.unwrapped.data.qpos.copy()
            init_qvel = env.unwrapped.data.qvel.copy()
            default_goal = np.asarray(env.unwrapped.goal, dtype=np.float64).copy()
            ee = np.asarray(obs["achieved_goal"], dtype=np.float64)
            obj_pose = ee.copy() if has_object else None
            # Reach: seed-only reset suffices; no payload to store.
            # Push/PickAndPlace: store the full qpos/qvel so the bank
            # is robust to future gymnasium-robotics seed-mapping drift.
            reset_payload: dict[str, Any] | None = (
                {"init_qpos": init_qpos, "init_qvel": init_qvel} if has_object else None
            )
            out.append(
                CalibrationResult(
                    context_id=i,
                    env_id=env_id,
                    reset_seed=seed,
                    desired_goal=default_goal,
                    initial_achieved_goal=ee,
                    initial_object_pose=obj_pose,
                    reset_payload=reset_payload,
                    success_rate=float("nan"),
                    mean_dense_return=float("nan"),
                    mean_final_goal_distance=float("nan"),
                )
            )
    finally:
        env.close()
    return out


# ---------------------------------------------------------------------------
# Starter rollout
# ---------------------------------------------------------------------------


def _roll_starter(
    pool: list[CalibrationResult],
    policy_fn: PolicyFn,
    episodes_per_context: int,
    max_steps_per_episode: int,
) -> list[CalibrationResult]:
    """Run the starter policy on every pool context; fill outcome fields.

    Builds a ``FetchTaskEnvironment`` keyed by a tiny calibration-only
    bank (pool, all in train split, dummy bins) so the wrapper's
    deterministic reset path is exercised — the same path the
    downstream loops will use. The bank is throwaway; the bank
    returned to the caller is built later from the surviving subset
    after classification.

    Each context is rolled ``episodes_per_context`` times. For each
    episode the policy is queried per step until ``max_steps_per_episode``
    or env-side termination/truncation.
    """
    from pcrc.envs.fetch_environment import FetchTaskEnvironment  # noqa: PLC0415

    # Build a throwaway bank wrapping the pool so the wrapper's reset
    # path is the same one downstream loops will use.
    contexts = [
        FetchTaskContext(
            context_id=r.context_id,
            env_id=r.env_id,
            reset_seed=r.reset_seed,
            desired_goal=r.desired_goal,
            reset_payload=r.reset_payload,
            initial_object_pose=r.initial_object_pose,
            distance_bin=0,
            direction_bin=0,
            pose_bin=0,
            split="train",
        )
        for r in pool
    ]
    ee = {r.context_id: r.initial_achieved_goal for r in pool}
    throwaway_bank = FetchTaskBank.from_contexts(contexts, ee)

    env_id = pool[0].env_id
    out: list[CalibrationResult] = []
    with FetchTaskEnvironment(throwaway_bank, env_id) as env:
        for r in pool:
            successes = 0
            total_return = 0.0
            final_distances: list[float] = []
            for _ep in range(episodes_per_context):
                obs = env.reset(r.context_id)
                ep_return = 0.0
                ep_success = False
                final_dist = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
                for _step in range(max_steps_per_episode):
                    action = np.asarray(policy_fn(obs), dtype=np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_return += reward
                    final_dist = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
                    # gymnasium-robotics Fetch flags success via
                    # info["is_success"]; gracefully fall back to the
                    # final-distance threshold (Fetch default 0.05 m)
                    # if the key is absent on some env variant.
                    if bool(info.get("is_success", final_dist < 0.05)):
                        ep_success = True
                    if terminated or truncated:
                        break
                if ep_success:
                    successes += 1
                total_return += ep_return
                final_distances.append(final_dist)
            out.append(
                CalibrationResult(
                    context_id=r.context_id,
                    env_id=r.env_id,
                    reset_seed=r.reset_seed,
                    desired_goal=r.desired_goal,
                    initial_achieved_goal=r.initial_achieved_goal,
                    initial_object_pose=r.initial_object_pose,
                    reset_payload=r.reset_payload,
                    success_rate=successes / episodes_per_context,
                    mean_dense_return=total_return / episodes_per_context,
                    mean_final_goal_distance=float(np.mean(final_distances)),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Classification + binning
# ---------------------------------------------------------------------------


def _assign_quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin a 1D float array into ``[0, n_bins-1]`` integers.

    Uses ``np.quantile`` with ``method="linear"`` so degenerate values
    cluster at bin 0 cleanly. Returns shape ``(len(values),)`` int.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be > 0; got {n_bins}")
    if len(values) == 0:
        return np.zeros(0, dtype=np.int64)
    if n_bins == 1:
        return np.zeros(len(values), dtype=np.int64)
    qs = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    # Right-closed bin edges; clip to [0, n_bins-1] so the maximum
    # value lands in the top bin rather than overflowing.
    bins = np.clip(np.searchsorted(qs[1:-1], values, side="right"), 0, n_bins - 1)
    return bins.astype(np.int64)


def _direction_bin(desired_goal: np.ndarray, initial_achieved_goal: np.ndarray, n_bins: int) -> int:
    """Project the goal direction into an integer bin in ``[0, n_bins-1]``.

    Uses the azimuth angle in the XY plane (the Fetch table's
    horizontal plane). Vertical-only goals (degenerate XY direction)
    map to bin 0.
    """
    diff = desired_goal - initial_achieved_goal
    angle = float(np.arctan2(diff[1], diff[0]))  # range (-π, π]
    # Map (-π, π] into [0, 1) then into [0, n_bins-1].
    norm = (angle + np.pi) / (2.0 * np.pi)
    return int(min(int(norm * n_bins), n_bins - 1))


@dataclass
class CalibrationOutcome:
    """Container for the full calibration result.

    Attributes:
        bank: The constructed :class:`FetchTaskBank` ready for
            downstream loops.
        per_context: Per-context calibration records (every candidate,
            including dropped ones).
        kept_context_ids: Subset of ``per_context`` ids actually
            included in ``bank``.
        drop_reasons: ``{context_id: reason}`` for every dropped
            candidate. Reasons: ``"saturated"`` (success_rate >= 1 -
            band_upper), ``"dead"`` (success_rate <= band_lower), or
            ``"unused"`` (not selected by the target train + held_out
            quota after sorting kept candidates).
        used_dense_fallback: Whether binary success was too sparse to
            stratify (variance below
            ``dense_fallback_variance_threshold``) and the bin
            assignment fell back to dense return / final goal distance.
        summary: Aggregate counts + the bank's on-disk path. Suitable
            for ``RunLogger.log_event``.
        original_to_new: ``{original_probe_pool_id: reindexed_bank_id}``
            for every kept context. The bank reindexes kept contexts so
            train ids are ``0..num_train-1`` (the invariant
            ``build_sampler_for_condition`` requires); this map recovers
            the original probe-pool id for per-context provenance.
    """

    bank: FetchTaskBank
    per_context: list[CalibrationResult]
    kept_context_ids: list[int]
    drop_reasons: dict[int, str]
    used_dense_fallback: bool
    summary: dict[str, Any]
    original_to_new: dict[int, int]


def _select_keep_set(
    pool: list[CalibrationResult],
    band_lower: float,
    band_upper: float,
    num_train: int,
    num_held_out: int,
    dense_fallback_variance_threshold: float,
) -> tuple[list[int], dict[int, str], bool]:
    """Decide which pool ids to keep and tag drop reasons.

    Returns ``(kept_ids, drop_reasons, used_dense_fallback)``. The keep
    set has at most ``num_train + num_held_out`` ids. If the band
    cannot supply enough contexts, the runner raises — the caller
    must redesign the goal grid.
    """
    successes = np.array([r.success_rate for r in pool], dtype=np.float64)
    variance = float(np.var(successes))
    used_dense_fallback = variance < dense_fallback_variance_threshold

    drop_reasons: dict[int, str] = {}
    if used_dense_fallback:
        # Binary success was too sparse to stratify reliably (variance
        # below threshold). Accept the whole pool — the bin assignment
        # downstream uses dense return / final goal distance instead of
        # success rate. Saturated / dead drops are skipped in this
        # path because the signal is below noise; trusting it to fire
        # the drops would leak that noise into the bank.
        pass
    else:
        for r in pool:
            if r.success_rate >= band_upper:
                drop_reasons[r.context_id] = "saturated"
            elif r.success_rate <= band_lower:
                drop_reasons[r.context_id] = "dead"

    survivors = [r for r in pool if r.context_id not in drop_reasons]
    needed = num_train + num_held_out
    if len(survivors) < needed:
        raise ValueError(
            f"Calibration: not enough surviving contexts to fill the "
            f"bank — have {len(survivors)} survivors but need "
            f"{needed} (train={num_train}, held_out={num_held_out}). "
            f"Drops: {len(drop_reasons)} (saturated/dead). Redesign "
            f"the goal grid or relax the success band."
        )

    # Keep the first ``needed`` survivors deterministically (already
    # ordered by context_id from the probe pool). Mark the rest as
    # unused.
    kept = [r.context_id for r in survivors[:needed]]
    for r in survivors[needed:]:
        drop_reasons[r.context_id] = "unused"
    return kept, drop_reasons, used_dense_fallback


def _assign_bins_for_kept(
    pool_by_id: dict[int, CalibrationResult],
    kept_ids: list[int],
    distance_bins: int,
    direction_bins: int,
    pose_bins: int,
    used_dense_fallback: bool,
) -> dict[int, tuple[int, int, int]]:
    """Compute (distance_bin, direction_bin, pose_bin) per kept id."""
    if used_dense_fallback:
        # Use mean final goal distance as the calibration signal for
        # distance binning since binary success was too sparse to
        # stratify reliably. Direction and pose bins are still
        # geometric (computed from the context, not the policy).
        dist_values = np.array(
            [pool_by_id[cid].mean_final_goal_distance for cid in kept_ids],
            dtype=np.float64,
        )
    else:
        # Use the inverse of starter success rate as the difficulty
        # axis — high success → low difficulty bin, low success → high
        # difficulty bin. Stable under ties because _assign_quantile_bins
        # uses np.quantile.
        dist_values = np.array(
            [1.0 - pool_by_id[cid].success_rate for cid in kept_ids],
            dtype=np.float64,
        )
    dist_idx = _assign_quantile_bins(dist_values, distance_bins)

    out: dict[int, tuple[int, int, int]] = {}
    for i, cid in enumerate(kept_ids):
        r = pool_by_id[cid]
        d_bin = int(dist_idx[i])
        dir_bin = _direction_bin(r.desired_goal, r.initial_achieved_goal, direction_bins)
        # Pose bin: only meaningful for Push / PickAndPlace. Reach uses
        # the constant 0 here; object envs get the real quantile bin
        # recomputed in the second pass below (single-pass needs the
        # full object_x array first).
        out[cid] = (d_bin, dir_bin, 0)

    # Pose bin (Push / PickAndPlace) over the kept set, if any object.
    has_object = any(pool_by_id[cid].initial_object_pose is not None for cid in kept_ids)
    if has_object and pose_bins > 0:
        obj_x = np.array(
            [
                float(pool_by_id[cid].initial_object_pose[0])
                if pool_by_id[cid].initial_object_pose is not None
                else 0.0
                for cid in kept_ids
            ],
            dtype=np.float64,
        )
        pose_idx = _assign_quantile_bins(obj_x, pose_bins)
        for i, cid in enumerate(kept_ids):
            d_bin, dir_bin, _ = out[cid]
            out[cid] = (d_bin, dir_bin, int(pose_idx[i]))

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def calibrate_fetch_bank(
    *,
    env_id: str,
    n_pool: int,
    num_train: int,
    num_held_out: int,
    policy_fn: PolicyFn,
    episodes_per_context: int,
    max_steps_per_episode: int,
    distance_bins: int,
    direction_bins: int,
    pose_bins: int,
    target_success_band: tuple[float, float],
    dense_fallback_variance_threshold: float,
    pool_base_seed: int,
    split_seed: int,
) -> CalibrationOutcome:
    """Run the full calibration pipeline; return the constructed bank.

    Args:
        env_id: Registered gymnasium env id.
        n_pool: Number of candidate seeds to probe.
        num_train: Train-bank size.
        num_held_out: Held-out-bank size.
        policy_fn: Starter policy callable. See :data:`PolicyFn`.
        episodes_per_context: Trials per candidate during the
            starter rollout.
        max_steps_per_episode: Per-episode horizon.
        distance_bins: Number of difficulty quantile bins (typically 5).
        direction_bins: Number of azimuth bins for goal direction.
        pose_bins: Number of object-pose quantile bins (Push /
            PickAndPlace only; ignored for Reach).
        target_success_band: ``(band_lower, band_upper)`` —
            ``success_rate <= band_lower`` is "dead",
            ``success_rate >= band_upper`` is "saturated". A design
            guideline (not a hard gate per §4.0 Priority 1a).
        dense_fallback_variance_threshold: If binary success variance
            across the pool is below this, fall back to dense return /
            final goal distance for bin assignment.
        pool_base_seed: Starting seed for the candidate-pool probe.
        split_seed: RNG seed for the deterministic train / held_out
            split among the kept survivors.

    Returns:
        :class:`CalibrationOutcome`.

    Raises:
        ValueError: not enough surviving contexts to fill the bank.
    """
    band_lower, band_upper = target_success_band
    if not 0.0 <= band_lower < band_upper <= 1.0:
        raise ValueError(
            f"target_success_band must satisfy 0 <= lower < upper <= 1; "
            f"got ({band_lower}, {band_upper})"
        )

    # Phase 1 — probe.
    pool = _probe_pool(env_id, n_pool, pool_base_seed)
    # Phase 2 — roll the starter on every candidate.
    pool = _roll_starter(pool, policy_fn, episodes_per_context, max_steps_per_episode)
    pool_by_id = {r.context_id: r for r in pool}

    # Phase 3 — keep / drop.
    kept_ids, drop_reasons, used_dense_fallback = _select_keep_set(
        pool,
        band_lower,
        band_upper,
        num_train,
        num_held_out,
        dense_fallback_variance_threshold,
    )

    # Phase 4 — bin assignment for the kept subset.
    bins_by_id = _assign_bins_for_kept(
        pool_by_id,
        kept_ids,
        distance_bins,
        direction_bins,
        pose_bins,
        used_dense_fallback,
    )

    # Phase 5 — deterministic train / held_out split, then REINDEX so the
    # bank's train contexts carry ids 0..num_train-1. The loop's
    # build_sampler_for_condition asserts train_ids() == range(0, n_train);
    # without reindexing, the random split scatters train ids across the
    # kept range and Stage D raises. The split decision stays random (fixed
    # by split_seed); only the final ids are renumbered. Original probe-pool
    # ids are preserved in original_to_new for per-context provenance.
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(kept_ids))
    train_local = set(perm[:num_train].tolist())
    train_originals = [cid for i, cid in enumerate(kept_ids) if i in train_local]
    heldout_originals = [cid for i, cid in enumerate(kept_ids) if i not in train_local]
    ordered_originals = train_originals + heldout_originals  # train block first
    original_to_new: dict[int, int] = {}
    contexts: list[FetchTaskContext] = []
    ee: dict[int, np.ndarray] = {}
    for new_id, orig_cid in enumerate(ordered_originals):
        r = pool_by_id[orig_cid]
        d_bin, dir_bin, pose_bin = bins_by_id[orig_cid]
        split: Literal["train", "held_out"] = "train" if new_id < num_train else "held_out"
        contexts.append(
            FetchTaskContext(
                context_id=new_id,
                env_id=r.env_id,
                reset_seed=r.reset_seed,
                desired_goal=r.desired_goal,
                reset_payload=r.reset_payload,
                initial_object_pose=r.initial_object_pose,
                distance_bin=d_bin,
                direction_bin=dir_bin,
                pose_bin=pose_bin,
                split=split,
            )
        )
        ee[new_id] = r.initial_achieved_goal
        original_to_new[orig_cid] = new_id
    bank = FetchTaskBank.from_contexts(contexts, ee)

    summary = {
        "env_id": env_id,
        "n_pool": n_pool,
        "num_train": num_train,
        "num_held_out": num_held_out,
        "kept": len(kept_ids),
        "dropped_saturated": sum(1 for v in drop_reasons.values() if v == "saturated"),
        "dropped_dead": sum(1 for v in drop_reasons.values() if v == "dead"),
        "dropped_unused": sum(1 for v in drop_reasons.values() if v == "unused"),
        "used_dense_fallback": used_dense_fallback,
        "target_success_band": list(target_success_band),
        "bins": {
            "distance": distance_bins,
            "direction": direction_bins,
            "pose": pose_bins,
        },
        "starter_success_mean": float(np.mean([pool_by_id[cid].success_rate for cid in kept_ids])),
        "starter_success_min": float(np.min([pool_by_id[cid].success_rate for cid in kept_ids])),
        "starter_success_max": float(np.max([pool_by_id[cid].success_rate for cid in kept_ids])),
    }
    return CalibrationOutcome(
        bank=bank,
        per_context=pool,
        kept_context_ids=kept_ids,
        drop_reasons=drop_reasons,
        used_dense_fallback=used_dense_fallback,
        summary=summary,
        original_to_new=original_to_new,
    )


def write_calibration_outputs(
    outcome: CalibrationOutcome,
    out_dir: Path,
) -> dict[str, Path]:
    """Persist a :class:`CalibrationOutcome` to ``out_dir``.

    Writes:
        - ``calibration/per_context.jsonl`` — one JSON object per
          candidate, columns ``context_id``, ``env_id``, ``reset_seed``,
          ``success_rate``, ``mean_dense_return``,
          ``mean_final_goal_distance``, ``kept``, ``drop_reason``,
          ``split``, ``distance_bin``, ``direction_bin``, ``pose_bin``.
        - ``calibration/summary.json`` — aggregate counts +
          ``bank_path`` pointer.
        - ``calibration/bank.json`` — serialized :class:`FetchTaskBank`.

    Returns a dict of the three written paths keyed by short name.
    """
    import json  # noqa: PLC0415

    out_dir = Path(out_dir)
    cal_dir = out_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    bank_path = cal_dir / "bank.json"
    outcome.bank.save(bank_path)

    per_context_path = cal_dir / "per_context.jsonl"
    kept = set(outcome.kept_context_ids)
    with per_context_path.open("w") as f:
        for r in outcome.per_context:
            # per_context rows are keyed by the original probe-pool id;
            # the bank reindexes kept contexts, so translate through
            # original_to_new before looking up split / bin metadata.
            new_id = outcome.original_to_new.get(r.context_id)
            ctx_in_bank = outcome.bank.contexts.get(new_id) if new_id is not None else None
            row = {
                "context_id": r.context_id,
                "env_id": r.env_id,
                "reset_seed": r.reset_seed,
                "success_rate": r.success_rate,
                "mean_dense_return": r.mean_dense_return,
                "mean_final_goal_distance": r.mean_final_goal_distance,
                "kept": r.context_id in kept,
                "drop_reason": outcome.drop_reasons.get(r.context_id),
                "split": ctx_in_bank.split if ctx_in_bank is not None else None,
                "distance_bin": (ctx_in_bank.distance_bin if ctx_in_bank is not None else None),
                "direction_bin": (ctx_in_bank.direction_bin if ctx_in_bank is not None else None),
                "pose_bin": ctx_in_bank.pose_bin if ctx_in_bank is not None else None,
            }
            f.write(json.dumps(row) + "\n")

    summary_path = cal_dir / "summary.json"
    summary = {**outcome.summary, "bank_path": str(bank_path)}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    return {
        "bank": bank_path,
        "per_context": per_context_path,
        "summary": summary_path,
    }
