"""Deterministic Gymnasium-Robotics Fetch wrapper for pre-VLA runs.

Constructs one Fetch env per worker (the env is expensive to build —
MuJoCo model parsing, OpenGL context); ``env.reset(context_id)`` is
then called per episode to switch to a stored task context without
reconstructing the env. Mirrors the SimpleVLA-RL LIBERO wrapper API
shape (deterministic ``context_id`` → reset config) so the rest of the
training loop is env-agnostic.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§3.3 (API sketch), §9 risk #9 (deterministic-reset caveat —
gymnasium-robotics Fetch envs re-sample goals + object pose inside
``_sample_goal`` each reset; we override after the env's own reset
completes), §7.1 ``test_fetch_environment_reset.py`` (Push /
PickAndPlace object state reproducibility is the §4 Priority 1a hard
gate).

Reset sequence per ``env.reset(context_id)``:
    1. ``self._env.reset(seed=ctx.reset_seed)``.
    2. If ``ctx.reset_payload`` is not None: restore the known keys
       (``init_qpos`` / ``init_qvel`` / ``mocap_pos`` / ``mocap_quat`` /
       ``object_pose`` / ``object_vel``) onto the MuJoCo data, then
       ``mujoco.mj_forward(model, data)`` to recompute derived
       quantities. Unknown keys raise (FetchTaskBank also rejects them
       on save).
    3. Override the goal: ``self._env.unwrapped.goal = ctx.desired_goal``,
       then call ``_get_obs()`` to refresh the returned obs dict's
       ``desired_goal`` slot.
    4. Return ``{observation, desired_goal, achieved_goal,
       source_context_id}``.

The env's own ``step`` reward path references ``self.goal`` (which we
set in step 3), so reward and ``info["is_success"]`` are computed
against the bank's stored ``desired_goal`` without per-step
intervention from the wrapper. Asserted by
``test_fetch_environment_reset.py::test_reward_uses_bank_desired_goal``.

The wrapper does NOT own the curator or the sampler — it is a
deterministic env factory. The loops construct one
``FetchTaskEnvironment(bank, env_id)`` per worker and call
``env.reset(context_id)`` per episode (no env reconstruction).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pcrc.envs.fetch_task_bank import (
    KNOWN_RESET_PAYLOAD_KEYS,
    FetchTaskBank,
    FetchTaskContext,
)
from pcrc.envs.trajectory_scaffold import ScaffoldLadder
from pcrc.registry import ScaffoldSource


def goal_conditioned_observation(obs_dict: dict[str, Any]) -> np.ndarray:
    """Policy input for a goal-conditioned Fetch task.

    Concatenates ``observation`` with ``desired_goal`` (float32). Fetch
    is goal-conditioned: ``observation`` alone never encodes WHERE the
    goal is (for Push it carries the object pose but not its target), so
    a policy fed only ``observation`` is goal-blind and cannot solve the
    task. The plan §Priority 1b requires the policy input to include
    ``desired_goal``. ``achieved_goal`` is already contained in
    ``observation`` for every Fetch family, so it is not re-appended.
    """
    return np.concatenate(
        [
            np.asarray(obs_dict["observation"], dtype=np.float32),
            np.asarray(obs_dict["desired_goal"], dtype=np.float32),
        ]
    )


def goal_conditioned_observation_dim(observation_space: Any) -> int:
    """Width of :func:`goal_conditioned_observation` for a Fetch env.

    ``observation_space`` is the env's Dict space; returns
    ``observation.shape[0] + desired_goal.shape[0]``.
    """
    return int(
        observation_space["observation"].shape[0] + observation_space["desired_goal"].shape[0]
    )


def _import_gym_and_register() -> Any:
    """Import gymnasium + gymnasium-robotics; register Fetch envs.

    Deferred so the rest of ``pcrc.envs`` (and the bank tests) can
    import without paying the gymnasium-robotics + mujoco import cost
    on hosts that only have the generic test extra installed.
    """
    try:
        import gymnasium as gym  # noqa: PLC0415 — deferred import is intentional
        import gymnasium_robotics  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "FetchTaskEnvironment requires the `robotics` extra. "
            "Install with `uv sync --extra test --extra robotics` "
            "and try again."
        ) from exc
    gym.register_envs(gymnasium_robotics)
    return gym


class FetchStateCodec:
    """Full MuJoCo state serialize/restore for Fetch (a
    :class:`~pcrc.envs.trajectory_scaffold.StateCodec`).

    Captures ``qpos``, ``qvel``, the actuator controls ``ctrl`` (the finger position targets that
    HOLD a grasp — without them a restored in-air grasp opens and the block drops on the first
    steps), and the welded gripper ``mocap`` target. ``restore`` writes them back and runs
    ``mj_forward``; a snapshot taken from a settled grasped state needs no re-stepping. Operates on
    the unwrapped Fetch env (``.data`` / ``.model``).
    """

    def serialize(self, u) -> dict:
        d = u.data
        state = {
            "qpos": [float(v) for v in d.qpos],
            "qvel": [float(v) for v in d.qvel],
            "ctrl": [float(v) for v in d.ctrl],
        }
        if getattr(d, "mocap_pos", None) is not None and d.mocap_pos.size:
            state["mocap_pos"] = np.asarray(d.mocap_pos, dtype=np.float64).tolist()
            state["mocap_quat"] = np.asarray(d.mocap_quat, dtype=np.float64).tolist()
        return state

    def restore(self, u, state) -> None:
        import mujoco  # noqa: PLC0415

        d = u.data
        d.qpos[:] = np.asarray(state["qpos"], dtype=np.float64)
        d.qvel[:] = np.asarray(state["qvel"], dtype=np.float64)
        d.ctrl[:] = np.asarray(state["ctrl"], dtype=np.float64)
        if "mocap_pos" in state and getattr(d, "mocap_pos", None) is not None and d.mocap_pos.size:
            d.mocap_pos[:] = np.asarray(state["mocap_pos"], dtype=np.float64)
            d.mocap_quat[:] = np.asarray(state["mocap_quat"], dtype=np.float64)
        mujoco.mj_forward(u.model, d)


class FetchTaskEnvironment:
    """Deterministic per-worker Fetch env keyed by ``context_id``.

    Construct once per worker; call ``reset(context_id)`` per episode.

    Attributes:
        bank: The shared :class:`FetchTaskBank` providing contexts.
        env_id: The registered gymnasium env id. For mixed-env banks
            this must still be a single id — one wrapper per env id;
            the loop selects the wrapper given the sampled context's
            ``env_id``.
        max_steps: Per-episode horizon override; ``None`` keeps the
            env's default time-limit wrapper (gymnasium-robotics Fetch
            defaults to 50 steps).
    """

    def __init__(
        self,
        bank: FetchTaskBank,
        env_id: str,
        max_steps: int | None = None,
        *,
        scaffold_source: str = ScaffoldSource.MANUAL,
        scaffold_ladder: ScaffoldLadder | None = None,
    ) -> None:
        """Instantiate the underlying gymnasium env once."""
        if bank.env_id != "mixed" and bank.env_id != env_id:
            raise ValueError(
                f"FetchTaskEnvironment: bank.env_id={bank.env_id!r} does "
                f"not match requested env_id={env_id!r}. Pass the right "
                f"env_id or use a mixed-env bank."
            )
        self.bank = bank
        self.env_id = env_id
        self.max_steps = max_steps

        gym = _import_gym_and_register()
        kwargs: dict[str, Any] = {}
        if max_steps is not None:
            kwargs["max_episode_steps"] = max_steps
        self._env = gym.make(env_id, **kwargs)
        self._current_context_id: int | None = None
        self._mujoco = None  # lazy-imported in _apply_reset_payload
        # Scaffold state source: "manual" runs the reverse-curriculum operator (object interpolation
        # / in-air grasp synthesis); "trajectory" restores a serialized MuJoCo snapshot per
        # (context, level) from an extracted ladder (Automatic_Trajectory_Scaffold_Execution_Plan.md).
        if scaffold_source not in (ScaffoldSource.MANUAL, ScaffoldSource.TRAJECTORY):
            raise ValueError(
                f"scaffold_source must be 'manual' or 'trajectory'; got {scaffold_source!r}"
            )
        if scaffold_source == ScaffoldSource.TRAJECTORY and scaffold_ladder is None:
            raise ValueError("scaffold_source='trajectory' requires scaffold_ladder")
        self._scaffold_source = scaffold_source
        self._ladder = scaffold_ladder
        self._codec = FetchStateCodec()
        # Per-env RNG for the Stage-4 multi-trajectory ablation (sample 1-of-N states per reset).
        # Unused for single-trajectory ladders, so the default scaffold path stays deterministic.
        self._traj_rng = np.random.default_rng(0) if scaffold_source == ScaffoldSource.TRAJECTORY else None

    # ----- properties -----

    @property
    def action_space(self) -> Any:
        """Gymnasium action space of the underlying env."""
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        """Gymnasium dict observation space of the underlying env."""
        return self._env.observation_space

    def current_context_id(self) -> int:
        """Return the currently active context id.

        Raises:
            RuntimeError: ``reset(context_id)`` has not been called yet.
        """
        if self._current_context_id is None:
            raise RuntimeError(
                "FetchTaskEnvironment: reset(context_id) must be called "
                "before current_context_id()."
            )
        return self._current_context_id

    # ----- core API -----

    def reset(self, context_id: int, start_progress: float | None = None) -> dict[str, Any]:
        """Deterministic reset to the stored ``context_id``.

        See module-level docstring for the four-step sequence. Returns
        the observation dict augmented with ``source_context_id`` so
        downstream replay / utility aggregation has the context tag
        without a separate lookup.

        Args:
            context_id: bank index to reset to.
            start_progress: optional reverse-curriculum knob in
                ``[0, 1]`` (Florensa et al. 2017, "Reverse Curriculum
                Generation"). After the normal reset, the object is
                moved a fraction ``start_progress`` along the line from
                its initial ``achieved_goal`` to the ``desired_goal``.
                ``0`` / ``None`` keeps the true start (no curriculum).
                For IN-AIR goals (desired z > 0.45) the object cannot
                rest at the interpolated point unsupported, so a grasped
                state is synthesized: the gripper is driven to the point
                and closed on the object so it is held in place. EVAL
                MUST always pass ``0`` / ``None`` (true start) so the
                measured success is the real task — never the curriculum.

        Returns:
            ``{observation, desired_goal, achieved_goal,
            source_context_id}``.

        Raises:
            KeyError: ``context_id`` is not in the bank.
            ValueError: the bank context's ``env_id`` does not match
                this wrapper's ``env_id``.
        """
        ctx: FetchTaskContext = self.bank[context_id]
        if ctx.env_id != self.env_id:
            raise ValueError(
                f"FetchTaskEnvironment.reset: context {context_id} has "
                f"env_id={ctx.env_id!r} but wrapper is bound to "
                f"env_id={self.env_id!r}."
            )

        # Step 1: env-level reset under the stored seed.
        self._env.reset(seed=int(ctx.reset_seed))

        # Step 2: restore simulator state from reset_payload if present.
        if ctx.reset_payload is not None:
            self._apply_reset_payload(ctx.reset_payload)

        # Step 3: override goal. The env's own step() reward path reads
        # self.goal, so setting it once here suffices for the whole
        # episode. _get_obs() then refreshes the obs dict's desired_goal
        # slot (which the wrapper returns).
        self._env.unwrapped.goal = np.asarray(ctx.desired_goal, dtype=np.float64).copy()

        # Step 3b (scaffold): for the "trajectory" source, restore the serialized snapshot for this
        # (context, level); for "manual", advance the object a fraction start_progress toward the
        # goal (reverse curriculum). Either way the start state recedes to the true start at level 0.
        if self._scaffold_source == ScaffoldSource.TRAJECTORY:
            level = self._recover_level(start_progress)
            if level > 0:
                self._codec.restore(
                    self._env.unwrapped,
                    self._ladder.sample_state(context_id, level, self._traj_rng),
                )
        elif start_progress is not None and start_progress > 0.0:
            self._apply_reverse_curriculum(
                np.asarray(ctx.desired_goal, dtype=np.float64), start_progress
            )

        obs = self._env.unwrapped._get_obs()

        self._current_context_id = int(context_id)
        return {
            "observation": np.asarray(obs["observation"]),
            "desired_goal": np.asarray(obs["desired_goal"]),
            "achieved_goal": np.asarray(obs["achieved_goal"]),
            "source_context_id": int(context_id),
        }

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Gymnasium 5-tuple: ``(obs, reward, terminated, truncated, info)``.

        ``obs`` mirrors the shape returned by :meth:`reset` (no
        ``source_context_id`` — the loop tracks that separately if
        needed). Reward is the env's own scalar from ``compute_reward``
        against the goal set in :meth:`reset`. Termination /
        truncation come straight from the underlying env.
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            {
                "observation": np.asarray(obs["observation"]),
                "desired_goal": np.asarray(obs["desired_goal"]),
                "achieved_goal": np.asarray(obs["achieved_goal"]),
            },
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self) -> None:
        """Close the underlying env."""
        self._env.close()

    # ----- context manager -----

    def __enter__(self) -> FetchTaskEnvironment:
        """Enter as a context manager; returns ``self`` unchanged."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the wrapped env on context exit."""
        self.close()

    # ----- internal helpers -----

    def _apply_reset_payload(self, payload: dict[str, Any]) -> None:
        """Apply each known key from a reset payload onto MuJoCo state.

        ``init_qpos`` / ``init_qvel`` overwrite the full qpos / qvel
        buffers. ``mocap_pos`` / ``mocap_quat`` overwrite the mocap
        buffers if present. ``object_pose`` / ``object_vel`` are
        currently delegated to qpos / qvel — gymnasium-robotics Fetch
        envs store object state inside the same qpos / qvel buffer,
        so a caller using these keys must already have a fully
        compatible qpos / qvel pair stored under ``init_qpos`` /
        ``init_qvel``. The ``object_*`` keys are kept in the
        ``KNOWN_RESET_PAYLOAD_KEYS`` whitelist for forward
        compatibility (env-pose-only datasets); if exercised today
        they are no-ops and a clear warning would aid debugging.

        After applying state, ``mujoco.mj_forward`` recomputes derived
        quantities so subsequent ``_get_obs()`` reads consistent
        kinematics.
        """
        # Defensive: even though FetchTaskBank.save also rejects unknown
        # keys, the wrapper sees the runtime payload directly (e.g. a
        # test fixture that constructs FetchTaskContext without going
        # through save / load).
        unknown = set(payload) - KNOWN_RESET_PAYLOAD_KEYS
        if unknown:
            raise ValueError(
                f"FetchTaskEnvironment._apply_reset_payload: unknown "
                f"reset_payload keys {sorted(unknown)}. Extend "
                f"KNOWN_RESET_PAYLOAD_KEYS and this method."
            )

        unwrapped = self._env.unwrapped
        data = unwrapped.data
        if "init_qpos" in payload:
            qpos = np.asarray(payload["init_qpos"], dtype=np.float64)
            if qpos.shape != data.qpos.shape:
                raise ValueError(
                    f"init_qpos shape {qpos.shape} != env qpos shape "
                    f"{data.qpos.shape} for env_id={self.env_id!r}"
                )
            data.qpos[:] = qpos
        if "init_qvel" in payload:
            qvel = np.asarray(payload["init_qvel"], dtype=np.float64)
            if qvel.shape != data.qvel.shape:
                raise ValueError(
                    f"init_qvel shape {qvel.shape} != env qvel shape "
                    f"{data.qvel.shape} for env_id={self.env_id!r}"
                )
            data.qvel[:] = qvel
        if "mocap_pos" in payload and hasattr(data, "mocap_pos"):
            mp = np.asarray(payload["mocap_pos"], dtype=np.float64)
            if mp.shape != data.mocap_pos.shape:
                raise ValueError(
                    f"mocap_pos shape {mp.shape} != env mocap_pos shape "
                    f"{data.mocap_pos.shape} for env_id={self.env_id!r}"
                )
            data.mocap_pos[:] = mp
        if "mocap_quat" in payload and hasattr(data, "mocap_quat"):
            mq = np.asarray(payload["mocap_quat"], dtype=np.float64)
            if mq.shape != data.mocap_quat.shape:
                raise ValueError(
                    f"mocap_quat shape {mq.shape} != env mocap_quat shape "
                    f"{data.mocap_quat.shape} for env_id={self.env_id!r}"
                )
            data.mocap_quat[:] = mq

        # Recompute derived state.
        if self._mujoco is None:
            import mujoco  # noqa: PLC0415

            self._mujoco = mujoco
        self._mujoco.mj_forward(unwrapped.model, data)

    def _recover_level(self, start_progress: float | None) -> int:
        """Nearest scaffold level to the emitted fraction, over the ladder's own quantiles.

        ``None`` (eval / true start) maps to level 0. Used only in the ``trajectory`` source.
        """
        sp = 0.0 if start_progress is None else float(start_progress)
        q = np.asarray(self._ladder.quantiles, dtype=np.float64)
        return int(np.argmin(np.abs(q - sp)))

    # Reverse-curriculum constants (FetchPickAndPlace MuJoCo model).
    # A desired-goal z above this is "in the air" (table surface ~0.42,
    # the env adds up to 0.45 to a height_offset of ~0.425): such goals
    # need a GRASPED start because the object cannot rest unsupported.
    _IN_AIR_Z_THRESHOLD: float = 0.45
    # Standard Fetch block half-width is 0.025; close fingers well inside
    # it so they firmly pinch the block (0.024 only kissed the surface and
    # the block slipped out under gravity within a few steps; 0.015 holds).
    # The two finger actuators are position-controlled (ctrl == finger
    # qpos target).
    _GRASP_FINGER_WIDTH: float = 0.015
    _OPEN_FINGER_WIDTH: float = 0.04
    # The grip site sits ~0.02 m below the mocap target; aim the mocap
    # this far above the object so the closed gripper holds it at z
    # rather than sagging the block down through the fingers.
    _MOCAP_Z_OFFSET: float = 0.02

    def _apply_reverse_curriculum(self, desired_goal: np.ndarray, start_progress: float) -> None:
        """Move the object ``start_progress`` of the way toward the goal.

        Florensa et al. 2017 reverse curriculum: the training start state
        is interpolated between the object's reset pose (achieved_goal)
        and the ``desired_goal``. For table-height goals the object is
        simply teleported to the interpolated point (it rests on the
        table). For IN-AIR goals (``desired_goal[2] > _IN_AIR_Z_THRESHOLD``)
        the object would fall, so a physically-plausible GRASPED state is
        synthesized: the gripper is driven to the interpolated point with
        open fingers, the object is dropped between the fingers, and the
        fingers close — leaving the block held in the gripper. A few
        settling steps then let the constraint solver stabilize the grip.

        Must be called AFTER the env reset + reset_payload restore and the
        goal override, before ``_get_obs()``.
        """
        if self._mujoco is None:
            import mujoco  # noqa: PLC0415

            self._mujoco = mujoco
        mujoco = self._mujoco
        unwrapped = self._env.unwrapped
        model, data = unwrapped.model, unwrapped.data
        utils = unwrapped._utils

        # Recompute kinematics so the object's current pose (achieved_goal)
        # is consistent with any reset_payload that was just applied.
        mujoco.mj_forward(model, data)
        start = np.asarray(unwrapped._get_obs()["achieved_goal"], dtype=np.float64)
        target = start + float(start_progress) * (desired_goal - start)

        in_air = float(desired_goal[2]) > self._IN_AIR_Z_THRESHOLD

        if not in_air:
            # Table-height goal: teleport the object along the table. The
            # block rests on the surface, no grasp needed.
            obj_qpos = utils.get_joint_qpos(model, data, "object0:joint").copy()
            obj_qpos[:3] = target
            obj_qpos[3:] = np.array([1.0, 0.0, 0.0, 0.0])
            utils.set_joint_qpos(model, data, "object0:joint", obj_qpos)
            utils.set_joint_qvel(model, data, "object0:joint", np.zeros(6))
            mujoco.mj_forward(model, data)
            return

        # In-air goal: synthesize a grasped state.
        grip_quat = np.array([1.0, 0.0, 1.0, 0.0])  # env's fixed EE rotation
        # Phase 1: open the fingers and drive the welded gripper to a
        # point ~_MOCAP_Z_OFFSET above the interpolated target.
        data.ctrl[:] = self._OPEN_FINGER_WIDTH
        utils.set_joint_qpos(model, data, "robot0:l_gripper_finger_joint", self._OPEN_FINGER_WIDTH)
        utils.set_joint_qpos(model, data, "robot0:r_gripper_finger_joint", self._OPEN_FINGER_WIDTH)
        utils.set_mocap_quat(model, data, "robot0:mocap", grip_quat)
        approach = target + np.array([0.0, 0.0, self._MOCAP_Z_OFFSET])
        for _ in range(80):
            utils.set_mocap_pos(model, data, "robot0:mocap", approach)
            mujoco.mj_step(model, data)

        grip = np.asarray(utils.get_site_xpos(model, data, "robot0:grip"), dtype=np.float64).copy()
        # Phase 2: drop the object at the gripper and close the fingers.
        obj_qpos = utils.get_joint_qpos(model, data, "object0:joint").copy()
        obj_qpos[:3] = grip
        obj_qpos[3:] = np.array([1.0, 0.0, 0.0, 0.0])
        utils.set_joint_qpos(model, data, "object0:joint", obj_qpos)
        utils.set_joint_qvel(model, data, "object0:joint", np.zeros(6))
        # Drive the finger actuators closed (ctrl == position target); the
        # solver then pinches the block. Setting ctrl (not per-step qpos)
        # lets the contact forces hold the object instead of being
        # overwritten each step.
        data.ctrl[:] = self._GRASP_FINGER_WIDTH
        hold = grip + np.array([0.0, 0.0, self._MOCAP_Z_OFFSET])
        for _ in range(60):
            utils.set_mocap_pos(model, data, "robot0:mocap", hold)
            data.ctrl[:] = self._GRASP_FINGER_WIDTH
            mujoco.mj_step(model, data)
        # Zero residual velocities so the episode starts from rest, then
        # recompute derived quantities for the returned observation.
        utils.set_joint_qvel(model, data, "object0:joint", np.zeros(6))
        mujoco.mj_forward(model, data)
