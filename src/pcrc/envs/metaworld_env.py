"""Meta-World task bank + environment — contact-rich sparse manipulation.

Duck-typed to the FetchTaskBank / FetchTaskEnvironment interfaces so the existing SAC+HER loop
(and the value-uncertainty curriculum) run on it unchanged.

Generalization spike (docs/metaworld_generalization_plan.md): test whether the value-uncertainty
curriculum win extends past sparse FetchPush to Meta-World's pushing tasks (push, sweep,
plate-slide, ...). Meta-World is flat-obs + dense-reward by default; we make it HER-compatible:

* A "context" is a frozen Meta-World ``rand_vec`` (object-init + goal), pinned via
  ``_freeze_rand_vec=True`` + ``_last_rand_vec`` so ``reset(context_id)`` is reproducible
  (verified: same context -> identical achieved/desired goal).
* ``achieved_goal`` = ``env._get_pos_objects()`` (per-task object accessor, NOT a hardcoded obs
  slice), ``desired_goal`` = ``env._target_pos``.
* HER-safe observation: the raw goal-observable obs ends in the goal (``obs[-3:]``); we STRIP it so
  the goal lives only in the ``desired_goal`` channel (else HER relabeling would be inconsistent).
* Sparse reward: 0 if ``||achieved - desired|| < success_distance_threshold`` else -1.
* Episodes are non-terminating (the loop controls length via ``max_steps``), matching Fetch.
"""

from __future__ import annotations

from collections import deque  # noqa: F401  (kept for parity with pointmaze_env; unused)
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class MetaWorldContext:
    """One Meta-World task instance: a frozen rand_vec (object-init + goal) + difficulty bin."""

    context_id: int
    rand_vec: list[float]
    distance_bin: int  # object->goal distance bucket (duck-types FetchTaskContext.distance_bin)
    split: str
    obj_init: list[float] = field(default_factory=list)  # captured object start (hand_near init)
    direction_bin: int = 0  # Meta-World difficulty is ~1-D here; present for evaluate_fetch parity
    pose_bin: int = 0
    task: str = ""  # Meta-World task id (e.g. "pick-place-v3"); "" = single-task bank default
    task_group: int = (
        0  # task index in a multi-task bank (the #5 oracle group: push=0, pick-place=1)
    )


def _go_env_class(env_id: str):
    """Resolve the goal-observable Meta-World env class for ``env_id`` (e.g. 'push-v3')."""
    import metaworld  # noqa: PLC0415

    go = metaworld.ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
    key = env_id if env_id.endswith("-goal-observable") else f"{env_id}-goal-observable"
    if key not in go:
        raise ValueError(f"unknown Meta-World task {env_id!r}; have e.g. {list(go)[:5]}")
    return go[key]


def _scripted_policy_for(task: str):
    """Resolve the Meta-World scripted EXPERT policy for ``task`` (e.g. 'pick-place-v3').

    Used only by the scripted-grasp reverse curriculum to synthesize a held-near-goal start state
    for in-air pick tasks (a teleport cannot produce a stable Meta-World grasp). 'pick-place-v3' ->
    ``metaworld.policies.SawyerPickPlaceV3Policy``.
    """
    import metaworld.policies as P  # noqa: PLC0415

    cls = (
        "Sawyer" + "".join(w.capitalize() for w in task.replace("-v3", "").split("-")) + "V3Policy"
    )
    return getattr(P, cls)()


def parse_metaworld_env_id(env_id: str) -> tuple[str, dict]:
    """Parse ``metaworld:push-v3[:handnear]`` -> (task, options).

    Modifier tokens after the task name select wrapper options so the env-agnostic loop and the
    driver can both construct the same env from one ``env_id`` string. Currently:
      * ``handnear`` -> start the hand AT the object (remove Meta-World's reach wall), making the
        task learnable from scratch like FetchPush — see ``MetaWorldTaskEnvironment``.
    """
    body = env_id.split(":", 1)[1] if env_id.startswith("metaworld:") else env_id
    parts = body.split(":")
    task = parts[0]
    opts: dict = {"hand_near_object": False}
    for p in parts[1:]:
        if p.startswith("handnear"):
            opts["hand_near_object"] = True
            suffix = p[len("handnear") :]  # e.g. "0.6" -> partial reach (fraction toward object)
            if suffix:
                opts["hand_near_fraction"] = float(suffix)
        elif p.startswith("graspcur"):
            # scripted-grasp reverse curriculum: for in-air (pick) goals, α>0 starts the object GRASPED
            # and held α-of-the-way to the goal (the expert maintains the grasp) — a teleport can't
            # synthesize a stable Meta-World grasp. α=0 (eval) is the true task. Needed because pick
            # tasks are unlearnable at ℓ=0 from a non-grasped scaffold (the object just falls).
            opts["grasp_curriculum"] = True
    return task, opts


class MetaWorldTaskBank:
    """Duck-typed FetchTaskBank for Meta-World: fixed (object-init, goal) rand_vec contexts."""

    def __init__(self, contexts: list[MetaWorldContext], env_id: str):
        """Build the bank from explicit contexts (use ``build`` to sample them)."""
        self.env_id = env_id
        self.contexts = {c.context_id: c for c in contexts}
        self._train = sorted(c.context_id for c in contexts if c.split == "train")
        self._held = sorted(c.context_id for c in contexts if c.split == "held_out")
        # Distinct Meta-World tasks present (multi-task bank); empty for a single-task bank.
        self.tasks = sorted({c.task for c in contexts if c.task})

    def __len__(self) -> int:
        """Number of contexts (train + held_out)."""
        return len(self.contexts)

    def __getitem__(self, context_id: int) -> MetaWorldContext:
        """Context lookup by id (duck-types FetchTaskBank.__getitem__)."""
        return self.contexts[context_id]

    def train_ids(self) -> list[int]:
        """Sorted training context ids."""
        return list(self._train)

    def held_out_ids(self) -> list[int]:
        """Sorted held-out context ids."""
        return list(self._held)

    # ----- (de)serialization + hashing (reproducibility: save a bank file + SHA-256) -----

    def _to_dict(self) -> dict:
        """Deterministic dict for save/hash (contexts sorted by id; stable field order)."""
        return {
            "env_id": self.env_id,
            "schema": "metaworld-1",
            "contexts": [
                {
                    "context_id": c.context_id,
                    "rand_vec": [float(x) for x in c.rand_vec],
                    "distance_bin": c.distance_bin,
                    "split": c.split,
                    "obj_init": [float(x) for x in c.obj_init],
                    "direction_bin": c.direction_bin,
                    "pose_bin": c.pose_bin,
                    "task": c.task,
                    "task_group": c.task_group,
                }
                for c in sorted(self.contexts.values(), key=lambda c: c.context_id)
            ],
        }

    def _payload(self) -> str:
        """Canonical JSON string used for both ``save`` and ``sha256`` (byte-stable)."""
        return json.dumps(self._to_dict(), sort_keys=True, indent=2)

    def sha256(self) -> str:
        """SHA-256 over the canonical serialization (matches ``hashlib.sha256(file.read_bytes())``)."""
        return hashlib.sha256(self._payload().encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        """Serialize the bank to JSON; the file bytes hash to :meth:`sha256`."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self._payload())
        return p

    @classmethod
    def load(cls, path: str | Path) -> MetaWorldTaskBank:
        """Load a bank saved by :meth:`save`."""
        data = json.loads(Path(path).read_text())
        ctxs = [
            MetaWorldContext(
                context_id=int(c["context_id"]),
                rand_vec=[float(x) for x in c["rand_vec"]],
                distance_bin=int(c["distance_bin"]),
                split=str(c["split"]),
                obj_init=[float(x) for x in c.get("obj_init", [])],
                direction_bin=int(c.get("direction_bin", 0)),
                pose_bin=int(c.get("pose_bin", 0)),
                task=str(c.get("task", "")),
                task_group=int(c.get("task_group", 0)),
            )
            for c in data["contexts"]
        ]
        return cls(ctxs, str(data["env_id"]))

    @classmethod
    def build(
        cls, env_id: str = "push-v3", *, n_train: int = 100, n_held: int = 50, seed: int = 0
    ) -> MetaWorldTaskBank:
        """Sample reproducible frozen contexts, binned by object->goal distance.

        Meta-World's raw ``_random_reset_space`` box contains invalid (off-table / in-collision)
        configs whose physics step can DIVERGE and hang ``reset`` (observed: a single bad vec
        stalls the whole build). So instead of sampling the box directly we let Meta-World sample
        its OWN *valid* ``rand_vec`` (``_freeze_rand_vec=False``) and CAPTURE ``_last_rand_vec`` per
        context; those captured vecs are frozen for reproducible ``reset(context_id)`` (Gate 0).
        """
        env = _go_env_class(env_id)(seed=seed)
        env._freeze_rand_vec = False  # let Meta-World sample VALID configs; we capture them
        # By default Meta-World draws rand_vec from the GLOBAL np.random (not reproducible);
        # seeded_rand_vec=True routes it through env.np_random, which reset(seed=) seeds.
        env.seeded_rand_vec = True
        env.reset(seed=seed)
        need = n_train + n_held
        recs = []  # (rand_vec, obj->goal distance, object-init pos)
        for _ in range(need):
            env.reset()
            vec = np.asarray(env._last_rand_vec, dtype=np.float64)
            obj = np.asarray(env._get_pos_objects())
            d = float(np.linalg.norm(obj - np.asarray(env._target_pos)))
            recs.append((vec.tolist(), d, obj.tolist()))
        env.close()
        dists = sorted({round(d, 4) for _, d, _ in recs})
        bin_of = {d: min(4, i * 5 // max(1, len(dists))) for i, d in enumerate(dists)}
        ctxs = []
        for k, (vec, d, obj) in enumerate(recs):
            split = "train" if k < n_train else "held_out"
            ctxs.append(
                MetaWorldContext(
                    context_id=k,
                    rand_vec=vec,
                    obj_init=obj,
                    distance_bin=bin_of[round(d, 4)],
                    split=split,
                )
            )
        return cls(ctxs, env_id)

    @classmethod
    def build_multi(
        cls,
        tasks: list[str],
        *,
        n_train: int = 100,
        n_held: int = 50,
        seed: int = 0,
    ) -> MetaWorldTaskBank:
        """Build a HETEROGENEOUS multi-task bank — the #5 conflict bank for the external benchmark.

        Each task contributes ``n_train``/``n_held`` frozen contexts (sampled exactly as ``build``),
        tagged with its ``task`` id and a ``task_group`` index (the per-context allocation groups:
        push vs pick-place). Contexts get GLOBAL unique ids across tasks; train/held splits are
        per-task so each task is represented in both. The hypothesis (FINDINGS §3.11b): push wants
        LOW scaffold (monotone-decreasing transfer) while pick-place wants a HIGH/curriculum scaffold
        (grasp wall at ℓ=0) — a no-compromise pacing conflict no single global schedule serves, so
        per-context allocation should beat the best global schedule on the worst group.
        """
        # Two-pass id assignment: ALL train contexts get contiguous ids 0..(n_tasks*n_train-1) FIRST,
        # then all held — the loop requires ``train_ids() == range(0, n_train_total)`` (the per-context
        # controller and the oracle level list index train contexts 0..n_train-1).
        train_recs: list[tuple[str, int, MetaWorldContext]] = []
        held_recs: list[tuple[str, int, MetaWorldContext]] = []
        for gi, task in enumerate(tasks):
            sub = cls.build(env_id=task, n_train=n_train, n_held=n_held, seed=seed)
            train_recs += [(task, gi, sub[i]) for i in sub.train_ids()]
            held_recs += [(task, gi, sub[i]) for i in sub.held_out_ids()]
        all_ctxs: list[MetaWorldContext] = []
        cid = 0
        for split, recs in (("train", train_recs), ("held_out", held_recs)):
            for task, gi, c in recs:
                all_ctxs.append(
                    MetaWorldContext(
                        context_id=cid,
                        rand_vec=list(c.rand_vec),
                        obj_init=list(c.obj_init),
                        distance_bin=c.distance_bin,
                        split=split,
                        task=task,
                        task_group=gi,
                    )
                )
                cid += 1
        return cls(all_ctxs, "+".join(tasks))


class MetaWorldTaskEnvironment:
    """Duck-typed FetchTaskEnvironment for Meta-World (HER-safe sparse goal reaching)."""

    def __init__(
        self,
        bank: MetaWorldTaskBank,
        env_id: str,
        max_steps: int | None = None,
        hand_near_object: bool = False,
        hand_offset_z: float = 0.05,
        hand_near_fraction: float = 1.0,
        grasp_curriculum: bool = False,
        grasp_goal_z: float = 0.10,
        grasp_max_steps: int = 80,
    ):
        """Wrap a goal-observable Meta-World env bound to ``bank``'s frozen contexts.

        ``hand_near_object``: start the hand directly above the object instead of at Meta-World's
        fixed home pose. This removes the *reach wall* (the arm normally starts far from the
        object, with no gradient to reach it — why push-v3 won't learn from scratch), making the
        task learnable from scratch like FetchPush, WITHOUT demos. The contact-rich push dynamics
        (and the leading uncertainty signal) are preserved; the bottleneck then comes from the
        cross-context object->goal difficulty spread rather than a per-context reach wall.
        ``hand_offset_z`` is how far ABOVE the object to start (a small clearance avoids the
        hand pre-disturbing the object during ``_reset_hand``).
        """
        self._bank = bank
        self._ctx: MetaWorldContext | None = None
        self._thr = 0.05  # Meta-World push success ~ obj_to_target < 0.05
        self._hand_near = bool(hand_near_object)
        self._hand_dz = float(hand_offset_z)
        # Fraction of the way from the default home pose to the object to start the hand
        # (1.0 = at the object; <1 keeps part of the reach/contact-initiation phase, where the
        # informative critic uncertainty may live).
        self._hand_frac = float(hand_near_fraction)
        # Scripted-grasp reverse curriculum (pick tasks): α>0 starts the object GRASPED + held α-of-
        # the-way to the (in-air) goal, bootstrapped by the Meta-World expert. Lazy per-task policy.
        self._grasp_curriculum = bool(grasp_curriculum)
        self._grasp_goal_z = float(grasp_goal_z)  # goal z above this => in-air => needs a grasp
        self._grasp_max_steps = int(grasp_max_steps)
        self._policies: dict[str, Any] = {}
        # Default task = the loop's env_id task (single-task path / fallback for "" contexts).
        self._default_task = env_id
        # MULTI-TASK bank (build_multi): one underlying MuJoCo env per distinct task, switched per
        # context at reset (the #5 conflict bank mixes push + pick-place). Single-task bank -> one env.
        tasks = list(bank.tasks) if getattr(bank, "tasks", None) else [env_id]
        self._envs: dict[str, Any] = {}
        self._hand_home: dict[str, np.ndarray] = {}
        for t in tasks:
            e = _go_env_class(t)(seed=0)
            e._freeze_rand_vec = True  # contexts pin the rand_vec; never re-sample on reset
            if self._hand_near:
                # Meta-World's DENSE reward assumes the hand starts AWAY from the object: its
                # gripper-caging margin (‖obj_init − init_tcp‖ − radius) goes negative when the hand
                # starts at the object and `reward_utils.tolerance` then raises. We override the reward
                # with sparse 0/−1 and compute success from distance, so bypass the dense computation
                # (env.step still returns a valid obs; we ignore its reward/info).
                e.evaluate_state = lambda obs, action: (0.0, {})
            self._envs[t] = e
            self._hand_home[t] = np.asarray(e.hand_init_pos, dtype=np.float64).copy()
        self._env = self._envs[tasks[0]]  # current underlying env (for spaces; switched per reset)
        del max_steps  # episode length is enforced by the training loop, like Fetch

    @property
    def action_space(self) -> Any:
        """Underlying gymnasium action space (4-D continuous, [-1, 1])."""
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        """Goal-conditioned Dict space: observation (goal stripped) + achieved/desired goal."""
        raw = self._env.observation_space
        n = int(np.asarray(raw.low).shape[0]) - 3  # strip the trailing goal
        box = lambda d: gym.spaces.Box(-np.inf, np.inf, shape=(d,), dtype=np.float64)  # noqa: E731
        return gym.spaces.Dict(
            {"observation": box(n), "achieved_goal": box(3), "desired_goal": box(3)}
        )

    def _obs_dict(self, raw_obs: np.ndarray) -> dict[str, Any]:
        # COPY every field: ``_get_pos_objects()`` / ``_target_pos`` return VIEWS into MuJoCo's live
        # qpos buffer, so a returned obs dict would silently mutate on the next reset/step (aliasing)
        # — corrupting any held reference (e.g. a stored replay transition). ``np.array`` copies.
        return {
            "observation": np.array(raw_obs[:-3], dtype=np.float64),  # HER-safe: goal removed
            "achieved_goal": np.array(self._env._get_pos_objects(), dtype=np.float64),
            "desired_goal": np.array(self._env._target_pos, dtype=np.float64),
        }

    def reset(self, context_id: int, start_progress: float = 0.0) -> dict[str, Any]:
        """Reset to ``context_id``'s frozen rand_vec; return a goal-conditioned obs dict.

        ``start_progress`` (α) is the PCRC reverse-curriculum scaffold: ``0`` is the true object
        start (eval always uses 0); ``α>0`` places the object a fraction α from its true start toward
        the target (``_set_obj_xyz`` of ``obj_init + α·(target − obj_init)``), so high α = object near
        the goal = easier, exactly like the Fetch reverse curriculum. The hand-reach scaffold
        (``hand_near_fraction``) is orthogonal and unchanged.
        """
        ctx = self._bank.contexts[context_id]
        self._ctx = ctx
        # Multi-task bank: switch the underlying MuJoCo env to this context's task before resetting.
        t = ctx.task or self._default_task
        self._env = self._envs[t]
        hand_home = self._hand_home[t]
        # Scaffolded object start: obj0 + α·(goal − obj0) (goal from the frozen rand_vec, computed
        # PRE-reset so the hand can be placed near the SCAFFOLDED object — otherwise hand_near pins the
        # gripper at the TRUE object start while the object is moved toward the goal, re-introducing the
        # reach wall at high α (a confound). α=0 ⇒ interp == obj0 (true start; eval unchanged).
        obj0 = np.asarray(ctx.obj_init, dtype=np.float64)
        goal = np.asarray(ctx.rand_vec[3:6], dtype=np.float64)
        alpha = float(np.clip(start_progress, 0.0, 1.0))
        interp = obj0 + alpha * (goal - obj0)
        # In-air (pick) goal + α>0 + grasp curriculum => bootstrap a GRASPED held start (the expert
        # carries the object α-of-the-way to the goal); a teleport can't hold a Meta-World grasp.
        use_grasp = self._grasp_curriculum and alpha > 0.0 and float(goal[2]) > self._grasp_goal_z
        if self._hand_near:
            # Start the hand a fraction of the way from home to the object — removes the reach wall.
            # For grasp-curriculum resets the expert grasps the object at its TRUE start, so aim the
            # hand at obj0 (not interp); otherwise aim at the scaffolded object (interp).
            base = obj0 if use_grasp else interp
            obj_target = base + np.array([0.0, 0.0, self._hand_dz])
            self._env.hand_init_pos = hand_home + self._hand_frac * (obj_target - hand_home)
        self._env._last_rand_vec = np.asarray(ctx.rand_vec, dtype=np.float64)
        res = self._env.reset()
        raw = np.asarray(res[0] if isinstance(res, tuple) else res, dtype=np.float64)
        if use_grasp:
            raw = self._scripted_grasp_bootstrap(t, obj0, goal, alpha)
        elif alpha > 0.0:
            # Object→goal reverse-curriculum scaffold: move the object α toward the goal, refresh obs.
            self._env._set_obj_xyz(interp)
            raw = np.asarray(self._env._get_obs(), dtype=np.float64)
        out = self._obs_dict(raw)
        out["source_context_id"] = context_id
        return out

    def _scripted_grasp_bootstrap(
        self, task: str, obj0: np.ndarray, goal: np.ndarray, alpha: float
    ) -> np.ndarray:
        """Drive the Meta-World expert until the GRASPED object is held α-of-the-way to the goal.

        The object starts on the table (true reset). The scripted expert grasps + lifts + transports
        it toward the in-air goal; we stop once ``‖obj − goal‖ ≤ (1−α)·‖obj0 − goal‖`` (high α => closer
        to the goal => easier) or after ``grasp_max_steps``. The RL episode then continues from this
        physically-grasped held state — the reverse-curriculum start the teleport scaffold could not
        produce. ``α=0`` never reaches here (eval is the true ungrasped task). Returns the raw obs.
        """
        pol = self._policies.get(task)
        if pol is None:
            pol = self._policies[task] = _scripted_policy_for(task)
        init_d = float(np.linalg.norm(obj0 - goal))
        target_d = (1.0 - alpha) * init_d
        raw = np.asarray(self._env._get_obs(), dtype=np.float64)
        for _ in range(self._grasp_max_steps):
            action = np.asarray(pol.get_action(raw), dtype=np.float32)
            self._env.step(action)
            raw = np.asarray(self._env._get_obs(), dtype=np.float64)
            if float(np.linalg.norm(self._env._get_pos_objects() - goal)) <= target_d:
                break
        # Hand the full episode budget to RL (the bootstrap steps shouldn't count toward truncation).
        self._env.curr_path_length = 0
        return raw

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step; sparse 0/-1 reward at the success threshold, non-terminating (loop sets length)."""
        raw, _r, _term, _trunc, info = self._env.step(np.asarray(action, dtype=np.float32))
        out = self._obs_dict(np.asarray(raw, dtype=np.float64))
        d = float(np.linalg.norm(out["achieved_goal"] - out["desired_goal"]))
        success = bool(info.get("success", d < self._thr)) or d < self._thr
        info = {**info, "is_success": success}
        return out, (0.0 if success else -1.0), False, False, info

    def close(self) -> None:
        """Close all underlying Meta-World envs (one per task in a multi-task bank)."""
        for e in self._envs.values():
            e.close()
