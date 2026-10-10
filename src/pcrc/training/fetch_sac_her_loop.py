"""Sparse Fetch SAC+HER loop with tagged global replay.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``fetch_sac_her_loop.py`` row), §4 Priority 3 (sparse driver,
HER on by default, tagged global replay is the single training-path
default), §8 (RunLogger streams), §7.2 SAC smoke (synthetic-prefill
replay-distribution check + plumbing exercise).

Per-iteration flow:

    1. Pick one context via ``sampler.sample_flat(1)``.
    2. Roll the SAC actor for ``max_steps_per_episode`` collecting
       :class:`GoalConditionedTransition` instances tagged with the
       source ``context_id``.
    3. Apply HER ``relabel`` over the episode (default ``"future"``,
       ``n_sampled_goal=4``); push every transition (original +
       relabeled) into the tagged replay buffer.
    4. Once the buffer holds ``learning_starts`` transitions, run
       ``updates_per_iter`` SAC update steps drawing minibatches from
       the buffer via ``sample_context_weighted``.
    5. After SAC updates, compute per-context critic-improvement
       utility (Priority 3 AC condition only); feed sampler.
    6. Update sampler with :class:`SamplerStats` from per-context
       episode counts.
    7. Log all mandatory streams (§8). Eval at the configured cadence.

The state passed to SAC is ``[observation, desired_goal]``
concatenated — the goal is part of the state, so SAC is goal-agnostic.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from pcrc.algorithms.crl_continuous import CRLAgent, CRLContinuousConfig
from pcrc.algorithms.her import CHERConfig, GoalConditionedTransition, relabel
from pcrc.algorithms.sac_continuous import SACAgent, SACContinuousConfig
from pcrc.algorithms.sac_critic_improvement import (
    sac_actor_objective_improvement,
    sac_critic_improvement,
)
from pcrc.algorithms.tagged_goal_replay import TaggedGlobalReplay
from pcrc.envs.fetch_environment import FetchTaskEnvironment
from pcrc.envs.fetch_task_bank import FetchTaskBank
from pcrc.envs.trajectory_scaffold import ScaffoldLadder
from pcrc.registry import DEFAULT_METHOD, Method, SamplerName, ScaffoldSource
from pcrc.samplers.base import SamplerStats
from pcrc.samplers.scaffold_controller import ScaffoldConfig, ScaffoldController
from pcrc.training.fetch_evaluation import evaluate_fetch
from pcrc.training.fetch_ppo_loop import build_sampler_for_condition
from pcrc.training.fetch_uniformized_reward import (
    compute_selection_vs_learning_decomposition,
)
from pcrc.training.kill_conditions import KillConditionMonitor, write_kill_reason
from pcrc.utils.results import RunLogger


@dataclass
class FetchSACHERLoopConfig:
    """All-in-one config for the Fetch SAC+HER driver."""

    exp_name: str
    condition: str
    env_id: str
    n_iterations: int = 100
    max_steps_per_episode: int = 50
    her_strategy: str = "future"
    her_n_sampled_goal: int = 4
    # CHER goal selection (Fang et al. 2019). When her_strategy="cher",
    # the n_sampled_goal future relabel goals are chosen by a
    # proximity(to true desired goal) + diversity(facility-location)
    # submodular score via lazier-greedy, with curriculum weight
    #   λ(t) = cher_lambda_starter * (1 + cher_lambda_rate)^t
    # annealed UP (diversity early -> proximity late). Default path
    # ("future"/"final") ignores these and stays bit-identical.
    cher_lambda_starter: float = 1.0
    cher_lambda_rate: float = 0.0001
    cher_fixed_lambda: float | None = None
    cher_sub_size: int = 3
    replay_capacity: int = 200_000
    minibatch_size: int = 64
    # Base learner: "sac" (default TD SAC+HER) or "crl" (contrastive critic, no TD — B2/M4).
    # CRL consumes HER "future" relabels as InfoNCE positives; crl_goal_dim tells the agent how to
    # split the concat(obs, goal) state (set by the driver from the env's desired_goal dim).
    learner: str = "sac"
    crl_goal_dim: int | None = None
    crl_repr_dim: int = 64
    learning_starts: int = 200
    updates_per_iter: int = 1
    eval_interval: int = 25
    n_eval_episodes_per_context: int = 4
    seed: int = 0
    epsilon: float = 0.05
    q_max: float | None = None
    success_distance_threshold: float = 0.05
    # HER relabel reward threshold (||achieved - goal|| < thr -> 0 else -1). 0.05 m for
    # Fetch; pass 0.45 for PointMaze (half a maze cell). Keep == success threshold.
    reward_distance_threshold: float = 0.05
    success_ema_alpha: float = 0.2
    # HER-paper action exploration (Andrychowicz et al. 2017 FetchPush
    # config). Sustained random/noisy actions seed the gripper-block
    # contacts that HER bootstraps from; SAC's collapsing entropy alpha
    # does not provide this on its own.
    random_eps: float = 0.3
    noise_eps: float = 0.2
    # Goal-INDEPENDENT dense reward shaping added to every transition (incl.
    # HER-relabeled copies) at training time: r += -coef * ||gripper - object||
    # (gripper = observation[:3], object = achieved_goal). Goal-independent so
    # it survives HER relabeling untouched. Intended to break the PnP grasping
    # wall (get the gripper to the object so HER can bootstrap grasps). Eval
    # uses sparse success only, so the measurement stays honest. 0 = off.
    reward_shaping_coef: float = 0.0
    # Shaping form (R2): "gripper_object" = -coef*||gripper-object|| (goal-
    # independent). "staged" = a multi-stage PROCESS reward
    #   coef*( -||gripper-object||  (reach)
    #          + 1.5*clip((obj_z-0.425)/0.1, 0, 1)  (grasp/lift; goal-independent)
    #          - ||object - training_desired_goal|| )  (place; goal-DEPENDENT, HER-safe)
    # Manufactures partial-progress-beyond-success so contexts differ in WHICH
    # stage they are stuck at — the property the AC paper says makes the actor
    # utility informative (R2: test whether this flips the within-band null).
    # "grasp_only" = coef*(-reach + 1.5*lift): shape the grasp/lift bottleneck only, leave
    # placement sparse — a PARTIAL unlock that preserves placement heterogeneity (vs "staged"
    # which shapes place too and saturates the band).
    reward_shaping_mode: str = "gripper_object"
    # Parallel rollout workers (CPU processes). 1 = sequential (default,
    # unchanged behavior). N>1 collects N episodes/iter across N spawned
    # worker processes, each with its own env + rollout-only SAC actor —
    # ~N x env-step throughput (MuJoCo rollout is CPU-bound and GIL-locked,
    # so threads don't help). The fixed updates_per_iter budget is amortized
    # over the N x data, landing at a HER-paper-like update ratio.
    num_workers: int = 1
    # Reverse curriculum (Florensa et al. 2017). When True, the TRAINING
    # reset moves the object a fraction `start_progress` of the way from
    # its reset pose to the desired goal (grasped for in-air goals), so
    # early episodes start near success and the start state recedes to the
    # true start over training. start_progress decays linearly from
    # reverse_curriculum_start_progress at iter 0 to 0 at
    # reverse_curriculum_decay_iters (and stays 0 after). EVAL always uses
    # start_progress=0 (true start) — the curriculum never touches eval.
    reverse_curriculum: bool = False
    reverse_curriculum_start_progress: float = 0.9
    reverse_curriculum_decay_iters: int = 1000
    # UNCERTAINTY-GATED reverse curriculum (the method extension): instead of decaying
    # start_progress uniformly across contexts, modulate it per-context by the critic's
    # value-uncertainty (mean |TD-error| EMA). Contexts the critic is most UNCERTAIN about keep a
    # nearer-goal start longer (more reverse-curriculum help); confident contexts recede to the
    # true start faster. start_progress(c) = base(it) * (1 - g + 2*g*û(c)), û = min-max normalized
    # uncertainty EMA in [0,1], g = gate_strength. Mean help ≈ vanilla RC (û mean ~0.5), just
    # REALLOCATED by uncertainty. Keeps value-uncertainty as the signal, applied to the start-state
    # lever rather than task selection. Requires reverse_curriculum=True.
    reverse_curriculum_uncertainty_gate: bool = False
    reverse_curriculum_gate_strength: float = 0.5
    # PCRC scaffold-allocation controller (docs/EXPERIMENTS.md).
    # When enabled, a ScaffoldController over (context, level) cells REPLACES the
    # sampler + per_context_start_progress path: it picks which cell to roll and emits the
    # per-episode start_progress (= the level's fraction), and moves a per-context frontier by
    # success + value-uncertainty. Eval is unchanged (start_progress=0). Default OFF → the
    # existing selection path is bit-for-bit identical.
    scaffold_enabled: bool = False
    scaffold_mode: str = DEFAULT_METHOD
    scaffold_n_levels: int = 5
    scaffold_level_fractions: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 0.9]
    )
    scaffold_kwargs: dict[str, Any] = field(default_factory=dict)
    # Per-context fixed level for scaffold_mode "oracle" (length n_train, indexed by
    # train context_id); the driver computes it from the bank's group labels. None
    # for every other mode.
    scaffold_oracle_levels: list[int] | None = None
    # Recorded [T, n_train] per-context level matrix for scaffold_mode "replay_*" (Gate E causal
    # controls); the driver loads it from a reference pcrc run's scaffold.jsonl. None otherwise.
    scaffold_replay_trajectory: Any | None = None
    # Per-context group label (length n_train, values 0..G-1) for scaffold_mode "per_group_success"
    # (campaign 5 C5.1); the driver computes it from the bank's oracle group labels. None otherwise.
    scaffold_group_ids: list[int] | None = None
    # Group-balanced replay sampling (campaign 5 C5.7, review Recommended #9): draw each SAC
    # minibatch with fixed per-group quotas (minibatch_size / G transitions from each group's
    # contexts, HER relabels included via their source_context_id tag) instead of one
    # context-weighted draw. Isolates off-policy buffer composition from reset allocation: if
    # global annealing + balanced replay rescues the sacrificed group, the collapse was partly a
    # replay artifact. A group with no stored transitions yet cedes its quota to the full
    # distribution. Requires replay_group_ids. Default OFF → sampling path is bit-identical.
    replay_group_balanced: bool = False
    replay_group_ids: list[int] | None = None
    # PointMaze scaffold reset geometry: "geodesic" (along shortest path) or "offpath" (straight-line
    # snap; bypasses bottlenecks for high-detour contexts — Nav-bypass conflict, Gate D / breadth).
    scaffold_geometry: str = "geodesic"
    # Scaffold reset-state SOURCE: "manual" runs the env's hand-written operator; "trajectory" restores
    # a serialized snapshot per (context, level) from `scaffold_ladder` (automatic trajectory-derived
    # scaffold; Automatic_Trajectory_Scaffold_Execution_Plan.md). The controller is identical for both,
    # and eval (level 0) is the true start regardless, so the eval env can stay "manual".
    scaffold_source: str = ScaffoldSource.MANUAL
    scaffold_ladder: ScaffoldLadder | None = None
    # Stage A (self-generated scaffold): after training, harvest the policy's own first ℓ=0-success
    # rollout per train context (serialized full sim states) for the trajectory extractor. None = off.
    harvest_path: str | None = None
    harvest_k: int = 5  # rollouts/context (attempt 0 deterministic, rest stochastic); keep shortest success
    harvest_seed: int = 99  # dedicated harvest seed, disjoint from eval seeds
    # PCGrad group threshold (goal-z): when sac.pcgrad is on, minibatch transitions split into
    # groups by desired_goal[2] >= this (the last state column) for per-group gradient projection.
    pcgrad_goalz_threshold: float = 0.45
    # Actor-Curator utility for the AC conditions: "critic_improvement"
    # (Q_{t+1}(s,π_{t+1}) - Q_t(s,π_t), default) or "actor_objective"
    # (Q_{t+1}(s,π_{t+1}) - Q_{t+1}(s,π_t) — isolates the policy update,
    # cleaner when the critic is noisy at near-zero sparse success).
    ac_utility: str = "critic_improvement"
    # AC easy-bias mitigations (the critic/actor-improvement utilities
    # over-concentrate on the easiest contexts). Both post-process the base
    # per-context utility before OSMD:
    #   ac_headroom_gate: multiply by s_ema*(1-s_ema) — restrict the
    #     actor-aware signal to the intermediate-difficulty frontier (combine
    #     with what makes `headroom` win).
    #   ac_ema_center: subtract each context's own running-mean utility —
    #     reward contexts improving faster than THEIR baseline (de-bias the
    #     systematic easy-improvement), mirroring the AC paper's advantage
    #     centering.
    ac_headroom_gate: bool = False
    ac_ema_center: bool = False
    ac_ema_alpha: float = 0.1
    # Diagnostic: compute + log the actor_objective utility every iter even when
    # the sampler is NOT an AC condition (e.g. uniform). The sampler still
    # ignores the utility, so sampling is unchanged — this only populates
    # utilities.jsonl, enabling the unbiased corr(u_actor, future Δsuccess)
    # diagnostic under uniform sampling. No effect on AC conditions.
    diagnostic_compute_utility: bool = False
    # Demonstrations + behavior-cloning lever (Nair et al. 2018, "Overcoming
    # Exploration in RL with Demonstrations"). The canonical literature fix for
    # in-air FetchPickAndPlace, which does not learn from scratch (grasp wall).
    #   demos_path: path to a demos .npz (scripts/collect_pnp_demos.py). When
    #     set + bc_lambda > 0, demo transitions are (a) HER-relabeled and added
    #     to the replay buffer so the critic sees them, and (b) supplied as a
    #     (state, action) BC minibatch to the SAC actor update.
    #   bc_lambda: weight of the BC auxiliary loss. 0.0 (default) = OFF — the
    #     training path is then bit-for-bit identical to current behavior (no
    #     demos loaded, no BC term).
    #   use_q_filter: Nair 2018 Q-filter — only clone demo actions that beat
    #     the current policy under the critic (Q(s,a_demo) > Q(s,π(s))).
    #   bc_minibatch_size: number of demo (s,a) pairs per BC step (sampled
    #     uniformly from the demo set each update).
    #   demo_replay_fraction: Nair 2018 protected demo buffer. When > 0, the
    #     HER-relabeled demos are kept in a SEPARATE pool (never evicted) instead
    #     of the main replay, and this fraction of every SAC minibatch is drawn
    #     from it — so the critic always sees successful demos and does not
    #     diverge once self-generated failures fill the main buffer. 0 = legacy
    #     (demos go into the main replay and dilute/evict).
    demos_path: str | None = None
    bc_lambda: float = 0.0
    use_q_filter: bool = True
    bc_minibatch_size: int = 128
    demo_replay_fraction: float = 0.0
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)
    sac: SACContinuousConfig | None = None
    kill_conditions_enabled: bool = True


def _concat_state(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Goal-conditioned state vector consumed by SAC."""
    return np.concatenate([np.asarray(obs, dtype=np.float32), np.asarray(goal, dtype=np.float32)])


def reverse_curriculum_progress(iteration: int, start_progress: float, decay_iters: int) -> float:
    """Per-episode reverse-curriculum start_progress schedule.

    Linear decay from ``start_progress`` at iteration 0 to 0 at
    ``decay_iters``; clamped to 0 thereafter. So early training starts the
    object near the goal (high start_progress) and the start state recedes
    to the true start as training proceeds (Florensa et al. 2017).
    """
    if decay_iters <= 0:
        return 0.0
    frac = 1.0 - float(iteration) / float(decay_iters)
    return float(np.clip(start_progress * frac, 0.0, 1.0))


def per_context_start_progress(
    base: float, contexts: list[int], unc_ema: np.ndarray, gate_strength: float, gate_on: bool
) -> list[float]:
    """Per-context reverse-curriculum start_progress, optionally uncertainty-gated.

    gate_on=False → every context gets ``base`` (vanilla RC). gate_on=True →
    ``base * (1 - g + 2*g*û(c))`` where û is the min-max normalized value-uncertainty EMA in
    [0,1] over the train contexts and g=gate_strength. High-uncertainty contexts keep a
    nearer-goal start; mean ≈ base (û mean ~0.5). Degenerate (flat/empty) û → vanilla.
    """
    if not gate_on or base <= 0.0:
        return [base] * len(contexts)
    lo, hi = float(np.min(unc_ema)), float(np.max(unc_ema))
    if hi - lo < 1e-9:
        return [base] * len(contexts)
    out = []
    for c in contexts:
        u = (float(unc_ema[c]) - lo) / (hi - lo)  # [0,1]
        sp = base * (1.0 - gate_strength + 2.0 * gate_strength * u)
        out.append(float(np.clip(sp, 0.0, 0.95)))
    return out


def _make_task_env(
    bank: Any,
    env_id: str,
    max_steps: int | None = None,
    scaffold_geometry: str = "geodesic",
    *,
    scaffold_source: str = ScaffoldSource.MANUAL,
    scaffold_ladder: ScaffoldLadder | None = None,
) -> Any:
    """Env factory: PointMaze/AntMaze -> maze, ``metaworld:<task>`` -> Meta-World, else Fetch.

    All wrappers share the reset(context_id)/step/spaces interface, so the loop is
    env-agnostic beyond this dispatch + the reward threshold. ``scaffold_geometry`` only
    affects PointMaze (geodesic vs off-path resets). ``scaffold_source="trajectory"`` (+ a
    ``scaffold_ladder``) makes PointMaze/Fetch restore extracted snapshots instead of the manual
    operator; Meta-World only supports the manual source today.
    """
    if env_id.startswith("metaworld:"):
        from pcrc.envs.metaworld_env import (  # noqa: PLC0415
            MetaWorldTaskEnvironment,
            parse_metaworld_env_id,
        )

        if scaffold_source != ScaffoldSource.MANUAL:
            raise NotImplementedError("Meta-World does not support scaffold_source='trajectory' yet")
        task, opts = parse_metaworld_env_id(env_id)
        return MetaWorldTaskEnvironment(bank, task, max_steps=max_steps, **opts)
    if env_id.startswith(("PointMaze", "AntMaze")):
        from pcrc.envs.pointmaze_env import PointMazeTaskEnvironment  # noqa: PLC0415

        return PointMazeTaskEnvironment(
            bank,
            env_id,
            max_steps=max_steps,
            scaffold_geometry=scaffold_geometry,
            scaffold_source=scaffold_source,
            scaffold_ladder=scaffold_ladder,
        )
    return FetchTaskEnvironment(
        bank,
        env_id,
        max_steps=max_steps,
        scaffold_source=scaffold_source,
        scaffold_ladder=scaffold_ladder,
    )


def _threshold_reward(threshold: float):
    """Sparse 0/-1 reward_fn at a given distance threshold (for HER relabel)."""

    def _fn(achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        del info
        d = float(np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal)))
        return 0.0 if d < threshold else -1.0

    return _fn


def _roll_one_episode(
    env: FetchTaskEnvironment,
    agent: SACAgent,
    context_id: int,
    episode_id: int,
    max_steps: int,
    success_distance_threshold: float,
    *,
    warmup: bool,
    rng: np.random.Generator,
    random_eps: float = 0.0,
    noise_eps: float = 0.0,
    update_rms: bool = True,
    start_progress: float = 0.0,
) -> tuple[list[GoalConditionedTransition], bool, float]:
    # start_progress > 0 (reverse curriculum) moves the object toward the
    # goal at reset; 0 is the true start. Eval always passes 0.
    obs_dict = env.reset(context_id, start_progress=start_progress)
    transitions: list[GoalConditionedTransition] = []
    success = False
    final_distance = float(np.linalg.norm(obs_dict["achieved_goal"] - obs_dict["desired_goal"]))
    low = env.action_space.low
    high = env.action_space.high
    for _step in range(max_steps):
        state = _concat_state(obs_dict["observation"], obs_dict["desired_goal"])
        if warmup:
            action = rng.uniform(low, high).astype(np.float32)
        else:
            # update_rms=False in parallel workers — the main process owns the
            # learner's obs_rms and folds the collected states into it after
            # each iteration (workers must not mutate a copied normalizer).
            action = agent.act(state, deterministic=False, update_rms=update_rms)
            # HER-paper / OpenAI-baselines exploration (Andrychowicz et al.
            # 2017, FetchPush config: noise_eps=0.2, random_eps=0.3). SAC's
            # auto-entropy alpha collapses on sparse Push before the gripper
            # ever contacts the block, so explicit action exploration is what
            # seeds the contacts HER then bootstraps from.
            if noise_eps > 0.0:
                action = np.clip(
                    action + rng.normal(0.0, noise_eps, size=action.shape).astype(np.float32),
                    low,
                    high,
                )
            if random_eps > 0.0 and rng.random() < random_eps:
                action = rng.uniform(low, high).astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        final_distance = float(np.linalg.norm(next_obs["achieved_goal"] - next_obs["desired_goal"]))
        ep_success = bool(info.get("is_success", final_distance < success_distance_threshold))
        if ep_success:
            success = True
        transitions.append(
            GoalConditionedTransition(
                observation=np.asarray(obs_dict["observation"], dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                next_observation=np.asarray(next_obs["observation"], dtype=np.float32),
                achieved_goal=np.asarray(obs_dict["achieved_goal"], dtype=np.float64),
                next_achieved_goal=np.asarray(next_obs["achieved_goal"], dtype=np.float64),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                source_context_id=int(context_id),
                original_desired_goal=np.asarray(obs_dict["desired_goal"], dtype=np.float64),
                training_desired_goal=np.asarray(obs_dict["desired_goal"], dtype=np.float64),
                relabel_strategy="none",
                relabel_source_step=None,
                episode_id=episode_id,
            )
        )
        obs_dict = next_obs
        if terminated or truncated:
            break
    return transitions, success, final_distance


def _harvest_ell0_sac(agent, env, train_ids, env_id, max_steps, k, seed):
    """Stage A harvest for SAC envs (Fetch / PointMaze): roll the trained policy at ℓ=0 (true start) on
    each train context — attempt 0 deterministic, the rest stochastic — up to ``k`` times; keep the
    *shortest* successful rollout's serialized full sim states (capture rule). Progress = cumulative
    achieved_goal arc length (matches the scripted builders). Returns ``{cid: [[state, progress], ...]}``.
    """
    if env_id.startswith(("PointMaze", "AntMaze")):
        from pcrc.envs.pointmaze_env import PointMazeStateCodec  # noqa: PLC0415

        codec = PointMazeStateCodec()
    else:
        from pcrc.envs.fetch_environment import FetchStateCodec  # noqa: PLC0415

        codec = FetchStateCodec()
    rng = np.random.default_rng(seed)
    u = env._env.unwrapped
    frames_by_ctx: dict[int, list] = {}
    for cid in train_ids:
        best = None
        for attempt in range(int(k)):
            obs = env.reset(int(cid), start_progress=0.0)
            prev = np.asarray(obs["achieved_goal"], dtype=np.float64)
            cum = 0.0
            frames = [[codec.serialize(u), 0.0]]
            success = False
            for _step in range(int(max_steps)):
                state = _concat_state(obs["observation"], obs["desired_goal"])
                action = agent.act(state, deterministic=(attempt == 0), update_rms=False)
                obs, _r, term, trunc, info = env.step(action)
                ag = np.asarray(obs["achieved_goal"], dtype=np.float64)
                cum += float(np.linalg.norm(ag - prev))
                prev = ag
                frames.append([codec.serialize(u), cum])
                if bool(info.get("is_success", False)):
                    success = True
                    break
                if term or trunc:
                    break
            if success and len(frames) >= 2 and (best is None or len(frames) < len(best)):
                best = frames
        if best is not None:
            frames_by_ctx[int(cid)] = best
    return frames_by_ctx


# ---------------------------------------------------------------------------
# Parallel rollout workers (CPU processes). MuJoCo rollout is CPU-bound and
# GIL-locked, so threads do not parallelize it (measured 0.65x); spawned
# processes do. Each worker holds a persistent env + a rollout-only SAC
# actor; the main process broadcasts the latest weights each iteration.
# ---------------------------------------------------------------------------

_WORKER_ENV: FetchTaskEnvironment | None = None
_WORKER_AGENT: SACAgent | None = None
_WORKER_MAX_STEPS: int = 50
_WORKER_SDT: float = 0.05


def _worker_init(
    bank: FetchTaskBank,
    env_id: str,
    sac_cfg: SACContinuousConfig,
    max_steps: int,
    success_distance_threshold: float,
    scaffold_geometry: str = "geodesic",
    scaffold_source: str = ScaffoldSource.MANUAL,
    scaffold_ladder: ScaffoldLadder | None = None,
) -> None:
    """Pool initializer: build one persistent env + actor per worker process."""
    import os  # noqa: PLC0415

    global _WORKER_ENV, _WORKER_AGENT, _WORKER_MAX_STEPS, _WORKER_SDT
    os.environ.pop("MUJOCO_GL", None)  # avoid the broken-EGL import path in workers
    torch.set_num_threads(1)  # N workers x M threads would oversubscribe the cores
    _WORKER_ENV = _make_task_env(
        bank,
        env_id,
        max_steps=max_steps,
        scaffold_geometry=scaffold_geometry,
        scaffold_source=scaffold_source,
        scaffold_ladder=scaffold_ladder,
    )
    _WORKER_AGENT = SACAgent(sac_cfg)
    _WORKER_MAX_STEPS = max_steps
    _WORKER_SDT = success_distance_threshold


def _worker_roll(task: tuple[Any, ...]) -> tuple[int, list[GoalConditionedTransition], bool, float]:
    """Roll one episode for ``context_id`` under the broadcast policy weights.

    Rollout-only: loads the actor + obs_rms snapshot, never mutates the
    normalizer (the main process folds collected states into the learner's
    obs_rms after the iteration).
    """
    (
        actor_sd,
        obs_rms_state,
        context_id,
        episode_id,
        warmup,
        random_eps,
        noise_eps,
        seed,
        start_progress,
    ) = task
    assert _WORKER_AGENT is not None and _WORKER_ENV is not None
    _WORKER_AGENT.actor.load_state_dict(actor_sd)
    _WORKER_AGENT.obs_rms.load_state_dict(obs_rms_state)
    transitions, success, final_distance = _roll_one_episode(
        _WORKER_ENV,
        _WORKER_AGENT,
        int(context_id),
        int(episode_id),
        _WORKER_MAX_STEPS,
        _WORKER_SDT,
        warmup=bool(warmup),
        rng=np.random.default_rng(int(seed)),
        random_eps=float(random_eps),
        noise_eps=float(noise_eps),
        update_rms=False,
        start_progress=float(start_progress),
    )
    return int(context_id), transitions, success, final_distance


def _build_state_action_reward_done(
    transitions: list[GoalConditionedTransition],
) -> dict[str, np.ndarray]:
    states = np.stack([_concat_state(t.observation, t.training_desired_goal) for t in transitions])
    next_states = np.stack(
        [_concat_state(t.next_observation, t.training_desired_goal) for t in transitions]
    )
    actions = np.stack([t.action for t in transitions])
    rewards = np.array([t.reward for t in transitions], dtype=np.float32)
    dones = np.array([1.0 if t.terminated else 0.0 for t in transitions], dtype=np.float32)
    return {
        "states": states.astype(np.float32),
        "actions": actions.astype(np.float32),
        "rewards": rewards,
        "next_states": next_states.astype(np.float32),
        "dones": dones,
    }


def _sac_actor_fn(net: SACAgent):
    """Deterministic (mean) action callable for a SAC agent's policy.

    Normalizes with the agent's running obs_rms — the networks were trained
    on normalized states (SACAgent.act/update), so the utility MUST normalize
    too or it evaluates Q / π out-of-distribution.
    """

    def _act(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        full_state = net.obs_rms.normalize(torch.cat([state, goal], dim=-1))
        mean, _ = net.actor(full_state)
        return torch.tanh(mean) * net.actor.action_scale + net.actor.action_bias

    return _act


def _sac_q_fn(net: SACAgent):
    """Double-Q min callable for a SAC agent (obs-normalized input)."""

    def _q(state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        full_state = net.obs_rms.normalize(torch.cat([state, goal], dim=-1))
        return torch.min(net.q1(full_state, action), net.q2(full_state, action))

    return _q


def _compute_critic_improvement(
    sampled: list[GoalConditionedTransition],
    agent_pre: SACAgent,
    agent: SACAgent,
) -> dict[int, float]:
    """Wrap :func:`sac_critic_improvement` against the SAC API."""
    return sac_critic_improvement(
        sampled,
        q_fn_t=_sac_q_fn(agent_pre),
        actor_fn_t=_sac_actor_fn(agent_pre),
        q_fn_tplus1=_sac_q_fn(agent),
        actor_fn_tplus1=_sac_actor_fn(agent),
    )


def _compute_actor_objective_improvement(
    sampled: list[GoalConditionedTransition],
    agent_pre: SACAgent,
    agent: SACAgent,
) -> dict[int, float]:
    """Wrap :func:`sac_actor_objective_improvement` (post critic for both)."""
    return sac_actor_objective_improvement(
        sampled,
        actor_fn_t=_sac_actor_fn(agent_pre),
        actor_fn_tplus1=_sac_actor_fn(agent),
        q_fn_tplus1=_sac_q_fn(agent),
    )


def _compute_is_advantage(
    rolled: list[tuple[int, list[GoalConditionedTransition], bool, float]],
    agent_pre: SACAgent,
    agent: SACAgent,
    ratio_clip: float = 5.0,
) -> dict[int, float]:
    """Actor-Curator IS-corrected per-context advantage (paper Eq. 7/10).

    For each rolled EPISODE (collected under π_old ≈ ``agent_pre``):
      log-ratio = Σ_t [log π_new(a_t|s_t) − log π_old(a_t|s_t)],  clipped to ±ratio_clip,
      w = exp(log-ratio),  r = episode success (0/1).
    Per context x: ``R_x = mean_y(w_y · r_y)`` (IS estimate of E_{π_new}[reward]),
    ``baseline_x = mean_y(r_y)`` (mean reward under π_old), ``Â_x = R_x − baseline_x``.

    This is the paper's actual utility (importance-weighted advantage), distinct from the
    critic-based proxies. Fetch's 50-step horizon keeps the trajectory IS ratio well-behaved
    (cf. technical_debt TD-09, a long-horizon LIBERO degeneracy). State = _concat_state(obs,
    original_desired_goal) — the goal the policy was conditioned on during rollout.
    """
    num: dict[int, float] = {}
    den: dict[int, float] = {}
    base: dict[int, float] = {}
    for cid, transitions, success, _fd in rolled:
        if not transitions:
            continue
        states = np.stack(
            [_concat_state(t.observation, t.original_desired_goal) for t in transitions]
        )
        actions = np.stack([np.asarray(t.action, dtype=np.float32) for t in transitions])
        lp_old = agent_pre.action_log_prob(states, actions).sum()
        lp_new = agent.action_log_prob(states, actions).sum()
        w = float(np.exp(np.clip(lp_new - lp_old, -ratio_clip, ratio_clip)))
        r = 1.0 if success else 0.0
        cid = int(cid)
        num[cid] = num.get(cid, 0.0) + w * r
        base[cid] = base.get(cid, 0.0) + r
        den[cid] = den.get(cid, 0.0) + 1.0
    return {cid: num[cid] / den[cid] - base[cid] / den[cid] for cid in num}


def _aggregate_per_context(
    sampled: list[GoalConditionedTransition], values: np.ndarray
) -> dict[int, float]:
    """Mean of per-transition ``values`` grouped by source_context_id."""
    acc: dict[int, list[float]] = {}
    for t, v in zip(sampled, values, strict=True):
        acc.setdefault(int(t.source_context_id), []).append(float(v))
    return {cid: float(np.mean(v)) for cid, v in acc.items()}


def _nt(agent: SACAgent, x: np.ndarray) -> torch.Tensor:
    """Numpy → device float tensor."""
    return torch.as_tensor(x, dtype=torch.float32, device=agent.device)


def _compute_td_error(
    sampled: list[GoalConditionedTransition], agent: SACAgent
) -> dict[int, float]:
    """Per-context mean |TD error| (critic surprise / learning opportunity).

    y = r + γ(1−d)[min_i Q^target_i(s',a') − α·logπ(a'|s')], a'~π(·|s');
    TD = |y − min_i Q_i(s,a)|. Distinct from the improvement utilities — measures where
    the critic is *wrong*, not where the policy *improved*.
    """
    arr = _build_state_action_reward_done(sampled)
    with torch.no_grad():
        s = agent.obs_rms.normalize(_nt(agent, arr["states"]))
        a = _nt(agent, arr["actions"])
        sn = agent.obs_rms.normalize(_nt(agent, arr["next_states"]))
        r = _nt(agent, arr["rewards"])
        d = _nt(agent, arr["dones"])
        a_next, logp_next, _ = agent.actor.sample(sn)
        alpha = agent.log_alpha.exp()
        q_next = (
            torch.min(agent.target_q1(sn, a_next), agent.target_q2(sn, a_next)) - alpha * logp_next
        )
        y = r + agent.cfg.gamma * (1.0 - d) * q_next
        q = torch.min(agent.q1(s, a), agent.q2(s, a))
        td = (y - q).abs().cpu().numpy()
    return _aggregate_per_context(sampled, td)


def _compute_q_disagreement(
    sampled: list[GoalConditionedTransition], agent: SACAgent
) -> dict[int, float]:
    """Per-context mean |Q1(s,a) − Q2(s,a)| (epistemic uncertainty via the double-Q heads)."""
    arr = _build_state_action_reward_done(sampled)
    with torch.no_grad():
        s = agent.obs_rms.normalize(_nt(agent, arr["states"]))
        a = _nt(agent, arr["actions"])
        dis = (agent.q1(s, a) - agent.q2(s, a)).abs().cpu().numpy()
    return _aggregate_per_context(sampled, dis)


def run_fetch_sac_her_loop(
    cfg: FetchSACHERLoopConfig,
    bank: FetchTaskBank,
    *,
    eval_env: FetchTaskEnvironment | None = None,
    logger: RunLogger | None = None,
    prefilled_replay: TaggedGlobalReplay | None = None,
) -> dict[str, Any]:
    """Run the Fetch SAC+HER loop. Returns the final metrics dict.

    Args:
        cfg: loop config.
        bank: pre-built bank with train + held-out split.
        eval_env: optional eval-bank-bound env. ``None`` skips eval.
        logger: optional RunLogger.
        prefilled_replay: optional buffer to start from (the §7.2
            smoke uses this to verify the distribution-matching path).
    """
    if cfg.sac is None:
        raise ValueError("FetchSACHERLoopConfig.sac must be set")
    if cfg.her_strategy not in ("future", "final", "cher"):
        raise ValueError(
            f"her_strategy must be 'future', 'final', or 'cher'; got {cfg.her_strategy!r}"
        )
    cher_cfg = (
        CHERConfig(
            lambda_starter=cfg.cher_lambda_starter,
            lambda_rate=cfg.cher_lambda_rate,
            fixed_lambda=cfg.cher_fixed_lambda,
            sub_size=cfg.cher_sub_size,
        )
        if cfg.her_strategy == "cher"
        else None
    )

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    train_ids = bank.train_ids()
    n_train = len(train_ids)
    train_mask = np.zeros(len(bank), dtype=np.bool_)
    train_mask[train_ids] = True

    env = _make_task_env(
        bank,
        cfg.env_id,
        max_steps=cfg.max_steps_per_episode,
        scaffold_geometry=cfg.scaffold_geometry,
        scaffold_source=cfg.scaffold_source,
        scaffold_ladder=cfg.scaffold_ladder,
    )
    _relabel_reward_fn = _threshold_reward(cfg.reward_distance_threshold)
    if cfg.learner == "crl":
        if cfg.crl_goal_dim is None:
            raise ValueError('learner="crl" requires crl_goal_dim')
        agent = CRLAgent(
            CRLContinuousConfig(
                sac=cfg.sac, goal_dim=int(cfg.crl_goal_dim), repr_dim=int(cfg.crl_repr_dim)
            )
        )
    else:
        agent = SACAgent(cfg.sac)
    sampler = build_sampler_for_condition(
        condition=cfg.condition,
        bank=bank,
        epsilon=cfg.epsilon,
        q_max=cfg.q_max,
        sampler_kwargs=cfg.sampler_kwargs,
    )
    # PCRC scaffold controller (None unless cfg.scaffold_enabled). When active it drives
    # cell selection + the start_progress; `sampler` still runs for distribution logging.
    controller: ScaffoldController | None = None
    if cfg.scaffold_enabled:
        controller = ScaffoldController(
            n_train=n_train,
            n_levels=cfg.scaffold_n_levels,
            level_fractions=list(cfg.scaffold_level_fractions),
            cfg=ScaffoldConfig(mode=cfg.scaffold_mode, **cfg.scaffold_kwargs),
            oracle_levels=cfg.scaffold_oracle_levels,
            replay_trajectory=cfg.scaffold_replay_trajectory,
            group_ids=cfg.scaffold_group_ids,
        )

    if cfg.replay_group_balanced and (
        cfg.replay_group_ids is None or len(cfg.replay_group_ids) != n_train
    ):
        raise ValueError(
            "replay_group_balanced requires replay_group_ids of length n_train "
            f"({n_train}); got {None if cfg.replay_group_ids is None else len(cfg.replay_group_ids)}"
        )
    replay = (
        prefilled_replay
        if prefilled_replay is not None
        else TaggedGlobalReplay(capacity=cfg.replay_capacity)
    )
    if logger is None:
        logger = RunLogger(cfg.exp_name)
    kill_monitor = KillConditionMonitor(n_arms=n_train) if cfg.kill_conditions_enabled else None

    # Demonstrations + behavior-cloning lever (Nair et al. 2018). Load demos
    # once: (a) HER-relabel + push them into the replay buffer so the critic
    # bootstraps from successful grasps (overcomes the in-air PnP grasp wall),
    # and (b) keep the raw (state, action) pairs for the BC auxiliary loss.
    # OFF unless bc_lambda > 0 AND demos_path is set — default path unchanged.
    demo_states: np.ndarray | None = None
    demo_actions: np.ndarray | None = None
    demo_pool: list[GoalConditionedTransition] | None = None  # protected demo buffer (Nair 2018)
    bc_active = cfg.bc_lambda > 0.0 and cfg.demos_path is not None
    if bc_active:
        from pcrc.algorithms.pnp_demos import load_demos  # noqa: PLC0415

        demos = load_demos(cfg.demos_path)
        demo_states = np.asarray(demos["state"], dtype=np.float32)
        demo_actions = np.asarray(demos["action"], dtype=np.float32)
        # Reconstruct GoalConditionedTransitions from the flat demo arrays so
        # they pass through the same HER relabel + replay.add path as rollouts.
        # state = [observation, desired_goal]. The achieved_goal (object pos) is
        # read from explicit demo columns when present (env-agnostic), falling
        # back to the Fetch obs[3:6] slice for legacy demos without them.
        has_ag = "achieved_goal" in demos and "next_achieved_goal" in demos
        if has_ag:
            goal_dim = int(np.asarray(demos["achieved_goal"]).shape[1])
        else:
            goal_dim = int(bank[bank.train_ids()[0]].desired_goal.shape[0])
        obs_dim_demo = demo_states.shape[1] - goal_dim
        demo_episode_offset = 1_000_000  # keep demo episode_ids disjoint from rollouts
        demo_transitions: list[GoalConditionedTransition] = []
        prev_ctx = None
        ep_counter = 0
        for i in range(demo_states.shape[0]):
            obs = demo_states[i, :obs_dim_demo]
            goal = demo_states[i, obs_dim_demo:]
            next_obs = demos["next_state"][i, :obs_dim_demo]
            cid = int(demos["source_context_id"][i])
            if cid != prev_ctx:
                ep_counter += 1
                prev_ctx = cid
            ag = demos["achieved_goal"][i] if has_ag else obs[3:6]
            next_ag = demos["next_achieved_goal"][i] if has_ag else next_obs[3:6]
            demo_transitions.append(
                GoalConditionedTransition(
                    observation=np.asarray(obs, dtype=np.float32),
                    action=np.asarray(demo_actions[i], dtype=np.float32),
                    next_observation=np.asarray(next_obs, dtype=np.float32),
                    achieved_goal=np.asarray(ag, dtype=np.float64),
                    next_achieved_goal=np.asarray(next_ag, dtype=np.float64),
                    reward=float(demos["reward"][i]),
                    terminated=bool(demos["terminated"][i]),
                    truncated=False,
                    source_context_id=cid,
                    original_desired_goal=np.asarray(goal, dtype=np.float64),
                    training_desired_goal=np.asarray(goal, dtype=np.float64),
                    relabel_strategy="none",
                    relabel_source_step=None,
                    episode_id=demo_episode_offset + ep_counter,
                )
            )
        demo_relabeled = relabel(
            demo_transitions,
            strategy=cfg.her_strategy,
            n_sampled_goal=cfg.her_n_sampled_goal,
            reward_fn=_relabel_reward_fn,
            rng=rng,
        )
        if cfg.demo_replay_fraction > 0.0:
            # Protected demo pool (Nair 2018): kept out of the main replay so it
            # is never evicted; a fixed fraction of each minibatch is drawn here.
            demo_pool = list(demo_relabeled)
        else:
            for t in demo_relabeled:
                replay.add(t)

    success_ema = np.zeros(len(bank), dtype=np.float64)
    visit_counts = np.zeros(len(bank), dtype=np.int64)
    # Per-context value-uncertainty EMA for uncertainty-gated reverse curriculum (stale by one
    # iter: updated after each iteration's SAC updates, read at the next iteration's rollout).
    rc_unc_ema = np.zeros(n_train, dtype=np.float64)
    utility_ema = np.zeros(len(bank), dtype=np.float64)  # for ac_ema_center
    last_metrics: dict[str, Any] = {}
    last_eval: dict[str, Any] | None = None
    episode_id = 0
    # Cumulative REAL env steps consumed by training rollouts (one per
    # GoalConditionedTransition, pre-HER). This is the budget axis for all
    # curriculum-vs-baseline comparisons — eval rollouts are NOT counted.
    # ("step" in the logs is iteration count and is NOT comparable across runs
    # with different num_workers; always normalize on env_steps.)
    env_steps_total = 0

    # Spawn a CPU worker pool for parallel rollout when num_workers > 1.
    # spawn (not fork) is required — MuJoCo state does not survive a fork.
    pool = None
    if cfg.num_workers > 1:
        import multiprocessing as mp  # noqa: PLC0415

        pool = mp.get_context("spawn").Pool(
            processes=cfg.num_workers,
            initializer=_worker_init,
            initargs=(
                bank,
                cfg.env_id,
                cfg.sac,
                cfg.max_steps_per_episode,
                cfg.success_distance_threshold,
                cfg.scaffold_geometry,
                cfg.scaffold_source,
                cfg.scaffold_ladder,
            ),
        )

    try:
        for it in range(cfg.n_iterations):
            t_iter = time.time()
            warmup = len(replay) < cfg.learning_starts
            n_eps = max(1, cfg.num_workers)
            # PCRC path: the controller selects (context, level) cells and emits the
            # per-episode start_progress directly. ep_levels carries the level each rolled
            # episode used (needed to attribute per-cell uncertainty in update below).
            if controller is not None:
                ep_contexts, ep_levels, sp_list = controller.select(n_eps, step=it, rng=rng)
            else:
                ep_levels = None
                batch = sampler.sample_flat(n_eps, step=it, rng=rng)
                ep_contexts = list(batch.sampled_context_ids)

                # Reverse-curriculum start_progress for THIS training iteration
                # (0 when disabled). Eval always uses 0 — see evaluate_fetch.
                _base_sp = (
                    reverse_curriculum_progress(
                        it,
                        cfg.reverse_curriculum_start_progress,
                        cfg.reverse_curriculum_decay_iters,
                    )
                    if cfg.reverse_curriculum
                    else 0.0
                )
                # Per-context start_progress (uncertainty-gated if enabled, else uniform).
                sp_list = per_context_start_progress(
                    _base_sp,
                    ep_contexts,
                    rc_unc_ema,
                    cfg.reverse_curriculum_gate_strength,
                    cfg.reverse_curriculum and cfg.reverse_curriculum_uncertainty_gate,
                )

            # rolled: list of (context_id, transitions, success, final_distance).
            if pool is not None:
                actor_sd = {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()}
                obs_rms_state = agent.obs_rms.state_dict()
                tasks = [
                    (
                        actor_sd,
                        obs_rms_state,
                        cid,
                        episode_id + i,
                        warmup,
                        cfg.random_eps,
                        cfg.noise_eps,
                        int(cfg.seed + it * n_eps + i),
                        sp_list[i],
                    )
                    for i, cid in enumerate(ep_contexts)
                ]
                rolled = pool.map(_worker_roll, tasks)
                episode_id += n_eps
                # Workers roll with update_rms=False on a copied normalizer; the
                # learner folds all collected states into its obs_rms here.
                all_states = np.concatenate(
                    [
                        np.stack(
                            [_concat_state(t.observation, t.training_desired_goal) for t in tr]
                        )
                        for (_c, tr, _s, _fd) in rolled
                    ]
                )
                agent.obs_rms.update(
                    torch.as_tensor(all_states, dtype=torch.float32, device=agent.device)
                )
            else:
                cid0 = ep_contexts[0]
                transitions, success, final_distance = _roll_one_episode(
                    env,
                    agent,
                    cid0,
                    episode_id,
                    cfg.max_steps_per_episode,
                    cfg.success_distance_threshold,
                    warmup=warmup,
                    rng=rng,
                    random_eps=cfg.random_eps,
                    noise_eps=cfg.noise_eps,
                    start_progress=sp_list[0],
                )
                episode_id += 1
                rolled = [(cid0, transitions, success, final_distance)]

            # Exact real env steps this iteration = total transitions rolled
            # (each GoalConditionedTransition is one env.step, pre-HER).
            env_steps_total += int(sum(len(tr) for (_c, tr, _s, _fd) in rolled))

            for _c, transitions, _s, _fd in rolled:
                relabeled = relabel(
                    transitions,
                    strategy=cfg.her_strategy,
                    n_sampled_goal=cfg.her_n_sampled_goal,
                    reward_fn=_relabel_reward_fn,
                    rng=rng,
                    cher_cfg=cher_cfg,
                    step=it,
                )
                if cfg.reward_shaping_coef > 0.0:
                    # Shaping on every transition (originals + HER copies).
                    # gripper = obs[:3], object = achieved_goal.
                    for t in relabeled:
                        gripper = np.asarray(t.observation[:3], dtype=np.float64)
                        obj = np.asarray(t.achieved_goal, dtype=np.float64)
                        reach = float(np.linalg.norm(gripper - obj))
                        if cfg.reward_shaping_mode == "staged":
                            # reach + grasp/lift (goal-indep) + place (goal-dep, HER-safe)
                            lift = float(np.clip((obj[2] - 0.425) / 0.1, 0.0, 1.0))
                            goal = np.asarray(t.training_desired_goal, dtype=np.float64)
                            place = float(np.linalg.norm(obj - goal))
                            shape = -reach + 1.5 * lift - place
                            t.reward = float(t.reward + cfg.reward_shaping_coef * shape)
                        elif cfg.reward_shaping_mode == "grasp_only":
                            # Partial unlock: shape reach + grasp/lift (the discrete skill
                            # bottleneck, both goal-INDEPENDENT and HER-safe), but leave PLACEMENT
                            # to the sparse 0/-1 success reward. Breaks the grasp wall while
                            # preserving within-difficulty placement heterogeneity (the band the
                            # value-uncertainty curriculum needs) — unlike "staged", which shapes
                            # place too and saturates the band.
                            lift = float(np.clip((obj[2] - 0.425) / 0.1, 0.0, 1.0))
                            shape = -reach + 1.5 * lift
                            t.reward = float(t.reward + cfg.reward_shaping_coef * shape)
                        else:
                            t.reward = float(t.reward - cfg.reward_shaping_coef * reach)
                for t in relabeled:
                    replay.add(t)

            context_id = int(rolled[0][0])
            mean_success = float(np.mean([float(s) for (_c, _t, s, _fd) in rolled]))
            mean_final_distance = float(np.mean([fd for (_c, _t, _s, fd) in rolled]))

            sac_metrics: dict[str, float] = {}
            utility_per_context: dict[int, float] = {}
            diag_td: dict[int, float] = {}
            diag_qdis: dict[int, float] = {}
            # Per-episode mean |TD-error| for the PCRC controller (0 during warmup /
            # when the controller is off). Filled post-update from the rolled episodes so
            # the level dimension is preserved (unlike the level-mixing replay path).
            ep_td = [0.0] * len(rolled)
            if len(replay) >= cfg.learning_starts:
                dist = sampler.distribution(step=it)
                # Extend the distribution to full bank length so the
                # buffer can index it by source_context_id.
                full = np.zeros(len(bank), dtype=np.float64)
                full[:n_train] = dist.final
                from pcrc.samplers.exploration_floor import (  # noqa: PLC0415
                    apply_floor_and_cap,
                )

                full_dist = apply_floor_and_cap(
                    full / max(full.sum(), 1e-12), epsilon=0.0, q_max=None
                )
                # C5.7 group-balanced replay: per-group restrictions of full_dist, rebuilt each
                # iteration so any sampler-driven change in the base distribution carries over.
                group_dists = None
                if cfg.replay_group_balanced:
                    gids = np.asarray(cfg.replay_group_ids, dtype=np.int64)
                    group_dists = []
                    for g in range(int(gids.max()) + 1):
                        m = np.zeros(len(bank), dtype=np.float64)
                        members = np.flatnonzero(gids == g)
                        m[members] = full_dist.final[members]
                        group_dists.append(
                            apply_floor_and_cap(m / max(m.sum(), 1e-12), epsilon=0.0, q_max=None)
                        )
                agent_pre = (
                    copy.deepcopy(agent)
                    if (
                        cfg.condition
                        in (
                            SamplerName.TABULAR_ACTOR_CURATOR,
                            SamplerName.MLP_ACTOR_CURATOR,
                            SamplerName.HEADROOM_ACTOR,
                            SamplerName.UNCERTAINTY,
                            SamplerName.PLR,
                        )
                        or cfg.diagnostic_compute_utility
                    )
                    else None
                )
                for _ in range(cfg.updates_per_iter):
                    if cfg.learner == "crl":
                        # CRL: minibatch of HER-"future" relabels — (obs, action, future achieved
                        # goal) are the InfoNCE positives; no reward/done/bootstrapping used.
                        cand = replay.sample_context_weighted(
                            cfg.minibatch_size * 2, full_dist, rng
                        )
                        fut = [t for t in cand if t.relabel_strategy == "future"]
                        fut = fut[: cfg.minibatch_size]
                        if len(fut) >= 8:
                            sac_metrics = agent.update_crl(
                                np.stack([t.observation for t in fut]).astype(np.float32),
                                np.stack([t.action for t in fut]).astype(np.float32),
                                np.stack([t.training_desired_goal for t in fut]).astype(
                                    np.float32
                                ),
                            )
                        continue
                    if group_dists is not None:
                        # C5.7: fixed per-group quotas; a group with nothing stored yet (or a
                        # dead masked distribution) cedes its quota to the full distribution.
                        sampled = []
                        quota = cfg.minibatch_size // len(group_dists)
                        for gd in group_dists:
                            try:
                                sampled.extend(replay.sample_context_weighted(quota, gd, rng))
                            except ValueError:
                                sampled.extend(
                                    replay.sample_context_weighted(quota, full_dist, rng)
                                )
                        if (rem := cfg.minibatch_size - len(sampled)) > 0:
                            sampled.extend(replay.sample_context_weighted(rem, full_dist, rng))
                    else:
                        sampled = replay.sample_context_weighted(
                            cfg.minibatch_size, full_dist, rng
                        )
                    # Protected demo buffer (Nair 2018): replace a fixed fraction of the
                    # minibatch with never-evicted demo transitions so the critic keeps
                    # seeing successes and does not diverge as failures fill the buffer.
                    if demo_pool and cfg.demo_replay_fraction > 0.0:
                        n_demo = min(
                            len(sampled),
                            int(round(cfg.demo_replay_fraction * cfg.minibatch_size)),
                        )
                        if n_demo > 0:
                            didx = rng.integers(0, len(demo_pool), size=n_demo)
                            sampled = sampled[: cfg.minibatch_size - n_demo] + [
                                demo_pool[i] for i in didx
                            ]
                    arrays = _build_state_action_reward_done(sampled)
                    # BC auxiliary loss (Nair 2018): sample a demo (s,a) minibatch
                    # uniformly each update. None when BC is off → vanilla SAC.
                    bc_s = bc_a = None
                    if bc_active and demo_states is not None:
                        n_demo = demo_states.shape[0]
                        idx = rng.integers(0, n_demo, size=min(cfg.bc_minibatch_size, n_demo))
                        bc_s = demo_states[idx]
                        bc_a = demo_actions[idx]
                    # PCGrad: split the minibatch into goal-z groups (z = last state column) so the
                    # update projects out conflicting per-group gradients (interference probe).
                    pc_gids = None
                    if cfg.sac.pcgrad:
                        pc_gids = (arrays["states"][:, -1] >= cfg.pcgrad_goalz_threshold).astype(
                            np.int64
                        )
                    sac_metrics = agent.update(
                        arrays["states"],
                        arrays["actions"],
                        arrays["rewards"],
                        arrays["next_states"],
                        arrays["dones"],
                        demo_states=bc_s,
                        demo_actions=bc_a,
                        bc_lambda=cfg.bc_lambda,
                        use_q_filter=cfg.use_q_filter,
                        group_ids=pc_gids,
                    )
                # Uncertainty-gated reverse curriculum: refresh the per-context |TD-error| EMA
                # (used at the NEXT iteration's rollout to allocate start_progress). Independent
                # of the sampler/ac_utility so it works under uniform sampling.
                if cfg.reverse_curriculum and cfg.reverse_curriculum_uncertainty_gate:
                    _us = replay.sample_context_weighted(cfg.minibatch_size, full_dist, rng)
                    for cid, u in _compute_td_error(_us, agent).items():
                        if cid < n_train:
                            rc_unc_ema[cid] = 0.8 * rc_unc_ema[cid] + 0.2 * float(u)
                # PCRC per-cell uncertainty: mean |TD-error| on each ROLLED episode's own
                # transitions (level known via ep_levels), keyed back per episode. Computed on
                # the post-update critic, mirroring rc_unc_ema, but never level-mixed.
                if controller is not None and getattr(agent, "supports_td", True):
                    for _i, (_cid_i, _tr_i, _s_i, _fd_i) in enumerate(rolled):
                        _td_i = _compute_td_error(_tr_i, agent)
                        ep_td[_i] = float(_td_i.get(int(_cid_i), 0.0))
                if agent_pre is not None:
                    sampled_for_util = replay.sample_context_weighted(
                        cfg.minibatch_size, full_dist, rng
                    )
                    if cfg.diagnostic_compute_utility:
                        # Extra diagnostic utilities (distinct families) logged
                        # alongside the main one: critic surprise + epistemic
                        # uncertainty. Sampler ignores them; for within-band corr.
                        diag_td = _compute_td_error(sampled_for_util, agent)
                        diag_qdis = _compute_q_disagreement(sampled_for_util, agent)
                    if cfg.ac_utility == "constant":
                        # Constant base (1.0 per observed context). With
                        # ac_headroom_gate this becomes pure s(1-s) routed
                        # through OSMD — a mechanism check isolating the
                        # OSMD/curator path from the actor-aware signal.
                        utility_per_context = {
                            int(t.source_context_id): 1.0 for t in sampled_for_util
                        }
                    elif cfg.ac_utility == "actor_objective":
                        utility_per_context = _compute_actor_objective_improvement(
                            sampled_for_util, agent_pre, agent
                        )
                    elif cfg.ac_utility == "td_error":
                        # Value-uncertainty curriculum (R5): critic surprise.
                        utility_per_context = _compute_td_error(sampled_for_util, agent)
                    elif cfg.ac_utility == "q_disagreement":
                        utility_per_context = _compute_q_disagreement(sampled_for_util, agent)
                    elif cfg.ac_utility == "is_advantage":
                        # Paper's actual utility: IS-corrected per-context advantage
                        # over the rolled EPISODES (collected under π_old≈agent_pre),
                        # NOT the replay minibatch.
                        utility_per_context = _compute_is_advantage(rolled, agent_pre, agent)
                    else:
                        utility_per_context = _compute_critic_improvement(
                            sampled_for_util, agent_pre, agent
                        )
                    # H3 EMA-center: subtract each context's own running-mean
                    # utility (de-bias the systematic easy-improvement). Update
                    # the EMA AFTER reading it so the centering uses the prior
                    # baseline.
                    if cfg.ac_ema_center:
                        centered = {}
                        for cid, u in utility_per_context.items():
                            centered[cid] = u - utility_ema[cid]
                            utility_ema[cid] = (1.0 - cfg.ac_ema_alpha) * utility_ema[
                                cid
                            ] + cfg.ac_ema_alpha * u
                        utility_per_context = centered
                    # H2 headroom-gate: restrict the actor-aware signal to the
                    # intermediate-difficulty frontier via s_ema*(1-s_ema).
                    if cfg.ac_headroom_gate:
                        utility_per_context = {
                            cid: u * success_ema[cid] * (1.0 - success_ema[cid])
                            for cid, u in utility_per_context.items()
                        }
                    # Center + scale to unit std before OSMD (the tabular
                    # sampler does NOT standardize — callers must, per its
                    # contract, exactly like the PPO loop's utility_scale).
                    # Critic-improvement and actor-objective have very
                    # different raw magnitudes; without this a shared eta
                    # over-concentrates one and barely moves the other.
                    if len(utility_per_context) > 1:
                        _uvals = np.array(list(utility_per_context.values()), dtype=np.float64)
                        _mu, _sd = _uvals.mean(), _uvals.std()
                        if _sd > 1e-8:
                            utility_per_context = {
                                k: (v - _mu) / _sd for k, v in utility_per_context.items()
                            }

            samples = np.zeros(n_train, dtype=np.int64)
            success_sum = np.zeros(n_train, dtype=np.float64)
            return_sum = np.zeros(n_train, dtype=np.float64)
            for cid, _tr, ep_success, ep_fd in rolled:
                if cid < n_train:
                    samples[cid] += 1
                    success_sum[cid] += 1.0 if ep_success else 0.0
                    return_sum[cid] += -ep_fd
            utility_sum = np.zeros(n_train, dtype=np.float64)
            for cid, u in utility_per_context.items():
                if cid < n_train:
                    utility_sum[cid] = u * max(samples[cid], 1)
            observed_mask = samples > 0
            stats = SamplerStats(
                observed_mask=observed_mask,
                samples_this_step=samples,
                success_sum_this_step=success_sum,
                return_sum_this_step=return_sum,
                utility_sum_this_step=utility_sum,
                utility_sq_sum_this_step=np.zeros(n_train, dtype=np.float64),
                success_ema=success_ema[:n_train].copy(),
                return_ema=np.zeros(n_train, dtype=np.float64),
                utility_ema=np.zeros(n_train, dtype=np.float64),
                n_visits_total=visit_counts[:n_train].copy(),
            )
            sampler.update(stats, step=it)
            # PCRC controller: fold this iteration's rolled cells into the cell table and
            # move per-context frontiers. ep_levels is set only on the controller path.
            if controller is not None:
                controller.update(
                    contexts=ep_contexts,
                    levels=ep_levels,
                    successes=[bool(s) for (_c, _t, s, _fd) in rolled],
                    td_errors=ep_td,
                    step=it,
                )

            for cid, _tr, ep_success, _fd in rolled:
                if cid < n_train:
                    success_ema[cid] = (1.0 - cfg.success_ema_alpha) * success_ema[
                        cid
                    ] + cfg.success_ema_alpha * (1.0 if ep_success else 0.0)
                    visit_counts[cid] += 1

            dist_log = sampler.distribution(step=it + 1)
            full_distribution = np.zeros(len(bank), dtype=np.float64)
            full_distribution[:n_train] = dist_log.final
            full_distribution = full_distribution / max(full_distribution.sum(), 1e-12)
            decomp = compute_selection_vs_learning_decomposition(
                success_ema=success_ema,
                distribution_final=full_distribution,
                train_mask=train_mask,
            )
            training_metrics = {
                "step": it + 1,
                "env_steps": env_steps_total,
                "wall_iter_s": time.time() - t_iter,
                "context_id": int(context_id),
                "n_episodes": len(rolled),
                "success": float(mean_success),
                "final_distance": float(mean_final_distance),
                "replay_size": len(replay),
                "pg_loss": float(sac_metrics.get("actor_loss", 0.0)),
                "rewards_mean": float(decomp["sampler_weighted_reward"]),
                **sac_metrics,
                **decomp,
                "sampler_entropy": float(dist_log.entropy),
                "sampler_max_probability": float(dist_log.max_probability),
            }
            last_metrics = training_metrics
            stream_prefix = f"training/{cfg.condition}"
            logger.log_step(f"{stream_prefix}/metrics.jsonl", training_metrics)
            logger.log_step(
                f"{stream_prefix}/sampler_distribution.jsonl",
                {
                    "step": it + 1,
                    "raw": dist_log.raw.tolist(),
                    "floored": dist_log.floored.tolist(),
                    "capped": dist_log.capped.tolist(),
                    "final": dist_log.final.tolist(),
                    "entropy": float(dist_log.entropy),
                    "max_probability": float(dist_log.max_probability),
                },
            )
            logger.log_step(
                f"{stream_prefix}/curator_weights.jsonl",
                {"step": it + 1, "weights": dist_log.raw.tolist()},
            )
            logger.log_step(
                f"{stream_prefix}/utilities.jsonl",
                {
                    "step": it + 1,
                    "per_context_utility": {
                        str(cid): float(u) for cid, u in utility_per_context.items()
                    },
                },
            )
            if cfg.diagnostic_compute_utility:
                for fname, dvals in (("utilities_td", diag_td), ("utilities_qdis", diag_qdis)):
                    logger.log_step(
                        f"{stream_prefix}/{fname}.jsonl",
                        {
                            "step": it + 1,
                            "per_context_utility": {str(c): float(v) for c, v in dvals.items()},
                        },
                    )
            logger.log_step(
                f"{stream_prefix}/uniformized_reward.jsonl",
                {"step": it + 1, "uniformized_reward": decomp["uniformized_reward"]},
            )
            logger.log_step(
                f"{stream_prefix}/selection_learning_decomposition.jsonl",
                {"step": it + 1, **decomp},
            )
            logger.log_step(
                f"{stream_prefix}/context_visit_counts.jsonl",
                {"step": it + 1, "visit_counts": visit_counts[:n_train].tolist()},
            )
            if controller is not None:
                logger.log_step(f"{stream_prefix}/scaffold.jsonl", controller.snapshot())

            if (
                eval_env is not None
                and cfg.eval_interval > 0
                and ((it + 1) % cfg.eval_interval == 0 or it == cfg.n_iterations - 1)
            ):

                def _greedy_actor(state: np.ndarray) -> np.ndarray:
                    # evaluate_fetch passes the goal-conditioned observation
                    # ([observation, desired_goal]) which already equals the
                    # SAC train state composition (_concat_state) — use it
                    # directly; re-concatenating would double the goal.
                    return agent.act(state, deterministic=True)

                eval_result = evaluate_fetch(
                    env=eval_env,
                    actor_fn=_greedy_actor,
                    bank=bank,
                    n_eval_episodes_per_context=cfg.n_eval_episodes_per_context,
                    max_steps_per_episode=cfg.max_steps_per_episode,
                    distribution=dist_log,
                    success_distance_threshold=cfg.success_distance_threshold,
                )
                last_eval = eval_result
                eval_prefix = f"eval/{cfg.condition}"
                logger.log_step(
                    f"{eval_prefix}/held_out.jsonl",
                    {
                        "step": it + 1,
                        "env_steps": env_steps_total,
                        "success_rate": eval_result["success_rate"],
                        "mean_dense_return": eval_result["mean_dense_return"],
                    },
                )
                logger.log_step(
                    f"{eval_prefix}/per_bin.jsonl",
                    {"step": it + 1, **eval_result["per_bin_success"]},
                )
                logger.log_step(
                    f"{eval_prefix}/per_context.jsonl",
                    {
                        "step": it + 1,
                        Method.PER_CONTEXT_SUCCESS: {
                            str(cid): v for cid, v in eval_result[Method.PER_CONTEXT_SUCCESS].items()
                        },
                    },
                )
                # Per-TRAIN-context success EMA (training distribution), logged at
                # eval cadence so it aligns with utilities.jsonl. Enables the
                # diagnostic corr(u_actor(x,t), future Δsuccess(x)) within
                # difficulty bands — the test of whether the actor signal carries
                # learnability info beyond the headroom difficulty signal.
                logger.log_step(
                    f"training/{cfg.condition}/train_context_success.jsonl",
                    {
                        "step": it + 1,
                        "env_steps": env_steps_total,
                        "success_ema": success_ema[:n_train].tolist(),
                    },
                )

            if kill_monitor is not None:
                reason = kill_monitor.check(
                    iter=it + 1,
                    metrics=training_metrics,
                    # Guard the SAMPLED distribution, not raw OSMD weights —
                    # the floor + q_max cap (plan §4) keep dist_log.final
                    # diverse; raw always concentrates. See fetch_ppo_loop.
                    osmd_weights=dist_log.final,
                    last_ckpt=None,
                )
                if reason is not None:
                    write_kill_reason(reason, logger.dir)
                    raise RuntimeError(
                        f"Fetch SAC+HER loop aborted at iter {reason.iter}: "
                        f"{reason.name}; diagnostic={reason.diagnostic}"
                    )
        if cfg.harvest_path:
            frames = _harvest_ell0_sac(
                agent, env, train_ids, cfg.env_id,
                cfg.max_steps_per_episode, cfg.harvest_k, cfg.harvest_seed,
            )
            out = {
                "env_id": cfg.env_id,
                "source": "policy_rollout",
                "n_train": n_train,
                "n_solved": len(frames),
                "harvest_k": cfg.harvest_k,
                "harvest_seed": cfg.harvest_seed,
                "frames": {str(c): f for c, f in frames.items()},
            }
            hp = Path(cfg.harvest_path)
            hp.parent.mkdir(parents=True, exist_ok=True)
            hp.write_text(json.dumps(out))
            print(
                f"[fetch harvest] ℓ=0 self-success on {len(frames)}/{n_train} train "
                f"contexts (k={cfg.harvest_k}) -> {hp}",
                flush=True,
            )
        # Final policy snapshot for post-hoc rollout and rendering: actor + obs_rms
        # are exactly what _worker_roll needs to reproduce deterministic behavior.
        ckpt_path = logger.dir / "checkpoint_final.pt"
        torch.save(
            {
                "actor": {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()},
                "obs_rms": agent.obs_rms.state_dict(),
                "condition": cfg.condition,
                "env_id": cfg.env_id,
            },
            ckpt_path,
        )
        print(f"[fetch loop] final checkpoint -> {ckpt_path}", flush=True)
    finally:
        env.close()
        logger.close()
        if pool is not None:
            pool.close()
            pool.join()
    return {"last_metrics": last_metrics, "last_eval": last_eval}
