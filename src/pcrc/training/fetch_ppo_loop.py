"""Dense Fetch PPO loop driving any :class:`Sampler` over a task bank.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``fetch_ppo_loop.py`` row), §4 Priority 1b / 1c (driver for dense
PPO Fetch runs), §8 (RunLogger streams).

Per-iteration flow:

    1. ``sampler.sample_flat`` → context ids for this iter.
    2. For each (context, rollout), reset env and roll the PPO actor
       to ``max_steps_per_episode`` collecting per-step (obs, action,
       old_logp, value, reward, done).
    3. Compute GAE per episode; flatten into the PPO update minibatch.
    4. ``PPOAgent.update`` runs ``n_epochs`` PPO gradient steps.
    5. Recompute new log-probs under the updated policy at the same
       (obs, action) pairs; call
       :func:`ppo_policy_improvement_utility` for AC conditions only.
    6. Build :class:`SamplerStats` from per-context counts and
       utilities; call ``sampler.update``.
    7. Update running per-context EMAs (drive uniformized reward).
    8. Log to all mandatory streams (§8).
    9. Eval every ``eval_interval`` iters via
       :func:`pcrc.training.fetch_evaluation.evaluate_fetch`.
   10. Kill-condition check via
       :class:`pcrc.training.kill_conditions.KillConditionMonitor`.

The condition → sampler factory is :func:`build_sampler_for_condition`.
Plumbing keeps the loop body env-agnostic — pass any Fetch env id with
matching action / observation dim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from pcrc.algorithms.policy_improvement_utility import ppo_policy_improvement_utility
from pcrc.algorithms.ppo_continuous import (
    PPOAgent,
    PPOContinuousConfig,
    compute_gae,
)
from pcrc.envs.fetch_environment import (
    FetchTaskEnvironment,
    goal_conditioned_observation,
)
from pcrc.envs.fetch_task_bank import FetchTaskBank
from pcrc.registry import Method, SamplerName
from pcrc.samplers.acdc_lite import ACDCLiteSampler
from pcrc.samplers.base import Sampler, SamplerStats
from pcrc.samplers.headroom import HeadroomSampler
from pcrc.samplers.headroom_actor import HeadroomActorSampler
from pcrc.samplers.learning_progress import LearningProgressSampler
from pcrc.samplers.mlp_actor_curator import MLPActorCuratorSampler
from pcrc.samplers.plr import PLRSampler
from pcrc.samplers.reward_greedy import RewardGreedySampler
from pcrc.samplers.tabular_actor_curator import TabularActorCuratorSampler
from pcrc.samplers.uncertainty import UncertaintySampler
from pcrc.samplers.uniform import UniformSampler
from pcrc.training.fetch_evaluation import evaluate_fetch
from pcrc.training.fetch_uniformized_reward import (
    compute_selection_vs_learning_decomposition,
)
from pcrc.training.group_diagnostics import classify_groups
from pcrc.training.kill_conditions import KillConditionMonitor, write_kill_reason
from pcrc.utils.results import RunLogger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FetchPPOLoopConfig:
    """All-in-one config for the Fetch PPO driver."""

    exp_name: str
    condition: str
    env_id: str
    n_iterations: int = 50
    episodes_per_iteration: int = 8
    max_steps_per_episode: int = 50
    eval_interval: int = 10
    n_eval_episodes_per_context: int = 4
    seed: int = 0
    epsilon: float = 0.05
    q_max: float | None = None
    utility_clip_eps: float = 0.2
    utility_center: bool = True
    utility_scale: bool = False
    success_distance_threshold: float = 0.05
    # Optional dense reward shaping for on-policy Push: add
    # -coef*||gripper - object|| each step to break the gripper-object
    # contact exploration wall (raw FetchPushDense reward only moves once
    # the block does). Training-only — eval still measures true is_success.
    # Lets PPO learn Push so the paper's on-policy advantage AC utility can
    # be tested in a learnable, non-saturated manipulation setting.
    reward_shaping_coef: float = 0.0
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)
    ppo: PPOContinuousConfig | None = None
    success_ema_alpha: float = 0.2
    # Kill-condition knobs (passed through to KillConditionMonitor).
    kill_conditions_enabled: bool = True


# ---------------------------------------------------------------------------
# Sampler factory
# ---------------------------------------------------------------------------


def build_sampler_for_condition(
    *,
    condition: str,
    bank: FetchTaskBank,
    epsilon: float,
    q_max: float | None,
    sampler_kwargs: dict[str, Any],
) -> Sampler:
    """Construct the sampler for ``condition`` against ``bank``.

    Train-only sampler: the support is restricted to the train split
    by capping the wrapper's effective bank size to the train ids and
    relying on the convention that ``train_ids() == range(0, n_train)``.
    """
    train_ids = bank.train_ids()
    if list(range(len(train_ids))) != train_ids:
        raise ValueError(
            "Loop expects train_ids() == range(0, n_train); reorder the bank "
            "so train contexts have ids 0..n_train-1."
        )
    n = len(train_ids)
    kwargs = dict(sampler_kwargs)
    kwargs.setdefault("epsilon", epsilon)
    kwargs.setdefault("q_max", q_max)
    if condition == SamplerName.UNIFORM:
        return UniformSampler(n)
    if condition == SamplerName.REWARD_GREEDY:
        return RewardGreedySampler(n, **kwargs)
    if condition == SamplerName.HEADROOM:
        return HeadroomSampler(n, **kwargs)
    if condition == SamplerName.HEADROOM_ACTOR:
        return HeadroomActorSampler(n, **kwargs)
    if condition == SamplerName.UNCERTAINTY:
        return UncertaintySampler(n, **kwargs)
    if condition == SamplerName.PLR:
        return PLRSampler(n, **kwargs)
    if condition == SamplerName.LEARNING_PROGRESS:
        return LearningProgressSampler(n, **kwargs)
    if condition == SamplerName.TABULAR_ACTOR_CURATOR:
        return TabularActorCuratorSampler(n, **kwargs)
    if condition == SamplerName.MLP_ACTOR_CURATOR:
        return MLPActorCuratorSampler(bank, **kwargs)
    if condition == SamplerName.ACDC_LITE:
        return ACDCLiteSampler(n, **kwargs)
    raise ValueError(f"unknown condition {condition!r}")


# ---------------------------------------------------------------------------
# Per-iteration helpers
# ---------------------------------------------------------------------------


def _roll_one_episode(
    env: FetchTaskEnvironment,
    agent: PPOAgent,
    context_id: int,
    max_steps: int,
    success_distance_threshold: float,
    reward_shaping_coef: float = 0.0,
) -> dict[str, Any]:
    obs_dict = env.reset(context_id)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []
    achieved_goals: list[np.ndarray] = []
    success = False
    final_distance = float(np.linalg.norm(obs_dict["achieved_goal"] - obs_dict["desired_goal"]))
    next_value = 0.0
    for _step in range(max_steps):
        obs_vec = goal_conditioned_observation(obs_dict)
        action, logp, value = agent.act(obs_vec, deterministic=False, update_rms=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if reward_shaping_coef > 0.0:
            # -coef*||gripper - object|| (gripper xyz = observation[:3];
            # object xyz = achieved_goal for Push). Encourages contact.
            gripper = np.asarray(next_obs["observation"][:3], dtype=np.float64)
            obj = np.asarray(next_obs["achieved_goal"], dtype=np.float64)
            reward = reward - reward_shaping_coef * float(np.linalg.norm(gripper - obj))
        observations.append(obs_vec)
        actions.append(action)
        log_probs.append(logp)
        values.append(value)
        rewards.append(reward)
        dones.append(1.0 if terminated else 0.0)
        achieved_goals.append(np.asarray(next_obs["achieved_goal"], dtype=np.float64))
        final_distance = float(np.linalg.norm(next_obs["achieved_goal"] - next_obs["desired_goal"]))
        if bool(info.get("is_success", final_distance < success_distance_threshold)):
            success = True
        obs_dict = next_obs
        if terminated or truncated:
            break
    if not (terminated or truncated):
        # Bootstrap final value from the last observation under the
        # current critic (same obs normalization as training).
        next_value = agent.value(goal_conditioned_observation(obs_dict))
    return {
        "context_id": int(context_id),
        "observations": np.stack(observations),
        "actions": np.stack(actions),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float64),
        "dones": np.array(dones, dtype=np.float64),
        "next_value": next_value,
        "achieved_goals": np.stack(achieved_goals),
        "final_distance": final_distance,
        "success": success,
    }


def _flatten_with_gae(
    episodes: list[dict[str, Any]],
    gamma: float,
    gae_lambda: float,
) -> dict[str, np.ndarray]:
    obs_chunks: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []
    logp_chunks: list[np.ndarray] = []
    adv_chunks: list[np.ndarray] = []
    ret_chunks: list[np.ndarray] = []
    ep_lengths: list[int] = []
    ep_context_ids: list[int] = []
    for ep in episodes:
        adv, ret = compute_gae(
            ep["rewards"],
            ep["values"],
            ep["dones"],
            ep["next_value"],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        obs_chunks.append(ep["observations"])
        act_chunks.append(ep["actions"])
        logp_chunks.append(ep["log_probs"])
        adv_chunks.append(adv.astype(np.float32))
        ret_chunks.append(ret.astype(np.float32))
        ep_lengths.append(int(ep["rewards"].shape[0]))
        ep_context_ids.append(int(ep["context_id"]))
    return {
        "observations": np.concatenate(obs_chunks),
        "actions": np.concatenate(act_chunks),
        "old_log_probs": np.concatenate(logp_chunks),
        "advantages": np.concatenate(adv_chunks),
        "returns": np.concatenate(ret_chunks),
        "episode_lengths": np.array(ep_lengths, dtype=np.int64),
        "episode_context_ids": np.array(ep_context_ids, dtype=np.int64),
    }


def _per_context_counts(
    episodes: list[dict[str, Any]], n_contexts: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (samples, success_sum, return_sum) per train-bank context."""
    samples = np.zeros(n_contexts, dtype=np.int64)
    success_sum = np.zeros(n_contexts, dtype=np.float64)
    return_sum = np.zeros(n_contexts, dtype=np.float64)
    for ep in episodes:
        cid = ep["context_id"]
        if cid >= n_contexts:
            continue
        samples[cid] += 1
        success_sum[cid] += 1.0 if ep["success"] else 0.0
        return_sum[cid] += float(ep["rewards"].sum())
    return samples, success_sum, return_sum


def _new_log_probs_for_episodes(agent: PPOAgent, episodes: list[dict[str, Any]]) -> np.ndarray:
    """Recompute log p_new(a|s) at the (s, a) pairs the actor collected.

    Routes through ``agent.evaluate_actions`` so the obs normalization
    matches training / rollout (the stored observations are raw).
    """
    obs = np.concatenate([ep["observations"] for ep in episodes])
    act = np.concatenate([ep["actions"] for ep in episodes])
    return agent.evaluate_actions(obs, act)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_fetch_ppo_loop(
    cfg: FetchPPOLoopConfig,
    bank: FetchTaskBank,
    *,
    eval_env: FetchTaskEnvironment | None = None,
    logger: RunLogger | None = None,
    checkpoint_out: Path | None = None,
) -> dict[str, Any]:
    """Run the Fetch PPO loop. Returns the final metrics dict.

    Args:
        cfg: loop configuration.
        bank: pre-built :class:`FetchTaskBank`. Train + held-out split
            must be present.
        eval_env: optional eval-bank-bound env. If ``None``, evaluation
            is skipped — useful for unit smokes.
        logger: optional logger. If ``None``, a temporary
            :class:`RunLogger` rooted at ``./results/<exp_name>`` is
            created.
        checkpoint_out: optional path. If set, the trained
            :class:`PPOAgent` is saved here via ``PPOAgent.save`` after
            the final iteration — used by the PPO warmup to emit a
            shared starter checkpoint.

    Returns:
        Aggregated metrics from the final iteration (last training
        metrics dict, plus last eval metrics if any).
    """
    if cfg.ppo is None:
        raise ValueError("FetchPPOLoopConfig.ppo must be set before running the loop")

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    train_ids = bank.train_ids()
    n_train = len(train_ids)
    train_mask = np.zeros(len(bank), dtype=np.bool_)
    train_mask[train_ids] = True

    env = FetchTaskEnvironment(bank, cfg.env_id, max_steps=cfg.max_steps_per_episode)
    agent = PPOAgent(cfg.ppo)
    sampler = build_sampler_for_condition(
        condition=cfg.condition,
        bank=bank,
        epsilon=cfg.epsilon,
        q_max=cfg.q_max,
        sampler_kwargs=cfg.sampler_kwargs,
    )

    if logger is None:
        logger = RunLogger(cfg.exp_name)
    kill_monitor = KillConditionMonitor(n_arms=n_train) if cfg.kill_conditions_enabled else None

    success_ema = np.zeros(len(bank), dtype=np.float64)
    visit_counts = np.zeros(len(bank), dtype=np.int64)
    last_metrics: dict[str, Any] = {}
    last_eval: dict[str, Any] | None = None
    # Cumulative REAL env steps from training rollouts (one per env.step). The
    # budget axis for all comparisons; "step" (iteration count) is NOT
    # comparable across episodes_per_iteration / episode-length differences.
    env_steps_total = 0

    try:
        for it in range(cfg.n_iterations):
            t_iter = time.time()
            # 1. Sample contexts.
            batch = sampler.sample_flat(cfg.episodes_per_iteration, step=it, rng=rng)
            # 2. Roll episodes.
            episodes = [
                _roll_one_episode(
                    env,
                    agent,
                    cid,
                    cfg.max_steps_per_episode,
                    cfg.success_distance_threshold,
                    cfg.reward_shaping_coef,
                )
                for cid in batch.sampled_context_ids
            ]
            env_steps_total += int(sum(len(ep["observations"]) for ep in episodes))
            # 3. Flatten + GAE.
            flat = _flatten_with_gae(episodes, cfg.ppo.gamma, cfg.ppo.gae_lambda)
            old_logp_t = torch.as_tensor(flat["old_log_probs"], dtype=torch.float32)
            adv_t = torch.as_tensor(flat["advantages"], dtype=torch.float32)
            # 4. PPO update.
            ppo_metrics = agent.update(
                flat["observations"],
                flat["actions"],
                flat["old_log_probs"],
                flat["advantages"],
                flat["returns"],
            )
            # 5. Per-context utility (AC conditions only).
            new_log_probs = _new_log_probs_for_episodes(agent, episodes)
            utility_per_context: dict[int, float] = {}
            if cfg.condition in (SamplerName.TABULAR_ACTOR_CURATOR, SamplerName.MLP_ACTOR_CURATOR):
                new_logp_t = torch.as_tensor(new_log_probs, dtype=torch.float32)
                utility_per_context = ppo_policy_improvement_utility(
                    old_logp_t,
                    new_logp_t,
                    adv_t,
                    context_ids=flat["episode_context_ids"].tolist(),
                    episode_lengths=flat["episode_lengths"].tolist(),
                    clip_eps=cfg.utility_clip_eps,
                    center=cfg.utility_center,
                    scale=cfg.utility_scale,
                )
            # 6. Build SamplerStats.
            samples, success_sum, return_sum = _per_context_counts(episodes, n_train)
            utility_sum = np.zeros(n_train, dtype=np.float64)
            for cid, u in utility_per_context.items():
                if cid < n_train:
                    utility_sum[cid] = u * samples[cid]
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
            # 7. Update sampler + feed any extra side hooks (e.g. ACDC).
            if isinstance(sampler, ACDCLiteSampler):
                for ep in episodes:
                    sampler.record_achieved_goals(ep["context_id"], ep["achieved_goals"])
            sampler.update(stats, step=it)

            # 7b. Per-context EMAs.
            for cid in range(n_train):
                if samples[cid] > 0:
                    success_ema[cid] = (1.0 - cfg.success_ema_alpha) * success_ema[
                        cid
                    ] + cfg.success_ema_alpha * (success_sum[cid] / samples[cid])
                    visit_counts[cid] += int(samples[cid])

            # 8. Diagnostics + logging.
            dist = sampler.distribution(step=it + 1)
            decomp = compute_selection_vs_learning_decomposition(
                success_ema=success_ema,
                distribution_final=np.concatenate([dist.final, np.zeros(len(bank) - n_train)])[
                    : len(bank)
                ],
                train_mask=train_mask,
            )
            group_stats = classify_groups(
                [
                    {"task_id": ep["context_id"], "reward": 1.0 if ep["success"] else 0.0}
                    for ep in episodes
                ]
            )
            training_metrics = {
                "step": it + 1,
                "env_steps": env_steps_total,
                "wall_iter_s": time.time() - t_iter,
                "rewards_mean": float(decomp["sampler_weighted_reward"]),
                "pg_loss": float(ppo_metrics["policy_loss"]),
                **ppo_metrics,
                **decomp,
                "sampler_entropy": float(dist.entropy),
                "sampler_max_probability": float(dist.max_probability),
            }
            last_metrics = training_metrics
            stream_prefix = f"training/{cfg.condition}"
            logger.log_step(f"{stream_prefix}/metrics.jsonl", training_metrics)
            logger.log_step(
                f"{stream_prefix}/sampler_distribution.jsonl",
                {
                    "step": it + 1,
                    "raw": dist.raw.tolist(),
                    "floored": dist.floored.tolist(),
                    "capped": dist.capped.tolist(),
                    "final": dist.final.tolist(),
                    "entropy": float(dist.entropy),
                    "max_probability": float(dist.max_probability),
                },
            )
            logger.log_step(
                f"{stream_prefix}/curator_weights.jsonl",
                {"step": it + 1, "weights": dist.raw.tolist()},
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
            logger.log_step(
                f"{stream_prefix}/group_stats.jsonl",
                {"step": it + 1, **group_stats.to_log_dict()},
            )

            # 9. Eval at the configured cadence.
            if (
                eval_env is not None
                and cfg.eval_interval > 0
                and ((it + 1) % cfg.eval_interval == 0 or it == cfg.n_iterations - 1)
            ):
                eval_result = evaluate_fetch(
                    env=eval_env,
                    actor_fn=lambda obs: agent.act(obs, deterministic=True)[0],
                    bank=bank,
                    n_eval_episodes_per_context=cfg.n_eval_episodes_per_context,
                    max_steps_per_episode=cfg.max_steps_per_episode,
                    distribution=dist,
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
                        "sampler_entropy": eval_result.get("sampler_entropy"),
                        "sampler_max_probability": eval_result.get("sampler_max_probability"),
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

            # 10. Kill conditions.
            if kill_monitor is not None:
                reason = kill_monitor.check(
                    iter=it + 1,
                    metrics=training_metrics,
                    # Entropy-collapse guards the SAMPLED distribution, not
                    # the raw OSMD weights: raw multiplicative-weights always
                    # concentrate, and the exploration floor + q_max cap
                    # (plan §4) are precisely the mechanisms that keep the
                    # sampled distribution diverse. Feeding raw here would
                    # kill essentially any concentrating curator and defeat
                    # the cap. dist.final is what actually draws rollouts.
                    osmd_weights=dist.final,
                    last_ckpt=None,
                )
                if reason is not None:
                    write_kill_reason(reason, logger.dir)
                    raise RuntimeError(
                        f"Fetch PPO loop aborted at iter {reason.iter}: "
                        f"{reason.name}; diagnostic={reason.diagnostic}"
                    )
    finally:
        env.close()
        logger.close()
    if checkpoint_out is not None:
        agent.save(checkpoint_out)
    return {"last_metrics": last_metrics, "last_eval": last_eval}
