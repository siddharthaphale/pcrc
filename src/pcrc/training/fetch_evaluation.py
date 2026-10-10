"""Fixed-distribution evaluation over the held-out split.

Greedy actor rollout, ``n_eval_episodes`` per held-out context, no
sampler bias. Reports overall success, mean dense return, per-bin
breakdowns, and per-context success — the per-context column is the
mandatory eval stream from plan §8.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``fetch_evaluation.py`` row), §8 mandatory streams
(``eval/per_context.jsonl``), §4 Priority 1b–4 (fixed-eval is the
production decision signal).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np

from pcrc.envs.fetch_environment import (
    FetchTaskEnvironment,
    goal_conditioned_observation,
)
from pcrc.envs.fetch_task_bank import FetchTaskBank
from pcrc.registry import Method
from pcrc.samplers.base import SamplerDistribution

# Greedy actor signature: ``act(observation_vector) -> action_vector``.
GreedyActorFn = Callable[[np.ndarray], np.ndarray]


def evaluate_fetch(
    *,
    env: FetchTaskEnvironment,
    actor_fn: GreedyActorFn,
    bank: FetchTaskBank,
    n_eval_episodes_per_context: int,
    max_steps_per_episode: int,
    distribution: SamplerDistribution | None = None,
    success_distance_threshold: float = 0.05,
) -> dict:
    """Roll ``actor_fn`` greedily on every held-out context.

    Args:
        env: a :class:`FetchTaskEnvironment` already bound to the
            held-out bank.
        actor_fn: greedy actor callable. Receives the ``observation``
            field of the obs dict, returns an action ndarray.
        bank: the bank that ``env`` wraps. Held-out ids are
            ``bank.held_out_ids()``.
        n_eval_episodes_per_context: trials per held-out context.
        max_steps_per_episode: per-episode horizon.
        distribution: optional sampler distribution to include in the
            returned dict (entropy + max prob diagnostics). ``None``
            omits those keys.
        success_distance_threshold: success threshold in metres;
            matches gymnasium-robotics Fetch default ``0.05``.

    Returns:
        dict with keys:
            - ``success_rate`` (float)
            - ``mean_dense_return`` (float)
            - ``per_context_success`` (dict[int, float])
            - ``per_bin_success`` (dict[str, dict[str, float]])
            - ``sampler_entropy``, ``sampler_max_probability`` (only
              when ``distribution`` is passed)
    """
    held_out = bank.held_out_ids()
    if not held_out:
        raise ValueError("Cannot evaluate: held-out split is empty")
    per_context_success: dict[int, float] = {}
    total_returns: list[float] = []

    distance_buckets: dict[int, list[float]] = defaultdict(list)
    direction_buckets: dict[int, list[float]] = defaultdict(list)
    pose_buckets: dict[int, list[float]] = defaultdict(list)

    for cid in held_out:
        ctx = bank[cid]
        successes = 0
        for _ in range(n_eval_episodes_per_context):
            obs = env.reset(cid)
            ep_return = 0.0
            ep_success = False
            final_distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            for _step in range(max_steps_per_episode):
                action = actor_fn(goal_conditioned_observation(obs))
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                final_distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
                if bool(info.get("is_success", final_distance < success_distance_threshold)):
                    ep_success = True
                if terminated or truncated:
                    break
            if ep_success:
                successes += 1
            total_returns.append(ep_return)
        rate = successes / n_eval_episodes_per_context
        per_context_success[cid] = rate
        distance_buckets[ctx.distance_bin].append(rate)
        direction_buckets[ctx.direction_bin].append(rate)
        pose_buckets[ctx.pose_bin].append(rate)

    success_rate = float(np.mean(list(per_context_success.values())))
    mean_dense_return = float(np.mean(total_returns))

    def _bin_summary(b: dict[int, list[float]]) -> dict[str, float]:
        return {f"bin_{idx}": float(np.mean(vals)) for idx, vals in sorted(b.items())}

    result: dict = {
        "success_rate": success_rate,
        "mean_dense_return": mean_dense_return,
        Method.PER_CONTEXT_SUCCESS: per_context_success,
        "per_bin_success": {
            "distance": _bin_summary(distance_buckets),
            "direction": _bin_summary(direction_buckets),
            "pose": _bin_summary(pose_buckets),
        },
    }
    if distribution is not None:
        result["sampler_entropy"] = float(distribution.entropy)
        result["sampler_max_probability"] = float(distribution.max_probability)
    return result
