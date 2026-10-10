"""Per-iter rollout-group diagnostics for the GRPO loop.

Classifies the flat rollout list into mixed / dead / saturated groups by
task_id and surfaces counters + per-task success rates for logging.

This is DIAGNOSTIC-only: it does not filter, drop, or re-order rollouts.
Rationale: the Phase 4 bandit signal requires all groups, including
dead/saturated, to compute the per-arm deviation from global_mean
(see simplevla_rl_curator_wrapper.CuratorTaskSampler.update).

Binary-reward dependency: `all_dead` is computed as `global_mean == 0.0`,
which is sound under LIBERO's binary 0/1 rewards. Revisit if dense
intermediate rewards are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GroupStats:
    """Per-iteration diagnostics over the task groups passed to the actor update."""

    n_total: int  # number of distinct task groups
    n_mixed: int  # 0 < successes < G
    n_dead: int  # successes == 0
    n_saturated: int  # successes == G
    per_task_success: dict[int, float]  # tid -> fraction in [0, 1]
    global_mean: float  # mean reward across all rollouts
    all_dead: bool  # True iff global_mean == 0.0

    def to_log_dict(self) -> dict:
        """Return a JSON-serializable view suitable for metric logging."""
        return {
            "groups/n_total": self.n_total,
            "groups/n_mixed": self.n_mixed,
            "groups/n_dead": self.n_dead,
            "groups/n_saturated": self.n_saturated,
            "groups/filter_rate": (
                (self.n_total - self.n_mixed) / self.n_total if self.n_total > 0 else 0.0
            ),
            "groups/global_mean": self.global_mean,
            "groups/all_dead": self.all_dead,
            # Stringify keys explicitly: json.dumps would coerce int keys to
            # strings anyway; doing it here keeps the in-memory dict and the
            # serialized form identical, so downstream readers don't need
            # to special-case the round-trip.
            "per_task_success": {str(k): v for k, v in self.per_task_success.items()},
        }


def classify_groups(rollouts: list[dict]) -> GroupStats:
    """Classify each task_id group in `rollouts` as mixed / dead / saturated.

    Args:
        rollouts: Flat list of trajectory dicts as produced by
                  collect_rollouts(). Each must have scalar `reward` (float
                  in {0.0, 1.0} for binary LIBERO success) and integer
                  `task_id`. Other keys are ignored.

    Returns:
        GroupStats with counts + per-task success-rate dict.

    Notes:
        - G is inferred per task from the number of rollouts sharing that
          task_id. With n_tasks_per_iter=4 and n_samples_per_group=4, each
          group has G=4. The function does not assume uniform G.
        - G=1 groups (should not occur under current defaults) are classified
          as dead OR saturated, never mixed (0 < n < 1 has no integer solution).
        - Empty input returns an all-zero GroupStats with all_dead=True.
    """
    if not rollouts:
        return GroupStats(0, 0, 0, 0, {}, 0.0, True)

    by_task: dict[int, list[float]] = {}
    for r in rollouts:
        by_task.setdefault(int(r["task_id"]), []).append(float(r["reward"]))

    n_mixed = n_dead = n_saturated = 0
    per_task_success: dict[int, float] = {}
    total_reward = 0.0
    total_n = 0
    for tid, rs in by_task.items():
        G = len(rs)
        n_succ = sum(1 for x in rs if x > 0.0)
        per_task_success[tid] = n_succ / G
        total_reward += sum(rs)
        total_n += G
        if 0 < n_succ < G:
            n_mixed += 1
        elif n_succ == 0:
            n_dead += 1
        else:
            n_saturated += 1

    global_mean = total_reward / total_n
    return GroupStats(
        n_total=len(by_task),
        n_mixed=n_mixed,
        n_dead=n_dead,
        n_saturated=n_saturated,
        per_task_success=per_task_success,
        global_mean=global_mean,
        all_dead=(global_mean == 0.0),
    )
