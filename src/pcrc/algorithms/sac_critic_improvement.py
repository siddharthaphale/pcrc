"""Per-context SAC critic-improvement utility for Priority 3.

Used as the Actor-Curator utility in the sparse-Fetch SAC + HER regime
(Plan §Priority 3) where the on-policy PPO surrogate is unavailable.
For each transition tagged with ``source_context_id``, compute the
critic-improvement signal

    ``Q_{t+1}(s, π_{t+1}(s, g), g) - Q_t(s, π_t(s, g), g)``

at the transition's state, with the *original* desired goal (HER
relabel does NOT mutate :attr:`source_context_id` or
:attr:`original_desired_goal`; per plan §3.4). Aggregate per
``source_context_id`` (mean over the transitions belonging to that
context).

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``sac_critic_improvement.py`` row), §4 Priority 3 (utility for
the sparse + SAC + HER setting), §7.1
``test_sac_critic_improvement.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from pcrc.algorithms.her import GoalConditionedTransition

# Type alias for a Q evaluator that consumes (state, action, goal)
# tensors and returns Q-values. Signature kept deliberately narrow —
# implementations decide whether to take the min over a double-Q pair
# internally; this module just consumes one scalar per call.
QFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
# Type alias for an actor that maps (state, goal) → action.
ActorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def sac_critic_improvement(
    transitions: list[GoalConditionedTransition],
    *,
    q_fn_t: QFn,
    actor_fn_t: ActorFn,
    q_fn_tplus1: QFn,
    actor_fn_tplus1: ActorFn,
    device: str = "cpu",
) -> dict[int, float]:
    """Aggregate per-``source_context_id`` critic improvement.

    Args:
        transitions: list of :class:`GoalConditionedTransition`. Uses
            :attr:`observation`, :attr:`source_context_id`, and
            :attr:`original_desired_goal` from each — explicitly NOT
            ``training_desired_goal`` (HER relabel keeps source
            identity unambiguous; see §3.4).
        q_fn_t, actor_fn_t: pre-update Q and policy.
        q_fn_tplus1, actor_fn_tplus1: post-update Q and policy.
        device: torch device for the tensor batches. ``"cpu"`` is the
            default and is fine for pre-VLA Fetch sizes.

    Returns:
        ``dict[source_context_id, float]`` where the value is the mean
        ``Q_{t+1} - Q_t`` evaluated on the buffer's transitions
        tagged with that ``source_context_id``. Contexts without any
        transitions in ``transitions`` are NOT in the returned dict —
        callers building a full length-``|X|`` vector fill those with
        ``0.0`` (per the :class:`TabularActorCuratorSampler`
        contract).
    """
    if not transitions:
        return {}

    # Batch the inputs once per device transfer.
    states = torch.as_tensor(
        np.stack([np.asarray(t.observation, dtype=np.float32) for t in transitions]),
        device=device,
    )
    goals = torch.as_tensor(
        np.stack([np.asarray(t.original_desired_goal, dtype=np.float32) for t in transitions]),
        device=device,
    )
    actions_t = actor_fn_t(states, goals)
    actions_tplus1 = actor_fn_tplus1(states, goals)
    q_t = q_fn_t(states, actions_t, goals).cpu().numpy().astype(np.float64)
    q_tplus1 = q_fn_tplus1(states, actions_tplus1, goals).cpu().numpy().astype(np.float64)
    improvement = q_tplus1 - q_t
    return _aggregate_per_context(improvement, transitions)


@torch.no_grad()
def sac_actor_objective_improvement(
    transitions: list[GoalConditionedTransition],
    *,
    actor_fn_t: ActorFn,
    actor_fn_tplus1: ActorFn,
    q_fn_tplus1: QFn,
    device: str = "cpu",
) -> dict[int, float]:
    """Per-``source_context_id`` SAC actor-objective improvement.

    Alternative Actor-Curator utility to :func:`sac_critic_improvement`.
    Critic-improvement (``Q_{t+1}(s, π_{t+1}) - Q_t(s, π_t)``) moves both
    the critic AND the policy, so on sparse Push — where the critic is
    noisy at near-zero success — it is a weak selection signal. This
    isolates the *policy* update by evaluating both policies under the
    **same post-update critic**:

        ``Q_{t+1}(s, π_{t+1}(s, g), g) - Q_{t+1}(s, π_t(s, g), g)``

    i.e. how much the actor update improved the action value on each
    context's states (the change in the actor objective, holding the
    critic fixed). Higher ⇒ the update was more productive on that
    context ⇒ more useful to sample next. ``actor_fn_*`` return the
    deterministic (mean) action; the entropy term is dropped because at
    the collapsed sparse-Push α it is negligible and only adds variance.

    Plan reference: §4 Priority 3 "Actor-objective improvement" utility row.
    """
    if not transitions:
        return {}
    states = torch.as_tensor(
        np.stack([np.asarray(t.observation, dtype=np.float32) for t in transitions]),
        device=device,
    )
    goals = torch.as_tensor(
        np.stack([np.asarray(t.original_desired_goal, dtype=np.float32) for t in transitions]),
        device=device,
    )
    a_pre = actor_fn_t(states, goals)
    a_post = actor_fn_tplus1(states, goals)
    q_post_new = q_fn_tplus1(states, a_post, goals).cpu().numpy().astype(np.float64)
    q_post_old = q_fn_tplus1(states, a_pre, goals).cpu().numpy().astype(np.float64)
    improvement = q_post_new - q_post_old
    return _aggregate_per_context(improvement, transitions)


def _aggregate_per_context(
    improvement: np.ndarray, transitions: list[GoalConditionedTransition]
) -> dict[int, float]:
    """Mean per-``source_context_id`` over transition-aligned values."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for i, t in enumerate(transitions):
        cid = int(t.source_context_id)
        sums[cid] = sums.get(cid, 0.0) + float(improvement[i])
        counts[cid] = counts.get(cid, 0) + 1
    return {cid: sums[cid] / counts[cid] for cid in sums}
