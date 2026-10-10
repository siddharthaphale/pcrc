"""Per-context policy-improvement utility from PPO-clipped surrogate.

Standalone — distinct from :func:`pcrc.algorithms.osmd.compute_ppo_clipped_osmd_loss`
(which is the curator parameter loss, audited before reuse here). All
input tensors are :meth:`torch.Tensor.detach`-ed from the actor
gradient graph; this function does NOT share code with
``compute_ppo_clipped_osmd_loss`` and the curator backward path does
NOT flow through these tensors.

Length-safe aggregation in two stages:

    1. Per-step PPO-clipped surrogate ``u_step = min(ρ · A,
       clip(ρ, 1-eps, 1+eps) · A)`` where ``ρ = exp(new_logp -
       old_logp)``.
    2. Per-episode mean of ``u_step`` over the episode's steps.
    3. Per-context mean of the episode means over episodes
       belonging to that ``source_context_id``.

Then optionally center (subtract the mean across observed contexts)
and / or scale (divide by std). Centering is the production default
because centered utilities feed cleanly into OSMD's
``log w' = log w + η · u`` step (a constant shift on all observed
arms is absorbed by the LSE renormalization, but if observed and
unobserved arms get different shifts the cap is not preserved — see
plan §3 Critical APIs).

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``policy_improvement_utility.py`` row), §4 Priority 1b (utility
fed into Tabular AC + MLP AC), §7.1 ``test_policy_improvement_utility.py``.
"""

from __future__ import annotations

import numpy as np
import torch


def ppo_policy_improvement_utility(
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    advantages: torch.Tensor,
    context_ids: list[int],
    episode_lengths: list[int],
    clip_eps: float,
    *,
    center: bool = True,
    scale: bool = False,
) -> dict[int, float]:
    """Return per-``source_context_id`` PPO policy-improvement utility.

    Args:
        old_logp: shape ``(n_total_steps,)`` log-prob of the actions
            under the pre-update policy. Detached from the actor graph
            internally.
        new_logp: shape ``(n_total_steps,)`` log-prob under the
            post-update policy. Detached.
        advantages: shape ``(n_total_steps,)`` GAE-style advantages.
            Detached.
        context_ids: length ``n_episodes``. Each entry is the
            ``source_context_id`` for one episode.
        episode_lengths: length ``n_episodes``. Each entry is the
            number of consecutive steps the episode occupies in the
            flattened tensors. Must satisfy
            ``sum(episode_lengths) == n_total_steps``.
        clip_eps: PPO clip width.
        center: subtract mean across observed contexts before returning.
        scale: divide by std across observed contexts (after centering
            if both flags are set). The std is floored at ``1e-6`` to
            avoid divide-by-zero on degenerate fixtures.

    Returns:
        ``dict[source_context_id, float]`` — one entry per distinct
        ``source_context_id`` appearing in ``context_ids``. Unobserved
        contexts (those without any episode in the input) are NOT in
        the dict — callers building a full length-``|X|`` utility
        vector fill those with ``0.0`` themselves (per the
        :class:`TabularActorCuratorSampler` contract).

    Raises:
        ValueError: shape / length mismatch.
    """
    if old_logp.shape != new_logp.shape or old_logp.shape != advantages.shape:
        raise ValueError(
            f"old_logp / new_logp / advantages shape mismatch: "
            f"{old_logp.shape} / {new_logp.shape} / {advantages.shape}"
        )
    if old_logp.ndim != 1:
        raise ValueError(f"old_logp / new_logp / advantages must be 1-D; got {old_logp.shape}")
    n_total = int(old_logp.shape[0])
    if len(context_ids) != len(episode_lengths):
        raise ValueError(
            f"context_ids ({len(context_ids)}) and episode_lengths "
            f"({len(episode_lengths)}) length mismatch"
        )
    if sum(episode_lengths) != n_total:
        raise ValueError(
            f"sum(episode_lengths) = {sum(episode_lengths)} != n_total_steps = {n_total}"
        )
    if any(L <= 0 for L in episode_lengths):
        raise ValueError("episode_lengths must all be > 0")

    # Detach inputs from the actor graph — utility is a number, not a
    # loss to be backpropped through.
    old = old_logp.detach()
    new = new_logp.detach()
    adv = advantages.detach()

    # Per-step PPO-clipped surrogate.
    ratio = (new - old).exp()
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    u_step = torch.minimum(ratio * adv, clipped * adv).cpu().numpy().astype(np.float64)

    # Per-episode mean.
    u_per_episode = np.empty(len(episode_lengths), dtype=np.float64)
    cursor = 0
    for ep_i, L in enumerate(episode_lengths):
        u_per_episode[ep_i] = float(u_step[cursor : cursor + L].mean())
        cursor += L

    # Per-context mean. Group episodes by context_id.
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for ep_i, cid in enumerate(context_ids):
        sums[cid] = sums.get(cid, 0.0) + float(u_per_episode[ep_i])
        counts[cid] = counts.get(cid, 0) + 1
    u_per_context = {cid: sums[cid] / counts[cid] for cid in sums}

    if center or scale:
        values = np.array(list(u_per_context.values()), dtype=np.float64)
        mean = float(values.mean())
        if center:
            u_per_context = {cid: u - mean for cid, u in u_per_context.items()}
        if scale:
            centered_values = np.array(list(u_per_context.values()), dtype=np.float64)
            std = float(max(centered_values.std(), 1e-6))
            u_per_context = {cid: u / std for cid, u in u_per_context.items()}

    return u_per_context
