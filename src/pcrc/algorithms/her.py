"""Hindsight Experience Replay (HER) relabeler.

Andrychowicz et al. 2017. Two relabel strategies are supported:

    - ``"final"``: replace the desired goal with the achieved goal at
      the final transition of the episode. One relabel per transition.
    - ``"future"``: replace the desired goal with the achieved goal at
      a randomly-chosen later transition within the same episode. Up
      to ``n_sampled_goal`` relabels per transition.

CRITICAL INVARIANT (plan §3.4 + §9 risk #10): ``source_context_id`` and
``original_desired_goal`` are immutable across relabel — only
``training_desired_goal`` mutates. Curator credit assignment always
uses ``source_context_id`` so HER does not blur task identity.
Verified by ``tests/cpu/algorithms/test_her_context_identity.py``.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``her.py`` row), §3.4 (transition schema), §4 Priority 3 (HER is
the sparse-reward training default).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

# Reward function signature: ``reward_fn(achieved_goal, desired_goal,
# info) -> float``. Matches gymnasium-robotics ``env.compute_reward``.
RewardFn = Callable[[np.ndarray, np.ndarray, dict], float]


def sparse_fetch_reward(achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
    """Standard sparse Fetch reward: ``0`` if within 0.05 m, else ``-1``.

    Matches gymnasium-robotics ``FetchPush-v4`` / ``FetchPickAndPlace-v4``
    in sparse mode. The 0.05 m threshold is the env's own
    ``distance_threshold`` default. Provided as a convenience default
    for tests and Priority 3 / 4 loops that do not pass an explicit
    ``reward_fn``.
    """
    del info
    d = float(np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal)))
    return 0.0 if d < 0.05 else -1.0


@dataclass
class GoalConditionedTransition:
    """One (s, a, s', goal-tagged) transition for HER + tagged replay.

    The fields separate concerns explicitly: sim state (mutable
    indirectly via the env), curator credit assignment (immutable
    under relabel), training target (mutable under relabel), and
    provenance (tracks where the relabeled goal came from).
    """

    # Sim state
    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    achieved_goal: np.ndarray
    next_achieved_goal: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    # Curator credit assignment — IMMUTABLE under HER
    source_context_id: int
    original_desired_goal: np.ndarray
    # Training target — MUTABLE under HER
    training_desired_goal: np.ndarray
    # Provenance
    relabel_strategy: Literal["none", "future", "final"]
    relabel_source_step: int | None
    episode_id: int
    info: dict = field(default_factory=dict)


def _group_by_episode(
    transitions: list[GoalConditionedTransition],
) -> dict[int, list[int]]:
    """Map episode_id → ordered list of indices into ``transitions``.

    Preserves within-episode order (matters for "future" relabel,
    which needs the index of the *later* step).
    """
    out: dict[int, list[int]] = {}
    for idx, t in enumerate(transitions):
        out.setdefault(t.episode_id, []).append(idx)
    return out


@dataclass
class CHERConfig:
    """Curriculum-guided HER goal-selection parameters (Fang et al. 2019).

    CHER replaces ``"future"``'s uniform pick of relabel goals with a
    submodular selection that balances PROXIMITY (closeness to the
    episode's true ``original_desired_goal`` — guides toward the task)
    and DIVERSITY (a facility-location coverage term over the candidate
    future achieved goals — maintains exploration), via lazier-greedy
    maximization. A curriculum weight ``lambda`` scales the proximity
    penalty and is annealed UP over training so emphasis shifts from
    diversity (early) to proximity (late):

        score(k | S) = Δdiversity(k | S) - λ(t) · ||g_k - desired_goal||

    where ``Δdiversity`` is the marginal facility-location coverage gain
    (each candidate's running max similarity to any selected goal) and
    similarity is ``exp(-d² / mean_d²)`` over the candidate set.

    ``λ(t) = lambda_starter · (1 + lambda_rate)^t`` when
    ``fixed_lambda is None``; otherwise the constant ``fixed_lambda``.
    ``t`` is the training step passed to :func:`relabel`.
    """

    lambda_starter: float = 1.0
    lambda_rate: float = 0.0001
    fixed_lambda: float | None = None
    sub_size: int = 3  # lazier-greedy candidate subset size per pick


def _cher_lambda(cfg: CHERConfig, step: int) -> float:
    """Curriculum weight λ(t) — annealed UP (more proximity over time)."""
    if cfg.fixed_lambda is not None:
        return float(cfg.fixed_lambda)
    return float(cfg.lambda_starter * (1.0 + cfg.lambda_rate) ** max(step, 0))


def _cher_select(
    candidate_goals: np.ndarray,
    desired_goal: np.ndarray,
    n_select: int,
    lam: float,
    sub_size: int,
    rng: np.random.Generator,
) -> list[int]:
    """Lazier-greedy submodular pick of ``n_select`` candidate indices.

    Faithful port of CHER's ``lazier_and_goals_sample_kg`` / ``fa``
    (github.com/mengf1/CHER), specialized to within-episode candidates:

    - DIVERSITY: facility-location coverage over the candidate set.
      ``coverage[i] = max_{s in S} sim(i, s)`` where
      ``sim(i, s) = exp(-||g_i - g_s||² / mean_d²)`` and ``mean_d`` is
      the mean nearest-neighbor distance among candidates. The marginal
      gain of adding ``k`` is ``Σ_i max(0, sim(i,k) - coverage[i])``.
    - PROXIMITY: ``-λ · ||g_k - desired_goal||`` (closeness to the true
      desired goal, per the paper). The original CHER code measured
      distance to the achieved goal; the paper's intent — and this
      task's — is proximity to the DESIRED goal, which is what guides
      the curriculum toward the task.
    - LAZIER-GREEDY: each pick scores only a random ``sub_size`` subset
      of remaining candidates and takes the best.

    Returns up to ``n_select`` indices into ``candidate_goals``.
    """
    n = len(candidate_goals)
    n_select = min(n_select, n)
    if n_select <= 0:
        return []
    if n == 1:
        return [0]

    # Pairwise similarity matrix over candidates (small: episode-length).
    diff = candidate_goals[:, None, :] - candidate_goals[None, :, :]
    dmat = np.linalg.norm(diff, axis=-1)  # (n, n)
    # mean nearest-neighbor distance (exclude self) — the CHER kernel width.
    # Mask the diagonal with +inf via np.where (NOT eye*inf, which yields
    # 0*inf = nan on the off-diagonal and corrupts the min).
    off = np.where(np.eye(n, dtype=bool), np.inf, dmat)
    nn_dist = off.min(axis=1)
    mean_d = float(np.mean(nn_dist))
    if mean_d <= 1e-12:
        mean_d = 1.0
    sim = np.exp(-(dmat**2) / (mean_d**2))  # (n, n), sim[i, k]

    prox = np.linalg.norm(candidate_goals - desired_goal[None, :], axis=1)  # (n,)

    coverage = np.zeros(n, dtype=np.float64)  # running max-sim to selected set
    remaining = list(range(n))
    selected: list[int] = []
    for _ in range(n_select):
        size = min(sub_size, len(remaining))
        subset = rng.choice(remaining, size=size, replace=False)
        best_idx = -1
        best_marginal = -np.inf
        best_cov: np.ndarray | None = None
        for k in subset:
            new_cov = np.maximum(coverage, sim[:, k])
            div_gain = float(np.sum(new_cov - coverage))
            marginal = div_gain - lam * float(prox[k])
            if marginal > best_marginal:
                best_marginal = marginal
                best_idx = int(k)
                best_cov = new_cov
        if best_idx < 0:  # all-equal / degenerate scores: take the first sampled
            best_idx = int(subset[0])
            best_cov = np.maximum(coverage, sim[:, best_idx])
        remaining.remove(best_idx)
        selected.append(best_idx)
        coverage = best_cov  # type: ignore[assignment]
    return selected


def relabel(
    transitions: list[GoalConditionedTransition],
    strategy: Literal["future", "final", "cher"],
    n_sampled_goal: int = 4,
    reward_fn: RewardFn | None = None,
    rng: np.random.Generator | None = None,
    cher_cfg: CHERConfig | None = None,
    step: int = 0,
) -> list[GoalConditionedTransition]:
    """Return original transitions plus the relabeled copies.

    Args:
        transitions: input list. Returned unchanged (re-included in
            the output before any relabel additions).
        strategy: ``"future"`` / ``"final"`` per Andrychowicz et al., or
            ``"cher"`` (Fang et al. 2019) — same "future" candidate pool
            (later achieved goals within the episode) but the
            ``n_sampled_goal`` picks are chosen by the CHER
            proximity+diversity submodular objective instead of
            uniformly. ``"cher"`` is provenance-tagged ``"future"`` so
            the curator's relabel bookkeeping is unchanged.
        n_sampled_goal: for ``"future"``/``"cher"``, number of relabels
            per transition. Ignored for ``"final"`` (which produces
            exactly one relabel per transition).
        reward_fn: per-transition reward recomputation. Defaults to
            :func:`sparse_fetch_reward`.
        rng: numpy Generator. Defaults to a fresh
            ``np.random.default_rng()``.
        cher_cfg: CHER parameters (used only when ``strategy="cher"``).
            Defaults to :class:`CHERConfig`.
        step: training step, for the CHER λ curriculum schedule.

    Returns:
        For ``"final"``: ``2 · len(transitions)`` items (originals
        followed by relabeled). For ``"future"``/``"cher"``: each
        transition contributes up to ``n_sampled_goal`` relabels (fewer
        if the transition is at or near the end of its episode and has
        no later steps available). Originals are always preserved.

    Raises:
        ValueError: invalid ``strategy`` or non-positive
            ``n_sampled_goal``.
    """
    if strategy not in ("future", "final", "cher"):
        raise ValueError(f"strategy must be 'future', 'final', or 'cher'; got {strategy!r}")
    if strategy in ("future", "cher") and n_sampled_goal <= 0:
        raise ValueError(f"n_sampled_goal must be > 0 for {strategy!r}; got {n_sampled_goal}")
    if reward_fn is None:
        reward_fn = sparse_fetch_reward
    if rng is None:
        rng = np.random.default_rng()
    if strategy == "cher" and cher_cfg is None:
        cher_cfg = CHERConfig()
    lam = _cher_lambda(cher_cfg, step) if strategy == "cher" else 0.0

    episode_indices = _group_by_episode(transitions)
    out: list[GoalConditionedTransition] = list(transitions)  # originals preserved
    for idx, transition in enumerate(transitions):
        ep_idx_list = episode_indices[transition.episode_id]
        # Position of this transition within its episode (0-based).
        within = ep_idx_list.index(idx)

        if strategy == "final":
            final_idx = ep_idx_list[-1]
            relabeled = _build_relabel(
                transition,
                source_transition=transitions[final_idx],
                relabel_source_step=final_idx,
                strategy="final",
                reward_fn=reward_fn,
            )
            out.append(relabeled)
            continue

        # "future"/"cher": candidate goals are the achieved goals of the
        # later steps within the same episode. If this transition is the
        # last in its episode there are no future steps — emit zero
        # relabels.
        later_indices = ep_idx_list[within + 1 :]
        if not later_indices:
            continue

        if strategy == "cher":
            candidate_goals = np.stack(
                [
                    np.asarray(transitions[i].next_achieved_goal, dtype=np.float64)
                    for i in later_indices
                ]
            )
            desired = np.asarray(transition.original_desired_goal, dtype=np.float64)
            sel = _cher_select(
                candidate_goals,
                desired,
                n_select=n_sampled_goal,
                lam=lam,
                sub_size=cher_cfg.sub_size,
                rng=rng,
            )
            picks: list[int] = [later_indices[i] for i in sel]
        else:
            # "future": pick n_sampled_goal random later steps.
            picks = [
                int(i)
                for i in rng.choice(
                    later_indices,
                    size=min(n_sampled_goal, len(later_indices)),
                    replace=False,
                )
            ]
        for src_idx in picks:
            relabeled = _build_relabel(
                transition,
                source_transition=transitions[int(src_idx)],
                relabel_source_step=int(src_idx),
                # CHER is provenance-tagged "future": same candidate
                # pool, only the selection differs.
                strategy="future",
                reward_fn=reward_fn,
            )
            out.append(relabeled)
    return out


def _build_relabel(
    transition: GoalConditionedTransition,
    *,
    source_transition: GoalConditionedTransition,
    relabel_source_step: int,
    strategy: Literal["future", "final"],
    reward_fn: RewardFn,
) -> GoalConditionedTransition:
    """Construct one relabeled copy with the invariant fields preserved.

    Mutates only ``training_desired_goal`` (set to
    ``source_transition.next_achieved_goal``), ``reward`` (recomputed
    against the new goal), ``relabel_strategy``, and
    ``relabel_source_step``. ``source_context_id`` and
    ``original_desired_goal`` are copied verbatim.
    """
    new_goal = np.asarray(source_transition.next_achieved_goal, dtype=np.float64).copy()
    new_reward = float(reward_fn(transition.next_achieved_goal, new_goal, transition.info))
    return replace(
        transition,
        training_desired_goal=new_goal,
        reward=new_reward,
        relabel_strategy=strategy,
        relabel_source_step=relabel_source_step,
    )
