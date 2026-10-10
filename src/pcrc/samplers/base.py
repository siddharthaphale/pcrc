"""Sampler base classes, statistics, and distribution bundles.

Defines the contracts that every concrete sampler under ``pcrc.samplers``
must satisfy. The contracts are deliberately strict — samplers MUST
gate internal state updates on :attr:`SamplerStats.observed_mask`, MUST
compose :func:`pcrc.samplers.exploration_floor.apply_floor_and_cap` to
build the returned :class:`SamplerDistribution`, and MUST return the
full bundle (raw / floored / capped / final + entropy + max) so the
floor / cap effect is visible to logging.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§3.1 (full API sketch + observed-mask semantics + sampling-mode
documentation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplerStats:
    """Per-update statistics fed back to adaptive samplers.

    Most contexts are unobserved on any given update; samplers MUST
    consult :attr:`observed_mask` before updating EMAs or
    sampler-internal state. The plan distinguishes four cases:

        - ``observed_mask[x] = False`` — no update; leave EMA / OSMD
          weight / utility unchanged.
        - ``observed_mask[x] = True`` and ``utility_sum_this_step[x] = 0``
          — real neutral observation.
        - ``samples_this_step[x] > 0`` and ``success_sum_this_step[x] = 0``
          — real all-fail observation.
        - ``samples_this_step[x] > 0`` and
          ``success_sum_this_step[x] = samples_this_step[x]`` — real
          saturated observation.

    Attributes:
        observed_mask: shape ``(|X|,)`` bool. True iff this context
            was sampled at least once in the current update window.
        samples_this_step: shape ``(|X|,)`` int. Number of rollouts
            per context this step. Zero on unobserved contexts.
        success_sum_this_step: shape ``(|X|,)`` float. Sum of
            successes per context this step (binary reward sum).
        return_sum_this_step: shape ``(|X|,)`` float. Sum of
            undiscounted returns this step.
        utility_sum_this_step: shape ``(|X|,)`` float. Sum of the AC
            utility this step (PPO-clipped surrogate or critic
            improvement, see §3 of the implementation plan).
        utility_sq_sum_this_step: shape ``(|X|,)`` float. Optional,
            for variance tracking. Zero by default — callers that do
            not track variance can pass ``np.zeros_like(...)``.
        success_ema: shape ``(|X|,)`` float. Running EMA of per-context
            success rate.
        return_ema: shape ``(|X|,)`` float. Running EMA of per-context
            return.
        utility_ema: shape ``(|X|,)`` float. Running EMA of per-context
            utility.
        n_visits_total: shape ``(|X|,)`` int. Cumulative visit counts.
    """

    observed_mask: np.ndarray
    samples_this_step: np.ndarray
    success_sum_this_step: np.ndarray
    return_sum_this_step: np.ndarray
    utility_sum_this_step: np.ndarray
    utility_sq_sum_this_step: np.ndarray
    success_ema: np.ndarray
    return_ema: np.ndarray
    utility_ema: np.ndarray
    n_visits_total: np.ndarray

    def __post_init__(self) -> None:
        """Validate consistent shape + dtype + non-negative counts."""
        n = self.observed_mask.shape[0]
        for name in (
            "observed_mask",
            "samples_this_step",
            "success_sum_this_step",
            "return_sum_this_step",
            "utility_sum_this_step",
            "utility_sq_sum_this_step",
            "success_ema",
            "return_ema",
            "utility_ema",
            "n_visits_total",
        ):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(
                    f"SamplerStats.{name} shape {arr.shape} does not match "
                    f"observed_mask shape ({n},)"
                )
        if self.observed_mask.dtype != np.bool_:
            raise TypeError(
                f"SamplerStats.observed_mask must be bool, got {self.observed_mask.dtype}"
            )
        if (self.samples_this_step < 0).any():
            raise ValueError("SamplerStats.samples_this_step contains negative entries")
        # Programmer-error check: observed_mask[x] = True must imply
        # samples_this_step[x] > 0 (you observed something; what was it?).
        observed_with_zero_samples = self.observed_mask & (self.samples_this_step == 0)
        if observed_with_zero_samples.any():
            bad = np.where(observed_with_zero_samples)[0]
            raise ValueError(
                f"SamplerStats: observed_mask=True but samples_this_step=0 at "
                f"context_ids {bad.tolist()[:5]} (programmer error — what was "
                f"observed?)"
            )

    @classmethod
    def zeros(cls, n_contexts: int) -> SamplerStats:
        """Construct an all-zero, all-unobserved instance for ``n_contexts`` arms.

        Convenience for samplers that need a starting state at step 0.
        """
        return cls(
            observed_mask=np.zeros(n_contexts, dtype=np.bool_),
            samples_this_step=np.zeros(n_contexts, dtype=np.int64),
            success_sum_this_step=np.zeros(n_contexts, dtype=np.float64),
            return_sum_this_step=np.zeros(n_contexts, dtype=np.float64),
            utility_sum_this_step=np.zeros(n_contexts, dtype=np.float64),
            utility_sq_sum_this_step=np.zeros(n_contexts, dtype=np.float64),
            success_ema=np.zeros(n_contexts, dtype=np.float64),
            return_ema=np.zeros(n_contexts, dtype=np.float64),
            utility_ema=np.zeros(n_contexts, dtype=np.float64),
            n_visits_total=np.zeros(n_contexts, dtype=np.int64),
        )


@dataclass(frozen=True)
class SamplerDistribution:
    """Sampler distribution snapshot for logging and downstream use.

    Every sampler returns this bundle from :meth:`Sampler.distribution`
    so the floor / cap effect is visible and reproducible. The four
    arrays are layered: ``raw`` is the sampler's own pre-floor
    distribution, ``floored`` after ``(1-ε)·raw + ε/|X|``, ``capped``
    after enforcing ``q ≤ q_max``, and ``final`` is what callers
    actually sample from (currently equal to ``capped``; documented
    separately for forward compatibility with e.g. temperature scaling).

    Attributes:
        raw: shape ``(|X|,)`` probabilities before exploration floor +
            cap. Sums to 1.
        floored: shape ``(|X|,)`` after ``(1-ε)·raw + ε/|X|``. Sums to 1.
        capped: shape ``(|X|,)`` after enforcing ``q ≤ q_max`` and
            renormalizing. Sums to 1.
        final: shape ``(|X|,)`` the distribution used for sampling.
        entropy: float, Shannon entropy of ``final`` in nats.
        max_probability: float, ``final.max()``.
    """

    raw: np.ndarray
    floored: np.ndarray
    capped: np.ndarray
    final: np.ndarray
    entropy: float
    max_probability: float

    def __post_init__(self) -> None:
        """Validate sum-to-1 within tolerance + shape coherence."""
        n = self.raw.shape[0]
        for name in ("raw", "floored", "capped", "final"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(
                    f"SamplerDistribution.{name} shape {arr.shape} does not match raw shape ({n},)"
                )
            s = float(arr.sum())
            if not np.isclose(s, 1.0, atol=1e-6):
                raise ValueError(f"SamplerDistribution.{name} must sum to 1 within 1e-6; got {s}")


@dataclass(frozen=True)
class SampledBatch:
    """One round of sampling — supports both flat and grouped rollouts.

    Attributes:
        sampled_context_ids: length == total rollouts requested.
            **Duplicates allowed** under flat (with-replacement)
            sampling. This is the source-of-truth for which rollout
            used which context.
        unique_context_ids: ``sorted(set(sampled_context_ids))``.
        rollout_counts: ``context_id → number of rollouts``. For flat
            with-replacement sampling values may exceed 1. For grouped
            sampling every value equals ``rollouts_per_group``.
        distribution: the :class:`SamplerDistribution` used to produce
            this batch (for logging and rank-correlation diagnostics).
    """

    sampled_context_ids: list[int]
    unique_context_ids: list[int]
    rollout_counts: dict[int, int]
    distribution: SamplerDistribution

    def __post_init__(self) -> None:
        """Validate internal coherence of the three views."""
        expected_unique = sorted(set(self.sampled_context_ids))
        if list(self.unique_context_ids) != expected_unique:
            raise ValueError(
                "SampledBatch.unique_context_ids must equal sorted(set(sampled_context_ids))"
            )
        expected_counts = Counter(self.sampled_context_ids)
        if dict(self.rollout_counts) != dict(expected_counts):
            raise ValueError("SampledBatch.rollout_counts must equal Counter(sampled_context_ids)")


class Sampler(ABC):
    """Adaptive task sampler over a fixed-size context bank.

    Subclasses MUST:
        - Update internal state only over the
          :attr:`SamplerStats.observed_mask` slice (gate every EMA /
          weight update on the mask).
        - Compose
          :func:`pcrc.samplers.exploration_floor.apply_floor_and_cap`
          to produce the returned :class:`SamplerDistribution`.

    Subclasses MAY ignore the ``step`` argument to :meth:`update` and
    :meth:`distribution` — most samplers do — but the parameter is
    exposed so curricula that schedule on training step (e.g. ACDC-lite
    λ schedule) have a single place to read it from.
    """

    name: str

    def __init__(self, n_contexts: int):
        """Bind the sampler to a fixed bank size."""
        if n_contexts <= 0:
            raise ValueError(f"n_contexts must be > 0; got {n_contexts}")
        self.n_contexts = int(n_contexts)

    @abstractmethod
    def update(self, stats: SamplerStats, step: int) -> None:
        """Update internal state from one round of sampler statistics."""

    @abstractmethod
    def distribution(self, step: int) -> SamplerDistribution:
        """Return the bundle. ``final`` is what callers sample from."""

    def sample_flat(self, n: int, step: int, rng: np.random.Generator) -> SampledBatch:
        """Draw ``n`` independent context ids with replacement.

        Default sampling mode for dense PPO (Priority 1b, 1c).
        Duplicates are expected and tracked in ``rollout_counts``.
        """
        if n <= 0:
            raise ValueError(f"n must be > 0; got {n}")
        dist = self.distribution(step)
        sampled = rng.choice(self.n_contexts, size=n, replace=True, p=dist.final)
        sampled_list = [int(c) for c in sampled]
        counts = Counter(sampled_list)
        return SampledBatch(
            sampled_context_ids=sampled_list,
            unique_context_ids=sorted(counts.keys()),
            rollout_counts=dict(counts),
            distribution=dist,
        )

    def sample_grouped(
        self,
        num_groups: int,
        rollouts_per_group: int,
        step: int,
        rng: np.random.Generator,
    ) -> SampledBatch:
        """Draw ``num_groups`` distinct contexts, ``rollouts_per_group`` each.

        Required for group-based diagnostics (``classify_groups``,
        GRPO-like settings) starting Priority 2.
        """
        if num_groups <= 0:
            raise ValueError(f"num_groups must be > 0; got {num_groups}")
        if rollouts_per_group <= 0:
            raise ValueError(f"rollouts_per_group must be > 0; got {rollouts_per_group}")
        if num_groups > self.n_contexts:
            raise ValueError(
                f"num_groups={num_groups} exceeds n_contexts={self.n_contexts} "
                f"— cannot sample that many distinct contexts without replacement"
            )
        dist = self.distribution(step)
        # rng.choice with replace=False requires p to have at least
        # num_groups non-zero entries; raise a clear error otherwise.
        nonzero = int((dist.final > 0).sum())
        if nonzero < num_groups:
            raise ValueError(
                f"sample_grouped: distribution.final has only {nonzero} "
                f"non-zero entries; cannot draw {num_groups} distinct contexts. "
                f"Increase the exploration floor or shrink num_groups."
            )
        ids = rng.choice(
            self.n_contexts,
            size=num_groups,
            replace=False,
            p=dist.final,
        )
        ids_list = [int(c) for c in ids]
        sampled = [cid for cid in ids_list for _ in range(rollouts_per_group)]
        return SampledBatch(
            sampled_context_ids=sampled,
            unique_context_ids=sorted(ids_list),
            rollout_counts={cid: rollouts_per_group for cid in ids_list},
            distribution=dist,
        )
