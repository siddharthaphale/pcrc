"""Tabular Actor-Curator: wraps :class:`OSMDUpdater` over the bank.

The wrapper enforces the plan's "full length-``|X|`` utility vector"
invariant — observed entries are filled with the mean per-rollout
utility from :class:`SamplerStats`, unobserved entries are set to
exactly ``0.0``. The full vector is passed to :meth:`OSMDUpdater.step`
so the multiplicative-weights step applies in full ``n``-arm space
(observed contexts receive a shift; unobserved receive a zero shift,
so their log-weights are unchanged before the global LSE
renormalization).

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``tabular_actor_curator.py`` row — full-vector contract), §3
Critical APIs (the "shortened observed-only vector raises" rule), §7.1
``test_tabular_actor_curator.py`` (hand-computed match assertions).

Utility centering / scaling is NOT done here. The plan separates the
per-context utility computation (``policy_improvement_utility.py``,
Stage A2 chunk 2) from the OSMD weight update; callers pre-center
their utility before populating ``SamplerStats.utility_sum_this_step``
if they want centering. The sampler passes the per-rollout mean
through verbatim so the OSMD step is exactly the hand-computable
``log w' = log w + η · u`` followed by LSE renormalization.
"""

from __future__ import annotations

import numpy as np

from pcrc.algorithms.osmd import OSMDUpdater
from pcrc.registry import SamplerName
from pcrc.samplers.base import Sampler, SamplerDistribution, SamplerStats
from pcrc.samplers.exploration_floor import apply_floor_and_cap


class TabularActorCuratorSampler(Sampler):
    """Per-arm multiplicative-weights curator over the full context bank.

    Wraps :class:`pcrc.algorithms.osmd.OSMDUpdater` (tabular mode).
    The OSMD weights are the raw sampler distribution; the standard
    exploration floor + cap from
    :func:`pcrc.samplers.exploration_floor.apply_floor_and_cap` is
    applied on top.

    Attributes:
        eta: OSMD learning rate forwarded to :class:`OSMDUpdater`.
        epsilon, q_max: floor + cap parameters.
    """

    name = SamplerName.TABULAR_ACTOR_CURATOR

    def __init__(
        self,
        n_contexts: int,
        *,
        eta: float,
        epsilon: float = 0.0,
        q_max: float | None = None,
    ):
        """Validate hyperparameters; initialize OSMD weights to uniform."""
        super().__init__(n_contexts)
        if eta < 0:
            raise ValueError(f"eta must be >= 0; got {eta}")
        self.eta = float(eta)
        self.epsilon = float(epsilon)
        self.q_max = q_max
        self._osmd = OSMDUpdater(n_arms=self.n_contexts, eta=self.eta, mode="tabular")

    def update(self, stats: SamplerStats, step: int) -> None:
        """Build the full-length utility vector and take one OSMD step.

        Observed entries are filled with
        ``utility_sum_this_step[x] / samples_this_step[x]`` (the
        mean per-rollout utility). Unobserved entries are exactly
        ``0.0``. The vector is passed verbatim to
        :meth:`OSMDUpdater.step` so the resulting weights are exactly
        the hand-computable ``log w' = log w + η · u``,
        LSE-renormalized.
        """
        del step
        if stats.observed_mask.shape != (self.n_contexts,):
            raise ValueError(
                f"stats shape mismatch: expected ({self.n_contexts},), got "
                f"{stats.observed_mask.shape}"
            )
        u_full = np.zeros(self.n_contexts, dtype=np.float64)
        obs = stats.observed_mask
        if obs.any():
            u_full[obs] = stats.utility_sum_this_step[obs] / stats.samples_this_step[obs]
        self._osmd.step(u_full)

    def distribution(self, step: int) -> SamplerDistribution:
        """OSMD weights as ``raw``; floor + cap composed on top."""
        del step
        raw = self._osmd.get_weights()
        return apply_floor_and_cap(raw, epsilon=self.epsilon, q_max=self.q_max)

    def get_osmd_weights(self) -> np.ndarray:
        """Diagnostic accessor for the underlying OSMD weights."""
        return self._osmd.get_weights()
