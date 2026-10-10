"""Headroom proxy sampler: ``q ∝ exp(α · ŝ_x · (1 - ŝ_x))``.

Concentrates probability on contexts at intermediate success rates —
those with maximum *expected information gain* under the binary
classifier heuristic. Peaks at ``ŝ = 0.5``; mass collapses both for
saturated (``ŝ → 1``) and dead (``ŝ → 0``) contexts.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``headroom.py`` row), §4 Priority 1b (proxy comparison condition).
"""

from __future__ import annotations

import numpy as np

from pcrc.registry import SamplerName
from pcrc.samplers.base import Sampler, SamplerDistribution, SamplerStats
from pcrc.samplers.exploration_floor import apply_floor_and_cap


class HeadroomSampler(Sampler):
    """``q ∝ exp(α · success_rate_ema · (1 - success_rate_ema))``.

    EMA is observed-mask gated — unobserved contexts retain their
    previous value.

    Attributes:
        alpha: softmax temperature on the headroom signal.
        ema_alpha: EMA mixing coefficient on observed contexts.
        initial_success_prior: starting EMA value (``0.5`` peaks the
            headroom signal at 0.25 uniformly, matching the
            symmetric-prior intuition).
        epsilon, q_max: passed through to :func:`apply_floor_and_cap`.
    """

    name = SamplerName.HEADROOM

    def __init__(
        self,
        n_contexts: int,
        *,
        alpha: float,
        ema_alpha: float = 0.1,
        initial_success_prior: float = 0.5,
        epsilon: float = 0.0,
        q_max: float | None = None,
    ):
        """Validate hyperparameters; initialize the EMA to the prior."""
        super().__init__(n_contexts)
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0; got {alpha}")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1]; got {ema_alpha}")
        self.alpha = float(alpha)
        self.ema_alpha = float(ema_alpha)
        self.epsilon = float(epsilon)
        self.q_max = q_max
        self._success_ema = np.full(self.n_contexts, float(initial_success_prior), dtype=np.float64)

    def update(self, stats: SamplerStats, step: int) -> None:
        """Update success EMA on observed contexts only."""
        del step
        if stats.observed_mask.shape != (self.n_contexts,):
            raise ValueError(
                f"stats shape mismatch: expected ({self.n_contexts},), got "
                f"{stats.observed_mask.shape}"
            )
        obs = stats.observed_mask
        if not obs.any():
            return
        rate = stats.success_sum_this_step[obs] / stats.samples_this_step[obs]
        self._success_ema[obs] = (1.0 - self.ema_alpha) * self._success_ema[
            obs
        ] + self.ema_alpha * rate

    def distribution(self, step: int) -> SamplerDistribution:
        """Softmax over ``alpha · s · (1-s)`` then floor / cap."""
        del step
        signal = self._success_ema * (1.0 - self._success_ema)
        logits = self.alpha * signal
        logits = logits - logits.max()  # stability
        raw = np.exp(logits)
        raw = raw / raw.sum()
        return apply_floor_and_cap(raw, epsilon=self.epsilon, q_max=self.q_max)
