"""Reward-greedy proxy sampler: ``q ∝ exp(α · ŝ_x_ema)``.

A standard "exploit highest-success contexts" curriculum baseline.
Concentrates probability on contexts the actor already does well on —
the opposite of headroom. Useful as a control for Priorities 1b–4.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``reward_greedy.py`` row, observed-mask gated EMA), §4 Priority 1b
(in the proxy-comparison condition list).
"""

from __future__ import annotations

import numpy as np

from pcrc.registry import SamplerName
from pcrc.samplers.base import Sampler, SamplerDistribution, SamplerStats
from pcrc.samplers.exploration_floor import apply_floor_and_cap


class RewardGreedySampler(Sampler):
    """``q ∝ exp(α · success_rate_ema)`` over the bank.

    The EMA is updated only on contexts with
    ``SamplerStats.observed_mask[x] = True``. Unobserved contexts
    retain their previous EMA value — this matters because most
    contexts are unobserved on any given update.

    Attributes:
        alpha: softmax temperature (positive scalar; large → greedy).
        ema_alpha: EMA mixing coefficient on observed contexts
            (``new ← (1-ema_alpha) · old + ema_alpha · obs``).
        initial_success_prior: starting value for every EMA entry.
            ``0.5`` gives a neutral prior so the initial distribution
            after the exploration floor is uniform regardless of
            ``alpha``.
        epsilon: exploration floor weight passed through to
            :func:`apply_floor_and_cap`.
        q_max: optional per-arm cap.
    """

    name = SamplerName.REWARD_GREEDY

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
        rate = np.zeros(self.n_contexts, dtype=np.float64)
        rate[obs] = stats.success_sum_this_step[obs] / stats.samples_this_step[obs]
        self._success_ema[obs] = (1.0 - self.ema_alpha) * self._success_ema[
            obs
        ] + self.ema_alpha * rate[obs]

    def distribution(self, step: int) -> SamplerDistribution:
        """Softmax over ``alpha · success_ema`` then floor / cap."""
        del step
        logits = self.alpha * self._success_ema
        logits = logits - logits.max()  # stability
        raw = np.exp(logits)
        raw = raw / raw.sum()
        return apply_floor_and_cap(raw, epsilon=self.epsilon, q_max=self.q_max)
