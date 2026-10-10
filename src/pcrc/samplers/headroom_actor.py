"""Headroom-Actor hybrid: ``q ∝ exp(α · s(1-s) + β · û_actor)``.

Combines the two signals that the AC-win campaign isolated as
individually useful on sparse FetchPush:

* the **headroom backbone** ``α · s(1-s)`` — intermediate-difficulty
  targeting with *instant* broad frontier coverage (the proven winner;
  see :class:`pcrc.samplers.headroom.HeadroomSampler`), and
* an **actor tilt** ``β · û_actor`` — the actor-objective improvement
  utility (``Q_{t+1}(s, π_{t+1}) - Q_{t+1}(s, π_t)``), EMA-smoothed,
  which Wave 4 showed adds real value *within* broad coverage.

Unlike :class:`pcrc.samplers.tabular_actor_curator.TabularActorCuratorSampler`
this does NOT route through OSMD multiplicative weights — so it inherits
headroom's instant reaction to ``s(1-s)`` (no accumulation lag) while the
actor utility only *tilts* mass between contexts of similar difficulty.

The caller (SAC+HER loop) populates ``SamplerStats.utility_sum_this_step``
with the per-rollout actor utility, **already standardized** to mean 0 /
unit std across observed contexts (so ``β`` is on a comparable scale to
``α · s(1-s) ∈ [0, α/4]``). ``ac_headroom_gate`` must be OFF for this
condition — the ``s(1-s)`` factor lives in this sampler's logits, not in
the utility.

Plan reference: ``docs/gymrobotics_results/ac_win_campaign_log.md`` Wave 5 (instant-weighting
hybrid to beat the headroom sampler's AUC).
"""

from __future__ import annotations

import numpy as np

from pcrc.registry import SamplerName
from pcrc.samplers.base import Sampler, SamplerDistribution, SamplerStats
from pcrc.samplers.exploration_floor import apply_floor_and_cap


class HeadroomActorSampler(Sampler):
    """``q ∝ exp(α · sᵉᵐᵃ(1-sᵉᵐᵃ) + β · ûᵉᵐᵃ_actor)``.

    Both EMAs are observed-mask gated — unobserved contexts retain their
    previous value (success at the prior, utility at 0).

    Attributes:
        alpha: temperature on the headroom backbone ``s(1-s)``.
        beta: temperature on the actor-utility tilt.
        ema_alpha: EMA mixing coefficient for the success rate.
        util_ema_alpha: EMA mixing coefficient for the actor utility.
        initial_success_prior: starting success EMA (``0.5`` peaks the
            backbone uniformly at 0.25).
        epsilon, q_max: passed through to :func:`apply_floor_and_cap`.
    """

    name = SamplerName.HEADROOM_ACTOR

    def __init__(
        self,
        n_contexts: int,
        *,
        alpha: float,
        beta: float,
        ema_alpha: float = 0.1,
        util_ema_alpha: float = 0.2,
        initial_success_prior: float = 0.5,
        epsilon: float = 0.0,
        q_max: float | None = None,
    ):
        """Validate hyperparameters; init success EMA to the prior, util EMA to 0."""
        super().__init__(n_contexts)
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0; got {alpha}")
        if beta < 0:
            raise ValueError(f"beta must be >= 0; got {beta}")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1]; got {ema_alpha}")
        if not 0.0 < util_ema_alpha <= 1.0:
            raise ValueError(f"util_ema_alpha must be in (0, 1]; got {util_ema_alpha}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ema_alpha = float(ema_alpha)
        self.util_ema_alpha = float(util_ema_alpha)
        self.epsilon = float(epsilon)
        self.q_max = q_max
        self._success_ema = np.full(self.n_contexts, float(initial_success_prior), dtype=np.float64)
        self._util_ema = np.zeros(self.n_contexts, dtype=np.float64)

    def update(self, stats: SamplerStats, step: int) -> None:
        """Update success + actor-utility EMAs on observed contexts only."""
        del step
        if stats.observed_mask.shape != (self.n_contexts,):
            raise ValueError(
                f"stats shape mismatch: expected ({self.n_contexts},), got "
                f"{stats.observed_mask.shape}"
            )
        obs = stats.observed_mask
        if not obs.any():
            return
        samples = stats.samples_this_step[obs]
        rate = stats.success_sum_this_step[obs] / samples
        self._success_ema[obs] = (1.0 - self.ema_alpha) * self._success_ema[
            obs
        ] + self.ema_alpha * rate
        util = stats.utility_sum_this_step[obs] / samples
        self._util_ema[obs] = (1.0 - self.util_ema_alpha) * self._util_ema[
            obs
        ] + self.util_ema_alpha * util

    def distribution(self, step: int) -> SamplerDistribution:
        """Softmax over ``α·s(1-s) + β·û`` then floor / cap."""
        del step
        headroom = self._success_ema * (1.0 - self._success_ema)
        logits = self.alpha * headroom + self.beta * self._util_ema
        logits = logits - logits.max()  # stability
        raw = np.exp(logits)
        raw = raw / raw.sum()
        return apply_floor_and_cap(raw, epsilon=self.epsilon, q_max=self.q_max)
