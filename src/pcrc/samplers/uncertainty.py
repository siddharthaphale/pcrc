"""Value-uncertainty curriculum sampler: ``q ∝ exp(α · û_ema(x))``.

Samples training contexts in proportion to a per-context **value-uncertainty**
utility (TD-error magnitude or double-Q disagreement), EMA-smoothed. Unlike the
policy-improvement utilities (which are redundant with task difficulty on sparse
goal-reaching control), value-uncertainty predicts *which* unsolved contexts are
learnable next — strongest precisely in the low-success band where success-rate /
``headroom`` is uninformative (every context ≈ 0). The dedicated, named version of
the `headroom_actor`-with-`alpha=0` stand-in used in the R5 screen.

The training loop computes the utility (`ac_utility="td_error"` / `"q_disagreement"`)
and feeds it through `SamplerStats.utility_sum_this_step` (standardized per iter by
the loop). This sampler maintains the EMA and softmaxes it; the exploration floor +
cap from `apply_floor_and_cap` are composed on top.

Plan reference: campaign result `docs/gymrobotics_results/RESULTS_SUMMARY.md` §3.7
(value-uncertainty curriculum beats headroom 3/3 seeds).
"""

from __future__ import annotations

import numpy as np

from pcrc.registry import SamplerName
from pcrc.samplers.base import Sampler, SamplerDistribution, SamplerStats
from pcrc.samplers.exploration_floor import apply_floor_and_cap


class UncertaintySampler(Sampler):
    """``q ∝ exp(α · û_ema)`` over a per-context value-uncertainty utility.

    Attributes:
        alpha: softmax temperature on the (standardized) uncertainty EMA.
        ema_alpha: EMA mixing coefficient on observed contexts.
        epsilon, q_max: passed through to :func:`apply_floor_and_cap`.
    """

    name = SamplerName.UNCERTAINTY

    def __init__(
        self,
        n_contexts: int,
        *,
        alpha: float,
        ema_alpha: float = 0.2,
        epsilon: float = 0.0,
        q_max: float | None = None,
    ):
        """Validate hyperparameters; init the uncertainty EMA to 0."""
        super().__init__(n_contexts)
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0; got {alpha}")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1]; got {ema_alpha}")
        self.alpha = float(alpha)
        self.ema_alpha = float(ema_alpha)
        self.epsilon = float(epsilon)
        self.q_max = q_max
        self._u_ema = np.zeros(self.n_contexts, dtype=np.float64)

    def update(self, stats: SamplerStats, step: int) -> None:
        """Update the uncertainty EMA on observed contexts only."""
        del step
        if stats.observed_mask.shape != (self.n_contexts,):
            raise ValueError(
                f"stats shape mismatch: expected ({self.n_contexts},), got "
                f"{stats.observed_mask.shape}"
            )
        obs = stats.observed_mask
        if not obs.any():
            return
        u = stats.utility_sum_this_step[obs] / stats.samples_this_step[obs]
        self._u_ema[obs] = (1.0 - self.ema_alpha) * self._u_ema[obs] + self.ema_alpha * u

    def distribution(self, step: int) -> SamplerDistribution:
        """Softmax over ``α · û_ema`` then floor / cap."""
        del step
        logits = self.alpha * self._u_ema
        logits = logits - logits.max()  # stability
        raw = np.exp(logits)
        raw = raw / raw.sum()
        return apply_floor_and_cap(raw, epsilon=self.epsilon, q_max=self.q_max)
