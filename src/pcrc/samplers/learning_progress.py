"""Learning-progress proxy sampler: ``q ∝ exp(α · Δŝ_x)``.

Tracks per-context success-rate improvement and prefers contexts where
the actor is currently improving fastest. The windowed-difference
estimator is implemented as a dual-EMA (fast EMA - slow EMA), which is
a smooth, cheap approximation of a true sliding-window difference and
matches the plan's "windowed-difference estimator computed only over
observed updates per context" intent without paying O(window · |X|)
memory.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``learning_progress.py`` row), §4 Priority 1b (proxy comparison
condition).
"""

from __future__ import annotations

import numpy as np

from pcrc.registry import SamplerName
from pcrc.samplers.base import Sampler, SamplerDistribution, SamplerStats
from pcrc.samplers.exploration_floor import apply_floor_and_cap


class LearningProgressSampler(Sampler):
    """``q ∝ exp(α · (fast_ema - slow_ema))`` on success rate.

    The signal is bounded in ``[-1, 1]`` (success rate range times the
    EMA-difference contraction). Positive Δ means the actor is
    improving on that context; negative means regression. The
    softmax temperature ``alpha`` controls how aggressively the
    sampler concentrates on improving contexts.

    Both EMAs update only on observed contexts. Initialized equal so
    the initial Δ is exactly zero and the initial distribution is
    uniform under any ``alpha``.

    Attributes:
        alpha: softmax temperature on Δ.
        fast_ema_alpha, slow_ema_alpha: per-context EMA mixing
            coefficients on observed updates. ``fast_ema_alpha >
            slow_ema_alpha`` so the difference is a meaningful
            "current minus history" signal.
        initial_success_prior: starting EMA value for both fast and
            slow.
        epsilon, q_max: passed through to :func:`apply_floor_and_cap`.
    """

    name = SamplerName.LEARNING_PROGRESS

    def __init__(
        self,
        n_contexts: int,
        *,
        alpha: float,
        fast_ema_alpha: float = 0.3,
        slow_ema_alpha: float = 0.05,
        initial_success_prior: float = 0.5,
        epsilon: float = 0.0,
        q_max: float | None = None,
    ):
        """Validate hyperparameters; initialize both EMAs to the prior."""
        super().__init__(n_contexts)
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0; got {alpha}")
        if not 0.0 < fast_ema_alpha <= 1.0:
            raise ValueError(f"fast_ema_alpha must be in (0, 1]; got {fast_ema_alpha}")
        if not 0.0 < slow_ema_alpha <= 1.0:
            raise ValueError(f"slow_ema_alpha must be in (0, 1]; got {slow_ema_alpha}")
        if slow_ema_alpha >= fast_ema_alpha:
            raise ValueError(
                f"slow_ema_alpha ({slow_ema_alpha}) must be < fast_ema_alpha "
                f"({fast_ema_alpha}) — otherwise the difference is not a "
                f"meaningful 'current minus history' signal."
            )
        self.alpha = float(alpha)
        self.fast_ema_alpha = float(fast_ema_alpha)
        self.slow_ema_alpha = float(slow_ema_alpha)
        self.epsilon = float(epsilon)
        self.q_max = q_max
        self._fast_ema = np.full(self.n_contexts, float(initial_success_prior), dtype=np.float64)
        self._slow_ema = np.full(self.n_contexts, float(initial_success_prior), dtype=np.float64)

    def update(self, stats: SamplerStats, step: int) -> None:
        """Update both EMAs on observed contexts only."""
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
        self._fast_ema[obs] = (1.0 - self.fast_ema_alpha) * self._fast_ema[
            obs
        ] + self.fast_ema_alpha * rate
        self._slow_ema[obs] = (1.0 - self.slow_ema_alpha) * self._slow_ema[
            obs
        ] + self.slow_ema_alpha * rate

    def distribution(self, step: int) -> SamplerDistribution:
        """Softmax over ``alpha · (fast - slow)`` then floor / cap."""
        del step
        delta = self._fast_ema - self._slow_ema
        logits = self.alpha * delta
        logits = logits - logits.max()  # stability
        raw = np.exp(logits)
        raw = raw / raw.sum()
        return apply_floor_and_cap(raw, epsilon=self.epsilon, q_max=self.q_max)
