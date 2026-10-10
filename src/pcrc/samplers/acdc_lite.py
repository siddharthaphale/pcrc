"""ACDC-lite: task-level diversity/quality proxy curriculum.

This is **ACDC-lite**, a clean-room re-implementation of the
task-level scoring rule from "ACDC: Adaptive Curriculum Planning with
Dynamic Contrastive Control..." — not full ACDC (which adds
replay-trajectory curriculum + contrastive trajectory selection;
neither is in scope here per plan §4 Priority 4).

Scoring rule:

    ``score_t(x) = λ_t · quality_t(x) + (1 - λ_t) · diversity_t(x)``

Quality: Gaussian over recent achieved-goal distance per context.
Diversity: coverage of an achieved-goal grid over recent rollouts
(default; cheaper than DPP and more robust on small banks per plan
§4 Priority 4). The DPP backend is available via
:func:`pcrc.algorithms.dpp_diversity.dpp_log_det_from_features` —
callers wanting DPP plug it in via the ``diversity_fn`` constructor
arg.

λ schedule: ``λ_t = λ_min + (λ_max - λ_min) · sigmoid(scale · (s̄ - center))``
where ``s̄`` is the global mean success rate. Defaults match the
paper's monotone-in-``s̄`` shape; the exact slope is configurable.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``acdc_lite.py`` row), §4 Priority 4 (framing + diversity-backend
choice), §7.1 ``test_acdc_lite.py``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable

import numpy as np

from pcrc.registry import SamplerName
from pcrc.samplers.base import Sampler, SamplerDistribution, SamplerStats
from pcrc.samplers.exploration_floor import apply_floor_and_cap

DiversityFn = Callable[[np.ndarray], float]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = float(np.exp(-x))
        return 1.0 / (1.0 + z)
    z = float(np.exp(x))
    return z / (1.0 + z)


def grid_coverage_diversity(
    achieved_goals: np.ndarray,
    *,
    grid_extent: tuple[tuple[float, float], ...] = ((0.0, 2.0), (0.0, 2.0), (0.0, 1.0)),
    grid_resolution: int = 8,
) -> float:
    """Number of distinct grid cells covered by ``achieved_goals``.

    ``achieved_goals`` shape ``(N, 3)``. Empty returns 0.
    """
    if achieved_goals.shape[0] == 0:
        return 0.0
    bins = []
    for k in range(achieved_goals.shape[1]):
        lo, hi = grid_extent[k]
        ix = np.floor((achieved_goals[:, k] - lo) / max(hi - lo, 1e-12) * grid_resolution)
        ix = np.clip(ix, 0, grid_resolution - 1).astype(np.int64)
        bins.append(ix)
    keys = bins[0] * (grid_resolution**2) + bins[1] * grid_resolution + bins[2]
    return float(np.unique(keys).size)


class ACDCLiteSampler(Sampler):
    """ACDC-lite curriculum: quality + diversity proxy.

    Attributes:
        sigma_quality: Gaussian σ for the quality kernel
            ``exp(-d^2 / (2 σ^2))``.
        alpha: softmax temperature on the combined score.
        lambda_min, lambda_max: bounds for the λ schedule.
        lambda_center, lambda_scale: sigmoid schedule parameters in
            ``s̄``-space (defaults: center=0.5, scale=10).
        recent_buffer_size: rolling window length per context for the
            achieved-goal coverage.
        epsilon, q_max: floor / cap parameters.
    """

    name = SamplerName.ACDC_LITE

    def __init__(
        self,
        n_contexts: int,
        *,
        sigma_quality: float = 0.2,
        alpha: float = 5.0,
        lambda_min: float = 0.1,
        lambda_max: float = 0.9,
        lambda_center: float = 0.5,
        lambda_scale: float = 10.0,
        recent_buffer_size: int = 8,
        ema_alpha: float = 0.2,
        epsilon: float = 0.0,
        q_max: float | None = None,
        diversity_fn: DiversityFn | None = None,
        initial_distance_prior: float = 1.0,
    ):
        """Validate hyperparameters; initialize per-context state to the priors."""
        super().__init__(n_contexts)
        if sigma_quality <= 0:
            raise ValueError(f"sigma_quality must be > 0; got {sigma_quality}")
        if not 0.0 <= lambda_min <= lambda_max <= 1.0:
            raise ValueError(
                f"need 0 <= lambda_min <= lambda_max <= 1; got ({lambda_min}, {lambda_max})"
            )
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1]; got {ema_alpha}")
        if recent_buffer_size <= 0:
            raise ValueError(f"recent_buffer_size must be > 0; got {recent_buffer_size}")

        self.sigma_quality = float(sigma_quality)
        self.alpha = float(alpha)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.lambda_center = float(lambda_center)
        self.lambda_scale = float(lambda_scale)
        self.recent_buffer_size = int(recent_buffer_size)
        self.ema_alpha = float(ema_alpha)
        self.epsilon = float(epsilon)
        self.q_max = q_max
        self._diversity_fn = diversity_fn or grid_coverage_diversity

        # Per-context EMA of mean final goal distance (quality input).
        self._distance_ema = np.full(n_contexts, float(initial_distance_prior), dtype=np.float64)
        # Per-context rolling buffer of recent achieved-goal vectors
        # (diversity input).
        self._recent_achieved: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.recent_buffer_size)
        )
        # Global mean success rate, drives the λ schedule.
        self._s_bar = 0.0

    # ----- update -----

    def update(self, stats: SamplerStats, step: int) -> None:
        """Update distance EMA, achieved-goal buffer, and ``s̄``.

        Diversity / quality inputs are stored here; the curriculum
        score is built in :meth:`distribution`.

        The runtime API expects per-context distance and achieved-goal
        observations to live in the standard :class:`SamplerStats`
        rather than ad-hoc side channels. We use:

            - ``return_sum_this_step`` / ``samples_this_step`` for the
              per-context mean *negative* distance (so positive return
              means closer to goal). The quality kernel inverts this
              back into a distance via ``-mean_return``.
            - ``utility_sum_this_step`` is unused here.
            - For diversity, achieved-goal vectors come from the
              optional :meth:`record_achieved_goals` side hook that
              loops call before / between :meth:`update` calls. The
              hook avoids forcing achieved goals into a fixed shape
              inside :class:`SamplerStats` (which is per-context
              scalar by design).

        Args:
            stats: standard SamplerStats. ``observed_mask`` gates the
                per-context EMA update.
            step: training step; not used here directly.
        """
        del step
        if stats.observed_mask.shape != (self.n_contexts,):
            raise ValueError(
                f"stats shape mismatch: expected ({self.n_contexts},), got "
                f"{stats.observed_mask.shape}"
            )
        obs = stats.observed_mask
        if obs.any():
            mean_neg_distance = stats.return_sum_this_step[obs] / stats.samples_this_step[obs]
            mean_distance = -mean_neg_distance
            self._distance_ema[obs] = (1.0 - self.ema_alpha) * self._distance_ema[
                obs
            ] + self.ema_alpha * mean_distance
            # Update global s̄ from the success-rate sum.
            success_rate = stats.success_sum_this_step[obs] / stats.samples_this_step[obs]
            obs_s_bar = float(success_rate.mean())
            self._s_bar = (1.0 - self.ema_alpha) * self._s_bar + self.ema_alpha * obs_s_bar

    def record_achieved_goals(self, context_id: int, achieved_goals: np.ndarray) -> None:
        """Append a per-context batch of achieved-goal vectors.

        Loops call this between :meth:`update` rounds, once per
        rollout. ``achieved_goals`` shape ``(K, 3)`` — each row is one
        achieved goal observation (end of episode or per-step,
        whichever the loop chooses to feed).
        """
        if not 0 <= context_id < self.n_contexts:
            raise IndexError(f"context_id {context_id} out of range")
        if achieved_goals.ndim != 2 or achieved_goals.shape[1] != 3:
            raise ValueError(f"achieved_goals must be shape (K, 3); got {achieved_goals.shape}")
        buf = self._recent_achieved[context_id]
        for row in achieved_goals:
            buf.append(np.asarray(row, dtype=np.float64))

    # ----- distribution -----

    def _quality_kernel(self) -> np.ndarray:
        return np.exp(-(self._distance_ema**2) / (2.0 * self.sigma_quality**2))

    def _diversity_per_context(self) -> np.ndarray:
        """Raw coverage per context — NOT normalized by global max.

        Normalizing by max would hide score-shape differences in
        tests where only one context has been populated (both
        normalize to 1.0). The softmax temperature ``alpha`` already
        handles relative-magnitude scaling for the production path.
        """
        diversity = np.zeros(self.n_contexts, dtype=np.float64)
        for cid, buf in self._recent_achieved.items():
            if not buf:
                continue
            arr = np.array(list(buf), dtype=np.float64)
            diversity[cid] = float(self._diversity_fn(arr))
        return diversity

    def _lambda_t(self) -> float:
        return self.lambda_min + (self.lambda_max - self.lambda_min) * _sigmoid(
            self.lambda_scale * (self._s_bar - self.lambda_center)
        )

    def distribution(self, step: int) -> SamplerDistribution:
        """Compute ``score_t`` and softmax to a distribution."""
        del step
        lam = self._lambda_t()
        quality = self._quality_kernel()
        diversity = self._diversity_per_context()
        score = lam * quality + (1.0 - lam) * diversity
        logits = self.alpha * score
        logits = logits - logits.max()
        raw = np.exp(logits)
        raw = raw / raw.sum()
        return apply_floor_and_cap(raw, epsilon=self.epsilon, q_max=self.q_max)

    # ----- diagnostics -----

    def diagnostics(self) -> dict[str, float | np.ndarray]:
        """Inspectable internals for logging / tests."""
        return {
            "s_bar": self._s_bar,
            "lambda_t": self._lambda_t(),
            "quality": self._quality_kernel(),
            "diversity": self._diversity_per_context(),
            "distance_ema": self._distance_ema.copy(),
        }
