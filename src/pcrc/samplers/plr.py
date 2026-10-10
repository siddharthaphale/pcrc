"""Prioritized Level Replay (PLR) sampler — closest prior method, included as a baseline.

PLR (Jiang et al., 2021, "Prioritized Level Replay") scores each level/task by its **L1 value
loss** (mean |TD-error| / |GAE| over the most recent visit) and samples by a **rank-based**
prioritization mixed with a **staleness** term that favours not-recently-seen levels:

    P_S(i)  ∝ (1 / rank_i) ** (1 / beta)            # rank by score, descending
    P_C(i)  ∝ (c − c_i)                              # staleness: steps since i was last sampled
    P(i)    = (1 − rho) · P_S(i) + rho · P_C(i)

This sampler consumes the SAME per-context signal as :class:`UncertaintySampler` — the loop's
``ac_utility="td_error"`` feeds mean |TD-error| via ``stats.utility_sum_this_step`` — so comparing
PLR vs UncertaintySampler isolates PLR's distinctive machinery (rank prioritization + staleness)
from the ``exp(alpha · u_ema)`` softmax. Deviation from the paper: PLR-original replaces the score
on each visit; off-policy SAC TD-error per visit is noisy, so we EMA-smooth with
``score_ema_alpha`` (default 0.3; set 1.0 for the faithful last-value rule).
"""

from __future__ import annotations

import numpy as np

from pcrc.registry import SamplerName
from pcrc.samplers.base import Sampler, SamplerDistribution, SamplerStats
from pcrc.samplers.exploration_floor import apply_floor_and_cap


class PLRSampler(Sampler):
    """Prioritized Level Replay: rank-prioritized |TD-error| + staleness mixing.

    Attributes:
        beta: rank-prioritization exponent — ``P_S ∝ (1/rank)**(1/beta)``. Smaller beta = sharper
            concentration on the highest-score contexts (paper uses 0.1–0.5).
        rho: staleness mixing coefficient in ``[0, 1]`` (paper default 0.1).
        score_ema_alpha: EMA on the per-context score (1.0 = faithful PLR last-value).
        epsilon, q_max: passed through to :func:`apply_floor_and_cap`.
    """

    name = SamplerName.PLR

    def __init__(
        self,
        n_contexts: int,
        *,
        beta: float = 0.5,
        rho: float = 0.1,
        score_ema_alpha: float = 0.3,
        epsilon: float = 0.0,
        q_max: float | None = None,
    ):
        """Validate hyperparameters; init scores to 0 and last-visit to -1 (unseen)."""
        super().__init__(n_contexts)
        if beta <= 0:
            raise ValueError(f"beta must be > 0; got {beta}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1]; got {rho}")
        if not 0.0 < score_ema_alpha <= 1.0:
            raise ValueError(f"score_ema_alpha must be in (0, 1]; got {score_ema_alpha}")
        self.beta = float(beta)
        self.rho = float(rho)
        self.score_ema_alpha = float(score_ema_alpha)
        self.epsilon = float(epsilon)
        self.q_max = q_max
        self._score = np.zeros(self.n_contexts, dtype=np.float64)
        self._seen = np.zeros(self.n_contexts, dtype=bool)
        self._last_visit = np.full(self.n_contexts, -1, dtype=np.int64)
        self._step = 0

    def update(self, stats: SamplerStats, step: int) -> None:
        """EMA the |TD-error| score and stamp last-visit on observed contexts only."""
        if stats.observed_mask.shape != (self.n_contexts,):
            raise ValueError(
                f"stats shape mismatch: expected ({self.n_contexts},), got "
                f"{stats.observed_mask.shape}"
            )
        self._step = int(step)
        obs = stats.observed_mask
        if not obs.any():
            return
        score = stats.utility_sum_this_step[obs] / stats.samples_this_step[obs]
        prev = self._score[obs]
        a = self.score_ema_alpha
        # first observation replaces; subsequent EMA-smooth
        self._score[obs] = np.where(self._seen[obs], (1.0 - a) * prev + a * score, score)
        self._seen[obs] = True
        self._last_visit[obs] = int(step)

    def distribution(self, step: int) -> SamplerDistribution:
        """Mix rank-prioritized score with staleness, then floor / cap."""
        del step
        n = self.n_contexts
        # rank prioritization: rank 1 = highest score (ties broken by stable order)
        order = np.argsort(-self._score, kind="stable")
        rank = np.empty(n, dtype=np.float64)
        rank[order] = np.arange(1, n + 1, dtype=np.float64)
        p_s = (1.0 / rank) ** (1.0 / self.beta)
        p_s = p_s / p_s.sum()
        # staleness: steps since last sampled; unseen (-1) get max staleness
        last = np.where(self._last_visit < 0, 0, self._last_visit)
        stale = self._step - last
        stale = np.where(self._seen, stale, self._step + 1)  # unseen → max staleness
        p_c = stale.astype(np.float64)
        p_c = p_c / p_c.sum() if p_c.sum() > 0 else np.full(n, 1.0 / n)
        raw = (1.0 - self.rho) * p_s + self.rho * p_c
        raw = raw / raw.sum()
        return apply_floor_and_cap(raw, epsilon=self.epsilon, q_max=self.q_max)
