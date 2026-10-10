"""OSMD (Online Mirror Descent) updater + PPO-clipped surrogate loss.

Implements:

- Tabular OSMD (Gu et al. 2026, Algorithm 2): closed-form multiplicative
  weights update `w_{t+1} ∝ w_t · exp(η · u_hat)`, computed in log space
  for numerical stability.
- PPO-clipped OSMD surrogate (Eq. 12) as a standalone function:
  `L = -mean(min(r·A, clip(r, 1-ε, 1+ε)·A))`, ratio r = exp(log_p_new - log_p_old).

The function-approximation path on `OSMDUpdater` is not implemented;
callers needing a function-approximation surrogate should use
`compute_ppo_clipped_osmd_loss` directly over a function-approximation
policy.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from pcrc.utils.precision_zones import fp32_zone, to_fp32


class OSMDUpdater:
    """Tabular OSMD over the candidate batch.

    Precision profile: all state (``self._weights``) and arithmetic
    (advantages cast to float64, log/exp in log-space with LSE
    stabilization) run in fp64. Inputs from a bf16 actor are upcast at
    the ``np.asarray(..., dtype=np.float64)`` entry point, so no
    additional ``fp32_zone`` wrap is needed.
    """

    def __init__(
        self,
        n_arms: int,
        eta: float,
        mode: Literal["tabular", "function_approx"] = "tabular",
    ):
        if mode not in ("tabular", "function_approx"):
            raise ValueError(f"mode must be 'tabular' or 'function_approx', got {mode!r}")
        if mode == "function_approx":
            raise NotImplementedError(
                "Function-approximation OSMD path is not implemented. "
                "Use compute_ppo_clipped_osmd_loss directly for the PPO-clipped "
                "surrogate over a function-approximation policy."
            )

        self.n_arms = int(n_arms)
        self.eta = float(eta)
        self.mode = mode

        # Probability simplex, float64 for stability over many multiplicative steps.
        self._weights = np.full(self.n_arms, 1.0 / self.n_arms, dtype=np.float64)

    def set_weights(self, w: np.ndarray) -> None:
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (self.n_arms,):
            raise ValueError(f"weights must have shape ({self.n_arms},), got {w.shape}")
        self._weights = w.copy()

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def step(self, advantages: np.ndarray) -> None:
        """One tabular OSMD step: `w ∝ w · exp(η·u_hat)`, in log space.

        With η=0 the update reduces to the identity (logits get a zero shift,
        the post-LSE probabilities equal the pre-step probabilities).
        """
        u = np.asarray(advantages, dtype=np.float64)
        if u.shape != (self.n_arms,):
            raise ValueError(f"advantages must have shape ({self.n_arms},), got {u.shape}")

        # log-space update: log w' = log w + η·u, then renormalize via LSE.
        # Add a tiny floor to guard against log(0) if a caller initialized a
        # weight to exactly zero. 1e-300 is safely below any meaningful weight.
        log_w = np.log(self._weights + 1e-300) + self.eta * u
        log_w -= log_w.max()  # subtract the max for stability
        new_w = np.exp(log_w)
        new_w /= new_w.sum()
        self._weights = new_w


def compute_ppo_clipped_osmd_loss(
    log_p_old: torch.Tensor,
    log_p_new: torch.Tensor,
    advantages: torch.Tensor,
    eps: float = 0.2,
) -> torch.Tensor:
    """PPO-clipped OSMD surrogate (Gu et al. 2026 Eq. 12, with PPO clip).

    Returns scalar `L = -mean(min(r·A, clip(r, 1-eps, 1+eps)·A))`,
    where `r = exp(log_p_new - log_p_old)`. Per-element clipping; final
    `mean()` reduces over the batch.

    Wrapped in ``fp32_zone`` because at a tight PPO clip (e.g. ``[3e-4, 4e-4]``)
    the bf16 ULP near 1.0 (~7.8e-3) is **larger than the entire clip range**:
    ``exp(3e-4)`` rounds to exactly 1.0 in bf16, killing both the unclipped and
    clipped surrogate signals (``min(r·A, clip(r)·A) == A`` becomes constant in
    the parameters, gradient = 0). The ``eps=0.2`` default would be bf16-safe,
    but callers may override with a tight clip, so the wrap is unconditional.
    """
    log_p_old = log_p_old.detach()
    advantages = advantages.detach()

    with fp32_zone():
        lpo_f, lpn_f, adv_f = to_fp32(log_p_old, log_p_new, advantages)
        ratio = (lpn_f - lpo_f).exp()
        clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        surrogate = torch.minimum(ratio * adv_f, clipped_ratio * adv_f)
        return -surrogate.mean()
