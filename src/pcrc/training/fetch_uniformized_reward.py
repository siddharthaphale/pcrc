"""Uniformized reward + selection-vs-learning decomposition.

Mandatory diagnostic stream from Priority 1b onward (plan §8). The
selection-vs-learning decomposition isolates whether the sampler is
gaining reward by *selecting* high-reward contexts (concentration) or
by *improving* the actor on the train bank (true learning).

Math:

    ``R_uniform_t = (1/|X_train|) · Σ_{x ∈ X_train} ŝ_t(x)``
    ``selection_signal_t = sampler_weighted_t - R_uniform_t``

Where ``ŝ_t(x)`` is the per-context running success EMA. Unobserved
contexts contribute their last EMA value (callers initialize EMAs to
the calibration starter prior so the early-iteration signal is
sensible).

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``fetch_uniformized_reward.py`` row), §8 (mandatory streams from
Priority 1b), §7.1 ``test_uniformized_reward.py``.
"""

from __future__ import annotations

import numpy as np


def compute_uniformized_reward(success_ema: np.ndarray, train_mask: np.ndarray) -> float:
    """Return ``(1/|X_train|) · Σ_x ŝ_t(x)`` over the train bank only.

    Args:
        success_ema: shape ``(|X|,)`` per-context running success EMA.
        train_mask: shape ``(|X|,)`` bool — True iff context is in the
            train split. Held-out contexts are excluded (they cannot
            be part of the curriculum-vs-uniform comparison).

    Returns:
        Float in ``[0, 1]`` assuming ``success_ema`` is in that range.
    """
    if success_ema.shape != train_mask.shape:
        raise ValueError(
            f"shape mismatch: success_ema {success_ema.shape} vs train_mask {train_mask.shape}"
        )
    train_ids = train_mask.nonzero()[0]
    if train_ids.size == 0:
        raise ValueError("train_mask is all False; no train bank to uniformize over")
    return float(success_ema[train_ids].mean())


def compute_sampler_weighted_reward(
    success_ema: np.ndarray, distribution_final: np.ndarray
) -> float:
    """Return ``Σ_x q_t(x) · ŝ_t(x)`` — the expected reward under the sampler.

    The sampler distribution is taken over the full bank index space;
    held-out entries (if present) should already have ``q = 0`` per the
    bank's split definition.
    """
    if success_ema.shape != distribution_final.shape:
        raise ValueError(
            f"shape mismatch: success_ema {success_ema.shape} vs "
            f"distribution_final {distribution_final.shape}"
        )
    return float((distribution_final * success_ema).sum())


def compute_selection_vs_learning_decomposition(
    success_ema: np.ndarray,
    distribution_final: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, float]:
    """Return the full decomposition dict ready for ``log_step``.

    Keys:
        - ``sampler_weighted_reward``
        - ``uniformized_reward``
        - ``selection_signal`` (the difference)
    """
    sw = compute_sampler_weighted_reward(success_ema, distribution_final)
    uni = compute_uniformized_reward(success_ema, train_mask)
    return {
        "sampler_weighted_reward": sw,
        "uniformized_reward": uni,
        "selection_signal": sw - uni,
    }
