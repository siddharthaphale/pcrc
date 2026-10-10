"""Per-iter kill-condition checks for the SimpleVLA-RL method validation.

The validation runs are short (15 iters per condition) and a silent
divergence wastes a meaningful share of the 10 H100-hr cap. Each iter
calls ``check_kill_conditions`` immediately after metrics are logged.
A triggered condition raises ``KillConditionTriggered`` carrying a
description; the outer loop is expected to catch it, flush logs, and
mark the run as ``killed_by_condition`` in the final report.

The thresholds here are calibrated for the SimpleVLA-RL setting:
N_ARMS=10, GRPO with N_SAMPLES_PER_GROUP=4, traj1 SFT base.
They are NOT general-purpose RL kill conditions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class KillConditionTriggered(Exception):  # noqa: N818
    """Raised by ``check_kill_conditions`` to halt the validation loop.

    Name kept without the conventional ``Error`` suffix because the exception
    is part of the existing public API consumed by several training loops.
    """


class _State:
    """Mutable counter for multi-iter kill conditions (e.g. 3-in-a-row grad=0)."""

    __slots__ = ("consec_grad_zero",)

    def __init__(self) -> None:
        self.consec_grad_zero = 0


def make_state() -> _State:
    return _State()


def check_kill_conditions(
    metrics: dict[str, Any],
    *,
    step: int,
    cumulative_hrs: float,
    state: _State,
    n_arms: int,
    budget_cap_hrs: float = 8.0,
    grad_zero_threshold: int = 3,
) -> None:
    """Raise ``KillConditionTriggered`` if any condition fires; else return.

    ``state`` carries multi-iter counters (must be reused across calls in the
    same run). ``budget_cap_hrs`` matches the plan's stop-projection level.
    """
    is_ratio = metrics.get("curator/is_ratio_mean")
    grad_norm = metrics.get("actor/grad_norm")
    actor_loss = metrics.get("actor/loss")
    entropy = metrics.get("curator/osmd_entropy")
    reward = metrics.get("reward")

    # 1. IS ratio NaN / Inf — bandit feedback broken.
    if is_ratio is not None:
        if not _is_finite(is_ratio):
            raise KillConditionTriggered(
                f"is_ratio_mean is non-finite ({is_ratio!r}) at step={step}"
            )
        # 2. IS ratio mean outside (0.1, 10) after warmup steps.
        if step >= 5 and (is_ratio > 10.0 or is_ratio < 0.1):
            raise KillConditionTriggered(
                f"is_ratio_mean={is_ratio:.4f} outside (0.1, 10) at step={step}"
            )

    # 3. Actor grad_norm == 0 for grad_zero_threshold consecutive iters.
    if grad_norm is not None:
        if grad_norm == 0.0:
            state.consec_grad_zero += 1
            if state.consec_grad_zero >= grad_zero_threshold:
                raise KillConditionTriggered(
                    f"actor/grad_norm == 0 for {grad_zero_threshold} consecutive "
                    f"iters (through step={step})"
                )
        else:
            state.consec_grad_zero = 0

    # 4. Actor loss diverged.
    if actor_loss is not None and not _is_finite(actor_loss):
        raise KillConditionTriggered(f"actor/loss is non-finite ({actor_loss!r}) at step={step}")

    # 5. OSMD entropy exceeds log(N_ARMS) + 0.1 — implementation bug.
    if entropy is not None and entropy > math.log(n_arms) + 0.1:
        raise KillConditionTriggered(
            f"osmd_entropy={entropy:.4f} > log({n_arms})+0.1="
            f"{math.log(n_arms) + 0.1:.4f} at step={step}"
        )

    # 6. Negative reward (LIBERO emits binary 0/1; a negative reading is a
    #    pipeline bug, not a learning signal).
    if reward is not None and reward < -0.1:
        raise KillConditionTriggered(
            f"reward={reward:.4f} negative at step={step} "
            f"(LIBERO binary reward should be in [0, 1])"
        )

    # 7. Budget cap reached.
    if cumulative_hrs > budget_cap_hrs:
        raise KillConditionTriggered(
            f"cumulative H100-hrs={cumulative_hrs:.2f} > cap={budget_cap_hrs:.2f}"
        )


def _is_finite(v: Any) -> bool:
    """Float-or-numpy-or-torch finite check that doesn't need torch imported."""
    try:
        return bool(np.isfinite(v))
    except (TypeError, ValueError):
        return False
