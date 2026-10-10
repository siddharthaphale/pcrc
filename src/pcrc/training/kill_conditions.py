"""In-code kill-condition monitor for actor-curator solo execution.

Per-iter abort logic with tolerance windows so noisy early iters don't
trigger false aborts. Required because Phase 3-5 of the actor-curator
loop run unattended on a colleague's hardware (handover-style); there
is no human watching mid-run, so kill checks must be code-enforced.

Four checks:

- ``nan_in_loss_or_advantages`` — immediate abort if any tracked
  metric is NaN/Inf.
- ``curator_entropy_collapse`` — sustained collapse of OSMD weights
  below ``0.1 * log(n_arms)`` for ``entropy_window`` iters at iter ≥ 30.
- ``ac_underperforms_uniform`` — at the named ``ac_check_iters`` (50
  and 75 by default), compare windowed mean reward of AC vs uniform;
  abort only if BOTH checks trigger (the "soft pathology" framing
  avoids early noisy aborts).
- ``offline_ev_negative`` — sustained negative offline explained
  variance for ``ev_floor_window`` iters at iter ≥ 25.

The trainer calls :meth:`KillConditionMonitor.check` after each iter;
on a trigger it writes ``kill_reason.json`` next to ``metrics.jsonl``
via :func:`write_kill_reason`, then raises ``SystemExit(2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pcrc.utils.atomic_io import atomic_write_json


@dataclass(frozen=True)
class KillReason:
    """Diagnostic payload for an auto-aborted run."""

    name: str
    iter: int
    diagnostic: dict[str, Any]
    last_good_ckpt: str | None


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _read_uniform_reference(path: Path | None) -> dict[int, float] | None:
    """Read per-iter mean training reward from a uniform-run metrics.jsonl.

    Returns a dict mapping iter (1-based step) → ``rewards_mean``, or
    ``None`` if ``path`` is ``None``. Missing or malformed rows are
    silently skipped; the AC-vs-uniform check no-ops if the lookup
    misses for the iter under examination.
    """
    if path is None:
        return None
    rewards: dict[int, float] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in row and "rewards_mean" in row:
                rewards[int(row["step"])] = float(row["rewards_mean"])
    return rewards


class KillConditionMonitor:
    """Per-iter abort checks; tolerance windows to avoid noisy aborts.

    Stateful across iters: tracks consecutive-violation counters for
    the entropy and EV checks, and a per-iter buffer of AC mean-rewards
    for the AC-vs-uniform comparison.
    """

    def __init__(
        self,
        n_arms: int,
        uniform_reference_path: Path | None = None,
        *,
        ev_floor_window: int = 3,
        entropy_window: int = 3,
        ac_check_iters: tuple[int, ...] = (50, 75),
        ac_underperformance_tolerance: float = 0.02,
        ac_window_start: int = 20,
        entropy_check_min_iter: int = 30,
        ev_check_min_iter: int = 25,
    ):
        """Configure per-check tolerances and load the uniform reference."""
        self.n_arms = int(n_arms)
        self.uniform_reference_path = uniform_reference_path
        self.ev_floor_window = int(ev_floor_window)
        self.entropy_window = int(entropy_window)
        self.ac_check_iters = tuple(int(c) for c in ac_check_iters)
        self.ac_underperformance_tolerance = float(ac_underperformance_tolerance)
        self.ac_window_start = int(ac_window_start)
        self.entropy_check_min_iter = int(entropy_check_min_iter)
        self.ev_check_min_iter = int(ev_check_min_iter)

        self._entropy_streak = 0
        self._ev_streak = 0
        self._ac_rewards: dict[int, float] = {}
        self._ac_check_results: dict[int, bool] = {}
        self._uniform_rewards = _read_uniform_reference(uniform_reference_path)

    def check(
        self,
        iter: int,  # noqa: A002 — matches the documented public API
        metrics: dict[str, float],
        osmd_weights: np.ndarray,
        last_ckpt: Path | None,
    ) -> KillReason | None:
        """Run all checks against this iter's metrics. Returns reason or None."""
        last_ckpt_str = str(last_ckpt) if last_ckpt is not None else None

        nan_diag = self._check_nan(metrics)
        if nan_diag is not None:
            return KillReason(
                name="nan_in_loss_or_advantages",
                iter=iter,
                diagnostic=nan_diag,
                last_good_ckpt=last_ckpt_str,
            )

        entropy_diag = self._check_entropy(iter, osmd_weights)
        if entropy_diag is not None:
            return KillReason(
                name="curator_entropy_collapse",
                iter=iter,
                diagnostic=entropy_diag,
                last_good_ckpt=last_ckpt_str,
            )

        ev_diag = self._check_offline_ev(iter, metrics)
        if ev_diag is not None:
            return KillReason(
                name="offline_ev_negative",
                iter=iter,
                diagnostic=ev_diag,
                last_good_ckpt=last_ckpt_str,
            )

        ac_diag = self._check_ac_vs_uniform(iter, metrics)
        if ac_diag is not None:
            return KillReason(
                name="ac_underperforms_uniform",
                iter=iter,
                diagnostic=ac_diag,
                last_good_ckpt=last_ckpt_str,
            )
        return None

    def _check_nan(self, metrics: dict[str, float]) -> dict[str, Any] | None:
        bad: dict[str, Any] = {}
        for k in (
            "pg_loss",
            "is_corrected_advantage_mean",
            "is_corrected_advantage_std",
            "rewards_mean",
            "log_ratio_mean",
        ):
            if k not in metrics:
                continue
            v = float(metrics[k])
            if not math.isfinite(v):
                bad[k] = v
        return bad or None

    def _check_entropy(
        self,
        iter: int,  # noqa: A002
        osmd_weights: np.ndarray,
    ) -> dict[str, Any] | None:
        if iter < self.entropy_check_min_iter:
            self._entropy_streak = 0
            return None
        H = _entropy(osmd_weights)
        H_floor = 0.1 * math.log(self.n_arms)
        if H_floor > H:
            self._entropy_streak += 1
        else:
            self._entropy_streak = 0
        if self._entropy_streak >= self.entropy_window:
            return {
                "entropy": H,
                "entropy_floor": H_floor,
                "consecutive_iters_below_floor": self._entropy_streak,
            }
        return None

    def _check_offline_ev(
        self,
        iter: int,  # noqa: A002
        metrics: dict[str, float],
    ) -> dict[str, Any] | None:
        if iter < self.ev_check_min_iter:
            self._ev_streak = 0
            return None
        ev = metrics.get("critic/offline_explained_variance")
        if ev is None:
            self._ev_streak = 0
            return None
        ev_v = float(ev)
        if not math.isfinite(ev_v):
            # Treat NaN as not-a-violation here; _check_nan handles overall NaN.
            self._ev_streak = 0
            return None
        if ev_v < 0:
            self._ev_streak += 1
        else:
            self._ev_streak = 0
        if self._ev_streak >= self.ev_floor_window:
            return {
                "offline_explained_variance": ev_v,
                "consecutive_negative_iters": self._ev_streak,
            }
        return None

    def _check_ac_vs_uniform(
        self,
        iter: int,  # noqa: A002
        metrics: dict[str, float],
    ) -> dict[str, Any] | None:
        # Buffer this iter's AC reward unconditionally so windowed
        # means at iter 50 and 75 have the data they need.
        if "rewards_mean" in metrics:
            self._ac_rewards[iter] = float(metrics["rewards_mean"])

        if iter not in self.ac_check_iters:
            return None
        if self._uniform_rewards is None:
            return None
        if iter <= self.ac_window_start:
            return None

        ac_window = [
            self._ac_rewards[s]
            for s in range(self.ac_window_start, iter + 1)
            if s in self._ac_rewards
        ]
        uni_window = [
            self._uniform_rewards[s]
            for s in range(self.ac_window_start, iter + 1)
            if s in self._uniform_rewards
        ]
        if not ac_window or not uni_window:
            return None

        ac_mean = float(np.mean(ac_window))
        uni_mean = float(np.mean(uni_window))
        passed = ac_mean >= (uni_mean - self.ac_underperformance_tolerance)
        self._ac_check_results[iter] = passed

        # Abort only if BOTH check iters have been evaluated AND both failed.
        if all(c in self._ac_check_results for c in self.ac_check_iters) and all(
            not self._ac_check_results[c] for c in self.ac_check_iters
        ):
            return {
                "ac_mean_window": ac_mean,
                "uniform_mean_window": uni_mean,
                "tolerance": self.ac_underperformance_tolerance,
                "ac_check_results": dict(self._ac_check_results),
                "ac_window_start": self.ac_window_start,
            }
        return None


def write_kill_reason(reason: KillReason, results_dir: Path) -> Path:
    """Write ``kill_reason.json`` next to ``metrics.jsonl`` atomically.

    Atomic write via :func:`pcrc.utils.atomic_io.atomic_write_json`:
    readers never see a partial file even if the container crashes
    mid-write.
    """
    payload = {
        "name": reason.name,
        "iter": reason.iter,
        "diagnostic": reason.diagnostic,
        "last_good_ckpt": reason.last_good_ckpt,
    }
    return atomic_write_json(results_dir / "kill_reason.json", payload)
