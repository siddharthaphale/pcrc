"""RFCL-style controller mode (Test Campaign 4 Stage B, campaign §B.3).

`rfcl_style` is a per-context frontier curriculum like PCRC, but with UNIFORM context sampling (no
headroom/recency/progress score) and NO anti-stall (`stale_advance_windows=0`). These six tests are the
campaign's B.3 checklist.
"""

from __future__ import annotations

import numpy as np

from pcrc.registry import TRAJECTORY_PREFIX, Method, ScaffoldSource
from pcrc.samplers.scaffold_controller import ScaffoldConfig, ScaffoldController
from pcrc.training.conditions import resolve_condition

FRACS3 = [0.0, 0.5, 0.9]


def _ctrl(n_train=3, n_levels=3, fractions=None, **cfg_kwargs) -> ScaffoldController:
    return ScaffoldController(n_train, n_levels, fractions or FRACS3, ScaffoldConfig(**cfg_kwargs))


def _drive(c, contexts, levels, successes, tds, n_windows, start_step=0):
    for w in range(n_windows):
        c.update(contexts, levels, successes, tds, step=start_step + w)


# 1. starts every context at the easiest level L
def test_rfcl_starts_at_easiest_level():
    c = _ctrl(n_train=4, mode=Method.RFCL_STYLE, stale_advance_windows=0)
    assert (c.ell == c.L).all()


# 2. high success advances one level (success-only frontier; uncertainty irrelevant)
def test_rfcl_high_success_advances():
    c = _ctrl(
        n_train=3, mode=Method.RFCL_STYLE, n_min=2, cooldown=0, hysteresis_windows=2,
        beta_s=1.0, tau_hi=0.8, tau_lo=0.15, stale_advance_windows=0,
    )
    _drive(c, [0, 1, 2], [2, 2, 2], [True, False, False], [5.0, 5.0, 5.0], n_windows=3)
    assert c.ell[0] == 1  # advanced on success alone (high TD/uncertainty does not block it)
    assert c.ell[1] == 2 and c.ell[2] == 2  # failers do not advance


# 3. low success retreats one level
def test_rfcl_low_success_retreats():
    c = _ctrl(
        n_train=3, mode=Method.RFCL_STYLE, n_min=2, cooldown=0, hysteresis_windows=2,
        beta_s=1.0, tau_hi=0.8, tau_lo=0.15, stale_advance_windows=0,
    )
    c.ell[:] = 1  # leave room to add assistance (→ ℓ=2)
    _drive(c, [0, 1, 2], [1, 1, 1], [False, True, True], [5.0, 5.0, 5.0], n_windows=3)
    assert c.ell[0] == 2  # retreated on failure (1 → 2, more assistance)
    assert c.ell[1] == 0 and c.ell[2] == 0  # the two that succeeded advanced (1 → 0)


# 4. dead-band success HOLDS, unlike PCRC anti-stall (the defining RFCL difference)
def test_rfcl_dead_band_holds_no_antistall():
    c = _ctrl(
        n_train=2, n_levels=3, mode=Method.RFCL_STYLE, n_min=2, cooldown=0, hysteresis_windows=2,
        beta_s=1.0, tau_hi=0.8, tau_lo=0.15, stale_advance_windows=0,
    )
    _drive(c, [0, 0], [2, 2], [True, False], [0.1, 0.1], n_windows=12)  # s≈0.5 ∈ [τ_lo, τ_hi]
    assert c.ell[0] == 2  # parks in the dead band — RFCL has no anti-stall to force descent
    # contrast: per_context_success with anti-stall WOULD descend out of the dead band
    c2 = _ctrl(
        n_train=2, n_levels=3, mode=Method.PER_CONTEXT_SUCCESS, n_min=2, cooldown=0, hysteresis_windows=2,
        beta_s=1.0, tau_hi=0.8, tau_lo=0.15, stale_advance_windows=3,
    )
    _drive(c2, [0, 0], [2, 2], [True, False], [0.1, 0.1], n_windows=4)
    assert c2.ell[0] == 1  # anti-stall forces one harder step


# 5. context sampling is uniform (no floor/cap/score), and levels follow the frontier
def test_rfcl_uniform_context_sampling():
    c = _ctrl(n_train=5, mode=Method.RFCL_STYLE, stale_advance_windows=0)
    contexts, levels, _ = c.select(10_000, step=0, rng=np.random.default_rng(0))
    counts = np.bincount(np.asarray(contexts), minlength=5)
    assert counts.min() > 1700 and counts.max() < 2300  # ~2000 each, uniform within noise
    assert all(lv == c.L for lv in levels)  # levels are the per-context frontier (all L initially)


# 6. uses the SAME automatic-ladder mechanism as trajectory_per_context_success (trajectory source)
def test_rfcl_auto_uses_same_trajectory_source():
    sb = {"n_levels": 5, "level_fractions": [0.0, 0.25, 0.5, 0.75, 0.9], "kwargs": {}}
    _, ov_rfcl = resolve_condition((TRAJECTORY_PREFIX + Method.RFCL_STYLE), sb, {})
    _, ov_pcrc = resolve_condition((TRAJECTORY_PREFIX + Method.PER_CONTEXT_SUCCESS), sb, {})
    assert ov_rfcl["scaffold_source"] == ScaffoldSource.TRAJECTORY == ov_pcrc["scaffold_source"]
    assert ov_rfcl["scaffold_mode"] == Method.RFCL_STYLE
    assert ov_rfcl["scaffold_kwargs"]["stale_advance_windows"] == 0  # anti-stall off
