"""Campaign 5 external-baseline modes: `rfcl_faithful`, `aced_style` (C5.5), `alp_cells` (C5.2b)."""

from __future__ import annotations

import numpy as np

from pcrc.registry import Method
from pcrc.samplers.scaffold_controller import ScaffoldConfig, ScaffoldController
from pcrc.training.conditions import resolve_condition

FRACS3 = [0.0, 0.5, 0.9]


def _ctrl(mode, n_train=3, n_levels=3, **kw) -> ScaffoldController:
    return ScaffoldController(n_train, n_levels, FRACS3, ScaffoldConfig(mode=mode, **kw))


# ------------------------------------------------------------- rfcl_faithful

def test_rfcl_faithful_windowed_advance_and_no_retreat():
    c = _ctrl(Method.RFCL_FAITHFUL, rfclf_window=5, rfclf_advance_successes=3)
    # 5 frontier rollouts, 3 successes -> advance exactly one level; window cleared
    for w, succ in enumerate([True, True, False, True, False]):
        c.update([0], [2], [succ], [0.1], step=w)
    assert c.ell[0] == 1
    assert len(c._rf_recent[0]) == 0
    # sustained failure at the new frontier: NO retreat, ever
    for w in range(5, 15):
        c.update([0], [1], [False], [0.1], step=w)
    assert c.ell[0] == 1


def test_rfcl_faithful_ignores_off_frontier_outcomes():
    c = _ctrl(Method.RFCL_FAITHFUL, rfclf_window=2, rfclf_advance_successes=2)
    for w in range(4):
        c.update([0], [1], [True], [0.1], step=w)  # level 1 != frontier (2): not counted
    assert c.ell[0] == 2 and len(c._rf_recent[0]) == 0


def test_rfcl_faithful_prioritizes_struggling_contexts():
    c = _ctrl(Method.RFCL_FAITHFUL, epsilon=0.0, q_max=None, rfclf_window=3, rfclf_advance_successes=3)
    # context 0 succeeds (low priority), context 1 fails (high priority), context 2 unseen (max)
    for w in range(2):
        c.update([0, 1], [2, 2], [True, False], [0.1, 0.1], step=w)
    rng = np.random.default_rng(0)
    contexts, levels, _ = c.select(600, step=2, rng=rng)
    counts = np.bincount(contexts, minlength=3)
    assert counts[1] > counts[0]  # failing context sampled more than succeeding one
    assert counts[2] > counts[0]  # unseen context outranks the succeeding one
    assert all(lvl == c.ell[x] for x, lvl in zip(contexts, levels))


# --------------------------------------------------------------- aced_style

def test_aced_style_advances_but_never_retreats():
    kw = dict(n_min=2, cooldown=0, hysteresis_windows=2, beta_s=1.0, stale_advance_windows=0)
    c = _ctrl(Method.ACED_STYLE, **kw)
    for w in range(3):
        c.update([0], [2], [True], [0.1], step=w)
    assert c.ell[0] == 1  # threshold advance works
    for w in range(3, 10):
        c.update([0], [1], [False], [0.1], step=w)
    assert c.ell[0] == 1  # retreat branch disabled (rfcl_style would retreat here)


def test_aced_style_resolve_condition_forces_antistall_off():
    block = {"n_levels": 3, "level_fractions": FRACS3, "kwargs": {"stale_advance_windows": 4}}
    _, overrides = resolve_condition(Method.ACED_STYLE, block, {})
    assert overrides["scaffold_mode"] == Method.ACED_STYLE
    assert overrides["scaffold_kwargs"]["stale_advance_windows"] == 0


# ---------------------------------------------------------------- alp_cells

def test_alp_cells_no_frontier_and_samples_all_levels():
    c = _ctrl(Method.ALP_CELLS, n_min=1, cooldown=0, hysteresis_windows=1, beta_s=1.0)
    for w in range(6):
        c.update([0, 1, 2], [2, 2, 2], [True, True, True], [0.1] * 3, step=w)
    assert (c.ell == c.L).all()  # no transitions
    rng = np.random.default_rng(0)
    _, levels, _ = c.select(500, step=6, rng=rng)
    assert set(levels) == {0, 1, 2}


def test_alp_cells_prioritizes_high_learning_progress():
    c = _ctrl(Method.ALP_CELLS, epsilon=0.0, q_max=None, lambda_stale=0.0, beta_s=1.0, beta_s_slow=0.05)
    # cell (0,1): success flips 0 -> 1 (high |s - s_slow|); cell (1,1): steady failure (ALP ~ 0)
    for w in range(4):
        c.update([0, 1], [1, 1], [w >= 2, False], [0.1, 0.1], step=w)
    rng = np.random.default_rng(0)
    contexts, levels, _ = c.select(800, step=4, rng=rng)
    hits = {}
    for x, lvl in zip(contexts, levels):
        hits[(x, lvl)] = hits.get((x, lvl), 0) + 1
    assert hits.get((0, 1), 0) > hits.get((1, 1), 0)
