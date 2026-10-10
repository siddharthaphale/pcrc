"""Campaign 5 controller modes: `per_group_success` (C5.1) and `plr_cells` (C5.2a).

`per_group_success` is the per_context_success rule at GROUP granularity — one pooled frontier, EMA,
evidence count, and cooldown per group; uniform context sampling within the chosen group.
`plr_cells` is PLR-style rank-prioritized |TD-error| + staleness sampling over ALL (x, ℓ)
cells with no frontier. Both must leave every other condition byte-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.registry import Method, SamplerName
from pcrc.samplers.scaffold_controller import ScaffoldConfig, ScaffoldController
from pcrc.training.conditions import resolve_condition

FRACS3 = [0.0, 0.5, 0.9]


def _gctrl(n_train=4, n_levels=3, group_ids=(0, 0, 1, 1), **cfg_kwargs) -> ScaffoldController:
    return ScaffoldController(
        n_train,
        n_levels,
        FRACS3,
        ScaffoldConfig(mode=Method.PER_GROUP_SUCCESS, **cfg_kwargs),
        group_ids=list(group_ids),
    )


# ------------------------------------------------------------- per_group_success

def test_per_group_success_requires_group_ids():
    with pytest.raises(ValueError, match="requires group_ids"):
        ScaffoldController(4, 3, FRACS3, ScaffoldConfig(mode=Method.PER_GROUP_SUCCESS))
    with pytest.raises(ValueError, match="not 'per_group_success'"):
        ScaffoldController(4, 3, FRACS3, ScaffoldConfig(mode=Method.PER_CONTEXT_SUCCESS), group_ids=[0] * 4)
    with pytest.raises(ValueError, match="contiguous"):
        _gctrl(group_ids=(0, 0, 2, 2))  # gap: group 1 missing


def test_per_group_success_starts_at_easiest_level():
    c = _gctrl()
    assert (c.g_ell == c.L).all()
    assert (c.ell == c.L).all()  # expanded per-context view


def test_per_group_success_pools_success_across_group():
    # Group 0's two contexts each see one rollout/window: one success + one failure → pooled
    # s=0.5 (dead band, holds). A per-context rule with beta_s=1.0 would advance context 0.
    c = _gctrl(n_min=2, cooldown=0, hysteresis_windows=2, beta_s=1.0, stale_advance_windows=0)
    for w in range(4):
        c.update([0, 1], [2, 2], [True, False], [0.1, 0.1], step=w)
    assert c.g_ell[0] == 2  # pooled dead band: no move
    assert c.g_ell[1] == 2  # group 1 never sampled: no evidence, no move


def test_per_group_success_advances_and_retreats_as_a_unit():
    c = _gctrl(n_min=2, cooldown=0, hysteresis_windows=2, beta_s=1.0, stale_advance_windows=0)
    for w in range(3):
        c.update([0, 1], [2, 2], [True, True], [0.1, 0.1], step=w)
    assert c.g_ell[0] == 1  # group 0 advanced after 2-window streak
    assert (c.ell[:2] == 1).all()  # BOTH members moved together
    assert (c.ell[2:] == 2).all()
    # now fail hard at the new level → the whole group retreats (2-window hysteresis; assert
    # right after the move — the persisted level-2 success EMA re-advances on later windows,
    # matching per-context PCRC's statistics-persist-on-revisit semantics)
    for w in range(3, 5):
        c.update([0, 1], [1, 1], [False, False], [0.1, 0.1], step=w)
    assert c.g_ell[0] == 2
    assert (c.ell[:2] == 2).all()


def test_per_group_success_evidence_gate_uses_pooled_counts():
    # n_min=4 with 2 rollouts/window: a single context would need 2 windows of ITS OWN data,
    # but the pooled group hits n_min via both members' rollouts.
    c = _gctrl(n_min=4, cooldown=0, hysteresis_windows=1, beta_s=1.0, stale_advance_windows=0)
    c.update([0, 1], [2, 2], [True, True], [0.1, 0.1], step=0)  # pooled count = 2 < 4: gated
    assert c.g_ell[0] == 2
    c.update([0, 1], [2, 2], [True, True], [0.1, 0.1], step=1)  # pooled count = 4: eligible
    assert c.g_ell[0] == 1


def test_per_group_success_antistall_moves_group():
    c = _gctrl(
        n_min=2, cooldown=0, hysteresis_windows=2, beta_s=1.0,
        tau_lo=0.15, tau_hi=0.8, stale_advance_windows=3,
    )
    # dead-band success (0.5) forever → antistall forces one harder step after 3 stuck windows
    for w in range(6):
        c.update([0, 1], [2, 2], [True, False], [0.1, 0.1], step=w)
    assert c.g_ell[0] == 1
    assert (c.ell[:2] == 1).all()


def test_per_group_success_select_levels_follow_per_group_success():
    c = _gctrl(n_min=2, cooldown=0, hysteresis_windows=2, beta_s=1.0, stale_advance_windows=0)
    for w in range(3):
        c.update([0, 1], [2, 2], [True, True], [0.1, 0.1], step=w)
    rng = np.random.default_rng(0)
    contexts, levels, fractions = c.select(200, step=3, rng=rng)
    for x, lvl, fr in zip(contexts, levels, fractions, strict=True):
        g = 0 if x < 2 else 1
        assert lvl == c.g_ell[g]
        assert fr == FRACS3[lvl]
    # both groups sampled (2-group distribution is uniform under the q_max cap)
    assert {0 if x < 2 else 1 for x in contexts} == {0, 1}


def test_per_group_success_resolve_condition_keeps_frozen_kwargs():
    block = {"n_levels": 3, "level_fractions": FRACS3, "kwargs": {"stale_advance_windows": 4}}
    loop_condition, overrides = resolve_condition(Method.PER_GROUP_SUCCESS, block, {})
    assert loop_condition == SamplerName.UNIFORM
    assert overrides["scaffold_enabled"] is True
    assert overrides["scaffold_mode"] == Method.PER_GROUP_SUCCESS
    assert overrides["scaffold_kwargs"]["stale_advance_windows"] == 4  # NOT forced off


# ----------------------------------------------------------------- plr_cells

def _pctrl(n_train=3, n_levels=3, **cfg_kwargs) -> ScaffoldController:
    return ScaffoldController(
        n_train, n_levels, FRACS3, ScaffoldConfig(mode=Method.PLR_CELLS, **cfg_kwargs)
    )


def test_plr_cells_samples_all_cells_not_a_frontier():
    c = _pctrl()
    rng = np.random.default_rng(0)
    contexts, levels, fractions = c.select(500, step=0, rng=rng)
    assert set(levels) == {0, 1, 2}  # no frontier constraint: every level reachable at once
    assert set(contexts) == {0, 1, 2}
    assert (c.ell == c.L).all()  # frontier state untouched


def test_plr_cells_never_transitions():
    c = _pctrl(n_min=1, cooldown=0, hysteresis_windows=1, beta_s=1.0)
    for w in range(6):
        c.update([0, 1, 2], [2, 2, 2], [True, True, True], [1.0, 1.0, 1.0], step=w)
    assert (c.ell == c.L).all()  # sustained success moves nothing


def test_plr_cells_prioritizes_high_td_cells():
    c = _pctrl(epsilon=0.0, q_max=None, plr_rho=0.0, plr_beta=0.1)
    # cell (0, 1) gets a large |TD| score; everything else small
    c.update([0, 1, 2], [1, 0, 2], [False, False, False], [10.0, 0.01, 0.01], step=0)
    dist = c._plr_cell_distribution()
    flat = np.asarray(dist.final).reshape(c.n_train, c.n_levels)
    assert flat[0, 1] == flat.max()  # highest-scoring cell gets the most mass


def test_plr_cells_score_replaces_on_first_visit_then_emas():
    c = _pctrl()
    c.update([0], [1], [False], [2.0], step=0)
    assert c._plr_score[0, 1] == pytest.approx(2.0)  # replace, not 0.3 * 2.0
    c.update([0], [1], [False], [4.0], step=1)
    assert c._plr_score[0, 1] == pytest.approx(0.7 * 2.0 + 0.3 * 4.0)


def test_plr_cells_staleness_favors_unvisited_cells():
    c = _pctrl(epsilon=0.0, q_max=None, plr_rho=1.0)  # pure staleness
    for w in range(5):
        c.update([0], [1], [False], [1.0], step=w)
    dist = c._plr_cell_distribution()
    flat = np.asarray(dist.final).reshape(c.n_train, c.n_levels)
    assert flat[0, 1] == flat.min()  # the only visited cell is the freshest → least mass


def test_plr_cells_resolve_condition():
    block = {"n_levels": 3, "level_fractions": FRACS3, "kwargs": {}}
    loop_condition, overrides = resolve_condition(Method.PLR_CELLS, block, {})
    assert loop_condition == SamplerName.UNIFORM
    assert overrides["scaffold_mode"] == Method.PLR_CELLS
