"""Tests for :class:`pcrc.samplers.scaffold_controller.ScaffoldController`.

Covers the framing-doc §6–§9 mechanics the plan (§3.5) requires: score
monotonicity, the ``λ_u=0`` ablation, frontier transitions (harder/easier/hold)
under ``n_min``/cooldown/hysteresis, valid ``select`` output, the
``global_anneal``/``random_scaffold`` baseline modes, and robust-z edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.registry import Method
from pcrc.samplers.scaffold_controller import ScaffoldConfig, ScaffoldController

FRACS3 = [0.0, 0.5, 0.9]  # 3 levels: ℓ=0 true start, ℓ=2 easiest


def _ctrl(n_train=3, n_levels=3, fractions=None, **cfg_kwargs) -> ScaffoldController:
    cfg_kwargs.setdefault("mode", Method.PER_CONTEXT_UNCERTAINTY_GATED)
    return ScaffoldController(
        n_train,
        n_levels,
        fractions or FRACS3,
        ScaffoldConfig(**cfg_kwargs),
    )


# --------------------------------------------------------------- construction


@pytest.mark.parametrize(
    "fracs",
    [
        [0.1, 0.5, 0.9],  # [0] != 0.0
        [0.0, 0.5],  # wrong length for n_levels=3
        [0.0, 0.9, 0.5],  # not non-decreasing
        [0.0, 0.5, 1.5],  # out of [0, 1]
    ],
)
def test_invalid_level_fractions_raise(fracs: list[float]) -> None:
    with pytest.raises(ValueError):
        ScaffoldController(3, 3, fracs, ScaffoldConfig())


@pytest.mark.parametrize(
    "bad", [{"mode": "nope"}, {"tau": 0.0}, {"beta_s": 0.0}, {"beta_u": 1.5}, {"epsilon": 2.0}]
)
def test_invalid_config_raises(bad: dict) -> None:
    with pytest.raises(ValueError):
        ScaffoldConfig(**bad)


def test_init_starts_at_easiest_level() -> None:
    c = _ctrl()
    assert (c.ell == c.L).all()
    assert c.L == 2


# ---------------------------------------------------------------------- select


def test_select_returns_valid_cells() -> None:
    c = _ctrl()
    rng = np.random.default_rng(0)
    contexts, levels, fractions = c.select(8, step=0, rng=rng)
    assert len(contexts) == len(levels) == len(fractions) == 8
    assert all(0 <= x < c.n_train for x in contexts)
    assert all(0 <= lvl < c.n_levels for lvl in levels)
    # fraction must be exactly the level's α and within [0, 1].
    assert all(fractions[i] == FRACS3[levels[i]] for i in range(8))
    assert all(0.0 <= f <= 1.0 for f in fractions)


def test_select_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError):
        _ctrl().select(0, step=0, rng=np.random.default_rng(0))


def test_global_anneal_levels_follow_schedule() -> None:
    c = _ctrl(mode=Method.GLOBAL_ANNEAL, anneal_iters=100)
    rng = np.random.default_rng(0)
    # step 0 → easiest level L for everything.
    _, levels0, _ = c.select(16, step=0, rng=rng)
    assert set(levels0) == {c.L}
    # past anneal_iters → true start (ℓ=0).
    _, levels_end, _ = c.select(16, step=999, rng=rng)
    assert set(levels_end) == {0}


def test_random_scaffold_spans_levels() -> None:
    c = _ctrl(mode=Method.RANDOM_SCAFFOLD)
    rng = np.random.default_rng(1)
    _, levels, _ = c.select(200, step=0, rng=rng)
    assert min(levels) == 0 and max(levels) == c.L


def test_pcrc_select_prefers_high_uncertainty_context() -> None:
    c = _ctrl(n_train=3, lambda_h=0.0, lambda_p=0.0, lambda_stale=0.0, lambda_u=5.0, n_min=1)
    # All at frontier level L; give context 1 the highest uncertainty.
    c.counts[:, c.L] = 10
    c.u[:, c.L] = np.array([0.1, 0.9, 0.1])
    dist = c._frontier_distribution()
    assert int(np.argmax(dist.final)) == 1


# ----------------------------------------------------------------------- score


def test_score_monotone_in_uncertainty() -> None:
    c = _ctrl(n_train=3, lambda_u=1.0, n_min=2)
    c.counts[:, c.L] = 5  # past reliability gate
    c.u[:, c.L] = np.array([0.1, 0.1, 0.1])  # flat
    base = c._score().copy()
    c.u[:, c.L] = np.array([0.1, 0.1, 0.9])  # raise context 2's uncertainty
    bumped = c._score()
    assert bumped[2] > base[2]


def test_lambda_u_zero_removes_uncertainty() -> None:
    c = _ctrl(n_train=3, mode=Method.PER_CONTEXT_SUCCESS, n_min=2)
    c.counts[:, c.L] = 5
    c.u[:, c.L] = np.array([0.1, 0.1, 0.1])
    base = c._score().copy()
    c.u[:, c.L] = np.array([0.1, 0.1, 0.9])
    assert np.allclose(c._score(), base)  # uncertainty has no effect when λ_u=0


# ----------------------------------------------------------------- robust-z


def test_robust_z_flat_is_zero_no_nan() -> None:
    c = _ctrl(n_train=4, n_min=1)
    c.counts[:, c.L] = 5
    c.u[:, c.L] = 0.3  # identical across contexts
    z = c._frontier_u_tilde()
    assert np.all(z == 0.0)
    assert not np.isnan(z).any()


def test_robust_z_clipped_and_gated() -> None:
    c = _ctrl(n_train=4, n_min=3, clip=3.0)
    c.u[:, c.L] = np.array([0.1, 0.1, 0.1, 100.0])  # one huge outlier
    c.counts[:, c.L] = np.array([5, 5, 5, 1])  # outlier below the n_min gate
    z = c._frontier_u_tilde()
    assert np.all(np.abs(z) <= 3.0 + 1e-9)
    assert z[3] == 0.0  # gated out (count < n_min)


# ----------------------------------------------------------------- transitions


def _drive(c, contexts, levels, successes, tds, n_windows, start_step=0):
    for w in range(n_windows):
        c.update(contexts, levels, successes, tds, step=start_step + w)


def test_solved_and_confident_moves_harder() -> None:
    # ctx0 solved + low uncertainty (confident vs peers) → frontier ℓ:2→1.
    c = _ctrl(
        n_train=3,
        n_min=2,
        cooldown=0,
        hysteresis_windows=2,
        beta_s=1.0,
        beta_u=1.0,
        u_lo=0.0,
        tau_hi=0.8,
        tau_lo=0.15,
    )
    _drive(c, [0, 1, 2], [2, 2, 2], [True, False, False], [0.1, 5.0, 5.0], n_windows=3)
    assert c.ell[0] == 1
    assert c.ell[1] == 2 and c.ell[2] == 2  # uncertain failers hold


def test_n_min_gate_blocks_early_transition() -> None:
    c = _ctrl(
        n_train=3, n_min=5, cooldown=0, hysteresis_windows=2, beta_s=1.0, beta_u=1.0, u_lo=0.0
    )
    _drive(c, [0, 1, 2], [2, 2, 2], [True, False, False], [0.1, 5.0, 5.0], n_windows=3)
    assert c.ell[0] == 2  # only 3 rollouts in the cell < n_min=5


def test_failing_and_confident_moves_easier() -> None:
    c = _ctrl(
        n_train=3,
        n_min=2,
        cooldown=0,
        hysteresis_windows=2,
        beta_s=1.0,
        beta_u=1.0,
        u_lo=0.0,
        tau_lo=0.15,
    )
    c.ell[:] = 1  # leave room to add assistance (→ ℓ=2)
    # ctx0 fails with LOW uncertainty (confident); peers high uncertainty.
    _drive(c, [0, 1, 2], [1, 1, 1], [False, True, True], [0.1, 5.0, 5.0], n_windows=3)
    assert c.ell[0] == 2


def test_anti_stall_disabled_parks_in_dead_band() -> None:
    # Moderate success (0.5 ∈ [τ_lo, τ_hi]) → neither advance nor retreat → frontier PARKS
    # when stale_advance_windows=0 (old behavior). This is the bug the anti-stall fixes.
    c = _ctrl(
        n_train=2,
        n_levels=3,
        mode=Method.PER_CONTEXT_SUCCESS,
        n_min=2,
        cooldown=0,
        hysteresis_windows=2,
        beta_s=1.0,
        tau_hi=0.8,
        tau_lo=0.15,
        stale_advance_windows=0,
    )
    _drive(c, [0, 0], [2, 2], [True, False], [0.1, 0.1], n_windows=8)
    assert c.ell[0] == 2  # parked at the start level — never reaches ℓ=0


def test_anti_stall_forces_descent_when_stuck() -> None:
    # Same moderate-success dead-band cell, but stale_advance_windows=3 forces ONE harder step
    # so the frontier progresses toward ℓ=0 instead of parking.
    c = _ctrl(
        n_train=2,
        n_levels=3,
        mode=Method.PER_CONTEXT_SUCCESS,
        n_min=2,
        cooldown=0,
        hysteresis_windows=2,
        beta_s=1.0,
        tau_hi=0.8,
        tau_lo=0.15,
        stale_advance_windows=3,
    )
    _drive(c, [0, 0], [2, 2], [True, False], [0.1, 0.1], n_windows=4)
    assert c.ell[0] == 1  # forced down out of the dead band


def test_anti_stall_does_not_fire_when_failing() -> None:
    # A failing cell (s < τ_lo) should RETREAT (easier), not be force-advanced by the anti-stall.
    c = _ctrl(
        n_train=2,
        n_levels=3,
        mode=Method.PER_CONTEXT_SUCCESS,
        n_min=2,
        cooldown=0,
        hysteresis_windows=2,
        beta_s=1.0,
        tau_hi=0.8,
        tau_lo=0.15,
        stale_advance_windows=3,
    )
    c.ell[0] = 1  # leave room to add assistance (→ 2)
    _drive(c, [0, 0], [1, 1], [False, False], [0.1, 0.1], n_windows=4)
    assert c.ell[0] == 2  # retreated (easier), not force-advanced


def test_uncertain_cell_holds() -> None:
    # ctx0 solved but HIGHEST uncertainty → not confident → holds; ctx1 advances.
    c = _ctrl(
        n_train=2, n_min=2, cooldown=0, hysteresis_windows=2, beta_s=1.0, beta_u=1.0, u_lo=0.0
    )
    _drive(c, [0, 1], [2, 2], [True, True], [5.0, 0.1], n_windows=4)
    assert c.ell[0] == 2  # held (uncertain)
    assert c.ell[1] == 1  # advanced (confident + solved)


def test_hysteresis_blocks_oscillation() -> None:
    # Alternating harder/easier conditions never reach the 2-window streak.
    c = _ctrl(
        n_train=2, n_min=1, cooldown=0, hysteresis_windows=2, beta_s=1.0, beta_u=1.0, u_lo=0.0
    )
    c.ell[:] = 1
    rng_levels = [1, 1]
    for w in range(6):
        solved = w % 2 == 0
        # ctx0 flips solved/failing each window; ctx1 stays uncertain.
        _drive(c, [0, 1], rng_levels, [solved, True], [0.1, 5.0], n_windows=1, start_step=w)
    assert c.ell[0] == 1  # never moved — streak kept resetting


def test_cooldown_blocks_immediate_second_move() -> None:
    c = _ctrl(
        n_train=3,
        n_min=1,
        cooldown=3,
        hysteresis_windows=1,
        beta_s=1.0,
        beta_u=1.0,
        u_lo=0.0,
        tau_hi=0.8,
    )
    # First move: ctx0 solved+confident at ℓ=2 → ℓ=1 (hysteresis_windows=1, immediate).
    _drive(c, [0, 1, 2], [2, 2, 2], [True, False, False], [0.1, 5.0, 5.0], n_windows=1)
    assert c.ell[0] == 1
    assert c.cooldown_left[0] == 3
    # Keep feeding harder evidence at the new frontier; cooldown must block.
    _drive(c, [0, 1, 2], [1, 2, 2], [True, False, False], [0.1, 5.0, 5.0], n_windows=2)
    assert c.ell[0] == 1  # still cooling down


def test_per_context_uncertainty_tiebreak_advances_on_success_alone() -> None:
    # per_context_uncertainty_tiebreak drops the confident gate: a SOLVED cell with HIGH uncertainty (which the
    # value cliff produces on sparse rewards) must still advance — unlike pcrc, which
    # blocks it. ctx0 solved + highest uncertainty among peers.
    c = _ctrl(
        n_train=2,
        mode=Method.PER_CONTEXT_UNCERTAINTY_TIEBREAK,
        n_min=2,
        cooldown=0,
        hysteresis_windows=2,
        beta_s=1.0,
        beta_u=1.0,
        u_lo=0.0,
    )
    _drive(c, [0, 1], [2, 2], [True, True], [9.0, 0.1], n_windows=3)
    assert c.ell[0] == 1  # advanced despite being the high-uncertainty cell
    # Contrast: plain pcrc blocks the same high-uncertainty solved cell.
    c2 = _ctrl(
        n_train=2,
        mode=Method.PER_CONTEXT_UNCERTAINTY_GATED,
        n_min=2,
        cooldown=0,
        hysteresis_windows=2,
        beta_s=1.0,
        beta_u=1.0,
        u_lo=0.0,
    )
    _drive(c2, [0, 1], [2, 2], [True, True], [9.0, 0.1], n_windows=3)
    assert c2.ell[0] == 2  # held (not confident)


def test_per_context_uncertainty_tiebreak_uncertainty_is_bounded_tiebreaker() -> None:
    # In per_context_uncertainty_tiebreak the uncertainty contribution is ±unc_tiebreak_weight, far below the
    # headroom term's range — so it cannot dominate selection the way pcrc's does.
    c = _ctrl(
        n_train=4,
        mode=Method.PER_CONTEXT_UNCERTAINTY_TIEBREAK,
        n_min=1,
        unc_tiebreak_weight=0.1,
        clip=3.0,
        lambda_h=1.0,
        lambda_p=0.0,
        lambda_stale=0.0,
    )
    c.counts[:, c.L] = 10
    c.s[:, c.L] = 0.5  # headroom h = 0.25 for all
    c.u[:, c.L] = np.array([0.1, 0.1, 0.1, 9.0])  # one outlier
    score = c._score()
    # uncertainty term magnitude is at most unc_tiebreak_weight (0.1) << headroom (0.25)
    assert float(np.ptp(score)) <= 0.1 + 1e-9


def test_dwell0_matches_per_context_success_advance() -> None:
    # dwell with post_solution_dwell=0 advances a solved cell exactly like per_context_success.
    c = _ctrl(
        n_train=2,
        mode=Method.PER_CONTEXT_SUCCESS_DWELL,
        post_solution_dwell=0,
        n_min=2,
        cooldown=0,
        hysteresis_windows=1,
        beta_s=1.0,
        beta_u=1.0,
    )
    _drive(c, [0, 1], [2, 2], [True, True], [0.1, 0.1], n_windows=2)
    assert c.ell[0] == 1 and c.ell[1] == 1  # both advanced immediately on solve


def test_dwell_delays_advance_by_post_solution_dwell() -> None:
    # A solved cell must stay solved for `post_solution_dwell` windows before advancing.
    D = 5
    c = _ctrl(
        n_train=2,
        mode=Method.PER_CONTEXT_SUCCESS_DWELL,
        post_solution_dwell=D,
        n_min=2,
        cooldown=0,
        hysteresis_windows=1,
        beta_s=1.0,
        beta_u=1.0,
        tau_hi=0.8,
    )
    # solved every window; should hold for D windows, then advance.
    _drive(c, [0, 1], [2, 2], [True, True], [0.1, 0.1], n_windows=D - 1)
    assert c.ell[0] == 2  # still held (dwell not yet satisfied)
    _drive(c, [0, 1], [2, 2], [True, True], [0.1, 0.1], n_windows=2)
    assert c.ell[0] == 1  # advanced after the dwell


def test_invalid_post_solution_dwell_raises() -> None:
    with pytest.raises(ValueError):
        ScaffoldConfig(post_solution_dwell=-1)


def test_fixed_mode_trains_at_one_level_only() -> None:
    c = _ctrl(n_train=4, mode=Method.FIXED, fixed_level=1)
    rng = np.random.default_rng(0)
    _, levels, fracs = c.select(50, step=0, rng=rng)
    assert set(levels) == {1}  # always the fixed level
    assert all(f == FRACS3[1] for f in fracs)
    # never advances
    _drive(c, [0, 1], [1, 1], [True, True], [0.1, 0.1], n_windows=5)
    assert (c.ell == c.L).all()  # frontier untouched (baseline mode)


def test_fixed_level_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        ScaffoldController(3, 3, FRACS3, ScaffoldConfig(mode=Method.FIXED, fixed_level=9))


def test_baseline_modes_never_move_frontier() -> None:
    for mode in (Method.GLOBAL_ANNEAL, Method.RANDOM_SCAFFOLD):
        c = _ctrl(n_train=3, mode=mode, n_min=1, beta_s=1.0, beta_u=1.0)
        before = c.ell.copy()
        _drive(c, [0, 1, 2], [2, 2, 2], [True, True, True], [0.1, 0.1, 0.1], n_windows=5)
        assert (c.ell == before).all()  # frontier controller is off for baselines


# ----------------------------------------------------------- mixture (Gate B)


def test_mixture_draws_from_categorical() -> None:
    # probs over levels {0,2}: only those levels appear, ~50/50, renormalized from raw.
    c = _ctrl(n_train=4, mode=Method.MIXTURE, mixture_probs=[0.5, 0.0, 0.5])
    rng = np.random.default_rng(0)
    _, levels, _ = c.select(4000, step=0, rng=rng)
    levels = np.asarray(levels)
    assert set(np.unique(levels).tolist()) == {0, 2}
    assert 0.4 < (levels == 0).mean() < 0.6  # ~half at each


def test_mixture_renormalizes_unnormalized_probs() -> None:
    c = _ctrl(n_train=2, mode=Method.MIXTURE, mixture_probs=[1.0, 0.0, 3.0])  # sum 4 → 0.25/0.75
    rng = np.random.default_rng(1)
    _, levels, _ = c.select(4000, step=0, rng=rng)
    assert 0.7 < (np.asarray(levels) == 2).mean() < 0.8


def test_mixture_requires_probs() -> None:
    with pytest.raises(ValueError):
        ScaffoldConfig(mode=Method.MIXTURE)  # no mixture_probs


def test_mixture_probs_wrong_length_raises() -> None:
    with pytest.raises(ValueError):
        ScaffoldController(3, 3, FRACS3, ScaffoldConfig(mode=Method.MIXTURE, mixture_probs=[0.5, 0.5]))


def test_mixture_never_moves_frontier() -> None:
    c = _ctrl(n_train=3, mode=Method.MIXTURE, mixture_probs=[0.0, 0.0, 1.0], n_min=1, beta_s=1.0)
    before = c.ell.copy()
    _drive(c, [0, 1, 2], [2, 2, 2], [True, True, True], [0.1, 0.1, 0.1], n_windows=5)
    assert (c.ell == before).all()


# ------------------------------------------------------------ oracle (Gate B)


def test_oracle_uses_per_context_levels() -> None:
    # Each context is pinned to its mapped level (on-table→0, in-air→2 analogue).
    oracle = [0, 2, 0, 2]
    c = ScaffoldController(4, 3, FRACS3, ScaffoldConfig(mode=Method.ORACLE), oracle_levels=oracle)
    rng = np.random.default_rng(0)
    ctxs, levels, _ = c.select(4000, step=0, rng=rng)
    for x, lvl in zip(ctxs, levels, strict=True):
        assert lvl == oracle[x]  # level is exactly the context's mapped level


def test_oracle_requires_levels() -> None:
    with pytest.raises(ValueError):
        ScaffoldController(3, 3, FRACS3, ScaffoldConfig(mode=Method.ORACLE))  # no oracle_levels


def test_oracle_levels_wrong_length_raises() -> None:
    with pytest.raises(ValueError):
        ScaffoldController(3, 3, FRACS3, ScaffoldConfig(mode=Method.ORACLE), oracle_levels=[0, 2])


def test_oracle_level_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        ScaffoldController(2, 3, FRACS3, ScaffoldConfig(mode=Method.ORACLE), oracle_levels=[0, 9])


def test_oracle_levels_without_oracle_mode_raises() -> None:
    with pytest.raises(ValueError):
        ScaffoldController(2, 3, FRACS3, ScaffoldConfig(mode=Method.FIXED), oracle_levels=[0, 1])


def test_oracle_never_moves_frontier() -> None:
    c = ScaffoldController(
        3, 3, FRACS3, ScaffoldConfig(mode=Method.ORACLE, n_min=1, beta_s=1.0), oracle_levels=[0, 1, 2]
    )
    before = c.ell.copy()
    _drive(c, [0, 1, 2], [0, 1, 2], [True, True, True], [0.1, 0.1, 0.1], n_windows=5)
    assert (c.ell == before).all()


# ------------------------------------------------------------------- bookkeeping


def test_update_length_mismatch_raises() -> None:
    c = _ctrl()
    with pytest.raises(ValueError):
        c.update([0, 1], [2], [True, True], [0.1, 0.2], step=0)


def test_staleness_ages_unsampled_cells() -> None:
    c = _ctrl(n_train=3, beta_s=1.0, beta_u=1.0)
    c.update([0], [2], [True], [0.1], step=0)
    assert c.stale[0, 2] == 0  # just sampled
    assert c.stale[1, 2] == 1 and c.stale[2, 2] == 1  # aged
    assert c.stale[0, 0] == 1  # non-frontier cell of ctx0 also ages


def test_snapshot_is_json_serializable() -> None:
    import json

    c = _ctrl()
    rng = np.random.default_rng(0)
    c.select(6, step=0, rng=rng)
    c.update([0, 1, 2], [2, 2, 2], [True, False, True], [0.1, 0.2, 0.3], step=0)
    snap = c.snapshot()
    json.dumps(snap)  # must not raise
    assert snap["mode"] == Method.PER_CONTEXT_UNCERTAINTY_GATED
    assert len(snap["ell"]) == c.n_train
    assert len(snap["selected_level_counts"]) == c.n_levels


# ----------------------------------------------------------------- closed loop


def test_closed_loop_frontier_migrates_toward_true_start() -> None:
    """Full select→roll→update cycle under a monotone-mastery signal.

    Mirrors how the SAC+HER loop drives the controller (plan §3.4): each context
    masters levels from easiest (ℓ=L) toward the true start (ℓ=0); solved cells
    report low uncertainty (confident), learning cells high. PCRC should walk the
    frontier toward ℓ=0 without exploding — a wiring/dynamics smoke the sim box
    can't run cheaply.
    """
    rng = np.random.default_rng(0)
    n, n_levels = 6, 5
    c = ScaffoldController(
        n,
        n_levels,
        [0.0, 0.25, 0.5, 0.75, 0.9],
        ScaffoldConfig(
            mode=Method.PER_CONTEXT_UNCERTAINTY_GATED,
            n_min=2,
            cooldown=1,
            hysteresis_windows=1,
            beta_s=0.6,
            beta_u=0.6,
            u_lo=1.0,
            tau_hi=0.6,
            tau_lo=0.15,
        ),
    )
    floor = [c.L] * n  # hardest level each context can currently solve
    learn = [0] * n  # practice counter at the learning frontier (ℓ = floor-1)
    k_master = [2, 3, 2, 4, 3, 2]  # staggered practice-to-master (de-synchronises cells)
    start_mean = float(c.ell.mean())

    for it in range(400):
        ctxs, lvls, fracs = c.select(4, step=it, rng=rng)
        assert all(0.0 <= f <= 1.0 for f in fracs)
        succ, tds = [], []
        for x, lvl in zip(ctxs, lvls, strict=True):
            if lvl >= floor[x]:  # at/easier than mastery → solved comfortably
                succ.append(True)
                tds.append(0.1)
            elif lvl == floor[x] - 1:  # the learning frontier
                learn[x] += 1
                mastered = learn[x] >= k_master[x] and floor[x] > 0
                if mastered:
                    floor[x] = lvl
                    learn[x] = 0
                succ.append(mastered)
                tds.append(0.1 if mastered else 1.5)
            else:  # too hard to learn yet
                succ.append(False)
                tds.append(1.5)
        c.update(ctxs, lvls, succ, tds, step=it)
        snap = c.snapshot()  # must stay finite/serializable throughout
        assert not np.isnan(snap["frontier_mean"])

    end_mean = float(c.ell.mean())
    assert end_mean < start_mean - 1.0  # frontier migrated toward the true start
    assert c.ell.min() == 0  # at least one context reached ℓ=0 (true start)


# --------------------------------------------------------- Gate E replay controls

FRACS5 = [0.0, 0.25, 0.5, 0.75, 0.9]
# [T=3, n_train=6] recorded per-context level trajectory (step 0 row has H_0={0:2,2:2,4:2}).
TRAJ = np.array([[0, 0, 4, 4, 2, 2], [1, 1, 3, 3, 2, 2], [0, 1, 2, 3, 4, 0]], dtype=np.int64)


def _replay_ctrl(mode, traj=TRAJ, **kw):
    return ScaffoldController(6, 5, FRACS5, ScaffoldConfig(mode=mode, **kw), replay_trajectory=traj)


def test_replay_openloop_matches_recording() -> None:
    # Each rolled context replays ITS OWN recorded level at step t (open-loop playback).
    c = _replay_ctrl(Method.REPLAY_OPENLOOP)
    rng = np.random.default_rng(0)
    for step in (0, 1, 2):
        ctxs, levels, _ = c.select(50, step=step, rng=rng)
        assert all(levels[i] == TRAJ[step][ctxs[i]] for i in range(len(ctxs)))


def test_replay_permuted_uses_permuted_context() -> None:
    # Context x replays perm(x)'s trajectory: right level multiset, wrong context.
    c = _replay_ctrl(Method.REPLAY_PERMUTED, replay_perm_seed=3)
    perm = c._replay_perm
    assert not np.all(perm == np.arange(6))  # not the identity
    rng = np.random.default_rng(0)
    ctxs, levels, _ = c.select(50, step=0, rng=rng)
    assert all(levels[i] == TRAJ[0][perm[ctxs[i]]] for i in range(len(ctxs)))


def test_replay_aggregate_reproduces_histogram() -> None:
    # Levels sampled from the recorded row's marginal H_t (NOT matched to contexts).
    c = _replay_ctrl(Method.REPLAY_AGGREGATE)
    rng = np.random.default_rng(0)
    _, levels, _ = c.select(6000, step=0, rng=rng)
    got = np.bincount(levels, minlength=5) / len(levels)
    want = np.bincount(TRAJ[0], minlength=5) / 6
    assert np.allclose(got, want, atol=0.03)


def test_replay_clamps_step_past_recording() -> None:
    # step beyond T uses the last recorded row (no crash).
    c = _replay_ctrl(Method.REPLAY_OPENLOOP)
    ctxs, levels, _ = c.select(20, step=999, rng=np.random.default_rng(0))
    assert all(levels[i] == TRAJ[-1][ctxs[i]] for i in range(len(ctxs)))


def test_replay_requires_trajectory() -> None:
    with pytest.raises(ValueError, match="requires replay_trajectory"):
        ScaffoldController(6, 5, FRACS5, ScaffoldConfig(mode=Method.REPLAY_OPENLOOP))


def test_replay_trajectory_rejected_for_non_replay_mode() -> None:
    with pytest.raises(ValueError, match="not a replay"):
        ScaffoldController(
            6, 5, FRACS5, ScaffoldConfig(mode=Method.PER_CONTEXT_SUCCESS), replay_trajectory=TRAJ
        )


def test_replay_trajectory_shape_validated() -> None:
    with pytest.raises(ValueError, match="n_train"):
        ScaffoldController(
            6,
            5,
            FRACS5,
            ScaffoldConfig(mode=Method.REPLAY_OPENLOOP),
            replay_trajectory=np.zeros((3, 4), dtype=np.int64),
        )


# --------------------------------------------------------- per_context_learning_progress (ALP frontier)


def test_per_context_learning_progress_prefers_high_learning_progress() -> None:
    # ALP = |s_fast - s_slow| at the frontier cell: the context whose success is CHANGING gets the
    # highest sampling score, even if another context has higher absolute success (which per_context_success's
    # headroom term would favor). Locks the learning-progress signal in.
    c = _ctrl(n_train=3, mode=Method.PER_CONTEXT_LEARNING_PROGRESS, n_min=1, lambda_p=0.0, lambda_stale=0.0, lambda_alp=1.0)
    L = c.L
    # ctx0: high but stable (s=s_slow=0.9 -> ALP 0); ctx1: mid, moving (s=0.6, s_slow=0.1 -> ALP 0.5);
    # ctx2: stable low (s=s_slow=0.1 -> ALP 0).
    c.s[:, L] = np.array([0.9, 0.6, 0.1])
    c.s_slow[:, L] = np.array([0.9, 0.1, 0.1])
    c.counts[:, L] = 10
    dist = c._frontier_distribution()
    assert int(np.argmax(dist.final)) == 1  # the high-ALP (actively-learning) context wins


def test_global_mixture_shifts_to_true_start_with_success():
    """Priority-4 baseline: a single global, time-varying level distribution q_t (no per-context
    frontiers). q_t starts easy-heavy and shifts mass to the true start as aggregate success rises,
    while keeping the easy-level floor and never advancing any per-context frontier."""
    import numpy as np

    from pcrc.samplers.scaffold_controller import ScaffoldConfig, ScaffoldController

    cfg = ScaffoldConfig(mode=Method.GLOBAL_MIXTURE)
    c = ScaffoldController(
        n_train=12, n_levels=5, level_fractions=[0.0, 0.25, 0.5, 0.75, 0.9], cfg=cfg
    )
    rng = np.random.default_rng(0)
    q0 = c._global_mixture_dist()
    assert abs(q0.sum() - 1.0) < 1e-9
    assert q0[0] == cfg.gmix_q0_lo and q0[-1] == cfg.gmix_easy_floor  # easy-heavy start
    for _ in range(300):
        ctx, lvl, _ = c.select(8, 0, rng)
        c.update(ctx, lvl, [True] * 8, [0.0] * 8, 0)
    q1 = c._global_mixture_dist()
    assert q1[0] > q0[0] + 0.4  # true-start mass grew with success
    assert q1[-1] == cfg.gmix_easy_floor  # easy floor retained
    # the per-context frontier is never used by this baseline (stays at init)
    assert (c.ell == c.L).all()
    ctx, lvl, _ = c.select(500, 0, rng)
    assert set(lvl).issubset(set(range(5)))


# ------------------------------------------------- per_context_wilson_confidence / per_context_beta_posterior (RLC upgrade)


@pytest.mark.parametrize(
    "bad",
    [
        {"tp_gamma": 0.0},
        {"tp_gamma": 1.5},
        {"tp_z": 0.0},
        {"bpcrc_gamma": -0.1},
        {"bpcrc_delta_hi": 0.0},
        {"bpcrc_delta_lo": 1.0},
        {"bpcrc_prior": 0.0},
    ],
)
def test_invalid_evidence_config_raises(bad: dict) -> None:
    with pytest.raises(ValueError):
        ScaffoldConfig(**bad)


def test_per_context_wilson_confidence_advance_waits_for_confident_evidence() -> None:
    # All-success evidence: EMA thresholding (beta_s=1) would advance after the hysteresis
    # streak, but the Wilson gate holds until n_eff makes lower-CI(p) > tau_hi.
    # 2 rollouts/window, tp_gamma=1, z=1: lower-CI = 0.667 (n=2), 0.800 (n=4), 0.857 (n=6).
    c = _ctrl(
        n_train=2,
        mode=Method.PER_CONTEXT_WILSON_CONFIDENCE,
        n_min=2,
        cooldown=0,
        beta_s=1.0,
        tau_hi=0.8,
        tau_lo=0.15,
        tp_gamma=1.0,
        tp_z=1.0,
        stale_advance_windows=0,
    )
    _drive(c, [0, 0], [2, 2], [True, True], [0.0, 0.0], n_windows=2)
    assert c.ell[0] == 2  # n_eff=4 → lower-CI = 0.8, not > 0.8 → still held
    _drive(c, [0, 0], [2, 2], [True, True], [0.0, 0.0], n_windows=1, start_step=2)
    assert c.ell[0] == 1  # n_eff=6 → lower-CI ≈ 0.857 > 0.8 → advance


def test_per_context_wilson_confidence_retreat_waits_for_confident_failure() -> None:
    # All-failure evidence: upper-CI = 0.333 (n=2), 0.200 (n=4), 0.143 (n=6) with z=1.
    c = _ctrl(
        n_train=2,
        mode=Method.PER_CONTEXT_WILSON_CONFIDENCE,
        n_min=2,
        cooldown=0,
        beta_s=1.0,
        tau_hi=0.8,
        tau_lo=0.15,
        tp_gamma=1.0,
        tp_z=1.0,
        stale_advance_windows=0,
    )
    c.ell[:] = 1
    _drive(c, [0, 0], [1, 1], [False, False], [0.0, 0.0], n_windows=2)
    assert c.ell[0] == 1  # upper-CI 0.2 > 0.15 → hold (an EMA at 0 would already retreat)
    _drive(c, [0, 0], [1, 1], [False, False], [0.0, 0.0], n_windows=1, start_step=2)
    assert c.ell[0] == 2  # upper-CI ≈ 0.143 < 0.15 → add assistance


def test_per_context_wilson_confidence_ignores_critic_uncertainty() -> None:
    # Success-only: huge TD errors must not block the advance (no confident gate).
    c = _ctrl(
        n_train=2,
        mode=Method.PER_CONTEXT_WILSON_CONFIDENCE,
        n_min=2,
        cooldown=0,
        beta_s=1.0,
        beta_u=1.0,
        u_lo=0.0,
        tp_gamma=1.0,
        tp_z=1.0,
    )
    _drive(c, [0, 0], [2, 2], [True, True], [100.0, 100.0], n_windows=3)
    assert c.ell[0] == 1


def test_per_context_wilson_confidence_score_prefers_learning_progress() -> None:
    c = _ctrl(
        n_train=2,
        mode=Method.PER_CONTEXT_WILSON_CONFIDENCE,
        lambda_h=0.0,
        lambda_p=0.0,
        lambda_stale=0.0,
        lambda_lp=1.0,
    )
    # Same evidence counts; ctx1 has a large fast-vs-slow EMA gap (recent learning progress).
    c.succ_d[:, c.L] = 1.0
    c.fail_d[:, c.L] = 1.0
    c.s[:, c.L] = np.array([0.5, 0.5])
    c.s_slow[:, c.L] = np.array([0.5, 0.1])
    score = c._score()
    assert score[1] > score[0]


def test_per_context_wilson_confidence_discount_forgets_old_evidence() -> None:
    c = _ctrl(n_train=1, mode=Method.PER_CONTEXT_WILSON_CONFIDENCE, tp_gamma=0.5, n_min=100)  # n_min blocks moves
    _drive(c, [0], [2], [True], [0.0], n_windows=1)
    assert c.succ_d[0, 2] == pytest.approx(1.0)
    _drive(c, [0], [2], [True], [0.0], n_windows=1, start_step=1)
    assert c.succ_d[0, 2] == pytest.approx(1.5)  # 0.5*1 + 1
    _drive(c, [0], [2], [False], [0.0], n_windows=1, start_step=2)
    assert c.succ_d[0, 2] == pytest.approx(0.75)  # decayed; failure adds to fail_d
    assert c.fail_d[0, 2] == pytest.approx(1.0)


def test_bpcrc_advances_on_posterior_mass() -> None:
    # Beta(1+k, 1) with all-success: P(p > 0.8) = 1 - 0.8^(1+k) > 0.9 needs k >= 10.
    c = _ctrl(
        n_train=2,
        mode=Method.PER_CONTEXT_BETA_POSTERIOR,
        n_min=2,
        cooldown=0,
        beta_s=1.0,
        tau_hi=0.8,
        tau_lo=0.15,
        bpcrc_gamma=1.0,
        bpcrc_delta_hi=0.1,
        bpcrc_prior=1.0,
        stale_advance_windows=0,
    )
    _drive(c, [0, 0], [2, 2], [True, True], [0.0, 0.0], n_windows=4)
    assert c.ell[0] == 2  # k=8 → P = 1 - 0.8^9 ≈ 0.866 < 0.9 → hold
    _drive(c, [0, 0], [2, 2], [True, True], [0.0, 0.0], n_windows=1, start_step=4)
    assert c.ell[0] == 1  # k=10 → P = 1 - 0.8^11 ≈ 0.914 > 0.9 → advance


def test_bpcrc_retreats_on_posterior_failure_mass() -> None:
    # Beta(1, 1+k) all-failure: P(p < 0.15) = 1 - 0.85^(1+k) > 0.9 needs k >= 14.
    c = _ctrl(
        n_train=2,
        mode=Method.PER_CONTEXT_BETA_POSTERIOR,
        n_min=2,
        cooldown=0,
        beta_s=1.0,
        tau_hi=0.8,
        tau_lo=0.15,
        bpcrc_gamma=1.0,
        bpcrc_delta_lo=0.1,
        bpcrc_prior=1.0,
        stale_advance_windows=0,
    )
    c.ell[:] = 1
    _drive(c, [0, 0], [1, 1], [False, False], [0.0, 0.0], n_windows=6)
    assert c.ell[0] == 1  # k=12 → P ≈ 0.879 < 0.9 → hold
    _drive(c, [0, 0], [1, 1], [False, False], [0.0, 0.0], n_windows=2, start_step=6)
    assert c.ell[0] == 2  # k=16 → P ≈ 0.937 > 0.9 → retreat


def test_bpcrc_score_has_posterior_sd_bonus() -> None:
    c = _ctrl(
        n_train=2,
        mode=Method.PER_CONTEXT_BETA_POSTERIOR,
        lambda_h=0.0,
        lambda_p=0.0,
        lambda_stale=0.0,
        w_unc_post=1.0,
        bpcrc_prior=1.0,
    )
    # Same posterior mean 0.5; ctx0 has 40 observations, ctx1 has 2 → ctx1 higher sd → higher score.
    c.succ_d[0, c.L] = 20.0
    c.fail_d[0, c.L] = 20.0
    c.succ_d[1, c.L] = 1.0
    c.fail_d[1, c.L] = 1.0
    score = c._score()
    assert score[1] > score[0]


def test_evidence_modes_antistall_still_fires() -> None:
    # Dead band under the CI gate (mixed successes, n_eff moderate): antistall must still
    # force one harder step after stale_advance_windows, exactly as for per_context_success.
    c = _ctrl(
        n_train=2,
        n_levels=3,
        mode=Method.PER_CONTEXT_WILSON_CONFIDENCE,
        n_min=2,
        cooldown=0,
        beta_s=1.0,
        tau_hi=0.8,
        tau_lo=0.15,
        tp_gamma=1.0,
        tp_z=1.0,
        stale_advance_windows=3,
    )
    _drive(c, [0, 0], [2, 2], [True, False], [0.0, 0.0], n_windows=4)
    assert c.ell[0] == 1  # p̂=0.5 in dead band → stuck 3 windows → antistall descent


def test_evidence_snapshot_serializes() -> None:
    import json

    for mode in (Method.PER_CONTEXT_WILSON_CONFIDENCE, Method.PER_CONTEXT_BETA_POSTERIOR):
        c = _ctrl(n_train=2, mode=mode, n_min=1, cooldown=0)
        _drive(c, [0, 1], [2, 2], [True, False], [0.0, 0.0], n_windows=2)
        c.select(4, step=2, rng=np.random.default_rng(0))
        json.dumps(c.snapshot())  # must not raise


# ---------------------------------------------------------- per_context_success_trend (drift-aware)


@pytest.mark.parametrize(
    "bad", [{"trend_q_level": 0.0}, {"trend_p0_level": -1.0}, {"trend_horizon": -0.5}]
)
def test_invalid_trend_config_raises(bad: dict) -> None:
    with pytest.raises(ValueError):
        ScaffoldConfig(**bad)


def _trend_ctrl(**kw):
    kw.setdefault("mode", Method.PER_CONTEXT_SUCCESS_TREND)
    kw.setdefault("n_min", 2)
    kw.setdefault("cooldown", 0)
    kw.setdefault("beta_s", 1.0)
    kw.setdefault("tau_hi", 0.8)
    kw.setdefault("tau_lo", 0.15)
    kw.setdefault("stale_advance_windows", 0)
    return _ctrl(n_train=2, **kw)


def test_trend_advances_on_sustained_success() -> None:
    c = _trend_ctrl()
    _drive(c, [0, 0], [2, 2], [True, True], [0.0, 0.0], n_windows=12)
    assert c.ell[0] == 1  # filter tracks up through logit(0.8) and predictive gate fires


def test_trend_retreats_on_sustained_failure() -> None:
    c = _trend_ctrl()
    c.ell[:] = 1
    _drive(c, [0, 0], [1, 1], [False, False], [0.0, 0.0], n_windows=12)
    assert c.ell[0] == 2


def test_trend_holds_in_dead_band_then_antistall() -> None:
    c = _trend_ctrl(stale_advance_windows=4)
    _drive(c, [0, 0], [2, 2], [True, False], [0.0, 0.0], n_windows=3)
    assert c.ell[0] == 2  # p̂≈0.5: neither gate fires
    _drive(c, [0, 0], [2, 2], [True, False], [0.0, 0.0], n_windows=4, start_step=3)
    assert c.ell[0] == 1  # antistall probe still works


def test_trend_anticipates_rise_faster_than_fixed_posterior() -> None:
    # A steadily RISING success sequence: the velocity term should let the predictive gate
    # advance no later than (and typically before) the fixed-discount Beta posterior.
    seq = [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def first_advance(mode):
        kw = dict(mode=mode, n_min=2, cooldown=0, beta_s=1.0, stale_advance_windows=0)
        if mode == Method.PER_CONTEXT_BETA_POSTERIOR:
            kw["bpcrc_gamma"] = 0.95
        c = _ctrl(n_train=2, **kw)
        for w, r in enumerate(seq):
            c.update([0, 0], [2, 2], [bool(r), bool(r)], [0.0, 0.0], step=w)
            if c.ell[0] == 1:
                return w
        return len(seq)

    assert first_advance(Method.PER_CONTEXT_SUCCESS_TREND) <= first_advance(Method.PER_CONTEXT_BETA_POSTERIOR)


def test_trend_velocity_positive_under_improvement() -> None:
    c = _trend_ctrl(n_min=100)  # block transitions; observe the filter only
    for w, r in enumerate([0, 0, 0, 1, 1, 1, 1, 1]):
        c.update([0], [2], [bool(r)], [0.0], step=w)
    assert c.kf_v[0, 2] > 0.0


def test_trend_snapshot_serializes() -> None:
    import json

    c = _trend_ctrl()
    _drive(c, [0, 1], [2, 2], [True, False], [0.0, 0.0], n_windows=3)
    c.select(4, step=3, rng=np.random.default_rng(0))
    json.dumps(c.snapshot())
