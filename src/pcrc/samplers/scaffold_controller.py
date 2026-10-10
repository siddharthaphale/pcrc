"""PCRC scaffold-allocation controller over ``(context, scaffold-level)`` cells.

The curriculum object is no longer a fixed bank of contexts ``x`` but a grid of
**context-scaffold cells** ``(x, ℓ)``, where ``ℓ=0`` is the true (unassisted)
start and ``ℓ=L`` is the easiest scaffold. The controller (a) tracks per-cell
success and value-uncertainty EMAs, (b) samples which cell to practice next, and
(c) moves a per-context *frontier* ``ell[x]`` harder/easier as competence and
uncertainty evolve. Evaluation is always at ``ℓ=0`` (handled by the loop, not
here).

This is deliberately **not** a :class:`pcrc.samplers.base.Sampler`: that ABC is
a pure ``q`` over a fixed bank with no reset/level state, whereas this controller
owns the cell table + frontier and emits *(context, level, fraction)* triples.
The ``fraction`` is the ``start_progress`` the loop feeds to
``env.reset(context_id, start_progress=…)`` — identical interface for Fetch and
(after the geodesic-reset change) PointMaze.

Plan reference + method spec: ``docs/EXPERIMENTS.md``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from pcrc.registry import DEFAULT_METHOD, SCAFFOLD_METHODS, Method
from pcrc.samplers.exploration_floor import apply_floor_and_cap

# Modes that use the PCRC score + frontier controller (vs. schedule/random
# baselines that bypass both). ``per_context_success`` / ``per_context_uncertainty_priority`` are the
# required ablations (framing doc §16.1 / §16.2).
# per_context_uncertainty_tiebreak = the §6-fix-#1 variant: uncertainty is a bounded sampling TIE-BREAKER
# (not a 12x score driver) and the frontier advances on SUCCESS ALONE (no confident
# gate, which the sparse-reward value cliff breaks — solved cells have high TD there).
# dwell = transfer-aware-pacing screen (next-steps doc §5): success-only frontier
# (= per_context_success) + a deterministic post-solution dwell of `post_solution_dwell`
# windows before advancing. The only manipulated variable vs per_context_success is dwell.
# per_context_wilson_confidence = TP-PCRC (RLC main-track upgrade, docs/rlc_maintrack_updgrade_experiments.md):
# transitions gated on a Wilson confidence interval over a DISCOUNTED success/failure count pair
# (advance iff lower-CI > τ_hi, retreat iff upper-CI < τ_lo), sampling score adds a learning-progress
# term |s - s_slow|. Success-only (no critic uncertainty anywhere).
# per_context_beta_posterior = B-PCRC-Beta (Bayesian ablation): discounted Beta(a,b) posterior per cell; advance iff
# P(p > τ_hi) > 1-δ_hi, retreat iff P(p < τ_lo) > 1-δ_lo; sampling adds a posterior-sd bonus.
# per_context_success_trend = B-PCRC-Trend (drift-aware Bayesian gating): per-cell Kalman filter on the LOGIT of
# cell success with a local level+velocity state (dynamic logistic state-space); transitions gate on
# the one-window-ahead PREDICTIVE distribution, so a rising competence advances by anticipation
# instead of lagging (the measured failure mode of the fixed-discount Beta posterior).
_PCRC_MODES = frozenset(
    {Method.PER_CONTEXT_UNCERTAINTY_GATED, Method.PER_CONTEXT_SUCCESS, Method.PER_CONTEXT_UNCERTAINTY_PRIORITY, Method.PER_CONTEXT_UNCERTAINTY_TIEBREAK, Method.PER_CONTEXT_SUCCESS_DWELL, Method.PER_CONTEXT_LEARNING_PROGRESS,
     Method.PER_CONTEXT_WILSON_CONFIDENCE, Method.PER_CONTEXT_BETA_POSTERIOR, Method.PER_CONTEXT_SUCCESS_TREND}
)
# fixed = train at one fixed scaffold level (no curriculum); the ℓ=0 held-out curve
# measures cross-level transfer from that level (Experiment 2 / Gate C).
# mixture = global stochastic mixture over levels (per-rollout categorical draw from
# `mixture_probs`); the autonomous-plan Gate B threat baseline — if a static level
# mixture matches online per-context control, the method claim dies. No curriculum.
# oracle = per-context FIXED level from a precomputed group→level map (`oracle_levels`,
# passed in by the driver from the bank); the deployability upper bound (not fair).
# replay_* = causal controls (autonomous-plan Gate E). They REPLAY a recorded per-context level
# trajectory (the online pcrc run's `ell[t]`, since selected level == ell[x] exactly) to isolate
# what about per-context allocation matters:
#   replay_aggregate — keep only the per-step aggregate level histogram H_t (levels sampled from the
#                      recorded row, NOT matched to contexts) → tests "aggregate schedule" alone.
#   replay_permuted  — recorded per-context trajectories permuted across contexts (right level multiset,
#                      wrong context) → tests "context matching".
#   replay_openloop  — each context replays ITS OWN trajectory from ANOTHER seed (no response to the
#                      current policy) → tests "online adaptation" vs a fixed per-context schedule.
_REPLAY_MODES = frozenset({Method.REPLAY_AGGREGATE, Method.REPLAY_PERMUTED, Method.REPLAY_OPENLOOP})
_BASELINE_MODES = (
    frozenset({Method.GLOBAL_ANNEAL, Method.RANDOM_SCAFFOLD, Method.FIXED, Method.MIXTURE, Method.ORACLE, Method.GLOBAL_MIXTURE})
    | _REPLAY_MODES
)
# rfcl_style = RFCL-style local trajectory curriculum (Test Campaign 4 Stage B): per-context frontier
# advance/retreat like PCRC, but UNIFORM context sampling (no headroom/recency/progress score) and NO
# anti-stall (stale_advance_windows=0). Tests whether simple per-trajectory reverse progression — without
# PCRC's scoring and anti-stall — explains the gains.
_RFCL_MODES = frozenset({Method.RFCL_STYLE})
# per_group_success = C5.1 (campaign 5, review Critical #2): PCRC at GROUP granularity. One shared
# frontier + pooled success EMA + evidence count + cooldown per GROUP (driver passes `group_ids`
# from oracle group labels); success pooled across the group's contexts; sampling = PCRC score
# over group frontier cells (degenerates to uniform for 2 groups under q_max=0.5), then uniform
# over the group's contexts. Transition rule, thresholds, hysteresis, and antistall are the frozen
# PCRC values — granularity is the only manipulated variable vs per_context_success.
# plr_cells = C5.2a (campaign 5, review Critical #3): PLR-style prioritized sampling over ALL
# (x, ℓ) cells with NO frontier constraint. Score = per-cell mean |TD-error| (PLR's L1 value
# loss), replace-on-first-visit then EMA — exactly samplers/plr.py's conventions (beta=0.5,
# rho=0.1, score_ema_alpha=0.3); staleness mixing over cells; PCRC's frozen floor/cap.
# alp_cells = C5.2b (campaign 5): the second reviewer-requested teacher — ALP-GMM-style
# learning progress |s_fast − s_slow| per CELL over ALL (x, ℓ) cells (no frontier), softmax with
# the frozen temperature, staleness term, and floor/cap. Distinct from per_context_learning_progress, which scores
# ALP only on the frontier.
# rfcl_faithful = C5.5 (review Critical #1): component-matched RFCL reverse curriculum —
# per-context frontier, ADVANCE-ONLY (no retreat, no antistall) on a windowed recent-success
# rule (advance iff >= rfclf_advance_successes of the last rfclf_window frontier rollouts
# succeeded), and struggling-demo-prioritized sampling (weight ∝ 1 − windowed success + floor).
# Documented deviations from the official stack: discrete quantile levels stand in for continuous
# demo timesteps; the learner stays our SAC+HER (curriculum-only comparison, no ensemble/UTD);
# the frozen exploration floor/cap replaces their unconstrained demo weights.
# aced_style = C5.5: ACED-style advance rule — the existing rfcl_style local frontier
# (uniform sampling, frozen EMA thresholds/evidence/hysteresis, no antistall) with the retreat
# branch REMOVED (ACED's threshold-triggered section advance has no backward move).
_GROUP_MODES = frozenset({Method.PER_GROUP_SUCCESS})
_CELL_TEACHER_MODES = frozenset({Method.PLR_CELLS, Method.ALP_CELLS})
_EXTERNAL_LOCAL_MODES = frozenset({Method.RFCL_FAITHFUL, Method.ACED_STYLE})

# Modes that run the per-context frontier transition (advance/retreat) on success.
# aced_style shares the machinery with its retreat branch disabled; rfcl_faithful runs its own
# windowed advance rule instead.
_FRONTIER_MODES = _PCRC_MODES | _RFCL_MODES | frozenset({Method.ACED_STYLE})
# Modes whose frontier advances on success thresholds alone (no ũ<u_lo gate).
_SUCCESS_ONLY_FRONTIER = frozenset(
    {Method.PER_CONTEXT_SUCCESS, Method.PER_CONTEXT_UNCERTAINTY_TIEBREAK, Method.PER_CONTEXT_SUCCESS_DWELL, Method.PER_CONTEXT_LEARNING_PROGRESS, Method.RFCL_STYLE, Method.ACED_STYLE, Method.PER_CONTEXT_WILSON_CONFIDENCE,
     Method.PER_CONTEXT_BETA_POSTERIOR, Method.PER_CONTEXT_SUCCESS_TREND}
)
# Modes with no uncertainty term in the sampling score.
_SUCCESS_SCORING_MODES = frozenset({Method.PER_CONTEXT_SUCCESS, Method.PER_CONTEXT_SUCCESS_DWELL, Method.PER_CONTEXT_LEARNING_PROGRESS})
# Modes whose transitions are gated on accumulated evidence models rather than the raw success EMA:
# per_context_wilson_confidence (Wilson CI), per_context_beta_posterior (Beta posterior), per_context_success_trend (logit-Kalman predictive).
_EVIDENCE_MODES = frozenset({Method.PER_CONTEXT_WILSON_CONFIDENCE, Method.PER_CONTEXT_BETA_POSTERIOR, Method.PER_CONTEXT_SUCCESS_TREND})


@dataclass
class ScaffoldConfig:
    """Hyperparameters for :class:`ScaffoldController`.

    Defaults are the framing-doc §7/§8 starting points, not tuned values.
    """

    mode: str = DEFAULT_METHOD
    # Score weights (framing doc §7).
    lambda_h: float = 1.0
    lambda_u: float = 1.0
    lambda_p: float = 0.2
    lambda_stale: float = 0.1
    tau: float = 0.5
    epsilon: float = 0.1  # uniform exploration floor over cells
    q_max: float | None = None
    # Success / uncertainty EMA rates (framing doc §6.3).
    beta_s: float = 0.2
    beta_u: float = 0.2
    # Frontier transition thresholds + hysteresis (framing doc §8).
    tau_lo: float = 0.15
    tau_hi: float = 0.8
    u_lo: float = -0.25
    u_hi: float = 0.5  # reserved (hold band); transitions use tau_lo/tau_hi + u_lo
    n_min: int = 15
    cooldown: int = 3
    hysteresis_windows: int = 2
    # Anti-stall (autonomous-plan Gate C2 rescue): a success-gated reverse curriculum can PARK a
    # context in the dead band [τ_lo, τ_hi] — moderate success at an intermediate level, never
    # advancing (s≤τ_hi) nor retreating (s≥τ_lo) — so its frontier never reaches ℓ=0 and it never
    # practices the true task (observed: hard PnP group stuck at L≈4, ℓ=0 eval=0.00 on some seeds).
    # If an eligible cell is stuck-but-not-failing (s>τ_lo) for this many windows, force ONE harder
    # step so the curriculum keeps progressing toward ℓ=0 at the context's own pace. 0 = disabled.
    stale_advance_windows: int = 0
    clip: float = 3.0  # robust-z clip (framing doc §6.4)
    # per_context_uncertainty_tiebreak tie-breaker weight: the uncertainty term becomes
    # ``unc_tiebreak_weight * (ũ / clip)`` ∈ ±unc_tiebreak_weight, so headroom (≤0.25)
    # leads and uncertainty only breaks ties within the difficulty band (framing §5.2 r4).
    unc_tiebreak_weight: float = 0.1
    # Deterministic post-solution dwell (mode "per_context_success_dwell"): a solved frontier cell must stay solved
    # (s>τ_hi) for this many update windows before it may advance. 0 ⇒ identical to per_context_success.
    post_solution_dwell: int = 0
    # Fixed scaffold level for mode "fixed" (transfer-landscape probe).
    fixed_level: int = 0
    # Per-rollout categorical over levels for mode "mixture" (autonomous-plan Gate B).
    # Length must equal n_levels (checked in the controller); non-negative; renormalized.
    mixture_probs: list[float] | None = None
    # Schedule length for the ``global_anneal`` baseline (levels L→0).
    anneal_iters: int = 1000
    # Seed for the fixed context permutation in mode "replay_permuted" (Gate E causal control).
    replay_perm_seed: int = 0
    # mode "per_context_learning_progress" (ALP-GMM/TSCL-style): allocate rollouts by Absolute Learning Progress
    # ALP=|s_fast - s_slow| at the frontier cell (a slow success EMA gives the baseline). Frontier
    # still advances on success (+ anti-stall). Tests a learning-progress signal vs the success score.
    beta_s_slow: float = 0.05
    lambda_alp: float = 1.0
    # mode "global_mixture" (deployable async global baseline, paper Priority 4): a single global,
    # time-varying distribution q_t over levels (NO per-context frontiers). The true-start (ell=0)
    # mass grows with aggregate success from gmix_q0_lo to gmix_q0_hi; the easiest level keeps a
    # constant floor gmix_easy_floor; remaining mass spreads over intermediate levels.
    gmix_q0_lo: float = 0.1
    gmix_q0_hi: float = 0.7
    gmix_easy_floor: float = 0.2
    gmix_beta: float = 0.05  # EMA rate of the aggregate-success signal driving q_t
    # mode "per_context_wilson_confidence" (TP-PCRC): discounted per-cell (success, failure) counts decay by tp_gamma
    # each update window the cell receives data (decay-on-update; unsampled cells keep their
    # evidence, mirroring the EMA's behavior). Wilson interval with the effective n = a+b:
    # advance iff lower-CI > tau_hi, retreat iff upper-CI < tau_lo; antistall unchanged but gated
    # on "not confidently failed" (upper-CI > tau_lo). tp_z=1.0 is a 68% interval — mild gating;
    # 1.96 makes transitions much more conservative (n_eff caps at 1/(1-tp_gamma)).
    tp_gamma: float = 0.97
    tp_z: float = 1.0
    # per_context_wilson_confidence sampling: score = λ_h·p̂(1-p̂) + λ_lp·|s - s_slow| + λ_p·progress + λ_stale·log1p(stale).
    lambda_lp: float = 1.0
    # mode "per_context_beta_posterior" (B-PCRC-Beta): discounted Beta posterior with symmetric prior pseudo-counts
    # bpcrc_prior; decay bpcrc_gamma per update window (decay-on-update, as above). Transitions:
    # advance iff P(p > tau_hi | a,b) > 1 - bpcrc_delta_hi; retreat iff P(p < tau_lo) > 1 - delta_lo.
    bpcrc_gamma: float = 0.95
    bpcrc_delta_hi: float = 0.1
    bpcrc_delta_lo: float = 0.1
    bpcrc_prior: float = 1.0
    # per_context_beta_posterior sampling: score = λ_h·E[p](1-E[p]) + w_unc_post·sqrt(Var[p]) + λ_p·progress
    # + λ_stale·log1p(stale) — posterior sd replaces the TD-uncertainty term.
    w_unc_post: float = 0.5
    # mode "per_context_success_trend": per-cell Kalman filter on logit success with state (level μ, velocity v).
    # Predict: μ←μ+v (local linear trend), P←FPFᵀ+Q with Q=diag(trend_q_level, trend_q_vel); EKF
    # update from each data window's Bernoulli mean. Transitions gate on the trend_horizon-window
    # -ahead predictive: advance iff P(σ(μ_pred)>τ_hi) > 1−bpcrc_delta_hi, retreat symmetric.
    # Filter steps run only on windows where the cell receives data (matches decay-on-update).
    trend_q_level: float = 0.02
    trend_q_vel: float = 0.002
    trend_p0_level: float = 4.0
    trend_p0_vel: float = 0.25
    trend_horizon: float = 1.0
    trend_logit_clip: float = 8.0
    # mode "plr_cells" (campaign 5 C5.2a): PLR rank/staleness knobs, defaults = samplers/plr.py's
    # established baseline values (NOT tuned on the conflicts; recorded in the C5.6 audit).
    plr_beta: float = 0.5
    plr_rho: float = 0.1
    plr_score_ema_alpha: float = 0.3
    # mode "rfcl_faithful" (campaign 5 C5.5): windowed advance rule + struggling-demo sampling.
    # Advance one level iff >= rfclf_advance_successes of the last rfclf_window frontier rollouts
    # succeeded (RFCL's recent-success criterion); sampling weight = (1 - windowed success) +
    # rfclf_w_min. Values follow the official implementation's 3-of-5 spirit; not tuned here.
    rfclf_window: int = 5
    rfclf_advance_successes: int = 3
    rfclf_w_min: float = 0.1

    def __post_init__(self) -> None:
        """Validate mode + numeric ranges."""
        if self.mode not in SCAFFOLD_METHODS:
            raise ValueError(f"mode must be one of {sorted(SCAFFOLD_METHODS)}; got {self.mode!r}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0; got {self.tau}")
        for name in ("beta_s", "beta_u"):
            v = getattr(self, name)
            if not 0.0 < v <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]; got {v}")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1]; got {self.epsilon}")
        if self.n_min < 0 or self.cooldown < 0 or self.hysteresis_windows < 1:
            raise ValueError("n_min/cooldown must be >= 0 and hysteresis_windows >= 1")
        if self.stale_advance_windows < 0:
            raise ValueError(
                f"stale_advance_windows must be >= 0; got {self.stale_advance_windows}"
            )
        if self.post_solution_dwell < 0:
            raise ValueError(f"post_solution_dwell must be >= 0; got {self.post_solution_dwell}")
        if self.fixed_level < 0:
            raise ValueError(f"fixed_level must be >= 0; got {self.fixed_level}")
        for name in ("tp_gamma", "bpcrc_gamma"):
            v = getattr(self, name)
            if not 0.0 < v <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]; got {v}")
        for name in (
            "trend_q_level", "trend_q_vel", "trend_p0_level", "trend_p0_vel", "trend_logit_clip",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0; got {getattr(self, name)}")
        if self.trend_horizon < 0:
            raise ValueError(f"trend_horizon must be >= 0; got {self.trend_horizon}")
        if self.tp_z <= 0:
            raise ValueError(f"tp_z must be > 0; got {self.tp_z}")
        for name in ("bpcrc_delta_hi", "bpcrc_delta_lo"):
            v = getattr(self, name)
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1); got {v}")
        if self.bpcrc_prior <= 0:
            raise ValueError(f"bpcrc_prior must be > 0; got {self.bpcrc_prior}")
        if self.plr_beta <= 0:
            raise ValueError(f"plr_beta must be > 0; got {self.plr_beta}")
        if not 0.0 <= self.plr_rho <= 1.0:
            raise ValueError(f"plr_rho must be in [0, 1]; got {self.plr_rho}")
        if not 0.0 < self.plr_score_ema_alpha <= 1.0:
            raise ValueError(
                f"plr_score_ema_alpha must be in (0, 1]; got {self.plr_score_ema_alpha}"
            )
        if self.rfclf_window < 1 or not 1 <= self.rfclf_advance_successes <= self.rfclf_window:
            raise ValueError(
                "need rfclf_window >= 1 and 1 <= rfclf_advance_successes <= rfclf_window; got "
                f"{self.rfclf_window}, {self.rfclf_advance_successes}"
            )
        if self.rfclf_w_min < 0:
            raise ValueError(f"rfclf_w_min must be >= 0; got {self.rfclf_w_min}")
        if self.mode == Method.MIXTURE:
            if self.mixture_probs is None:
                raise ValueError("mode 'mixture' requires mixture_probs")
            mp = np.asarray(self.mixture_probs, dtype=np.float64)
            if np.any(mp < 0) or mp.sum() <= 0:
                raise ValueError(f"mixture_probs must be non-negative with positive sum; got {mp}")


@dataclass
class _SnapshotState:
    """Last-step selection state, for ``scaffold.jsonl`` logging."""

    step: int = -1
    selected_levels: list[int] = field(default_factory=list)
    selection_entropy: float | None = None
    n_harder_moves: int = 0
    n_easier_moves: int = 0


class ScaffoldController:
    """Maintain ``(context × level)`` cells; sample cells; move the frontier.

    Args:
        n_train: number of training contexts ``|X|``.
        n_levels: number of scaffold levels ``L + 1`` (``ℓ = 0 … L``).
        level_fractions: per-level assist fraction ``α_ℓ ∈ [0, 1]`` passed to the
            env as ``start_progress``. Must be length ``n_levels``,
            non-decreasing, with ``level_fractions[0] == 0`` (true start).
        cfg: :class:`ScaffoldConfig`.
    """

    def __init__(
        self,
        n_train: int,
        n_levels: int,
        level_fractions: list[float],
        cfg: ScaffoldConfig | None = None,
        oracle_levels: list[int] | None = None,
        replay_trajectory: np.ndarray | None = None,
        group_ids: list[int] | None = None,
    ):
        """Allocate the cell table; start every context at the easiest level.

        ``oracle_levels`` (mode ``"oracle"`` only): a per-context fixed scaffold
        level, length ``n_train``, indexed by controller context (= train bank
        context_id), computed by the driver from the bank's group labels.

        ``replay_trajectory`` (modes ``"replay_*"`` only): a recorded per-step
        per-context level matrix of shape ``[T, n_train]`` (the online pcrc run's
        ``ell[t]``), loaded by the driver. ``replay_openloop`` passes ANOTHER seed's
        recording; ``replay_aggregate``/``replay_permuted`` pass the same run's.

        ``group_ids`` (mode ``"per_group_success"`` only): a per-context group label,
        length ``n_train``, values ``0..G-1`` (contiguous), computed by the driver
        from the bank's oracle group labels (goal-z for Fetch, task for Meta-World).
        """
        if n_train <= 0:
            raise ValueError(f"n_train must be > 0; got {n_train}")
        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1; got {n_levels}")
        frac = np.asarray(level_fractions, dtype=np.float64)
        if frac.shape != (n_levels,):
            raise ValueError(
                f"level_fractions must have length n_levels={n_levels}; got {frac.shape}"
            )
        if frac[0] != 0.0:
            raise ValueError(f"level_fractions[0] must be 0.0 (true start); got {frac[0]}")
        if np.any(np.diff(frac) < 0):
            raise ValueError("level_fractions must be non-decreasing")
        if np.any((frac < 0) | (frac > 1)):
            raise ValueError("level_fractions must lie in [0, 1]")

        self.n_train = int(n_train)
        self.n_levels = int(n_levels)
        self.L = int(n_levels - 1)
        self.level_fractions = frac
        self.cfg = cfg or ScaffoldConfig()
        if self.cfg.mode == Method.FIXED and self.cfg.fixed_level > self.L:
            raise ValueError(f"fixed_level {self.cfg.fixed_level} exceeds L={self.L}")

        # mode "mixture": validate + cache a normalized categorical over the n_levels.
        self._mixture_probs: np.ndarray | None = None
        if self.cfg.mode == Method.MIXTURE:
            mp = np.asarray(self.cfg.mixture_probs, dtype=np.float64)
            if mp.shape != (self.n_levels,):
                raise ValueError(
                    f"mixture_probs must have length n_levels={self.n_levels}; got {mp.shape}"
                )
            self._mixture_probs = mp / mp.sum()

        # mode "oracle": validate the per-context level map (length n_train, in [0, L]).
        self._oracle_levels: np.ndarray | None = None
        if self.cfg.mode == Method.ORACLE:
            if oracle_levels is None:
                raise ValueError("mode 'oracle' requires oracle_levels")
            ol = np.asarray(oracle_levels, dtype=np.int64)
            if ol.shape != (self.n_train,):
                raise ValueError(
                    f"oracle_levels must have length n_train={self.n_train}; got {ol.shape}"
                )
            if np.any((ol < 0) | (ol > self.L)):
                raise ValueError(f"oracle_levels must lie in [0, L={self.L}]; got {ol}")
            self._oracle_levels = ol
        elif oracle_levels is not None:
            raise ValueError("oracle_levels given but mode is not 'oracle'")

        # modes "replay_*": validate the recorded [T, n_train] level matrix + fix a context
        # permutation for replay_permuted (Gate E causal controls).
        self._replay: np.ndarray | None = None
        self._replay_perm: np.ndarray | None = None
        if self.cfg.mode in _REPLAY_MODES:
            if replay_trajectory is None:
                raise ValueError(f"mode {self.cfg.mode!r} requires replay_trajectory")
            rt = np.asarray(replay_trajectory, dtype=np.int64)
            if rt.ndim != 2 or rt.shape[1] != self.n_train:
                raise ValueError(
                    f"replay_trajectory must be [T, n_train={self.n_train}]; got {rt.shape}"
                )
            if np.any((rt < 0) | (rt > self.L)):
                raise ValueError(f"replay_trajectory levels must lie in [0, L={self.L}]")
            self._replay = rt
            if self.cfg.mode == Method.REPLAY_PERMUTED:
                perm = np.random.default_rng(self.cfg.replay_perm_seed).permutation(self.n_train)
                # avoid the identity (would equal replay_openloop on the same recording)
                if self.n_train > 1 and np.all(perm == np.arange(self.n_train)):
                    perm = np.roll(perm, 1)
                self._replay_perm = perm
        elif replay_trajectory is not None:
            raise ValueError("replay_trajectory given but mode is not a replay_* mode")

        # mode "per_group_success": validate group labels; allocate per-(group, level) pooled
        # statistics + one frontier per group (C5.1; state mirrors the per-context arrays below).
        self.group_ids: np.ndarray | None = None
        if self.cfg.mode in _GROUP_MODES:
            if group_ids is None:
                raise ValueError("mode 'per_group_success' requires group_ids")
            gid = np.asarray(group_ids, dtype=np.int64)
            if gid.shape != (self.n_train,):
                raise ValueError(
                    f"group_ids must have length n_train={self.n_train}; got {gid.shape}"
                )
            n_groups = int(gid.max()) + 1 if gid.size else 0
            if np.any(gid < 0) or len(np.unique(gid)) != n_groups:
                raise ValueError(f"group_ids must be contiguous 0..G-1; got {np.unique(gid)}")
            self.group_ids = gid
            self.n_groups = n_groups
            self._group_members = [np.flatnonzero(gid == g) for g in range(n_groups)]
            gshape = (n_groups, self.n_levels)
            self.g_s = np.zeros(gshape, dtype=np.float64)
            self.g_counts = np.zeros(gshape, dtype=np.int64)
            self.g_stale = np.zeros(gshape, dtype=np.int64)
            self.g_ell = np.full(n_groups, self.L, dtype=np.int64)
            self.g_cooldown = np.zeros(n_groups, dtype=np.int64)
            self._g_harder_streak = np.zeros(n_groups, dtype=np.int64)
            self._g_easier_streak = np.zeros(n_groups, dtype=np.int64)
            self._g_stuck = np.zeros(n_groups, dtype=np.int64)
        elif group_ids is not None:
            raise ValueError("group_ids given but mode is not 'per_group_success'")

        shape = (self.n_train, self.n_levels)
        self.s = np.zeros(shape, dtype=np.float64)  # success EMA per cell
        self.s_slow = np.zeros(shape, dtype=np.float64)  # slow success EMA (for ALP = |s - s_slow|)
        self.u = np.zeros(shape, dtype=np.float64)  # |TD-error| EMA per cell
        # Discounted (success, failure) evidence counts for per_context_wilson_confidence / per_context_beta_posterior (decay-on-update).
        self.succ_d = np.zeros(shape, dtype=np.float64)
        self.fail_d = np.zeros(shape, dtype=np.float64)
        # per_context_success_trend: per-cell logit-Kalman state (level μ, velocity v) + covariance P (2x2).
        self.kf_mu = np.zeros(shape, dtype=np.float64)  # μ0 = logit(0.5)
        self.kf_v = np.zeros(shape, dtype=np.float64)
        self.kf_P = np.zeros(shape + (2, 2), dtype=np.float64)
        cfg_ = self.cfg
        self.kf_P[..., 0, 0] = cfg_.trend_p0_level
        self.kf_P[..., 1, 1] = cfg_.trend_p0_vel
        self.counts = np.zeros(shape, dtype=np.int64)  # rollouts seen per cell
        self.stale = np.zeros(shape, dtype=np.int64)  # iters since cell sampled
        # Frontier: start at the easiest level (framing doc Algorithm 1 init).
        self.ell = np.full(self.n_train, self.L, dtype=np.int64)
        self.cooldown_left = np.zeros(self.n_train, dtype=np.int64)
        self._harder_streak = np.zeros(self.n_train, dtype=np.int64)
        self._easier_streak = np.zeros(self.n_train, dtype=np.int64)
        self._solved_dwell = np.zeros(self.n_train, dtype=np.int64)  # windows solved at frontier
        self._stuck = np.zeros(
            self.n_train, dtype=np.int64
        )  # eligible windows with no frontier move
        self._g_bar = 0.0  # aggregate-success EMA driving the global_mixture level distribution
        # mode "plr_cells": per-cell PLR score (mean |TD-error|, replace-on-first then EMA).
        self._plr_score = np.zeros(shape, dtype=np.float64)
        self._plr_seen = np.zeros(shape, dtype=bool)
        # mode "rfcl_faithful": per-context deque of recent frontier-rollout outcomes (cleared on
        # every frontier move so the windowed advance rule always reads the CURRENT cell).
        self._rf_recent: list[deque[float]] = [
            deque(maxlen=self.cfg.rfclf_window) for _ in range(self.n_train)
        ]
        self._snap = _SnapshotState()

    # ------------------------------------------------------------------ select

    def select(
        self, n: int, step: int, rng: np.random.Generator
    ) -> tuple[list[int], list[int], list[float]]:
        """Choose ``n`` cells to roll this iteration.

        Returns ``(contexts, levels, fractions)``, each length ``n`` (contexts
        may repeat — sampled with replacement). ``fractions[i] =
        level_fractions[levels[i]]`` is the env ``start_progress``.
        """
        if n <= 0:
            raise ValueError(f"n must be > 0; got {n}")
        mode = self.cfg.mode
        entropy: float | None = None

        if mode == Method.GLOBAL_ANNEAL:
            g = self._global_level(step)
            contexts = rng.integers(0, self.n_train, size=n)
            levels = np.full(n, g, dtype=np.int64)
        elif mode == Method.FIXED:
            # Transfer-landscape probe: always train at one fixed scaffold level (no
            # curriculum). The ℓ=0 held-out curve then IS the transfer from `fixed_level`.
            contexts = rng.integers(0, self.n_train, size=n)
            levels = np.full(n, self.cfg.fixed_level, dtype=np.int64)
        elif mode == Method.RANDOM_SCAFFOLD:
            contexts = rng.integers(0, self.n_train, size=n)
            levels = rng.integers(0, self.n_levels, size=n)
        elif mode == Method.MIXTURE:
            # Global stochastic mixture: context uniform, level ~ fixed categorical.
            contexts = rng.integers(0, self.n_train, size=n)
            levels = rng.choice(self.n_levels, size=n, p=self._mixture_probs)
        elif mode == Method.GLOBAL_MIXTURE:
            # Deployable async global baseline: context uniform, level ~ time-varying q_t (no
            # per-context frontiers). q_t shifts toward true-start as aggregate success rises.
            q = self._global_mixture_dist()
            contexts = rng.integers(0, self.n_train, size=n)
            levels = rng.choice(self.n_levels, size=n, p=q)
        elif mode == Method.ORACLE:
            # Per-context fixed level from the group→level map (upper bound).
            contexts = rng.integers(0, self.n_train, size=n)
            levels = self._oracle_levels[contexts]
        elif mode in _REPLAY_MODES:
            # Gate E causal controls — play back a recorded per-context level trajectory.
            row = self._replay[min(int(step), self._replay.shape[0] - 1)]  # recorded levels at t
            contexts = rng.integers(0, self.n_train, size=n)
            if mode == Method.REPLAY_AGGREGATE:
                levels = rng.choice(row, size=n)  # marginal H_t only; NOT matched to contexts
            elif mode == Method.REPLAY_PERMUTED:
                levels = row[self._replay_perm[contexts]]  # right multiset, wrong context
            else:  # replay_openloop — each context replays its own (other-seed) trajectory
                levels = row[contexts]
        elif mode in (Method.RFCL_STYLE, Method.ACED_STYLE):
            # RFCL-style / ACED-style: UNIFORM context sampling over the per-context frontier cells
            # (no PCRC score). The frontier advances (and, rfcl_style only, retreats) on success.
            contexts = rng.integers(0, self.n_train, size=n)
            levels = self.ell[contexts]
        elif mode == Method.RFCL_FAITHFUL:
            # C5.5: struggling-demo-prioritized sampling — weight ∝ (1 − windowed frontier
            # success) + floor; contexts with no recent evidence get maximal priority.
            s_hat = np.asarray(
                [float(np.mean(q)) if q else 0.0 for q in self._rf_recent], dtype=np.float64
            )
            w = (1.0 - s_hat) + self.cfg.rfclf_w_min
            dist = apply_floor_and_cap(w / w.sum(), epsilon=self.cfg.epsilon, q_max=self.cfg.q_max)
            contexts = rng.choice(self.n_train, size=n, replace=True, p=dist.final)
            levels = self.ell[contexts]
            entropy = float(dist.entropy)
        elif mode == Method.ALP_CELLS:
            # C5.2b: ALP-GMM-style learning-progress teacher over ALL (x, ℓ) cells, no frontier.
            score = (
                self.cfg.lambda_alp * np.abs(self.s - self.s_slow)
                + self.cfg.lambda_stale * np.log1p(self.stale)
            ).reshape(self.n_train * self.n_levels)
            logits = score / self.cfg.tau
            raw = np.exp(logits - logits.max())
            dist = apply_floor_and_cap(
                raw / raw.sum(), epsilon=self.cfg.epsilon, q_max=self.cfg.q_max
            )
            flat = rng.choice(self.n_train * self.n_levels, size=n, replace=True, p=dist.final)
            contexts, levels = np.divmod(flat, self.n_levels)
            entropy = float(dist.entropy)
        elif mode == Method.PER_GROUP_SUCCESS:
            # C5.1: PCRC score over GROUP frontier cells (for 2 groups the q_max=0.5 cap makes
            # this exactly uniform over groups), then a uniform context within the chosen group.
            dist = self._group_distribution()
            gsel = rng.choice(self.n_groups, size=n, replace=True, p=dist.final)
            contexts = np.array(
                [rng.choice(self._group_members[g]) for g in gsel], dtype=np.int64
            )
            levels = self.g_ell[gsel]
            entropy = float(dist.entropy)
        elif mode == Method.PLR_CELLS:
            # C5.2a: PLR rank-prioritized |TD-error| + staleness over ALL (x, ℓ) cells; no frontier.
            dist = self._plr_cell_distribution()
            flat = rng.choice(self.n_train * self.n_levels, size=n, replace=True, p=dist.final)
            contexts, levels = np.divmod(flat, self.n_levels)
            entropy = float(dist.entropy)
        else:  # pcrc family — score over frontier cells, softmax-sample
            dist = self._frontier_distribution()
            contexts = rng.choice(self.n_train, size=n, replace=True, p=dist.final)
            levels = self.ell[contexts]
            entropy = float(dist.entropy)

        contexts = np.asarray(contexts, dtype=np.int64)
        levels = np.asarray(levels, dtype=np.int64)
        fractions = self.level_fractions[levels]
        self._snap.selected_levels = levels.tolist()
        self._snap.selection_entropy = entropy
        return contexts.tolist(), levels.tolist(), fractions.tolist()

    def _global_level(self, step: int) -> int:
        """``global_anneal`` schedule: level ``L`` at step 0, ``0`` by ``anneal_iters``."""
        if self.cfg.anneal_iters <= 0:
            return 0
        frac = max(0.0, 1.0 - float(step) / float(self.cfg.anneal_iters))
        return int(round(self.L * frac))

    def _global_mixture_dist(self) -> np.ndarray:
        """Global, time-varying level distribution ``q_t`` for mode ``global_mixture``.

        True-start (``ell=0``) mass grows linearly with aggregate success ``g_bar`` from
        ``gmix_q0_lo`` to ``gmix_q0_hi``; the easiest level keeps a constant floor ``gmix_easy_floor``;
        the remaining mass spreads uniformly over the intermediate levels. No per-context state.
        """
        if self.n_levels == 1:
            return np.ones(1)
        c = float(np.clip(self._g_bar, 0.0, 1.0))
        q = np.zeros(self.n_levels, dtype=np.float64)
        q[0] = self.cfg.gmix_q0_lo + (self.cfg.gmix_q0_hi - self.cfg.gmix_q0_lo) * c
        q[self.n_levels - 1] = self.cfg.gmix_easy_floor
        inter = list(range(1, self.n_levels - 1))
        rem = max(0.0, 1.0 - q[0] - q[self.n_levels - 1])
        if inter:
            for i in inter:
                q[i] = rem / len(inter)
        else:
            q[self.n_levels - 1] += rem
        total = q.sum()
        return q / total if total > 0 else np.full(self.n_levels, 1.0 / self.n_levels)

    def _frontier_distribution(self):
        """``apply_floor_and_cap(softmax(score/τ))`` over the per-context frontier cells."""
        score = self._score()
        logits = score / self.cfg.tau
        logits = logits - logits.max()  # stability
        raw = np.exp(logits)
        raw = raw / raw.sum()
        return apply_floor_and_cap(raw, epsilon=self.cfg.epsilon, q_max=self.cfg.q_max)

    def _group_distribution(self):
        """Mode ``per_group_success``: the per_context_success score over GROUP frontier cells (C5.1)."""
        rows = np.arange(self.n_groups)
        s_gf = self.g_s[rows, self.g_ell]
        progress = (self.L - self.g_ell) / self.L if self.L > 0 else np.zeros(self.n_groups)
        score = (
            self.cfg.lambda_h * s_gf * (1.0 - s_gf)
            + self.cfg.lambda_p * progress
            + self.cfg.lambda_stale * np.log1p(self.g_stale[rows, self.g_ell])
        )
        logits = score / self.cfg.tau
        logits = logits - logits.max()
        raw = np.exp(logits)
        raw = raw / raw.sum()
        return apply_floor_and_cap(raw, epsilon=self.cfg.epsilon, q_max=self.cfg.q_max)

    def _plr_cell_distribution(self):
        """Mode ``plr_cells``: PLR rank-prioritization + staleness over the flat cell grid (C5.2a).

        ``P_S ∝ (1/rank)**(1/plr_beta)`` over the per-cell |TD-error| score (never-visited cells
        rank last with score 0); ``P_C ∝ staleness`` (never-visited cells age monotonically since
        ``stale`` only resets on a visit, so they get maximal staleness for free);
        ``P = (1-ρ)·P_S + ρ·P_C``, then PCRC's frozen floor/cap.
        """
        n = self.n_train * self.n_levels
        score = self._plr_score.reshape(n)
        order = np.argsort(-score, kind="stable")
        rank = np.empty(n, dtype=np.float64)
        rank[order] = np.arange(1, n + 1, dtype=np.float64)
        p_s = (1.0 / rank) ** (1.0 / self.cfg.plr_beta)
        p_s = p_s / p_s.sum()
        stale = self.stale.reshape(n).astype(np.float64)
        p_c = stale / stale.sum() if stale.sum() > 0 else np.full(n, 1.0 / n)
        raw = (1.0 - self.cfg.plr_rho) * p_s + self.cfg.plr_rho * p_c
        raw = raw / raw.sum()
        return apply_floor_and_cap(raw, epsilon=self.cfg.epsilon, q_max=self.cfg.q_max)

    def _score(self) -> np.ndarray:
        """PCRC score over each context's current frontier cell (framing doc §7)."""
        rows = np.arange(self.n_train)
        s_xl = self.s[rows, self.ell]
        progress = (self.L - self.ell) / self.L if self.L > 0 else np.zeros(self.n_train)
        stale_term = np.log1p(self.stale[rows, self.ell])
        if self.cfg.mode == Method.PER_CONTEXT_LEARNING_PROGRESS:
            # ALP-GMM-style: allocate by Absolute Learning Progress at the frontier cell.
            alp = np.abs(s_xl - self.s_slow[rows, self.ell])
            return (
                self.cfg.lambda_alp * alp
                + self.cfg.lambda_p * progress
                + self.cfg.lambda_stale * stale_term
            )
        if self.cfg.mode == Method.PER_CONTEXT_WILSON_CONFIDENCE:
            # TP-PCRC: competence (from the discounted-count p̂) + learning progress + staleness.
            p_hat, _, _ = self._wilson_bounds()
            lp = np.abs(s_xl - self.s_slow[rows, self.ell])
            return (
                self.cfg.lambda_h * p_hat * (1.0 - p_hat)
                + self.cfg.lambda_lp * lp
                + self.cfg.lambda_p * progress
                + self.cfg.lambda_stale * stale_term
            )
        if self.cfg.mode == Method.PER_CONTEXT_BETA_POSTERIOR:
            # B-PCRC-Beta: posterior-mean competence + posterior-sd exploration bonus.
            a, b = self._beta_params()
            mean = a / (a + b)
            var = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
            return (
                self.cfg.lambda_h * mean * (1.0 - mean)
                + self.cfg.w_unc_post * np.sqrt(var)
                + self.cfg.lambda_p * progress
                + self.cfg.lambda_stale * stale_term
            )
        if self.cfg.mode == Method.PER_CONTEXT_SUCCESS_TREND:
            # B-PCRC-Trend: filtered competence + level-sd bonus (delta method on σ(μ)).
            mu = self.kf_mu[rows, self.ell]
            P00 = self.kf_P[rows, self.ell][:, 0, 0]
            p_hat = 1.0 / (1.0 + np.exp(-mu))
            sd_p = p_hat * (1.0 - p_hat) * np.sqrt(np.maximum(P00, 0.0))
            return (
                self.cfg.lambda_h * p_hat * (1.0 - p_hat)
                + self.cfg.w_unc_post * sd_p
                + self.cfg.lambda_p * progress
                + self.cfg.lambda_stale * stale_term
            )
        h = s_xl * (1.0 - s_xl)
        u_tilde = self._frontier_u_tilde()
        lam_h = 0.0 if self.cfg.mode == Method.PER_CONTEXT_UNCERTAINTY_PRIORITY else self.cfg.lambda_h
        if self.cfg.mode in _SUCCESS_SCORING_MODES:
            unc_term = np.zeros(self.n_train)
        elif self.cfg.mode == Method.PER_CONTEXT_UNCERTAINTY_TIEBREAK:
            # Bounded tie-breaker: normalize z to [-1, 1], small weight → headroom leads.
            unc_term = self.cfg.unc_tiebreak_weight * (u_tilde / self.cfg.clip)
        else:  # pcrc, per_context_uncertainty_priority — uncertainty as a (dominant) driver
            unc_term = self.cfg.lambda_u * u_tilde
        return (
            lam_h * h + unc_term + self.cfg.lambda_p * progress + self.cfg.lambda_stale * stale_term
        )

    def _frontier_u_tilde(self) -> np.ndarray:
        """Robust z-score of frontier-cell uncertainty *within each level* (§6.4).

        Compares a cell only to other frontier cells at the same scaffold level
        (median/MAD), clips to ``±clip``, and zeroes any cell below the ``n_min``
        reliability gate. Flat/degenerate groups → 0 (no NaN).
        """
        rows = np.arange(self.n_train)
        levels = self.ell
        u_vals = self.u[rows, levels]
        cell_counts = self.counts[rows, levels]
        z = np.zeros(self.n_train, dtype=np.float64)
        for lvl in np.unique(levels):
            mask = levels == lvl
            grp = u_vals[mask]
            med = np.median(grp)
            mad = np.median(np.abs(grp - med))
            z[mask] = (grp - med) / (mad + 1e-9)
        z = np.clip(z, -self.cfg.clip, self.cfg.clip)
        z[cell_counts < self.cfg.n_min] = 0.0
        return z

    # -------------------------------------------------- evidence (tp / bpcrc)

    def _wilson_bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(p̂, lower, upper)`` Wilson bounds at each context's frontier cell (mode per_context_wilson_confidence).

        Uses the discounted counts' effective sample size ``n_eff = a + b`` (docs: "exponential-window
        effective sample size"). Cells with no evidence yet get ``(0, 0, 1)`` — maximally uncertain,
        which can neither advance (lower 0 < τ_hi) nor retreat (upper 1 > τ_lo).
        """
        rows = np.arange(self.n_train)
        a = self.succ_d[rows, self.ell]
        b = self.fail_d[rows, self.ell]
        n = a + b
        z = self.cfg.tp_z
        p_hat = np.where(n > 0, a / np.maximum(n, 1e-12), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = 1.0 + z * z / np.maximum(n, 1e-12)
            center = (p_hat + z * z / (2.0 * np.maximum(n, 1e-12))) / denom
            half = (
                z
                * np.sqrt(
                    p_hat * (1.0 - p_hat) / np.maximum(n, 1e-12)
                    + z * z / (4.0 * np.maximum(n, 1e-12) ** 2)
                )
                / denom
            )
        lower = np.where(n > 0, np.clip(center - half, 0.0, 1.0), 0.0)
        upper = np.where(n > 0, np.clip(center + half, 0.0, 1.0), 1.0)
        return p_hat, lower, upper

    def _beta_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Discounted Beta posterior ``(a, b)`` at each context's frontier cell (mode per_context_beta_posterior)."""
        rows = np.arange(self.n_train)
        a = self.cfg.bpcrc_prior + self.succ_d[rows, self.ell]
        b = self.cfg.bpcrc_prior + self.fail_d[rows, self.ell]
        return a, b

    def _kf_step(self, x: int, lvl: int, mean_succ: float, k: int) -> None:
        """One predict+EKF-update of the cell's logit-Kalman filter (mode per_context_success_trend).

        State (μ, v) with local-linear-trend dynamics; observation = the window's Bernoulli mean
        with variance p̂(1-p̂)/k (floored). Runs only on windows where the cell receives data.
        """
        cfg = self.cfg
        mu, v = self.kf_mu[x, lvl], self.kf_v[x, lvl]
        P = self.kf_P[x, lvl]
        # predict: μ ← μ + v, P ← F P Fᵀ + Q with F = [[1,1],[0,1]]
        mu = mu + v
        P = np.array(
            [
                [P[0, 0] + 2 * P[0, 1] + P[1, 1] + cfg.trend_q_level, P[0, 1] + P[1, 1]],
                [P[0, 1] + P[1, 1], P[1, 1] + cfg.trend_q_vel],
            ]
        )
        # EKF update with H = [σ'(μ), 0]
        p = 1.0 / (1.0 + np.exp(-mu))
        h = max(p * (1.0 - p), 1e-4)
        r = max(p * (1.0 - p) / max(k, 1), 1e-3)
        s = h * P[0, 0] * h + r
        k_gain = np.array([P[0, 0] * h / s, P[0, 1] * h / s])
        e = float(mean_succ) - p
        mu += k_gain[0] * e
        v += k_gain[1] * e
        # P ← (I - K H) P, H = [h, 0]
        ikh = np.array([[1.0 - k_gain[0] * h, 0.0], [-k_gain[1] * h, 1.0]])
        P = ikh @ P
        P = (P + P.T) / 2.0
        P[0, 0] = max(P[0, 0], 1e-6)
        P[1, 1] = max(P[1, 1], 1e-8)
        c = cfg.trend_logit_clip
        self.kf_mu[x, lvl] = float(np.clip(mu, -c, c))
        self.kf_v[x, lvl] = float(np.clip(v, -2.0, 2.0))
        self.kf_P[x, lvl] = P

    def _trend_signals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predictive ``(harder, easier, s_est)`` at each frontier cell (mode per_context_success_trend).

        μ_pred = μ + h·v with predictive variance [1,h] P [1,h]ᵀ; advance iff
        P(σ(μ_pred) > τ_hi) > 1−δ_hi ⇔ (μ_pred − logit τ_hi)/σ_pred > z_{1−δ}, retreat symmetric.
        s_est = σ(μ) (current point estimate) feeds the shared antistall guard.
        """
        from scipy.stats import norm as _norm

        cfg = self.cfg
        rows = np.arange(self.n_train)
        mu = self.kf_mu[rows, self.ell]
        v = self.kf_v[rows, self.ell]
        P = self.kf_P[rows, self.ell]
        h = cfg.trend_horizon
        mu_pred = mu + h * v
        var_pred = P[:, 0, 0] + 2 * h * P[:, 0, 1] + h * h * P[:, 1, 1]
        sd = np.sqrt(np.maximum(var_pred, 1e-12))
        z_hi = float(_norm.ppf(1.0 - cfg.bpcrc_delta_hi))
        z_lo = float(_norm.ppf(1.0 - cfg.bpcrc_delta_lo))
        logit = lambda q: float(np.log(q / (1.0 - q)))  # noqa: E731
        harder = (mu_pred - logit(cfg.tau_hi)) / sd > z_hi
        easier = (logit(cfg.tau_lo) - mu_pred) / sd > z_lo
        s_est = 1.0 / (1.0 + np.exp(-mu))
        return harder, easier, s_est

    def _evidence_signals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-context ``(harder, easier, s_est)`` transition signals for the evidence modes.

        per_context_wilson_confidence: harder iff Wilson lower > τ_hi; easier iff Wilson upper < τ_lo; s_est = p̂.
        per_context_beta_posterior: harder iff P(p > τ_hi) > 1-δ_hi; easier iff P(p < τ_lo) > 1-δ_lo;
        s_est = posterior mean. per_context_success_trend: predictive gate (see _trend_signals). ``s_est``
        feeds the shared antistall guard (probe only if the estimate is above τ_lo, i.e. the
        cell is not confidently failed).
        """
        if self.cfg.mode == Method.PER_CONTEXT_WILSON_CONFIDENCE:
            p_hat, lower, upper = self._wilson_bounds()
            return lower > self.cfg.tau_hi, upper < self.cfg.tau_lo, p_hat
        if self.cfg.mode == Method.PER_CONTEXT_SUCCESS_TREND:
            return self._trend_signals()
        from scipy.stats import beta as _beta

        a, b = self._beta_params()
        harder = _beta.sf(self.cfg.tau_hi, a, b) > 1.0 - self.cfg.bpcrc_delta_hi
        easier = _beta.cdf(self.cfg.tau_lo, a, b) > 1.0 - self.cfg.bpcrc_delta_lo
        return harder, easier, a / (a + b)

    # ------------------------------------------------------------------ update

    def update(
        self,
        contexts: list[int],
        levels: list[int],
        successes: list[bool],
        td_errors: list[float],
        step: int,
    ) -> None:
        """Fold one iteration's rollouts into the cell table, then move frontiers.

        The four lists are aligned per rollout. ``td_errors[i]`` is the mean
        ``|TD-error|`` of rollout ``i``'s transitions — computed by the loop on
        the **freshly-rolled** episode (whose level is known), never from a
        level-mixing replay minibatch.
        """
        if not (len(contexts) == len(levels) == len(successes) == len(td_errors)):
            raise ValueError("contexts/levels/successes/td_errors must be the same length")

        # Aggregate-success EMA (drives the global_mixture level distribution; no per-context state).
        if successes:
            sm_all = float(np.mean([1.0 if s else 0.0 for s in successes]))
            self._g_bar = (1.0 - self.cfg.gmix_beta) * self._g_bar + self.cfg.gmix_beta * sm_all

        # Staleness: age every cell, then reset the ones sampled this step.
        self.stale += 1

        # Aggregate per cell so EMA updates are order-independent.
        agg: dict[tuple[int, int], list[list[float]]] = defaultdict(lambda: [[], []])
        for x, lvl, succ, td in zip(contexts, levels, successes, td_errors, strict=True):
            agg[(int(x), int(lvl))][0].append(1.0 if succ else 0.0)
            agg[(int(x), int(lvl))][1].append(float(td))

        evidence_gamma = (
            self.cfg.tp_gamma if self.cfg.mode == Method.PER_CONTEXT_WILSON_CONFIDENCE else self.cfg.bpcrc_gamma
        )
        for (x, lvl), (succs, tds) in agg.items():
            sm = float(np.mean(succs))
            self.s[x, lvl] = (1.0 - self.cfg.beta_s) * self.s[x, lvl] + self.cfg.beta_s * sm
            self.s_slow[x, lvl] = (1.0 - self.cfg.beta_s_slow) * self.s_slow[
                x, lvl
            ] + self.cfg.beta_s_slow * sm
            self.u[x, lvl] = (1.0 - self.cfg.beta_u) * self.u[x, lvl] + self.cfg.beta_u * float(
                np.mean(tds)
            )
            self.counts[x, lvl] += len(succs)
            self.stale[x, lvl] = 0
            if self.cfg.mode in (Method.PER_CONTEXT_WILSON_CONFIDENCE, Method.PER_CONTEXT_BETA_POSTERIOR):
                k = float(np.sum(succs))
                self.succ_d[x, lvl] = evidence_gamma * self.succ_d[x, lvl] + k
                self.fail_d[x, lvl] = evidence_gamma * self.fail_d[x, lvl] + (len(succs) - k)
            elif self.cfg.mode == Method.PER_CONTEXT_SUCCESS_TREND:
                self._kf_step(x, lvl, sm, len(succs))
            elif self.cfg.mode == Method.PLR_CELLS:
                td_m = float(np.mean(tds))
                a = self.cfg.plr_score_ema_alpha
                self._plr_score[x, lvl] = (
                    (1.0 - a) * self._plr_score[x, lvl] + a * td_m
                    if self._plr_seen[x, lvl]
                    else td_m
                )
                self._plr_seen[x, lvl] = True

        # rfcl_faithful: fold this window's frontier-cell outcomes into the recent-outcome
        # windows (the windowed advance itself runs after the snapshot reset below).
        if self.cfg.mode == Method.RFCL_FAITHFUL:
            for x, lvl, succ in zip(contexts, levels, successes, strict=True):
                if int(lvl) == int(self.ell[int(x)]):
                    self._rf_recent[int(x)].append(1.0 if succ else 0.0)

        # per_group_success: pool this window's rollouts per (group, level), then run the
        # group-granularity transition with the frozen PCRC rule.
        if self.cfg.mode in _GROUP_MODES:
            g_agg: dict[tuple[int, int], list[list[float]]] = defaultdict(lambda: [[]])
            for (x, lvl), (succs, _tds) in agg.items():
                g_agg[(int(self.group_ids[x]), int(lvl))][0].extend(succs)
            self.g_stale += 1
            for (g, lvl), (succs,) in g_agg.items():
                sm = float(np.mean(succs))
                self.g_s[g, lvl] = (1.0 - self.cfg.beta_s) * self.g_s[g, lvl] + self.cfg.beta_s * sm
                self.g_counts[g, lvl] += len(succs)
                self.g_stale[g, lvl] = 0
            self.g_cooldown = np.maximum(0, self.g_cooldown - 1)

        self.cooldown_left = np.maximum(0, self.cooldown_left - 1)

        self._snap.step = int(step)
        self._snap.n_harder_moves = 0
        self._snap.n_easier_moves = 0
        if self.cfg.mode in _FRONTIER_MODES:
            self._transition()
        elif self.cfg.mode in _GROUP_MODES:
            self._group_transition()
        elif self.cfg.mode == Method.RFCL_FAITHFUL:
            # Windowed ADVANCE-ONLY rule (no retreat, no antistall, no cooldown): advance one
            # level iff >= rfclf_advance_successes of the last rfclf_window frontier rollouts
            # succeeded; the window clears on every move so it always reads the current cell.
            for x in range(self.n_train):
                q = self._rf_recent[x]
                if (
                    len(q) == self.cfg.rfclf_window
                    and sum(q) >= self.cfg.rfclf_advance_successes
                    and self.ell[x] > 0
                ):
                    self.ell[x] -= 1
                    q.clear()
                    self._snap.n_harder_moves += 1

    def _transition(self) -> None:
        """Advance/retreat frontiers for any context with enough accumulated evidence (§8).

        Eligibility is the ``n_min`` count gate + cooldown on a context's CURRENT
        frontier cell — evaluated for ALL contexts, NOT gated on being sampled this
        step. (Gating on current-window sampling couples advancement to the sampler;
        since pcrc's sampler is uncertainty-seeking it starves the low-uncertainty
        *solved* cells that are exactly the ones eligible to advance, stalling the
        frontier.) The frontier EMAs persist, so they are the accumulated evidence.
        Hysteresis: a direction must hold for ``hysteresis_windows`` consecutive
        update windows before the frontier moves.
        """
        u_tilde = self._frontier_u_tilde()
        rows = np.arange(self.n_train)
        s_front = self.s[rows, self.ell]
        cnt_front = self.counts[rows, self.ell]
        use_unc = self.cfg.mode not in _SUCCESS_ONLY_FRONTIER
        # Evidence modes: harder/easier come from the CI / posterior gate, the antistall guard
        # from the evidence estimate, and hysteresis is subsumed by the confidence gate itself
        # (a one-window streak suffices — the gate already demands accumulated evidence).
        evidence = self.cfg.mode in _EVIDENCE_MODES
        if evidence:
            harder_sig, easier_sig, s_front = self._evidence_signals()
        hyst = 1 if evidence else self.cfg.hysteresis_windows

        for x in range(self.n_train):
            if cnt_front[x] < self.cfg.n_min or self.cooldown_left[x] > 0:
                continue
            confident = (u_tilde[x] < self.cfg.u_lo) if use_unc else True
            # Post-solution dwell: a solved frontier cell must stay solved for
            # post_solution_dwell windows before it may advance (deterministic pacing screen).
            solved = bool(harder_sig[x]) if evidence else s_front[x] > self.cfg.tau_hi
            self._solved_dwell[x] = self._solved_dwell[x] + 1 if solved else 0
            dwell_ok = self._solved_dwell[x] >= self.cfg.post_solution_dwell
            harder = solved and confident and dwell_ok
            # aced_style (C5.5): ACED's threshold-triggered section advance has no backward move.
            easier = (
                (bool(easier_sig[x]) if evidence else s_front[x] < self.cfg.tau_lo)
                and confident
                and self.cfg.mode != Method.ACED_STYLE
            )

            if harder:
                self._harder_streak[x] += 1
                self._easier_streak[x] = 0
            elif easier:
                self._easier_streak[x] += 1
                self._harder_streak[x] = 0
            else:
                self._harder_streak[x] = 0
                self._easier_streak[x] = 0

            moved = False
            if self._harder_streak[x] >= hyst and self.ell[x] > 0:
                self.ell[x] -= 1  # remove assistance
                self._reset_after_move(x)
                self._snap.n_harder_moves += 1
                moved = True
            elif self._easier_streak[x] >= hyst and self.ell[x] < self.L:
                self.ell[x] += 1  # add assistance
                self._reset_after_move(x)
                self._snap.n_easier_moves += 1
                moved = True

            # Anti-stall: a context stuck-but-not-failing in the dead band is forced one harder
            # step so its frontier keeps progressing toward ℓ=0 (see ScaffoldConfig docstring).
            if not moved:
                self._stuck[x] += 1
                if (
                    self.cfg.stale_advance_windows > 0
                    and self._stuck[x] >= self.cfg.stale_advance_windows
                    and s_front[x] > self.cfg.tau_lo
                    and self.ell[x] > 0
                ):
                    self.ell[x] -= 1
                    self._reset_after_move(x)
                    self._snap.n_harder_moves += 1

    def _reset_after_move(self, x: int) -> None:
        """Clear streaks + start the cooldown after a frontier move."""
        self._harder_streak[x] = 0
        self._easier_streak[x] = 0
        self._solved_dwell[x] = 0
        self._stuck[x] = 0
        self.cooldown_left[x] = self.cfg.cooldown

    def _group_transition(self) -> None:
        """Mode ``per_group_success``: the ``_transition`` rule at GROUP granularity (C5.1).

        Byte-for-byte the per_context_success success-only semantics — n_min evidence gate, cooldown,
        two-window hysteresis, tau_lo/tau_hi thresholds, antistall — but over the pooled
        per-(group, level) statistics and one frontier per group. After any move the expanded
        per-context ``ell`` is refreshed so scaffold.jsonl logs stay comparable across conditions.
        """
        rows = np.arange(self.n_groups)
        s_front = self.g_s[rows, self.g_ell]
        cnt_front = self.g_counts[rows, self.g_ell]
        for g in range(self.n_groups):
            if cnt_front[g] < self.cfg.n_min or self.g_cooldown[g] > 0:
                continue
            harder = s_front[g] > self.cfg.tau_hi
            easier = s_front[g] < self.cfg.tau_lo
            if harder:
                self._g_harder_streak[g] += 1
                self._g_easier_streak[g] = 0
            elif easier:
                self._g_easier_streak[g] += 1
                self._g_harder_streak[g] = 0
            else:
                self._g_harder_streak[g] = 0
                self._g_easier_streak[g] = 0

            moved = False
            if self._g_harder_streak[g] >= self.cfg.hysteresis_windows and self.g_ell[g] > 0:
                self.g_ell[g] -= 1
                self._g_reset_after_move(g)
                self._snap.n_harder_moves += 1
                moved = True
            elif self._g_easier_streak[g] >= self.cfg.hysteresis_windows and self.g_ell[g] < self.L:
                self.g_ell[g] += 1
                self._g_reset_after_move(g)
                self._snap.n_easier_moves += 1
                moved = True

            if not moved:
                self._g_stuck[g] += 1
                if (
                    self.cfg.stale_advance_windows > 0
                    and self._g_stuck[g] >= self.cfg.stale_advance_windows
                    and s_front[g] > self.cfg.tau_lo
                    and self.g_ell[g] > 0
                ):
                    self.g_ell[g] -= 1
                    self._g_reset_after_move(g)
                    self._snap.n_harder_moves += 1
        self.ell = self.g_ell[self.group_ids]

    def _g_reset_after_move(self, g: int) -> None:
        """Clear the group's streaks + start its cooldown after a frontier move."""
        self._g_harder_streak[g] = 0
        self._g_easier_streak[g] = 0
        self._g_stuck[g] = 0
        self.g_cooldown[g] = self.cfg.cooldown

    # ---------------------------------------------------------------- snapshot

    def snapshot(self) -> dict:
        """JSON-serializable diagnostics for ``scaffold.jsonl`` (framing doc §15.1)."""
        level_counts = np.bincount(
            np.asarray(self._snap.selected_levels, dtype=np.int64), minlength=self.n_levels
        ).tolist()
        has_data = self.counts > 0
        mean_s_by_level: list[float | None] = []
        mean_u_by_level: list[float | None] = []
        for lvl in range(self.n_levels):
            col = has_data[:, lvl]
            mean_s_by_level.append(float(self.s[col, lvl].mean()) if col.any() else None)
            mean_u_by_level.append(float(self.u[col, lvl].mean()) if col.any() else None)
        return {
            "step": self._snap.step,
            "mode": self.cfg.mode,
            "frontier_mean": float(self.ell.mean()),
            "ell": self.ell.tolist(),
            "selected_level_counts": level_counts,
            "mean_success_by_level": mean_s_by_level,
            "mean_uncertainty_by_level": mean_u_by_level,
            "n_harder_moves": self._snap.n_harder_moves,
            "n_easier_moves": self._snap.n_easier_moves,
            "selection_entropy": self._snap.selection_entropy,
        }
