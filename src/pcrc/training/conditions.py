"""PCRC-Nav condition → loop-config bundle mapping (pure; CPU-testable).

The Stage-2 driver (``scripts/pointmaze_sac_her.py``) maps each ``--conditions``
name to a ``FetchSACHERLoopConfig`` bundle. PCRC scaffold modes all run the
uniform sampler (the :class:`ScaffoldController` drives selection), so they
differ only in loop overrides. Kept out of the script so it imports no sim/torch
deps and can be unit-tested in the CPU suite.

Plan / provenance: docs/EXPERIMENTS.md.
"""

from __future__ import annotations

from pcrc.registry import SCAFFOLD_METHODS, TRAJECTORY_PREFIX, Method, SamplerName, ScaffoldSource

# PCRC modes handled by ScaffoldController (each runs the uniform sampler).
PCRC_SCAFFOLD_MODES = tuple(sorted(SCAFFOLD_METHODS))


def resolve_condition(condition: str, scaffold_block: dict, rc_block: dict) -> tuple[str, dict]:
    """Map a ``--conditions`` name to ``(loop_condition, loop_overrides)``.

    - PCRC modes → uniform sampler + ``scaffold_enabled`` and the shared
      ``scaffold`` block (n_levels / level_fractions / kwargs), with
      ``scaffold_mode`` set to the condition name.
    - ``global_linear_continuous`` → uniform sampler + the legacy continuous reverse-curriculum
      block (the pilot bridge; not budget-matched to the discrete cells).
    - ``uniform`` and any other name → pass through as a plain sampler condition.

    Returns the loop ``condition`` (a valid sampler name) and a dict of overrides
    to splat over the base ``loop`` config.
    """
    # trajectory_<mode>: automatic trajectory-derived scaffold. Resolve the base <mode> unchanged, then —
    # only when it enables the scaffold controller — flip the env reset-state source to "trajectory"
    # (the driver injects the loaded ScaffoldLadder). The controller config is byte-identical to
    # <mode>; §2.3 of docs/Automatic_Trajectory_Scaffold_Execution_Plan.md.
    if condition == Method.UNASSISTED:
        return SamplerName.UNIFORM, {}
    if condition.startswith(TRAJECTORY_PREFIX):
        loop_condition, overrides = resolve_condition(
            condition[len(TRAJECTORY_PREFIX) :], scaffold_block, rc_block
        )
        if overrides.get("scaffold_enabled"):
            overrides = {**overrides, "scaffold_source": ScaffoldSource.TRAJECTORY}
        return loop_condition, overrides
    # dwell<N>: deterministic pacing screen — mode "per_context_success_dwell" with post_solution_dwell=N.
    if condition.startswith(Method.PER_CONTEXT_SUCCESS_DWELL) and condition[len(Method.PER_CONTEXT_SUCCESS_DWELL) :].isdigit():
        kwargs = dict(scaffold_block.get("kwargs", {}))
        kwargs["post_solution_dwell"] = int(condition[len(Method.PER_CONTEXT_SUCCESS_DWELL) :])
        return SamplerName.UNIFORM, {
            "scaffold_enabled": True,
            "scaffold_mode": Method.PER_CONTEXT_SUCCESS_DWELL,
            "scaffold_n_levels": int(scaffold_block["n_levels"]),
            "scaffold_level_fractions": list(scaffold_block["level_fractions"]),
            "scaffold_kwargs": kwargs,
        }
    # fixed<N>: transfer-landscape probe — train always at scaffold level N (no curriculum).
    if condition.startswith(Method.FIXED) and condition[len(Method.FIXED) :].isdigit():
        kwargs = dict(scaffold_block.get("kwargs", {}))
        kwargs["fixed_level"] = int(condition[len(Method.FIXED) :])
        return SamplerName.UNIFORM, {
            "scaffold_enabled": True,
            "scaffold_mode": Method.FIXED,
            "scaffold_n_levels": int(scaffold_block["n_levels"]),
            "scaffold_level_fractions": list(scaffold_block["level_fractions"]),
            "scaffold_kwargs": kwargs,
        }
    # global_anneal<N>: global anneal schedule with anneal_iters=N (autonomous-plan Gate C2
    # rescue — anneal-RATE sweep to separate per-context allocation from "just more scaffold").
    if condition.startswith(Method.GLOBAL_ANNEAL) and condition[len(Method.GLOBAL_ANNEAL) :].isdigit():
        kwargs = dict(scaffold_block.get("kwargs", {}))
        kwargs["anneal_iters"] = int(condition[len(Method.GLOBAL_ANNEAL) :])
        return SamplerName.UNIFORM, {
            "scaffold_enabled": True,
            "scaffold_mode": Method.GLOBAL_ANNEAL,
            "scaffold_n_levels": int(scaffold_block["n_levels"]),
            "scaffold_level_fractions": list(scaffold_block["level_fractions"]),
            "scaffold_kwargs": kwargs,
        }
    # mixture_<l>:<p>_<l>:<p>...: global static mixture over levels (autonomous-plan Gate B
    # threat baseline). e.g. "mixture_0:0.5_2:0.5". Probs are renormalized by the controller.
    if condition.startswith("mixture_"):
        n_levels = int(scaffold_block["n_levels"])
        probs = [0.0] * n_levels
        for part in condition[len("mixture_") :].split("_"):
            lvl_s, sep, p_s = part.partition(":")
            if not sep:
                raise ValueError(f"mixture spec {part!r} must be '<level>:<prob>'")
            lvl = int(lvl_s)
            if not 0 <= lvl < n_levels:
                raise ValueError(f"mixture level {lvl} out of range [0,{n_levels})")
            probs[lvl] += float(p_s)
        kwargs = dict(scaffold_block.get("kwargs", {}))
        kwargs["mixture_probs"] = probs
        return SamplerName.UNIFORM, {
            "scaffold_enabled": True,
            "scaffold_mode": Method.MIXTURE,
            "scaffold_n_levels": n_levels,
            "scaffold_level_fractions": list(scaffold_block["level_fractions"]),
            "scaffold_kwargs": kwargs,
        }
    # oracle: per-context fixed level from a group→level map. resolve_condition only sets
    # the mode; the DRIVER computes scaffold_oracle_levels from the bank (it has goal-z).
    if condition == Method.ORACLE:
        return SamplerName.UNIFORM, {
            "scaffold_enabled": True,
            "scaffold_mode": Method.ORACLE,
            "scaffold_n_levels": int(scaffold_block["n_levels"]),
            "scaffold_level_fractions": list(scaffold_block["level_fractions"]),
            "scaffold_kwargs": dict(scaffold_block.get("kwargs", {})),
        }
    # replay_{aggregate,permuted,openloop}: Gate E causal controls. resolve_condition only sets
    # the mode; the DRIVER loads scaffold_replay_trajectory from a reference pcrc run's scaffold.jsonl.
    if condition in (Method.REPLAY_AGGREGATE, Method.REPLAY_PERMUTED, Method.REPLAY_OPENLOOP):
        return SamplerName.UNIFORM, {
            "scaffold_enabled": True,
            "scaffold_mode": condition,
            "scaffold_n_levels": int(scaffold_block["n_levels"]),
            "scaffold_level_fractions": list(scaffold_block["level_fractions"]),
            "scaffold_kwargs": dict(scaffold_block.get("kwargs", {})),
        }
    # rfcl_style: RFCL-style local trajectory curriculum (Test Campaign 4 Stage B) — per-context frontier
    # with UNIFORM sampling and NO anti-stall. Shares thresholds with PCRC; only anti-stall is forced off.
    if condition == Method.RFCL_STYLE:
        kwargs = dict(scaffold_block.get("kwargs", {}))
        kwargs["stale_advance_windows"] = 0
        return SamplerName.UNIFORM, {
            "scaffold_enabled": True,
            "scaffold_mode": Method.RFCL_STYLE,
            "scaffold_n_levels": int(scaffold_block["n_levels"]),
            "scaffold_level_fractions": list(scaffold_block["level_fractions"]),
            "scaffold_kwargs": kwargs,
        }
    # per_group_success (campaign 5 C5.1): PCRC at GROUP granularity — one pooled frontier per group.
    # resolve_condition only sets the mode; the DRIVER computes scaffold_group_ids from the bank's
    # oracle group labels (goal-z for Fetch, task for Meta-World). Frozen PCRC kwargs untouched.
    # plr_cells / alp_cells (campaign 5 C5.2a/b): cell teachers over ALL (x, ℓ) cells, no frontier.
    # rfcl_faithful (C5.5): component-matched RFCL — windowed advance-only rule + struggling-demo
    # sampling (its own rfclf_* defaults; antistall/thresholds unused by construction).
    if condition in (Method.PER_GROUP_SUCCESS, Method.PLR_CELLS, Method.ALP_CELLS, Method.RFCL_FAITHFUL):
        return SamplerName.UNIFORM, {
            "scaffold_enabled": True,
            "scaffold_mode": condition,
            "scaffold_n_levels": int(scaffold_block["n_levels"]),
            "scaffold_level_fractions": list(scaffold_block["level_fractions"]),
            "scaffold_kwargs": dict(scaffold_block.get("kwargs", {})),
        }
    # aced_style (campaign 5 C5.5): the rfcl_style local frontier with the retreat branch removed
    # (ACED's threshold advance has no backward move). Antistall off, like rfcl_style.
    if condition == Method.ACED_STYLE:
        kwargs = dict(scaffold_block.get("kwargs", {}))
        kwargs["stale_advance_windows"] = 0
        return SamplerName.UNIFORM, {
            "scaffold_enabled": True,
            "scaffold_mode": Method.ACED_STYLE,
            "scaffold_n_levels": int(scaffold_block["n_levels"]),
            "scaffold_level_fractions": list(scaffold_block["level_fractions"]),
            "scaffold_kwargs": kwargs,
        }
    if condition in SCAFFOLD_METHODS:
        return SamplerName.UNIFORM, {
            "scaffold_enabled": True,
            "scaffold_mode": condition,
            "scaffold_n_levels": int(scaffold_block["n_levels"]),
            "scaffold_level_fractions": list(scaffold_block["level_fractions"]),
            "scaffold_kwargs": dict(scaffold_block.get("kwargs", {})),
        }
    if condition == Method.GLOBAL_LINEAR_CONTINUOUS:
        return SamplerName.UNIFORM, dict(rc_block)
    return condition, {}
