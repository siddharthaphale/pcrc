"""Tests for the PCRC-Nav condition → loop-config bundle mapping."""

from __future__ import annotations

import pytest

from pcrc.registry import TRAJECTORY_PREFIX, Method, SamplerName, ScaffoldSource
from pcrc.training.conditions import PCRC_SCAFFOLD_MODES, resolve_condition

SCAFFOLD_BLOCK = {
    "n_levels": 5,
    "level_fractions": [0.0, 0.25, 0.5, 0.75, 0.9],
    "kwargs": {"n_min": 15, "tau_hi": 0.8},
}
RC_BLOCK = {
    "reverse_curriculum": True,
    "reverse_curriculum_start_progress": 0.9,
    "reverse_curriculum_decay_iters": 1500,
}


@pytest.mark.parametrize("mode", PCRC_SCAFFOLD_MODES)
def test_pcrc_modes_enable_scaffold_with_uniform_sampler(mode: str) -> None:
    loop_condition, ov = resolve_condition(mode, SCAFFOLD_BLOCK, RC_BLOCK)
    assert loop_condition == SamplerName.UNIFORM  # controller drives selection
    assert ov["scaffold_enabled"] is True
    assert ov["scaffold_mode"] == mode
    assert ov["scaffold_n_levels"] == 5
    assert ov["scaffold_level_fractions"] == [0.0, 0.25, 0.5, 0.75, 0.9]
    assert ov["scaffold_kwargs"]["n_min"] == 15
    # mode must NOT be inside kwargs (would collide with ScaffoldConfig(mode=...)).
    assert "mode" not in ov["scaffold_kwargs"]


def test_pcrc_kwargs_are_copies_not_aliases() -> None:
    _, ov = resolve_condition(Method.PER_CONTEXT_UNCERTAINTY_GATED, SCAFFOLD_BLOCK, RC_BLOCK)
    ov["scaffold_kwargs"]["n_min"] = 999
    ov["scaffold_level_fractions"].append(1.0)
    assert SCAFFOLD_BLOCK["kwargs"]["n_min"] == 15  # source untouched
    assert SCAFFOLD_BLOCK["level_fractions"] == [0.0, 0.25, 0.5, 0.75, 0.9]


def test_global_linear_continuous_uses_continuous_reverse_curriculum() -> None:
    loop_condition, ov = resolve_condition(Method.GLOBAL_LINEAR_CONTINUOUS, SCAFFOLD_BLOCK, RC_BLOCK)
    assert loop_condition == SamplerName.UNIFORM
    assert ov["reverse_curriculum"] is True
    assert ov["reverse_curriculum_start_progress"] == 0.9
    assert "scaffold_enabled" not in ov  # not the discrete controller


def test_uniform_is_passthrough_no_overrides() -> None:
    loop_condition, ov = resolve_condition(SamplerName.UNIFORM, SCAFFOLD_BLOCK, RC_BLOCK)
    assert loop_condition == SamplerName.UNIFORM
    assert ov == {}


def test_plain_sampler_name_passthrough() -> None:
    # A non-PCRC, non-rc name maps to itself as a sampler condition.
    loop_condition, ov = resolve_condition(SamplerName.HEADROOM, SCAFFOLD_BLOCK, RC_BLOCK)
    assert loop_condition == SamplerName.HEADROOM
    assert ov == {}


def test_mixture_condition_parses_probs() -> None:
    # mix_0:0.5_2:0.5 → mode "mixture" with a length-n_levels prob vector.
    loop_condition, ov = resolve_condition("mixture_0:0.5_2:0.5", SCAFFOLD_BLOCK, RC_BLOCK)
    assert loop_condition == SamplerName.UNIFORM
    assert ov["scaffold_mode"] == Method.MIXTURE
    assert ov["scaffold_kwargs"]["mixture_probs"] == [0.5, 0.0, 0.5, 0.0, 0.0]
    assert "mode" not in ov["scaffold_kwargs"]


def test_mixture_single_level_spec() -> None:
    _, ov = resolve_condition("mixture_1:1.0", SCAFFOLD_BLOCK, RC_BLOCK)
    assert ov["scaffold_kwargs"]["mixture_probs"] == [0.0, 1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("bad", ["mixture_9:1.0", "mixture_0", "mixture_0:0.5_7:0.5"])
def test_mixture_bad_spec_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        resolve_condition(bad, SCAFFOLD_BLOCK, RC_BLOCK)


def test_global_anneal_rate_condition_sets_anneal_iters() -> None:
    # global_anneal<N> → mode "global_anneal" with anneal_iters overridden to N (rate sweep).
    loop_condition, ov = resolve_condition("global_anneal600", SCAFFOLD_BLOCK, RC_BLOCK)
    assert loop_condition == SamplerName.UNIFORM
    assert ov["scaffold_mode"] == Method.GLOBAL_ANNEAL
    assert ov["scaffold_kwargs"]["anneal_iters"] == 600


def test_global_anneal_bare_unchanged() -> None:
    # bare "global_anneal" still routes through the generic PCRC-mode branch (no rate override).
    _, ov = resolve_condition(Method.GLOBAL_ANNEAL, SCAFFOLD_BLOCK, RC_BLOCK)
    assert ov["scaffold_mode"] == Method.GLOBAL_ANNEAL


def test_auto_prefix_matches_base_plus_trajectory_source() -> None:
    # trajectory_<mode> resolves to the SAME controller config as <mode>, only adding the env-level
    # scaffold_source="trajectory" (§2.3). Prove the controller config is byte-identical.
    base_lc, base_ov = resolve_condition(Method.PER_CONTEXT_SUCCESS, SCAFFOLD_BLOCK, RC_BLOCK)
    auto_lc, auto_ov = resolve_condition((TRAJECTORY_PREFIX + Method.PER_CONTEXT_SUCCESS), SCAFFOLD_BLOCK, RC_BLOCK)
    assert auto_lc == base_lc
    assert auto_ov.get("scaffold_source") == ScaffoldSource.TRAJECTORY
    assert {k: v for k, v in auto_ov.items() if k != "scaffold_source"} == base_ov


def test_auto_prefix_works_for_baselines() -> None:
    _, ov = resolve_condition((TRAJECTORY_PREFIX + Method.GLOBAL_ANNEAL), SCAFFOLD_BLOCK, RC_BLOCK)
    assert ov["scaffold_mode"] == Method.GLOBAL_ANNEAL and ov["scaffold_source"] == ScaffoldSource.TRAJECTORY
    _, ov = resolve_condition((TRAJECTORY_PREFIX + Method.RANDOM_SCAFFOLD), SCAFFOLD_BLOCK, RC_BLOCK)
    assert ov["scaffold_mode"] == Method.RANDOM_SCAFFOLD and ov["scaffold_source"] == ScaffoldSource.TRAJECTORY


def test_auto_prefix_no_op_when_scaffold_disabled() -> None:
    # trajectory_unassisted: the base enables no controller, so there is no ladder to restore -> stay manual.
    _, ov = resolve_condition((TRAJECTORY_PREFIX + Method.UNASSISTED), SCAFFOLD_BLOCK, RC_BLOCK)
    assert "scaffold_source" not in ov and ov == {}


def test_oracle_condition_sets_mode_only() -> None:
    # resolve only sets the mode; the driver injects scaffold_oracle_levels from the bank.
    loop_condition, ov = resolve_condition(Method.ORACLE, SCAFFOLD_BLOCK, RC_BLOCK)
    assert loop_condition == SamplerName.UNIFORM
    assert ov["scaffold_mode"] == Method.ORACLE
    assert ov["scaffold_enabled"] is True
    assert "scaffold_oracle_levels" not in ov  # driver's job, not resolve_condition's


def test_auto_prefix_works_for_upgrade_modes() -> None:
    # RLC-upgrade controllers ride the same auto_ trajectory-source path as the incumbent.
    for mode in (Method.PER_CONTEXT_WILSON_CONFIDENCE, Method.PER_CONTEXT_BETA_POSTERIOR):
        _, ov = resolve_condition(f"{TRAJECTORY_PREFIX}{mode}", SCAFFOLD_BLOCK, RC_BLOCK)
        assert ov["scaffold_mode"] == mode and ov["scaffold_source"] == ScaffoldSource.TRAJECTORY
