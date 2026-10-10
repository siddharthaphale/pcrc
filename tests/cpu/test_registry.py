"""Public names must validate before a task starts and resolve into controller settings."""

import json

import pytest

from pcrc.registry import (
    METHODS,
    SCAFFOLD_METHODS,
    Method,
    SamplerName,
    ScaffoldSource,
    base_method,
    method_label,
    parse_conditions,
)
from pcrc.training.conditions import resolve_condition

SCAFFOLD = {"n_levels": 5, "level_fractions": [0, 0.25, 0.5, 0.75, 0.9], "kwargs": {}}


@pytest.mark.parametrize("method", sorted(SCAFFOLD_METHODS))
def test_registered_scaffolds_resolve_for_manual_and_trajectory(method):
    for prefix in ("", "trajectory_"):
        condition = prefix + method
        task = sorted(METHODS[method].tasks)[0]
        if task == "minigrid" and method == Method.FIXED:
            condition += "2"
        assert parse_conditions(condition, task) == [condition]
        sampler, overrides = resolve_condition(condition, SCAFFOLD, {})
        assert sampler == SamplerName.UNIFORM
        assert overrides["scaffold_mode"] == method
        assert overrides["scaffold_enabled"]
        if prefix:
            assert overrides["scaffold_source"] == ScaffoldSource.TRAJECTORY
        assert base_method(condition) == method
        assert method_label(condition)


@pytest.mark.parametrize(
    "condition,task",
    [
        ("unknown_method", "fetch"),
        ("uniform", "fetch"),
        ("true_start", "minigrid"),
        ("trajectory_unassisted", "fetch"),
        ("per_group_success", "pointmaze"),
        ("replay_openloop", "metaworld"),
        ("fixed", "minigrid"),
        ("", "fetch"),
    ],
)
def test_invalid_task_conditions_fail_early(condition, task):
    with pytest.raises(ValueError):
        parse_conditions(condition, task)


def test_ids_serialize_as_plain_strings():
    assert json.loads(json.dumps({"mode": Method.PER_CONTEXT_SUCCESS})) == {
        "mode": "per_context_success"
    }


def test_unassisted_uses_plain_uniform_sampler():
    assert resolve_condition("unassisted", SCAFFOLD, {}) == ("uniform", {})
