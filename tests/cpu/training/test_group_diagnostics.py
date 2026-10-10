import pytest

from pcrc.training.group_diagnostics import classify_groups


def _make(task_reward_pairs):
    """task_reward_pairs: list[(task_id, reward)]."""
    return [{"task_id": tid, "reward": r} for tid, r in task_reward_pairs]


def test_empty_input_returns_zeros_and_all_dead():
    s = classify_groups([])
    assert s.n_total == 0 and s.n_mixed == 0
    assert s.n_dead == 0 and s.n_saturated == 0
    assert s.all_dead is True
    assert s.per_task_success == {}


def test_single_mixed_group():
    rollouts = _make([(3, 0.0), (3, 1.0), (3, 0.0), (3, 1.0)])
    s = classify_groups(rollouts)
    assert s.n_total == 1
    assert s.n_mixed == 1
    assert s.n_dead == 0 and s.n_saturated == 0
    assert s.per_task_success[3] == 0.5
    assert s.global_mean == 0.5
    assert s.all_dead is False


def test_all_dead_iter():
    rollouts = _make([(0, 0.0), (0, 0.0), (1, 0.0), (1, 0.0)])
    s = classify_groups(rollouts)
    assert s.n_dead == 2 and s.n_mixed == 0
    assert s.global_mean == 0.0
    assert s.all_dead is True


def test_mixed_dead_saturated_combo():
    rollouts = _make(
        [
            (0, 0.0),
            (0, 0.0),
            (1, 0.0),
            (1, 1.0),
            (2, 1.0),
            (2, 1.0),
        ]
    )
    s = classify_groups(rollouts)
    assert (s.n_dead, s.n_mixed, s.n_saturated) == (1, 1, 1)
    assert s.per_task_success == {0: 0.0, 1: 0.5, 2: 1.0}
    assert s.global_mean == pytest.approx(3 / 6)
    assert s.all_dead is False


def test_g1_group_is_never_mixed():
    s = classify_groups(_make([(0, 1.0)]))
    assert s.n_saturated == 1 and s.n_mixed == 0
    s = classify_groups(_make([(0, 0.0)]))
    assert s.n_dead == 1 and s.n_mixed == 0


def test_nonuniform_g_across_tasks():
    rollouts = _make(
        [
            (0, 1.0),
            (0, 0.0),
            (1, 0.0),
            (1, 0.0),
            (1, 0.0),
            (1, 0.0),
        ]
    )
    s = classify_groups(rollouts)
    assert s.n_mixed == 1 and s.n_dead == 1
    assert s.per_task_success[0] == 0.5
    assert s.per_task_success[1] == 0.0


def test_all_saturated_is_not_all_dead():
    rollouts = _make([(0, 1.0), (0, 1.0), (1, 1.0), (1, 1.0)])
    s = classify_groups(rollouts)
    assert s.n_saturated == 2
    assert s.n_dead == 0
    assert s.n_mixed == 0
    assert s.global_mean == 1.0
    assert s.all_dead is False


def test_per_task_success_keys_are_strings_in_log_dict():
    s = classify_groups(_make([(2, 1.0), (2, 0.0)]))
    log = s.to_log_dict()
    assert all(isinstance(k, str) for k in log["per_task_success"])
    assert all(isinstance(k, int) for k in s.per_task_success)


def test_log_dict_keys_and_filter_rate():
    rollouts = _make(
        [
            (0, 0.0),
            (0, 0.0),
            (1, 0.0),
            (1, 1.0),
            (2, 1.0),
            (2, 1.0),
        ]
    )
    s = classify_groups(rollouts)
    log = s.to_log_dict()
    expected = {
        "groups/n_total",
        "groups/n_mixed",
        "groups/n_dead",
        "groups/n_saturated",
        "groups/filter_rate",
        "groups/global_mean",
        "groups/all_dead",
        "per_task_success",
    }
    assert expected == set(log.keys())
    assert log["groups/filter_rate"] == pytest.approx(2 / 3)
