"""Multi-task Meta-World bank (the #5 conflict bank) — build_multi + per-context env switching.

The external-benchmark #5 test mixes push (wants LOW scaffold) + pick-place (wants HIGH/curriculum)
so a single global schedule cannot serve both. These tests pin the contracts that path relies on:
task-tagged contexts, per-task groups, picklability (workers receive the bank), and that
``reset(context_id)`` switches the underlying MuJoCo env + applies the object->goal scaffold per task.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

metaworld = pytest.importorskip("metaworld")

from pcrc.envs.metaworld_env import (  # noqa: E402
    MetaWorldTaskBank,
    MetaWorldTaskEnvironment,
    parse_metaworld_env_id,
)

TASKS = ["push-v3", "pick-place-v3"]


@pytest.fixture(scope="module")
def multi_bank() -> MetaWorldTaskBank:
    return MetaWorldTaskBank.build_multi(TASKS, n_train=2, n_held=1, seed=0)


def test_build_multi_tags_tasks_and_groups(multi_bank: MetaWorldTaskBank) -> None:
    assert multi_bank.tasks == sorted(TASKS)
    # 2 train + 1 held per task, global unique ids, each task in both splits.
    assert len(multi_bank.train_ids()) == 2 * len(TASKS)
    assert len(multi_bank.held_out_ids()) == 1 * len(TASKS)
    ids = list(multi_bank.contexts)
    assert len(ids) == len(set(ids))  # unique context ids across tasks
    # the loop requires train ids to be a contiguous 0..n_train-1 block (controller/oracle indexing)
    assert multi_bank.train_ids() == list(range(2 * len(TASKS)))
    by_task = {t: 0 for t in TASKS}
    for c in multi_bank.contexts.values():
        assert c.task in TASKS
        by_task[c.task] += 1
        # task_group is the task index (the per-context allocation group)
        assert c.task_group == TASKS.index(c.task)
    assert by_task == {t: 3 for t in TASKS}  # 2 train + 1 held each


def test_build_multi_is_picklable(multi_bank: MetaWorldTaskBank) -> None:
    # Workers receive the bank via ProcessPool initargs -> must round-trip through pickle.
    bank2 = pickle.loads(pickle.dumps(multi_bank))
    assert len(bank2) == len(multi_bank)
    assert bank2.tasks == multi_bank.tasks


def test_env_switches_task_and_scaffolds(multi_bank: MetaWorldTaskBank) -> None:
    task, opts = parse_metaworld_env_id("metaworld:push-v3+pick-place-v3:handnear")
    env = MetaWorldTaskEnvironment(multi_bank, task, **opts)
    try:
        assert set(env._envs) == set(TASKS)  # one underlying env per task
        push_c = next(c for c in multi_bank.train_ids() if multi_bank[c].task == "push-v3")
        pp_c = next(c for c in multi_bank.train_ids() if multi_bank[c].task == "pick-place-v3")

        # push goal is on the table; pick-place goal is in the air (the grasp/lift task).
        o_push = env.reset(push_c, start_progress=0.0)
        assert env._ctx.task == "push-v3"
        o_pp = env.reset(pp_c, start_progress=0.0)
        assert env._ctx.task == "pick-place-v3"
        assert o_pp["desired_goal"][2] > o_push["desired_goal"][2] + 0.05

        # α>0 scaffolds the object toward the goal; the hand tracks the SCAFFOLDED object (no reach
        # wall) — for pick-place that means the object is lifted off the table.
        for cid in (push_c, pp_c):
            o0 = env.reset(cid, start_progress=0.0)
            o9 = env.reset(cid, start_progress=0.9)
            d0 = np.linalg.norm(o0["achieved_goal"] - o0["desired_goal"])
            d9 = np.linalg.norm(o9["achieved_goal"] - o9["desired_goal"])
            assert d9 < d0  # high scaffold = object near goal
            tcp = np.asarray(env._env._get_obs()[:3])
            assert np.linalg.norm(tcp - o9["achieved_goal"]) < 0.15  # hand tracks scaffolded object
        # pick-place object is lifted into the air at high scaffold
        o_pp9 = env.reset(pp_c, start_progress=0.9)
        assert o_pp9["achieved_goal"][2] > 0.10

        # step returns the sparse-reward contract
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert r in (0.0, -1.0)
        assert "is_success" in info
        assert not term and not trunc
    finally:
        env.close()


def test_grasp_curriculum_starts_held_and_anneals() -> None:
    """The scripted-grasp reverse curriculum: α>0 starts pick-place GRASPED + held α-toward the goal;
    α=0 is the true ungrasped task (object on the table). Without this, pick-place is unlearnable at
    ℓ=0 (the teleported object just falls)."""
    task, opts = parse_metaworld_env_id("metaworld:pick-place-v3:handnear:graspcur")
    assert opts.get("grasp_curriculum") is True
    bank = MetaWorldTaskBank.build("pick-place-v3", n_train=2, n_held=1, seed=0)
    env = MetaWorldTaskEnvironment(bank, task, **opts)
    try:
        cid = bank.train_ids()[0]
        o0 = env.reset(cid, start_progress=0.0)
        o5 = env.reset(cid, start_progress=0.5)
        o9 = env.reset(cid, start_progress=0.9)
        # α=0: object on the table (ungrasped true start). α>0: lifted (held) and nearer the goal,
        # higher α => lifted higher (monotone).
        assert o0["achieved_goal"][2] < 0.05
        assert o5["achieved_goal"][2] > 0.05  # α=0.5 is held off the table
        assert o9["achieved_goal"][2] > o5["achieved_goal"][2] > o0["achieved_goal"][2]
        d = lambda o: float(np.linalg.norm(o["achieved_goal"] - o["desired_goal"]))
        assert d(o9) < d(o5) < d(o0)  # higher α => held closer to the goal
        # the grasp is stable: holding the gripper closed keeps the object up (no drop)
        env.reset(cid, start_progress=0.9)
        last = None
        for _ in range(20):
            last, _, _, _, _ = env.step(np.array([0, 0, 0, 1.0], dtype=np.float32))
        assert last["achieved_goal"][2] > 0.12
    finally:
        env.close()
