"""CPU integration smoke for ``pcrc.training.fetch_ppo_loop.run_fetch_ppo_loop``.

§7.2 row: tiny end-to-end smoke on FetchReachDense-v4 with tabular AC
+ KillConditionMonitor. Exercises plumbing, not learning. Reduced
budget vs the plan's nominal 10-update target to stay under the 60-s
``make test-cpu-fetch`` budget envelope.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pcrc.registry import SamplerName

pytest.importorskip("gymnasium_robotics")

from pcrc.algorithms.ppo_continuous import PPOContinuousConfig  # noqa: E402
from pcrc.envs.fetch_environment import (  # noqa: E402
    FetchTaskEnvironment,
    goal_conditioned_observation_dim,
)
from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext  # noqa: E402
from pcrc.training.fetch_ppo_loop import (  # noqa: E402
    FetchPPOLoopConfig,
    build_sampler_for_condition,
    run_fetch_ppo_loop,
)
from pcrc.utils.results import RunLogger  # noqa: E402

ENV_ID = "FetchReachDense-v4"


def _build_bank(n_train: int = 3, n_held_out: int = 2) -> FetchTaskBank:
    rng = np.random.default_rng(0)
    contexts: list[FetchTaskContext] = []
    ee = {}
    for i in range(n_train):
        contexts.append(
            FetchTaskContext(
                context_id=i,
                env_id=ENV_ID,
                reset_seed=10_000 + i,
                desired_goal=rng.uniform(low=[1.1, 0.5, 0.4], high=[1.5, 1.0, 0.7]),
                reset_payload=None,
                initial_object_pose=None,
                distance_bin=i % 2,
                direction_bin=i % 2,
                pose_bin=0,
                split="train",
            )
        )
        ee[i] = rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6])
    for j in range(n_held_out):
        cid = n_train + j
        contexts.append(
            FetchTaskContext(
                context_id=cid,
                env_id=ENV_ID,
                reset_seed=20_000 + j,
                desired_goal=rng.uniform(low=[1.1, 0.5, 0.4], high=[1.5, 1.0, 0.7]),
                reset_payload=None,
                initial_object_pose=None,
                distance_bin=j % 2,
                direction_bin=j % 2,
                pose_bin=0,
                split="held_out",
            )
        )
        ee[cid] = rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6])
    return FetchTaskBank.from_contexts(contexts, ee)


def _build_ppo_cfg(obs_dim: int, action_dim: int) -> PPOContinuousConfig:
    return PPOContinuousConfig(
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=(8, 8),
        n_epochs=2,
        minibatch_size=8,
        learning_rate=1.0e-3,
    )


def test_build_sampler_factory_handles_all_conditions() -> None:
    bank = _build_bank()
    for condition in (
        SamplerName.UNIFORM,
        SamplerName.REWARD_GREEDY,
        SamplerName.HEADROOM,
        SamplerName.LEARNING_PROGRESS,
        SamplerName.TABULAR_ACTOR_CURATOR,
        SamplerName.MLP_ACTOR_CURATOR,
        SamplerName.ACDC_LITE,
    ):
        kwargs: dict = {}
        if condition == SamplerName.REWARD_GREEDY:
            kwargs["alpha"] = 1.0
        if condition == SamplerName.HEADROOM:
            kwargs["alpha"] = 1.0
        if condition == SamplerName.LEARNING_PROGRESS:
            kwargs["alpha"] = 1.0
        if condition == SamplerName.TABULAR_ACTOR_CURATOR:
            kwargs["eta"] = 1.0
        if condition == SamplerName.MLP_ACTOR_CURATOR:
            kwargs["alpha"] = 1.0
            kwargs["hidden_sizes"] = (4,)
        if condition == SamplerName.ACDC_LITE:
            kwargs["sigma_quality"] = 0.2
        sampler = build_sampler_for_condition(
            condition=condition,
            bank=bank,
            epsilon=0.05,
            q_max=None,
            sampler_kwargs=kwargs,
        )
        assert sampler.n_contexts == len(bank.train_ids())


def test_fetch_ppo_loop_smoke_with_tabular_ac(tmp_path: Path) -> None:
    bank = _build_bank()
    probe = FetchTaskEnvironment(bank, ENV_ID)
    obs_dim = goal_conditioned_observation_dim(probe.observation_space)
    action_dim = int(probe.action_space.shape[0])
    probe.close()

    cfg = FetchPPOLoopConfig(
        exp_name="ppo_smoke",
        condition=SamplerName.TABULAR_ACTOR_CURATOR,
        env_id=ENV_ID,
        n_iterations=2,
        episodes_per_iteration=2,
        max_steps_per_episode=5,
        eval_interval=2,
        n_eval_episodes_per_context=1,
        seed=0,
        sampler_kwargs={"eta": 1.0},
        ppo=_build_ppo_cfg(obs_dim, action_dim),
        kill_conditions_enabled=False,  # avoid the offline_ev / etc. machinery
    )
    eval_env = FetchTaskEnvironment(bank, ENV_ID)
    logger = RunLogger("ppo_smoke", root=str(tmp_path))
    try:
        result = run_fetch_ppo_loop(cfg, bank, eval_env=eval_env, logger=logger)
    finally:
        eval_env.close()
    assert "last_metrics" in result
    # Mandatory streams written.
    stream_root = tmp_path / "ppo_smoke" / "training" / SamplerName.TABULAR_ACTOR_CURATOR
    for stream in (
        "metrics.jsonl",
        "sampler_distribution.jsonl",
        "uniformized_reward.jsonl",
        "selection_learning_decomposition.jsonl",
        "context_visit_counts.jsonl",
    ):
        assert (stream_root / stream).exists(), stream
    # No NaN in the metrics rows.
    with (stream_root / "metrics.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            for k, v in row.items():
                if isinstance(v, float):
                    assert np.isfinite(v), f"{k}={v}"


def test_fetch_ppo_loop_smoke_with_uniform(tmp_path: Path) -> None:
    bank = _build_bank()
    probe = FetchTaskEnvironment(bank, ENV_ID)
    obs_dim = goal_conditioned_observation_dim(probe.observation_space)
    action_dim = int(probe.action_space.shape[0])
    probe.close()

    cfg = FetchPPOLoopConfig(
        exp_name="ppo_uniform_smoke",
        condition=SamplerName.UNIFORM,
        env_id=ENV_ID,
        n_iterations=2,
        episodes_per_iteration=2,
        max_steps_per_episode=5,
        eval_interval=0,  # no eval to keep the smoke tight
        seed=0,
        sampler_kwargs={},
        ppo=_build_ppo_cfg(obs_dim, action_dim),
        kill_conditions_enabled=False,
    )
    logger = RunLogger("ppo_uniform_smoke", root=str(tmp_path))
    result = run_fetch_ppo_loop(cfg, bank, eval_env=None, logger=logger)
    assert result["last_metrics"]["step"] == 2
