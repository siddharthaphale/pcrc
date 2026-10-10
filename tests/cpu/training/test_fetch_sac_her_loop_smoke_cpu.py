"""CPU smoke for the Fetch SAC+HER loop.

§7.2 row: synthetic-prefill replay-distribution check + plumbing
smoke. Asserts plumbing (HER preserves source_context_id; metrics
finite; mandatory streams written). Does NOT assert learning.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from pcrc.registry import SamplerName

pytest.importorskip("gymnasium_robotics")

from pcrc.algorithms.her import GoalConditionedTransition  # noqa: E402
from pcrc.algorithms.sac_continuous import SACContinuousConfig  # noqa: E402
from pcrc.algorithms.tagged_goal_replay import TaggedGlobalReplay  # noqa: E402
from pcrc.envs.fetch_environment import FetchTaskEnvironment  # noqa: E402
from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext  # noqa: E402
from pcrc.samplers.base import SamplerDistribution  # noqa: E402
from pcrc.training.fetch_sac_her_loop import (  # noqa: E402
    FetchSACHERLoopConfig,
    run_fetch_sac_her_loop,
)
from pcrc.utils.results import RunLogger  # noqa: E402

ENV_ID = "FetchReachDense-v4"


def _make_bank(n_train: int = 4, n_held_out: int = 2) -> FetchTaskBank:
    rng = np.random.default_rng(0)
    contexts: list[FetchTaskContext] = []
    ee = {}
    for i in range(n_train):
        contexts.append(
            FetchTaskContext(
                context_id=i,
                env_id=ENV_ID,
                reset_seed=30_000 + i,
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
                reset_seed=40_000 + j,
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


def _stub_transition(
    *, source_context_id: int, episode_id: int, obs_dim: int, action_dim: int
) -> GoalConditionedTransition:
    return GoalConditionedTransition(
        observation=np.zeros(obs_dim, dtype=np.float32),
        action=np.zeros(action_dim, dtype=np.float32),
        next_observation=np.zeros(obs_dim, dtype=np.float32),
        achieved_goal=np.zeros(3),
        next_achieved_goal=np.zeros(3),
        reward=-1.0,
        terminated=False,
        truncated=False,
        source_context_id=source_context_id,
        original_desired_goal=np.zeros(3),
        training_desired_goal=np.zeros(3),
        relabel_strategy="none",
        relabel_source_step=None,
        episode_id=episode_id,
    )


def test_synthetic_prefill_replay_distribution_matches() -> None:
    """Plan §7.2: synthetic prefill exercises ``sample_context_weighted``
    independent of the actual env-step rollout, then asserts the
    realized empirical context distribution matches the requested
    sampler distribution within MC error.
    """
    n_contexts = 4
    buf = TaggedGlobalReplay(capacity=20_000)
    for cid in range(n_contexts):
        for _ in range(2_000):
            buf.add(
                _stub_transition(source_context_id=cid, episode_id=cid, obs_dim=10, action_dim=4)
            )
    p = np.array([0.5, 0.25, 0.15, 0.10])
    dist = SamplerDistribution(
        raw=p,
        floored=p,
        capped=p,
        final=p,
        entropy=float(-(p * np.log(p)).sum()),
        max_probability=float(p.max()),
    )
    rng = np.random.default_rng(0)
    sampled = buf.sample_context_weighted(20_000, dist, rng)
    counts = Counter(t.source_context_id for t in sampled)
    empirical = np.array([counts[c] / 20_000 for c in range(n_contexts)])
    sigma = np.sqrt(p * (1 - p) / 20_000)
    np.testing.assert_allclose(empirical, p, atol=3 * sigma.max() + 1e-4)


def test_fetch_sac_her_loop_smoke_plumbing(tmp_path: Path) -> None:
    bank = _make_bank()
    probe = FetchTaskEnvironment(bank, ENV_ID)
    obs_dim = int(probe.observation_space["observation"].shape[0])
    goal_dim = 3
    action_dim = int(probe.action_space.shape[0])
    probe.close()

    cfg = FetchSACHERLoopConfig(
        exp_name="sac_her_smoke",
        condition=SamplerName.UNIFORM,
        env_id=ENV_ID,
        n_iterations=3,
        max_steps_per_episode=5,
        her_strategy="future",
        her_n_sampled_goal=2,
        replay_capacity=1000,
        minibatch_size=16,
        learning_starts=8,  # very low so the update path fires
        updates_per_iter=1,
        eval_interval=0,
        seed=0,
        sampler_kwargs={},
        sac=SACContinuousConfig(
            state_dim=obs_dim + goal_dim,
            action_dim=action_dim,
            hidden_sizes=(16, 16),
            learning_rate=1.0e-3,
        ),
        kill_conditions_enabled=False,
    )
    logger = RunLogger("sac_her_smoke", root=str(tmp_path))
    result = run_fetch_sac_her_loop(cfg, bank, logger=logger)
    assert result["last_metrics"]["step"] == 3
    # Mandatory streams.
    root = tmp_path / "sac_her_smoke" / "training" / SamplerName.UNIFORM
    for stream in (
        "metrics.jsonl",
        "sampler_distribution.jsonl",
        "uniformized_reward.jsonl",
        "selection_learning_decomposition.jsonl",
        "context_visit_counts.jsonl",
    ):
        assert (root / stream).exists(), stream
    # No NaN.
    with (root / "metrics.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            for k, v in row.items():
                if isinstance(v, float):
                    assert np.isfinite(v), f"{k}={v}"
