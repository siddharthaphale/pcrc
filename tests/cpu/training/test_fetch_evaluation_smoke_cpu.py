"""CPU smoke for ``pcrc.training.fetch_evaluation.evaluate_fetch``.

Asserts on a tiny held-out split: returns finite metrics, per-bin /
per-context dicts have the expected key sets.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.registry import Method

pytest.importorskip("gymnasium_robotics")

from pcrc.envs.fetch_environment import FetchTaskEnvironment  # noqa: E402
from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext  # noqa: E402
from pcrc.training.fetch_evaluation import evaluate_fetch  # noqa: E402

ENV_ID = "FetchReachDense-v4"


def _make_bank(n_train: int = 3, n_held_out: int = 3) -> FetchTaskBank:
    rng = np.random.default_rng(0)
    contexts: list[FetchTaskContext] = []
    ee = {}
    for i in range(n_train):
        contexts.append(
            FetchTaskContext(
                context_id=i,
                env_id=ENV_ID,
                reset_seed=1000 + i,
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
                reset_seed=2000 + j,
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


def _zero_actor(obs: np.ndarray) -> np.ndarray:
    del obs
    return np.zeros(4, dtype=np.float32)


def test_evaluate_fetch_returns_expected_keys() -> None:
    bank = _make_bank()
    with FetchTaskEnvironment(bank, ENV_ID) as env:
        result = evaluate_fetch(
            env=env,
            actor_fn=_zero_actor,
            bank=bank,
            n_eval_episodes_per_context=2,
            max_steps_per_episode=5,
        )
    assert set(result.keys()) >= {
        "success_rate",
        "mean_dense_return",
        Method.PER_CONTEXT_SUCCESS,
        "per_bin_success",
    }
    assert np.isfinite(result["success_rate"])
    assert np.isfinite(result["mean_dense_return"])
    assert isinstance(result[Method.PER_CONTEXT_SUCCESS], dict)
    assert set(result[Method.PER_CONTEXT_SUCCESS].keys()) == set(bank.held_out_ids())
    assert "distance" in result["per_bin_success"]
    assert "direction" in result["per_bin_success"]
    assert "pose" in result["per_bin_success"]


def test_evaluate_fetch_distribution_diagnostics() -> None:
    from pcrc.samplers.uniform import UniformSampler  # noqa: PLC0415

    bank = _make_bank()
    sampler = UniformSampler(3)
    dist = sampler.distribution(step=0)
    with FetchTaskEnvironment(bank, ENV_ID) as env:
        result = evaluate_fetch(
            env=env,
            actor_fn=_zero_actor,
            bank=bank,
            n_eval_episodes_per_context=1,
            max_steps_per_episode=3,
            distribution=dist,
        )
    assert "sampler_entropy" in result
    assert "sampler_max_probability" in result


def test_evaluate_fetch_rejects_empty_held_out() -> None:
    bank = _make_bank(n_train=3, n_held_out=0)
    with (
        FetchTaskEnvironment(bank, ENV_ID) as env,
        pytest.raises(ValueError, match="held-out split is empty"),
    ):
        evaluate_fetch(
            env=env,
            actor_fn=_zero_actor,
            bank=bank,
            n_eval_episodes_per_context=1,
            max_steps_per_episode=1,
        )
