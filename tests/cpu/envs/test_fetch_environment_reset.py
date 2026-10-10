"""Tests for ``pcrc.envs.fetch_environment.FetchTaskEnvironment``.

§7.1 row "Env reset determinism" — the Priority 1a hard gate:
    - same ``context_id`` → reproducible observable simulator state
      across two ``env.reset(context_id)`` calls
    - Push / PickAndPlace covered (not only Reach), so the object
      state is asserted reproducible, not only the EE
    - ``env.step(zero_action)`` from the reset state is deterministic
    - reward computed by ``env.step(...)`` uses the bank's stored
      ``desired_goal``, not the env's freshly-sampled goal

Module-level ``pytest.importorskip("gymnasium_robotics")`` so plain
``make test-cpu`` (no robotics extra) skips this file cleanly;
``make test-cpu-fetch`` runs it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium_robotics")

from pcrc.envs.fetch_environment import FetchTaskEnvironment  # noqa: E402
from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext  # noqa: E402

ENV_ID_REACH = "FetchReachDense-v4"
ENV_ID_PUSH = "FetchPushDense-v4"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _reach_bank() -> FetchTaskBank:
    """Tiny reach bank: 3 train contexts with distinct goals + seeds."""
    rng = np.random.default_rng(0)
    contexts = []
    ee: dict[int, np.ndarray] = {}
    for i in range(3):
        contexts.append(
            FetchTaskContext(
                context_id=i,
                env_id=ENV_ID_REACH,
                reset_seed=1000 + i,
                desired_goal=rng.uniform(low=[1.1, 0.5, 0.4], high=[1.5, 1.0, 0.7]),
                reset_payload=None,
                initial_object_pose=None,
                distance_bin=i,
                direction_bin=0,
                pose_bin=0,
                split="train",
            )
        )
        ee[i] = rng.uniform(low=[1.2, 0.5, 0.4], high=[1.4, 0.9, 0.6])
    return FetchTaskBank.from_contexts(contexts, ee)


def _push_bank() -> FetchTaskBank:
    """Tiny push bank: 3 train contexts. Reset payload is constructed
    from a probe rollout below so the saved qpos / qvel actually match
    a reachable env state (rather than fabricated numbers that would
    diverge under ``mj_forward``).
    """
    import gymnasium as gym  # noqa: PLC0415
    import gymnasium_robotics  # noqa: PLC0415

    gym.register_envs(gymnasium_robotics)
    probe = gym.make(ENV_ID_PUSH)
    rng = np.random.default_rng(1)
    contexts = []
    ee: dict[int, np.ndarray] = {}
    for i in range(3):
        seed = 2000 + i
        obs, _ = probe.reset(seed=seed)
        # Capture the env's own initial qpos / qvel + object pose for
        # this seed — that is the deterministic ground truth.
        init_qpos = probe.unwrapped.data.qpos.copy()
        init_qvel = probe.unwrapped.data.qvel.copy()
        # achieved_goal on Push is the object position (per
        # gymnasium-robotics convention) — same as the object pose for
        # diagnostics. Override desired_goal so we can verify the
        # bank's value is what drives reward.
        contexts.append(
            FetchTaskContext(
                context_id=i,
                env_id=ENV_ID_PUSH,
                reset_seed=seed,
                desired_goal=rng.uniform(low=[1.1, 0.5, 0.42], high=[1.5, 1.0, 0.42]),
                reset_payload={"init_qpos": init_qpos, "init_qvel": init_qvel},
                initial_object_pose=np.asarray(obs["achieved_goal"]),
                distance_bin=i,
                direction_bin=0,
                pose_bin=i,
                split="train",
            )
        )
        ee[i] = np.asarray(obs["achieved_goal"])
    probe.close()
    return FetchTaskBank.from_contexts(contexts, ee)


# ---------------------------------------------------------------------------
# Reach: same context_id → reproducible reset
# ---------------------------------------------------------------------------


def test_reach_reset_reproducible_across_calls() -> None:
    bank = _reach_bank()
    with FetchTaskEnvironment(bank, ENV_ID_REACH) as env:
        cid = bank.train_ids()[1]
        obs_a = env.reset(cid)
        obs_b = env.reset(cid)
        for k in ("observation", "desired_goal", "achieved_goal"):
            assert np.allclose(obs_a[k], obs_b[k]), k
        assert obs_a["source_context_id"] == obs_b["source_context_id"] == cid


def test_reach_reset_desired_goal_matches_bank() -> None:
    bank = _reach_bank()
    with FetchTaskEnvironment(bank, ENV_ID_REACH) as env:
        cid = bank.train_ids()[0]
        obs = env.reset(cid)
        assert np.allclose(obs["desired_goal"], bank[cid].desired_goal)


def test_reach_zero_action_step_deterministic() -> None:
    bank = _reach_bank()
    with FetchTaskEnvironment(bank, ENV_ID_REACH) as env:
        cid = bank.train_ids()[0]
        env.reset(cid)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        obs_a, r_a, *_ = env.step(a)
        env.reset(cid)
        obs_b, r_b, *_ = env.step(a)
        for k in ("observation", "desired_goal", "achieved_goal"):
            assert np.allclose(obs_a[k], obs_b[k]), k
        assert r_a == r_b


# ---------------------------------------------------------------------------
# Push: same context_id → reproducible reset INCLUDING object state
# ---------------------------------------------------------------------------


def test_push_reset_object_state_reproducible() -> None:
    bank = _push_bank()
    with FetchTaskEnvironment(bank, ENV_ID_PUSH) as env:
        cid = bank.train_ids()[1]
        obs_a = env.reset(cid)
        obs_b = env.reset(cid)
        # achieved_goal is the object pose on Push — assert object state
        # reproducible (not only the EE or the goal).
        assert np.allclose(obs_a["achieved_goal"], obs_b["achieved_goal"])
        # And the full observation block (joints + EE + object pose +
        # object velocity + ...) matches too.
        assert np.allclose(obs_a["observation"], obs_b["observation"])
        assert np.allclose(obs_a["desired_goal"], obs_b["desired_goal"])


def test_push_reset_desired_goal_matches_bank() -> None:
    bank = _push_bank()
    with FetchTaskEnvironment(bank, ENV_ID_PUSH) as env:
        for cid in bank.train_ids():
            obs = env.reset(cid)
            assert np.allclose(obs["desired_goal"], bank[cid].desired_goal), cid


def test_push_zero_action_step_deterministic() -> None:
    bank = _push_bank()
    with FetchTaskEnvironment(bank, ENV_ID_PUSH) as env:
        cid = bank.train_ids()[0]
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        env.reset(cid)
        obs_a, r_a, *_ = env.step(a)
        env.reset(cid)
        obs_b, r_b, *_ = env.step(a)
        assert np.allclose(obs_a["observation"], obs_b["observation"])
        assert np.allclose(obs_a["achieved_goal"], obs_b["achieved_goal"])
        assert r_a == r_b


# ---------------------------------------------------------------------------
# Reward computed by env uses the BANK's desired_goal
# ---------------------------------------------------------------------------


def test_reward_uses_bank_desired_goal() -> None:
    """The env's own step() reward path reads ``self.goal``; the wrapper
    sets it to the bank's ``desired_goal`` post-reset. If reward had
    been left referring to the env's freshly-sampled goal, two
    contexts with the same seed but different bank goals would
    produce the same reward — which we assert it does NOT.
    """
    import gymnasium as gym  # noqa: PLC0415
    import gymnasium_robotics  # noqa: PLC0415

    gym.register_envs(gymnasium_robotics)
    seed = 4242
    # Two contexts sharing the same reset_seed (so the env's freshly
    # sampled goal would be identical for both) but different bank
    # desired_goals; if reward references the bank goal, the per-step
    # reward must differ.
    goal_a = np.array([1.20, 0.55, 0.45])  # close to default EE
    goal_b = np.array([1.55, 1.00, 0.70])  # far
    contexts = [
        FetchTaskContext(
            context_id=0,
            env_id=ENV_ID_REACH,
            reset_seed=seed,
            desired_goal=goal_a,
            reset_payload=None,
            initial_object_pose=None,
            distance_bin=0,
            direction_bin=0,
            pose_bin=0,
            split="train",
        ),
        FetchTaskContext(
            context_id=1,
            env_id=ENV_ID_REACH,
            reset_seed=seed,
            desired_goal=goal_b,
            reset_payload=None,
            initial_object_pose=None,
            distance_bin=0,
            direction_bin=0,
            pose_bin=0,
            split="train",
        ),
    ]
    ee = {0: np.array([1.3, 0.7, 0.5]), 1: np.array([1.3, 0.7, 0.5])}
    bank = FetchTaskBank.from_contexts(contexts, ee)
    with FetchTaskEnvironment(bank, ENV_ID_REACH) as env:
        env.reset(0)
        _, r_a, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        env.reset(1)
        _, r_b, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    # Reach dense reward is negative L2 distance to goal — must differ
    # by far more than floating-point noise across these two distinct
    # goals.
    assert abs(r_a - r_b) > 0.1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_wrong_env_id_for_bank_rejected() -> None:
    bank = _reach_bank()
    with pytest.raises(ValueError, match="does not match"):
        FetchTaskEnvironment(bank, ENV_ID_PUSH)


def test_context_env_id_mismatch_rejected_on_reset() -> None:
    # Build a single-env bank but try to load a context whose env_id was
    # mutated post-construction. This is a defensive check on the
    # wrapper rather than something the bank would normally allow.
    bank = _reach_bank()
    cid = bank.train_ids()[0]
    bad = FetchTaskContext(
        context_id=cid,
        env_id=ENV_ID_PUSH,  # mismatched
        reset_seed=bank[cid].reset_seed,
        desired_goal=bank[cid].desired_goal,
        reset_payload=None,
        initial_object_pose=None,
        distance_bin=bank[cid].distance_bin,
        direction_bin=bank[cid].direction_bin,
        pose_bin=bank[cid].pose_bin,
        split=bank[cid].split,
    )
    bank.contexts[cid] = bad  # bypass save / load validation
    with (
        FetchTaskEnvironment(bank, ENV_ID_REACH) as env,
        pytest.raises(ValueError, match="env_id"),
    ):
        env.reset(cid)


def test_current_context_id_before_reset_raises() -> None:
    bank = _reach_bank()
    with (
        FetchTaskEnvironment(bank, ENV_ID_REACH) as env,
        pytest.raises(RuntimeError, match="reset"),
    ):
        env.current_context_id()


def test_current_context_id_after_reset() -> None:
    bank = _reach_bank()
    with FetchTaskEnvironment(bank, ENV_ID_REACH) as env:
        cid = bank.train_ids()[2]
        env.reset(cid)
        assert env.current_context_id() == cid
