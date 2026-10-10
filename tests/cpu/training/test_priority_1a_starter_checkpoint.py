"""CPU smoke for ``priority_1a_fetch_calibration._load_starter_policy``.

Saves a tiny :class:`PPOAgent` to a temp file, loads it through the
starter-policy helper, and verifies:

    - the helper returns a callable that maps a Fetch obs dict to an
      ndarray of the env's action dim
    - the returned actions are deterministic across calls (greedy
      decoding contract)
    - the full calibration pipeline runs end-to-end against the
      loaded policy without raising
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("gymnasium_robotics")

from pcrc.algorithms.ppo_continuous import PPOAgent, PPOContinuousConfig  # noqa: E402
from pcrc.envs.fetch_calibration import calibrate_fetch_bank  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import priority_1a_fetch_calibration as p1a  # noqa: E402, PLC0415

ENV_ID_REACH = "FetchReachDense-v4"


def _save_tiny_agent(tmp_path: Path, obs_dim: int = 10, action_dim: int = 4) -> Path:
    """Save a freshly-initialized PPOAgent for the Fetch Reach dims."""
    cfg = PPOContinuousConfig(
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=(8, 8),
        n_epochs=1,
        minibatch_size=4,
    )
    agent = PPOAgent(cfg)
    ckpt = tmp_path / "tiny_starter.pt"
    agent.save(ckpt)
    return ckpt


def test_load_starter_policy_returns_deterministic_callable(tmp_path: Path) -> None:
    ckpt = _save_tiny_agent(tmp_path)
    policy_fn = p1a._load_starter_policy(ckpt)
    obs_dict = {
        "observation": np.zeros(10, dtype=np.float32),
        "desired_goal": np.zeros(3, dtype=np.float32),
        "achieved_goal": np.zeros(3, dtype=np.float32),
    }
    a = policy_fn(obs_dict)
    b = policy_fn(obs_dict)
    assert a.shape == (4,)
    assert np.array_equal(a, b)


def test_load_starter_policy_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        p1a._load_starter_policy(tmp_path / "nonexistent.pt")


def test_load_starter_policy_rejects_non_ppo_checkpoint(tmp_path: Path) -> None:
    import torch  # noqa: PLC0415

    bad = tmp_path / "bad.pt"
    torch.save({"actor": "junk"}, bad)
    with pytest.raises(ValueError, match="missing the 'config'"):
        p1a._load_starter_policy(bad)


def test_calibration_runs_end_to_end_with_real_starter(tmp_path: Path) -> None:
    """Wire the starter-policy helper into ``calibrate_fetch_bank``.

    Verifies the same end-to-end path that the GPU mid-run will exercise:
    PPO checkpoint → starter policy → calibrate_fetch_bank → bank.
    """
    ckpt = _save_tiny_agent(tmp_path)
    policy_fn = p1a._load_starter_policy(ckpt)
    outcome = calibrate_fetch_bank(
        env_id=ENV_ID_REACH,
        n_pool=8,
        num_train=4,
        num_held_out=2,
        policy_fn=policy_fn,
        episodes_per_context=1,
        max_steps_per_episode=5,
        distance_bins=2,
        direction_bins=2,
        pose_bins=1,
        # Permissive band so the freshly-initialized PPO doesn't drop
        # everything as dead.
        target_success_band=(0.05, 0.70),
        dense_fallback_variance_threshold=1.0,
        pool_base_seed=10_000,
        split_seed=42,
    )
    assert len(outcome.bank.train_ids()) == 4
    assert len(outcome.bank.held_out_ids()) == 2
