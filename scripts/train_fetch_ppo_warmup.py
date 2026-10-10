#!/usr/bin/env python3
"""Train a short uniform-PPO warmup → shared starter checkpoint.

Stage A1 mid-run, Fetch-side prerequisite. Priority 1a calibration
(``scripts/priority_1a_fetch_calibration.py``) requires a starter policy;
the resolved decision (plan §9 risk #1, handover "Resolved decisions" #2)
is a short uniform PPO warmup of 10k-25k env steps, saved as one shared
``PPOAgent.save`` artifact per env id that every Stage D condition then
loads.

Route (a) from the handover "PPO warmup design choice" block: build a
throwaway bank of diverse Fetch contexts via ``_probe_pool`` (probe phase
only, no policy rollout), wrap with ``FetchTaskBank.from_contexts``, and
drive the existing ``run_fetch_ppo_loop`` under the Uniform sampler. The
Uniform sampler spreads rollouts evenly across the throwaway contexts →
diverse-seed coverage. The bank does not need calibration-quality
contexts here; it only provides seed diversity during warmup.

Usage::

    uv run python scripts/train_fetch_ppo_warmup.py \\
        --config config/fetch_priority_1a.yaml \\
        --env reach_dense

    # Saves <CHECKPOINTS_ROOT>/fetch_starter_FetchReachDense-v4.pt
    # Repeat for --env push_dense and --env push_sparse.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pcrc.algorithms.ppo_continuous import PPOAgent, PPOContinuousConfig
from pcrc.envs.fetch_calibration import _probe_pool
from pcrc.envs.fetch_environment import goal_conditioned_observation_dim
from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext
from pcrc.registry import SamplerName
from pcrc.training.fetch_ppo_loop import FetchPPOLoopConfig, run_fetch_ppo_loop
from pcrc.utils.paths import CHECKPOINTS_ROOT


def _resolve_env_id(cfg: dict[str, Any], cli_env_key: str | None) -> str:
    """Pick the env id from config; CLI ``--env`` key override wins.

    Mirrors ``scripts/priority_1a_fetch_calibration.py`` so the warmup
    and calibration always resolve the same id from the same YAML.
    """
    env_ids = cfg.get("env_ids", {})
    key = cli_env_key or cfg.get("default_env")
    if key is None:
        raise KeyError("config has no `default_env` and --env not passed")
    if key not in env_ids:
        raise KeyError(f"env key {key!r} not in config.env_ids (known: {sorted(env_ids)})")
    return env_ids[key]


def _observation_dim(env_id: str) -> int:
    """Reset the env once to read the goal-conditioned input width.

    The PPO policy is goal-conditioned, so its input is ``observation``
    concatenated with ``desired_goal`` (Reach 10+3=13, Push 25+3=28).
    Must match :func:`goal_conditioned_observation` in the loop.
    """
    import gymnasium as gym  # noqa: PLC0415
    import gymnasium_robotics  # noqa: PLC0415

    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id)
    try:
        return goal_conditioned_observation_dim(env.observation_space)
    finally:
        env.close()


def _build_throwaway_bank(env_id: str, n_contexts: int, base_seed: int) -> FetchTaskBank:
    """Probe ``n_contexts`` seeds → an all-train throwaway bank.

    Identical construction to ``fetch_calibration._roll_starter``'s
    throwaway bank: probe-phase contexts, dummy bins, all in the train
    split so ``train_ids() == range(0, n_contexts)`` (the invariant
    ``build_sampler_for_condition`` asserts).
    """
    pool = _probe_pool(env_id, n_contexts, base_seed)
    contexts = [
        FetchTaskContext(
            context_id=r.context_id,
            env_id=r.env_id,
            reset_seed=r.reset_seed,
            desired_goal=r.desired_goal,
            reset_payload=r.reset_payload,
            initial_object_pose=r.initial_object_pose,
            distance_bin=0,
            direction_bin=0,
            pose_bin=0,
            split="train",
        )
        for r in pool
    ]
    ee = {r.context_id: r.initial_achieved_goal for r in pool}
    return FetchTaskBank.from_contexts(contexts, ee)


def _ppo_config_from_yaml(
    ppo_cfg: dict[str, Any], observation_dim: int, device: str
) -> PPOContinuousConfig:
    """Map the YAML ``starter.ppo`` block to ``PPOContinuousConfig``.

    ``n_steps_per_update`` from the YAML is not used here — the warmup
    drives env-step budget through ``episodes_per_iteration *
    n_iterations`` in the loop config — so it is ignored intentionally.
    """
    return PPOContinuousConfig(
        observation_dim=observation_dim,
        action_dim=4,  # Fetch action_space is always Box(-1, 1, (4,)).
        hidden_sizes=tuple(ppo_cfg["hidden_sizes"]),
        learning_rate=float(ppo_cfg["learning_rate"]),
        clip_eps=float(ppo_cfg["clip_eps"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        minibatch_size=int(ppo_cfg["minibatch_size"]),
        entropy_coef=float(ppo_cfg["entropy_coef"]),
        value_coef=float(ppo_cfg["value_coef"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        device=device,
    )


def _sanity_check_checkpoint(out_path: Path, env_id: str) -> None:
    """Reload the saved artifact and assert greedy actions are finite.

    Cheap guard so a NaN/garbage warmup does not silently propagate into
    Priority 1a calibration.
    """
    import torch  # noqa: PLC0415

    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    agent = PPOAgent(payload["config"])
    agent.load(out_path)

    import gymnasium as gym  # noqa: PLC0415
    import gymnasium_robotics  # noqa: PLC0415

    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id)
    try:
        obs, _ = env.reset(seed=12345)
        for _ in range(5):
            action, _, _ = agent.act(
                np.asarray(obs["observation"], dtype=np.float32), deterministic=True
            )
            if not np.all(np.isfinite(action)):
                raise ValueError(f"warmup checkpoint {out_path} produced non-finite action")
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=12345)
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    """PPO warmup entry point. Returns shell-style exit code."""
    parser = argparse.ArgumentParser(prog="train_fetch_ppo_warmup")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/fetch_priority_1a.yaml"),
        help="Priority 1a YAML (env_ids + starter.* block).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="env key in config.env_ids (e.g. reach_dense). Defaults to config.default_env.",
    )
    parser.add_argument(
        "--warmup-env-steps",
        type=int,
        default=None,
        help="Override starter.warmup_env_steps (10k-25k range).",
    )
    parser.add_argument(
        "--episodes-per-iteration",
        type=int,
        default=8,
        help="Episodes rolled per PPO update (drives env-step budget).",
    )
    parser.add_argument(
        "--n-pool",
        type=int,
        default=500,
        help="Throwaway-bank size (diverse-seed coverage during warmup).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the loop / bank seed (defaults to config.bank.seed).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device for the tiny PPO nets (cpu is fastest here).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Explicit checkpoint path. Defaults to "
        "<CHECKPOINTS_ROOT>/<starter.checkpoint_filename for env_id>.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=None,
        help="Override CHECKPOINTS_ROOT for the output path.",
    )
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    env_id = _resolve_env_id(cfg, args.env)
    bank_cfg = cfg["bank"]
    starter_cfg = cfg["starter"]
    ppo_yaml = starter_cfg["ppo"]

    warmup_env_steps = args.warmup_env_steps or int(starter_cfg["warmup_env_steps"])
    seed = args.seed if args.seed is not None else int(bank_cfg["seed"])
    max_steps = int(cfg["calibration"]["max_steps_per_episode"])
    episodes_per_iter = args.episodes_per_iteration
    n_iterations = max(1, math.ceil(warmup_env_steps / (episodes_per_iter * max_steps)))

    # Resolve the output checkpoint path.
    if args.out is not None:
        out_path = args.out
    else:
        ckpt_root = args.checkpoints_root or CHECKPOINTS_ROOT
        filename = starter_cfg["checkpoint_filename"].format(env_id=env_id)
        out_path = Path(ckpt_root) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[warmup] env_id={env_id} warmup_env_steps={warmup_env_steps} "
        f"n_iterations={n_iterations} episodes_per_iter={episodes_per_iter} "
        f"max_steps={max_steps} n_pool={args.n_pool} seed={seed} "
        f"device={args.device}\n[warmup] -> {out_path}",
        flush=True,
    )

    obs_dim = _observation_dim(env_id)
    bank = _build_throwaway_bank(env_id, args.n_pool, base_seed=seed)
    ppo_config = _ppo_config_from_yaml(ppo_yaml, obs_dim, args.device)

    loop_cfg = FetchPPOLoopConfig(
        exp_name=f"warmup_{env_id}",
        condition=SamplerName.UNIFORM,
        env_id=env_id,
        n_iterations=n_iterations,
        episodes_per_iteration=episodes_per_iter,
        max_steps_per_episode=max_steps,
        eval_interval=0,  # no held-out eval during warmup
        seed=seed,
        ppo=ppo_config,
        kill_conditions_enabled=False,  # throwaway run; sanity-checked post-save
    )

    result = run_fetch_ppo_loop(loop_cfg, bank, eval_env=None, checkpoint_out=out_path)

    last = result.get("last_metrics", {})
    print(
        f"[warmup] done. final rewards_mean={last.get('rewards_mean')!r} "
        f"pg_loss={last.get('pg_loss')!r} "
        f"explained_variance={last.get('explained_variance')!r}",
        flush=True,
    )

    _sanity_check_checkpoint(out_path, env_id)
    print(f"[warmup] checkpoint saved + sanity-checked: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
