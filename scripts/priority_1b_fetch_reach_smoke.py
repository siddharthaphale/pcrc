"""Priority 1b launcher — Dense PPO smoke on FetchReachDense-v4.

Reads config/fetch_priority_1b.yaml + a calibrated bank, then runs
:func:`pcrc.training.fetch_ppo_loop.run_fetch_ppo_loop` per condition.

Plan reference: docs/actor_curator_gymrobotics_implementation_plan.md
§Priority 1b, §10 Stage D step 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from pcrc.algorithms.ppo_continuous import PPOContinuousConfig
from pcrc.envs.fetch_environment import (
    FetchTaskEnvironment,
    goal_conditioned_observation_dim,
)
from pcrc.envs.fetch_task_bank import FetchTaskBank
from pcrc.training.fetch_ppo_loop import (
    FetchPPOLoopConfig,
    run_fetch_ppo_loop,
)
from pcrc.utils.paths import RESULTS_ROOT
from pcrc.utils.results import RunLogger


def _build_ppo_config(ppo_cfg: dict, observation_dim: int, action_dim: int) -> PPOContinuousConfig:
    return PPOContinuousConfig(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_sizes=tuple(ppo_cfg.get("hidden_sizes", (64, 64))),
        learning_rate=float(ppo_cfg.get("learning_rate", 3.0e-4)),
        clip_eps=float(ppo_cfg.get("clip_eps", 0.2)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        minibatch_size=int(ppo_cfg.get("minibatch_size", 64)),
        entropy_coef=float(ppo_cfg.get("entropy_coef", 0.0)),
        value_coef=float(ppo_cfg.get("value_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        log_std_init=float(ppo_cfg.get("log_std_init", 0.0)),
    )


def main(argv: list[str] | None = None) -> int:
    """Priority 1b entry point."""
    parser = argparse.ArgumentParser(prog="priority_1b_fetch_reach_smoke")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/fetch_priority_1b.yaml"),
    )
    parser.add_argument("--exp-name", required=True, type=str)
    parser.add_argument("--bank-path", type=Path, default=None)
    parser.add_argument(
        "--conditions",
        type=str,
        default="uniform,tabular_actor_curator",
        help="Comma-separated list of conditions to run.",
    )
    parser.add_argument("--results-root", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    defaults = cfg.get("defaults", {})
    loop_cfg = cfg["loop"]
    ppo_cfg_dict = cfg["ppo"]
    sampler_kwargs_by_condition = cfg.get("sampler_kwargs", {})
    env_id = defaults["env_id"]
    bank_path = args.bank_path or Path(defaults["bank_path"])
    if not bank_path.exists():
        raise FileNotFoundError(
            f"bank file not found: {bank_path}; run priority_1a_fetch_calibration first"
        )
    bank = FetchTaskBank.load(bank_path)

    # Probe the env once to read observation/action dims. The policy is
    # goal-conditioned, so its input width is observation + desired_goal.
    probe = FetchTaskEnvironment(bank, env_id)
    obs_dim = goal_conditioned_observation_dim(probe.observation_space)
    action_dim = int(probe.action_space.shape[0])
    probe.close()

    results_root = args.results_root or RESULTS_ROOT
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(
        f"[priority-1b] env_id={env_id} obs_dim={obs_dim} "
        f"action_dim={action_dim} conditions={conditions}",
        flush=True,
    )

    for condition in conditions:
        sampler_kwargs = sampler_kwargs_by_condition.get(condition, {})
        ppo_config = _build_ppo_config(ppo_cfg_dict, obs_dim, action_dim)
        loop_config = FetchPPOLoopConfig(
            exp_name=args.exp_name,
            condition=condition,
            env_id=env_id,
            ppo=ppo_config,
            sampler_kwargs=sampler_kwargs,
            **loop_cfg,
        )
        logger = RunLogger(args.exp_name, root=str(results_root))
        print(f"[priority-1b] running condition={condition}", flush=True)
        eval_env = FetchTaskEnvironment(bank, env_id)
        try:
            result = run_fetch_ppo_loop(loop_config, bank, eval_env=eval_env, logger=logger)
        finally:
            eval_env.close()
        last_eval = result.get("last_eval")
        if last_eval is not None:
            print(
                f"[priority-1b] condition={condition} last_eval_success_rate="
                f"{last_eval['success_rate']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
