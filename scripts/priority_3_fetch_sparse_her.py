"""Priority 3 launcher — Sparse Fetch + SAC + HER on FetchPush-v4.

Reads config/fetch_priority_3.yaml + a sparse-calibrated bank, then
runs :func:`pcrc.training.fetch_sac_her_loop.run_fetch_sac_her_loop`
per condition.

Plan reference: docs/actor_curator_gymrobotics_implementation_plan.md
§Priority 3, §10 Stage D step 4.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from pcrc.algorithms.sac_continuous import SACContinuousConfig
from pcrc.envs.fetch_environment import FetchTaskEnvironment
from pcrc.envs.fetch_task_bank import FetchTaskBank
from pcrc.training.fetch_sac_her_loop import (
    FetchSACHERLoopConfig,
    run_fetch_sac_her_loop,
)
from pcrc.utils.paths import RESULTS_ROOT
from pcrc.utils.results import RunLogger


def _build_sac_config(sac_cfg: dict, state_dim: int, action_dim: int) -> SACContinuousConfig:
    return SACContinuousConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=tuple(sac_cfg.get("hidden_sizes", (256, 256))),
        learning_rate=float(sac_cfg.get("learning_rate", 3.0e-4)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        tau=float(sac_cfg.get("tau", 0.005)),
        target_entropy=sac_cfg.get("target_entropy"),
        init_log_alpha=float(sac_cfg.get("init_log_alpha", -1.6)),
        action_low=float(sac_cfg.get("action_low", -1.0)),
        action_high=float(sac_cfg.get("action_high", 1.0)),
    )


def main(argv: list[str] | None = None) -> int:
    """Priority 3 entry point."""
    parser = argparse.ArgumentParser(prog="priority_3_fetch_sparse_her")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/fetch_priority_3.yaml"),
    )
    parser.add_argument("--exp-name", required=True, type=str)
    parser.add_argument("--bank-path", type=Path, default=None)
    parser.add_argument(
        "--conditions",
        type=str,
        default="uniform,tabular_actor_curator",
    )
    parser.add_argument("--results-root", type=Path, default=None)
    # Demonstrations + behavior-cloning lever (Nair 2018) overrides. When unset,
    # the YAML loop values (default bc_lambda=0 → BC off) are used unchanged.
    parser.add_argument("--demos-path", type=str, default=None)
    parser.add_argument("--bc-lambda", type=float, default=None)
    parser.add_argument(
        "--use-q-filter",
        type=lambda s: s.lower() in ("1", "true", "yes"),
        default=None,
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=None,
        help="override loop.n_iterations (e.g. for the BC smoke test).",
    )
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    defaults = cfg.get("defaults", {})
    loop_cfg = dict(cfg["loop"])
    if args.demos_path is not None:
        loop_cfg["demos_path"] = args.demos_path
    if args.bc_lambda is not None:
        loop_cfg["bc_lambda"] = args.bc_lambda
    if args.use_q_filter is not None:
        loop_cfg["use_q_filter"] = args.use_q_filter
    if args.n_iterations is not None:
        loop_cfg["n_iterations"] = args.n_iterations
    sac_cfg_dict = cfg["sac"]
    sampler_kwargs_by_condition = cfg.get("sampler_kwargs", {})
    env_id = defaults["env_id"]
    bank_path = args.bank_path or Path(defaults["bank_path"])
    if not bank_path.exists():
        raise FileNotFoundError(
            f"bank file not found: {bank_path}; calibrate first via "
            "priority_1a_fetch_calibration with --env push_sparse"
        )
    bank = FetchTaskBank.load(bank_path)

    probe = FetchTaskEnvironment(bank, env_id)
    obs_dim = int(probe.observation_space["observation"].shape[0])
    goal_dim = int(probe.observation_space["desired_goal"].shape[0])
    action_dim = int(probe.action_space.shape[0])
    probe.close()

    results_root = args.results_root or RESULTS_ROOT
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(
        f"[priority-3] env_id={env_id} state_dim={obs_dim + goal_dim} "
        f"action_dim={action_dim} conditions={conditions}",
        flush=True,
    )

    for condition in conditions:
        sampler_kwargs = sampler_kwargs_by_condition.get(condition, {})
        sac_config = _build_sac_config(sac_cfg_dict, obs_dim + goal_dim, action_dim)
        loop_config = FetchSACHERLoopConfig(
            exp_name=args.exp_name,
            condition=condition,
            env_id=env_id,
            sac=sac_config,
            sampler_kwargs=sampler_kwargs,
            **loop_cfg,
        )
        logger = RunLogger(args.exp_name, root=str(results_root))
        print(f"[priority-3] running condition={condition}", flush=True)
        eval_env = FetchTaskEnvironment(bank, env_id)
        try:
            result = run_fetch_sac_her_loop(loop_config, bank, eval_env=eval_env, logger=logger)
        finally:
            eval_env.close()
        last_eval = result.get("last_eval")
        if last_eval is not None:
            print(
                f"[priority-3] condition={condition} last_eval_success_rate="
                f"{last_eval['success_rate']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
