"""PointMaze launcher — cross-domain port of the SAC+HER + value-uncertainty
curriculum to a NON-Fetch sparse goal-reaching env.

Builds a deterministic PointMazeTaskBank (seed from config), then runs
:func:`pcrc.training.fetch_sac_her_loop.run_fetch_sac_her_loop` per condition —
the loop is env-agnostic via the env_id-prefix dispatch in ``_make_task_env``.

Usage::

    uv run python scripts/pointmaze_sac_her.py --config config/pointmaze_curriculum.yaml \
        --exp-name pm_probe --conditions unassisted

Plan reference: cross-domain generality follow-up to the Fetch uncertainty result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

from pcrc.algorithms.sac_continuous import SACContinuousConfig
from pcrc.envs.pointmaze_env import PointMazeTaskBank, PointMazeTaskEnvironment
from pcrc.envs.trajectory_scaffold import ScaffoldLadder
from pcrc.registry import DEFAULT_METHOD, Method, ScaffoldSource, condition_help, parse_conditions
from pcrc.training.conditions import resolve_condition
from pcrc.training.fetch_sac_her_loop import (
    FetchSACHERLoopConfig,
    run_fetch_sac_her_loop,
)
from pcrc.utils.paths import RESULTS_ROOT
from pcrc.utils.results import RunLogger


def _build_sac_config(
    sac_cfg: dict, state_dim: int, action_dim: int, action_low: float, action_high: float
) -> SACContinuousConfig:
    return SACContinuousConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=tuple(sac_cfg.get("hidden_sizes", (256, 256))),
        learning_rate=float(sac_cfg.get("learning_rate", 3.0e-4)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        tau=float(sac_cfg.get("tau", 0.005)),
        target_entropy=sac_cfg.get("target_entropy"),
        init_log_alpha=float(sac_cfg.get("init_log_alpha", -1.6)),
        action_low=action_low,
        action_high=action_high,
        layer_norm=bool(sac_cfg.get("layer_norm", False)),  # RLPD critic LayerNorm
    )


def main(argv: list[str] | None = None) -> int:
    """PointMaze SAC+HER entry point."""
    parser = argparse.ArgumentParser(prog="pointmaze_sac_her")
    parser.add_argument("--config", type=Path, default=Path("config/pointmaze_curriculum.yaml"))
    parser.add_argument("--exp-name", required=True, type=str)
    parser.add_argument("--conditions", type=str, default=DEFAULT_METHOD, help=condition_help("pointmaze"))
    parser.add_argument("--seed", type=int, default=None, help="override loop.seed (paired runs)")
    parser.add_argument("--n-iterations", type=int, default=None, help="override loop.n_iterations")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--bank-path", type=Path, default=None, help="load a saved strata bank")
    parser.add_argument(
        "--harvest", action="store_true",
        help="Stage A: after training, harvest the policy's own ℓ=0 first-success rollout per train "
        "context to <out_dir>/harvest_ell0.json (self-generated scaffold source)",
    )
    parser.add_argument("--harvest-k", type=int, default=5)
    parser.add_argument("--harvest-seed", type=int, default=99)
    parser.add_argument(
        "--ladder-path",
        type=Path,
        default=None,
        help="ScaffoldLadder JSON for trajectory_* (trajectory-source) conditions",
    )
    parser.add_argument(
        "--scaffold-geometry",
        type=str,
        default=None,
        choices=("geodesic", "offpath"),
        help="PointMaze reset geometry (offpath = Nav-bypass conflict; default geodesic)",
    )
    parser.add_argument(
        "--learner",
        type=str,
        default="sac",
        choices=("sac", "crl"),
        help="base learner: sac (TD, default) or crl (contrastive critic, B2/M4)",
    )
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    defaults = cfg.get("defaults", {})
    loop_cfg = dict(cfg["loop"])
    sac_cfg_dict = cfg["sac"]
    sampler_kwargs_by_condition = cfg.get("sampler_kwargs", {})
    scaffold_block = cfg.get("scaffold", {})
    rc_block = cfg.get("reverse_curriculum", {})
    if args.seed is not None:
        loop_cfg["seed"] = int(args.seed)
    if args.n_iterations is not None:
        loop_cfg["n_iterations"] = int(args.n_iterations)
    geometry = args.scaffold_geometry or defaults.get("scaffold_geometry", "geodesic")
    loop_cfg["scaffold_geometry"] = geometry
    env_id = defaults["env_id"]

    if args.bank_path is not None:
        bank = PointMazeTaskBank.load(args.bank_path)
        env_id = bank.env_id
        print(
            f"[pointmaze] loaded bank {args.bank_path} (env {env_id}, {len(bank)} ctx)", flush=True
        )
    else:
        bank = PointMazeTaskBank.build(
            env_id=env_id,
            n_train=int(defaults.get("n_train", 100)),
            n_held=int(defaults.get("n_held", 50)),
            seed=int(defaults.get("bank_seed", 0)),
        )

    probe = PointMazeTaskEnvironment(bank, env_id, scaffold_geometry=geometry)
    obs_dim = int(probe.observation_space["observation"].shape[0])
    goal_dim = int(probe.observation_space["desired_goal"].shape[0])
    action_dim = int(probe.action_space.shape[0])
    action_low = float(np.min(probe.action_space.low))
    action_high = float(np.max(probe.action_space.high))
    probe.close()

    results_root = args.results_root or RESULTS_ROOT
    conditions = parse_conditions(args.conditions, "pointmaze")
    print(
        f"[pointmaze] env_id={env_id} state_dim={obs_dim + goal_dim} action_dim={action_dim} "
        f"action_bounds=[{action_low},{action_high}] n_train={len(bank.train_ids())} "
        f"n_held={len(bank.held_out_ids())} conditions={conditions}",
        flush=True,
    )

    for condition in conditions:
        loop_condition, overrides = resolve_condition(condition, scaffold_block, rc_block)
        # mode "oracle": per-context level from the route group (Nav-bypass bank). direct→high
        # scaffold (increasing transfer), bypass→low (decreasing transfer) — levels from the config.
        if overrides.get("scaffold_mode") == Method.ORACLE:
            direct_lvl = int(
                scaffold_block.get("oracle_direct_level", scaffold_block["n_levels"] - 1)
            )
            bypass_lvl = int(scaffold_block.get("oracle_bypass_level", 0))
            overrides["scaffold_oracle_levels"] = [
                bypass_lvl if bank[cid].route_group == 1 else direct_lvl for cid in bank.train_ids()
            ]
        # trajectory_* (trajectory source): load the extracted ladder and hand it to the loop config.
        if overrides.get("scaffold_source") == ScaffoldSource.TRAJECTORY:
            if args.ladder_path is None:
                raise ValueError(
                    f"condition {condition!r} is trajectory-source but no --ladder-path was given"
                )
            overrides["scaffold_ladder"] = ScaffoldLadder.load(args.ladder_path)
        if args.harvest:  # Stage A: harvest the policy's own ℓ=0 first-success rollouts post-training
            overrides["harvest_path"] = str(
                results_root / f"{args.exp_name}/{condition}" / "harvest_ell0.json"
            )
            overrides["harvest_k"] = args.harvest_k
            overrides["harvest_seed"] = args.harvest_seed
        sampler_kwargs = sampler_kwargs_by_condition.get(loop_condition, {})
        sac_config = _build_sac_config(
            sac_cfg_dict, obs_dim + goal_dim, action_dim, action_low, action_high
        )
        # PCRC conditions all share loop_condition="uniform", so namespace the output by
        # the real condition name to avoid collisions in one exp_name.
        cond_exp = f"{args.exp_name}/{condition}"
        loop_config = FetchSACHERLoopConfig(
            exp_name=cond_exp,
            condition=loop_condition,
            env_id=env_id,
            sac=sac_config,
            sampler_kwargs=sampler_kwargs,
            learner=args.learner,
            crl_goal_dim=goal_dim if args.learner == "crl" else None,
            **{**loop_cfg, **overrides},
        )
        logger = RunLogger(cond_exp, root=str(results_root))
        print(
            f"[pointmaze] running condition={condition} (loop_condition={loop_condition})",
            flush=True,
        )
        # Eval env MUST use the same episode budget as training. Without max_steps the env defaults
        # its gym TimeLimit to 300 (pointmaze_env.py), so a config with max_steps_per_episode>300
        # (e.g. the long-horizon band [14,18]) would train at 500 but truncate eval at 300 — a
        # train/eval horizon mismatch that structurally pins held-out ℓ=0 near zero. (No-op for the
        # Stage-0 configs, which already use 300.)
        eval_env = PointMazeTaskEnvironment(
            bank, env_id, max_steps=loop_config.max_steps_per_episode, scaffold_geometry=geometry
        )
        try:
            result = run_fetch_sac_her_loop(loop_config, bank, eval_env=eval_env, logger=logger)
        finally:
            eval_env.close()
        last_eval = result.get("last_eval")
        if last_eval is not None:
            print(
                f"[pointmaze] condition={condition} "
                f"last_eval_success_rate={last_eval['success_rate']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
