"""Meta-World launcher — generalization spike for the value-uncertainty curriculum.

Builds a deterministic MetaWorldTaskBank (frozen rand_vec contexts), then runs
:func:`pcrc.training.fetch_sac_her_loop.run_fetch_sac_her_loop` per condition — the loop is
env-agnostic via the ``metaworld:<task>`` dispatch in ``_make_task_env``.

Usage::

    uv run python scripts/metaworld_sac_her.py --config config/metaworld_pickplace.yaml \
        --exp-name mw_push_probe --conditions unassisted

Plan: docs/metaworld_generalization_plan.md (Phase 1 learnability probe).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

from pcrc.algorithms.sac_continuous import SACContinuousConfig
from pcrc.envs.metaworld_env import (
    MetaWorldTaskBank,
    MetaWorldTaskEnvironment,
    parse_metaworld_env_id,
)
from pcrc.registry import DEFAULT_METHOD, Method, condition_help, parse_conditions
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
        alpha_min=float(sac_cfg.get("alpha_min", 0.0)),
        action_low=action_low,
        action_high=action_high,
        device=str(sac_cfg.get("device", "cpu")),
    )


def main(argv: list[str] | None = None) -> int:
    """Meta-World SAC+HER entry point."""
    parser = argparse.ArgumentParser(prog="metaworld_sac_her")
    parser.add_argument("--config", type=Path, default=Path("config/metaworld_pickplace.yaml"))
    parser.add_argument("--exp-name", required=True, type=str)
    parser.add_argument("--conditions", type=str, default=DEFAULT_METHOD, help=condition_help("metaworld"))
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="override loop seed (per-seed screen)"
    )
    parser.add_argument("--n-iterations", type=int, default=None, help="override loop.n_iterations")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="override task(s) for the conflict search: single 'reach-v3' or pair "
        "'reach-v3+pick-place-v3' (builds the env_id as metaworld:<tasks>:handnear:graspcur)",
    )
    parser.add_argument(
        "--ac-utility",
        type=str,
        default=None,
        help="override ac_utility (td_error / q_disagreement) for uncertainty runs",
    )
    parser.add_argument(
        "--demo-replay-fraction",
        type=float,
        default=None,
        help="override demo_replay_fraction (bottleneck-band sweep)",
    )
    parser.add_argument(
        "--bc-lambda", type=float, default=None, help="override bc_lambda (bottleneck-band sweep)"
    )
    parser.add_argument(
        "--pcgrad",
        action="store_true",
        help="enable per-group PCGrad gradient surgery (multi-task interference fix)",
    )
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    defaults = cfg.get("defaults", {})
    loop_cfg = dict(cfg["loop"])
    if args.seed is not None:
        loop_cfg["seed"] = int(args.seed)
    if args.n_iterations is not None:
        loop_cfg["n_iterations"] = int(args.n_iterations)
    if args.ac_utility is not None:
        loop_cfg["ac_utility"] = str(args.ac_utility)
    if args.demo_replay_fraction is not None:
        loop_cfg["demo_replay_fraction"] = float(args.demo_replay_fraction)
    if args.bc_lambda is not None:
        loop_cfg["bc_lambda"] = float(args.bc_lambda)
    sac_cfg_dict = dict(cfg["sac"])
    if getattr(args, "pcgrad", False):
        sac_cfg_dict["pcgrad"] = True
    sampler_kwargs_by_condition = cfg.get("sampler_kwargs", {})
    scaffold_block = cfg.get("scaffold", {})
    rc_block = cfg.get("reverse_curriculum", {})
    # --tasks override (conflict search): single task or '+'-joined pair. handnear aids non-grasp
    # learnability; graspcur self-gates to elevated-goal (pick/insert/assembly) tasks only, so the
    # option string is safe for any task. Falls back to the config's env_id/tasks when not given.
    if args.tasks is not None:
        tlist = [t.strip() for t in args.tasks.split("+") if t.strip()]
        env_id = f"metaworld:{'+'.join(tlist)}:handnear:graspcur"
        tasks_cfg = tlist if len(tlist) > 1 else None
    else:
        env_id = defaults["env_id"]  # e.g. "metaworld:push-v3+pick-place-v3:handnear:graspcur"
        tasks_cfg = defaults.get("tasks")  # list -> heterogeneous multi-task bank (build_multi)
    task, env_opts = parse_metaworld_env_id(env_id)  # task for the bank; opts for the env wrapper

    if tasks_cfg:
        bank = MetaWorldTaskBank.build_multi(
            list(tasks_cfg),
            n_train=int(defaults.get("n_train", 100)),
            n_held=int(defaults.get("n_held", 50)),
            seed=int(defaults.get("bank_seed", 0)),
        )
    else:
        bank = MetaWorldTaskBank.build(
            env_id=task,
            n_train=int(defaults.get("n_train", 100)),
            n_held=int(defaults.get("n_held", 50)),
            seed=int(defaults.get("bank_seed", 0)),
        )

    probe = MetaWorldTaskEnvironment(bank, task, **env_opts)
    obs_dim = int(probe.observation_space["observation"].shape[0])
    goal_dim = int(probe.observation_space["desired_goal"].shape[0])
    action_dim = int(probe.action_space.shape[0])
    action_low = float(np.min(probe.action_space.low))
    action_high = float(np.max(probe.action_space.high))
    probe.close()

    results_root = args.results_root or RESULTS_ROOT
    conditions = parse_conditions(args.conditions, "metaworld")
    print(
        f"[metaworld] task={task} state_dim={obs_dim + goal_dim} action_dim={action_dim} "
        f"bounds=[{action_low},{action_high}] n_train={len(bank.train_ids())} "
        f"n_held={len(bank.held_out_ids())} conditions={conditions}",
        flush=True,
    )

    for condition in conditions:
        # PCRC scaffold conditions (per_context_success/fixed<N>/global_anneal/oracle/...) resolve to the
        # uniform sampler + scaffold overrides; plain sampler names (uniform/headroom/...) pass through.
        # <base>_balanced_replay (campaign 5 C5.7): resolve <base> unchanged, then add group-balanced
        # minibatch quotas (groups = tasks). Only the replay sampling differs from <base>.
        balanced_replay = condition.endswith("_balanced_replay")
        resolve_name = condition.removesuffix("_balanced_replay")
        loop_condition, overrides = resolve_condition(resolve_name, scaffold_block, rc_block)
        if balanced_replay:
            if not bank.tasks:
                raise ValueError("_balanced_replay needs a multi-task bank (groups = tasks)")
            overrides["replay_group_balanced"] = True
            overrides["replay_group_ids"] = [
                int(bank[cid].task_group) for cid in bank.train_ids()
            ]
        # oracle: per-context level from the object→goal distance band (driver computes it from the
        # bank, like Fetch's goal-z map). near→low scaffold, far→high (or per the transfer landscape).
        if overrides.get("scaffold_mode") == Method.ORACLE:
            d_thr = float(scaffold_block.get("oracle_dist_threshold", 0.18))
            low = int(scaffold_block.get("oracle_low_level", 0))
            high = int(scaffold_block.get("oracle_high_level", 4))
            import numpy as _np

            if bank.tasks:
                # multi-task #5 oracle: per-context level by TASK group (e.g. push->low, pick-place->high)
                overrides["scaffold_oracle_levels"] = [
                    (high if bank[cid].task_group > 0 else low) for cid in bank.train_ids()
                ]
            else:

                def _dist(cid):
                    c = bank[cid]
                    return float(
                        _np.linalg.norm(_np.asarray(c.obj_init) - _np.asarray(c.rand_vec[3:6]))
                    )

                overrides["scaffold_oracle_levels"] = [
                    (high if _dist(cid) >= d_thr else low) for cid in bank.train_ids()
                ]
        # mode "per_group_success" (campaign 5 C5.1): per-context group ids = the bank's task groups
        # (e.g. push=0, shelf-place=1) — the oracle's group labels without its level map.
        if overrides.get("scaffold_mode") == Method.PER_GROUP_SUCCESS:
            if not bank.tasks:
                raise ValueError("per_group_success needs a multi-task bank (groups = tasks)")
            overrides["scaffold_group_ids"] = [
                int(bank[cid].task_group) for cid in bank.train_ids()
            ]
        sampler_kwargs = sampler_kwargs_by_condition.get(loop_condition, {})
        sac_config = _build_sac_config(
            sac_cfg_dict, obs_dim + goal_dim, action_dim, action_low, action_high
        )
        cond_exp = f"{args.exp_name}/{condition}"
        loop_config = FetchSACHERLoopConfig(
            exp_name=cond_exp,
            condition=loop_condition,
            env_id=env_id,
            sac=sac_config,
            sampler_kwargs=sampler_kwargs,
            **{**loop_cfg, **overrides},
        )
        logger = RunLogger(cond_exp, root=str(results_root))
        print(
            f"[metaworld] running condition={condition} (loop_condition={loop_condition})",
            flush=True,
        )
        eval_env = MetaWorldTaskEnvironment(bank, task, **env_opts)
        try:
            result = run_fetch_sac_her_loop(loop_config, bank, eval_env=eval_env, logger=logger)
        finally:
            eval_env.close()
        last_eval = result.get("last_eval")
        if last_eval is not None:
            print(
                f"[metaworld] condition={condition} "
                f"last_eval_success_rate={last_eval['success_rate']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
