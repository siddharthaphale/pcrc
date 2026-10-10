"""PCRC-RC (Stage 3) launcher — in-air FetchPickAndPlace scaffold SAC+HER.

Fetch analogue of scripts/pointmaze_sac_her.py: loads a saved FetchTaskBank and
runs run_fetch_sac_her_loop per condition, mapping each --conditions name to a
config bundle via pcrc.training.conditions.resolve_condition (PCRC scaffold
levels = reverse-curriculum start_progress; the env synthesizes the grasped
state for in-air goals). Output is namespaced per condition.

    uv run python scripts/fetch_sac_her.py --config config/fetch_pick_and_place.yaml \
        --exp-name pcrc_rc/s42 --conditions pcrc --seed 42

Plan / provenance: docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from pcrc.algorithms.sac_continuous import SACContinuousConfig
from pcrc.envs.fetch_environment import FetchTaskEnvironment
from pcrc.envs.fetch_task_bank import FetchTaskBank
from pcrc.envs.trajectory_scaffold import ScaffoldLadder
from pcrc.registry import DEFAULT_METHOD, Method, ScaffoldSource, condition_help, parse_conditions
from pcrc.training.conditions import resolve_condition
from pcrc.training.fetch_sac_her_loop import FetchSACHERLoopConfig, run_fetch_sac_her_loop
from pcrc.utils.paths import RESULTS_ROOT
from pcrc.utils.results import RunLogger


def _load_replay_trajectory(path: Path) -> np.ndarray:
    """Read a reference pcrc run's scaffold.jsonl → [T, n_train] per-context level matrix.

    The pcrc-family selected level == the frontier ``ell[x]`` exactly (controller.select), so the
    logged ``ell`` array per snapshot IS the per-context level trajectory. Gate E replays it.
    """
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ell = json.loads(line).get("ell")
            if ell is not None:
                rows.append(ell)
    if not rows:
        raise ValueError(f"no 'ell' rows in {path}")
    return np.asarray(rows, dtype=np.int64)


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
        pcgrad=bool(sac_cfg.get("pcgrad", False)),
    )


def main(argv: list[str] | None = None) -> int:
    """Fetch PCRC-RC entry point."""
    parser = argparse.ArgumentParser(prog="fetch_sac_her")
    parser.add_argument("--config", type=Path, default=Path("config/fetch_pick_and_place.yaml"))
    parser.add_argument("--exp-name", required=True, type=str)
    parser.add_argument("--conditions", type=str, default=DEFAULT_METHOD, help=condition_help("fetch"))
    parser.add_argument("--seed", type=int, default=None, help="override loop.seed (paired runs)")
    parser.add_argument("--n-iterations", type=int, default=None, help="override loop.n_iterations")
    parser.add_argument("--bank-path", type=Path, default=None)
    parser.add_argument(
        "--ladder-path",
        type=Path,
        default=None,
        help="ScaffoldLadder JSON for trajectory_* (trajectory-source) conditions",
    )
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument(
        "--harvest", action="store_true",
        help="Stage A: after training, harvest the policy's own ℓ=0 first-success rollout per train "
        "context to <out_dir>/harvest_ell0.json (self-generated scaffold source)",
    )
    parser.add_argument("--harvest-k", type=int, default=5)
    parser.add_argument("--harvest-seed", type=int, default=99)
    parser.add_argument(
        "--replay-source",
        type=Path,
        default=None,
        help="scaffold.jsonl of a reference pcrc run; required for replay_* conditions (Gate E)",
    )
    parser.add_argument(
        "--pcgrad",
        action="store_true",
        help="enable PCGrad gradient surgery across goal-z groups (interference probe)",
    )
    parser.add_argument(
        "--stale-advance-windows",
        type=int,
        default=None,
        help="override scaffold.kwargs.stale_advance_windows (anti-stall ablation sweep)",
    )
    parser.add_argument(
        "--bpcrc-gamma",
        type=float,
        default=None,
        help="override scaffold.kwargs.bpcrc_gamma (B-PCRC discount grid, T0.3)",
    )
    parser.add_argument(
        "--learner",
        type=str,
        default="sac",
        choices=("sac", "crl"),
        help="base learner: sac (TD, default) or crl (contrastive critic, B2/M4)",
    )
    parser.add_argument(
        "--scaffold-kwargs-json",
        type=str,
        default=None,
        help="JSON dict merged over scaffold.kwargs (BO gating-knob search; e.g. "
        '\'{"trend_q_level":0.05,"trend_horizon":2.0}\')',
    )
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    defaults = cfg.get("defaults", {})
    loop_cfg = dict(cfg["loop"])
    sac_cfg_dict = dict(cfg["sac"])
    if args.pcgrad:
        sac_cfg_dict["pcgrad"] = True
    sampler_kwargs_by_condition = cfg.get("sampler_kwargs", {})
    scaffold_block = cfg.get("scaffold", {})
    rc_block = cfg.get("reverse_curriculum", {})
    if args.seed is not None:
        loop_cfg["seed"] = int(args.seed)
    if args.n_iterations is not None:
        loop_cfg["n_iterations"] = int(args.n_iterations)
    env_id = defaults["env_id"]

    bank_path = args.bank_path or Path(defaults["bank_path"])
    if not bank_path.exists():
        raise FileNotFoundError(
            f"bank not found: {bank_path}. Build the in-air PnP bank first "
            "(scripts/build_inair_pnp_bank.py)."
        )
    bank = FetchTaskBank.load(bank_path)

    probe = FetchTaskEnvironment(bank, env_id)
    obs_dim = int(probe.observation_space["observation"].shape[0])
    goal_dim = int(probe.observation_space["desired_goal"].shape[0])
    action_dim = int(probe.action_space.shape[0])
    action_low = float(np.min(probe.action_space.low))
    action_high = float(np.max(probe.action_space.high))
    probe.close()

    results_root = args.results_root or RESULTS_ROOT
    conditions = parse_conditions(args.conditions, "fetch")
    print(
        f"[pcrc-rc] env_id={env_id} state_dim={obs_dim + goal_dim} action_dim={action_dim} "
        f"n_train={len(bank.train_ids())} n_held={len(bank.held_out_ids())} conditions={conditions}",
        flush=True,
    )

    for condition in conditions:
        loop_condition, overrides = resolve_condition(condition, scaffold_block, rc_block)
        # anti-stall ablation: override stale_advance_windows for PCRC-family conditions.
        if args.stale_advance_windows is not None and "scaffold_kwargs" in overrides:
            overrides["scaffold_kwargs"] = {
                **overrides["scaffold_kwargs"],
                "stale_advance_windows": int(args.stale_advance_windows),
            }
        # B-PCRC discount grid (T0.3): override bpcrc_gamma for the per_context_beta_posterior condition.
        if args.bpcrc_gamma is not None and "scaffold_kwargs" in overrides:
            overrides["scaffold_kwargs"] = {
                **overrides["scaffold_kwargs"],
                "bpcrc_gamma": float(args.bpcrc_gamma),
            }
        # BO gating-knob search: merge an arbitrary JSON dict over scaffold kwargs.
        if args.scaffold_kwargs_json is not None and "scaffold_kwargs" in overrides:
            overrides["scaffold_kwargs"] = {
                **overrides["scaffold_kwargs"],
                **json.loads(args.scaffold_kwargs_json),
            }
        # mode "oracle": compute the per-context group→level map from the bank's goal-z.
        # Map + threshold come from the scaffold block (default = Gate-B mixed bank:
        # on-table z<0.45 → L0, in-air → L2). Indexed by train context_id (= controller idx).
        if overrides.get("scaffold_mode") == Method.ORACLE:
            z_thr = float(scaffold_block.get("oracle_z_threshold", 0.45))
            low = int(scaffold_block.get("oracle_low_level", 0))
            high = int(scaffold_block.get("oracle_high_level", 2))
            overrides["scaffold_oracle_levels"] = [
                low if float(bank[cid].desired_goal[2]) < z_thr else high
                for cid in bank.train_ids()
            ]
        # mode "per_group_success" (campaign 5 C5.1): per-context group ids from the same goal-z
        # split the oracle uses (0 = on-table z<thr, 1 = in-air) — oracle group labels, no levels.
        # Relabeled contiguously so single-group banks (e.g. the C5.10 all-in-air bank, where the
        # group frontier degenerates to ONE synchronized frontier) pass controller validation.
        if overrides.get("scaffold_mode") == Method.PER_GROUP_SUCCESS:
            z_thr = float(scaffold_block.get("oracle_z_threshold", 0.45))
            raw = np.asarray(
                [0 if float(bank[cid].desired_goal[2]) < z_thr else 1 for cid in bank.train_ids()]
            )
            overrides["scaffold_group_ids"] = np.unique(raw, return_inverse=True)[1].tolist()
        # replay_* (Gate E): load the recorded [T, n_train] per-context level matrix from a reference
        # pcrc run's scaffold.jsonl (selected level == ell[x] exactly, so `ell` IS the trajectory).
        if str(overrides.get("scaffold_mode", "")).startswith("replay_"):
            if args.replay_source is None or not args.replay_source.exists():
                raise FileNotFoundError(
                    f"replay condition {condition!r} needs --replay-source (got {args.replay_source})"
                )
            traj = _load_replay_trajectory(args.replay_source)
            if traj.shape[1] != len(bank.train_ids()):
                raise ValueError(
                    f"replay trajectory n_train={traj.shape[1]} != bank train {len(bank.train_ids())}"
                )
            overrides["scaffold_replay_trajectory"] = traj
            print(
                f"[pcrc-rc] replay {condition}: loaded {traj.shape[0]}x{traj.shape[1]} from "
                f"{args.replay_source}",
                flush=True,
            )
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
            f"[pcrc-rc] running condition={condition} (loop_condition={loop_condition})",
            flush=True,
        )
        eval_env = FetchTaskEnvironment(bank, env_id)
        try:
            result = run_fetch_sac_her_loop(loop_config, bank, eval_env=eval_env, logger=logger)
        finally:
            eval_env.close()
        last_eval = result.get("last_eval")
        if last_eval is not None:
            print(
                f"[pcrc-rc] condition={condition} last_eval_success_rate="
                f"{last_eval['success_rate']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
