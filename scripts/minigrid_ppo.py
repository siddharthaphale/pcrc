"""MiniGrid DoorKey PCRC driver: discrete PPO + scaffold controller, one or more conditions.

Builds a deterministic MiniGridTaskBank (layout seeds) and runs ``run_minigrid_ppo_loop`` per condition.
Outputs land under ``$PCRC_SCRATCH_ROOT/results/<exp-name>/<condition>/{eval,training}/uniform/*.jsonl``,
the same layout the analyzer reads.

    uv run python scripts/minigrid_ppo.py --config config/minigrid_doorkey.yaml \
        --exp-name pcrc_minigrid_doorkey_dev/s40 --conditions per_context_success --seed 40 --n-iterations 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pcrc.envs.minigrid_env import MiniGridTaskBank, MiniGridTaskEnvironment
from pcrc.envs.trajectory_scaffold import ScaffoldLadder
from pcrc.registry import (
    DEFAULT_METHOD,
    TRAJECTORY_PREFIX,
    Method,
    ScaffoldSource,
    condition_help,
    parse_conditions,
)
from pcrc.samplers.scaffold_controller import ScaffoldConfig, ScaffoldController
from pcrc.training.minigrid_ppo_loop import MiniGridPPOConfig, run_minigrid_ppo_loop
from pcrc.utils.paths import RESULTS_ROOT


def make_controller(
    condition: str,
    n_train: int,
    n_levels: int,
    level_fractions: list[float],
    anneal_iters: int,
    stale_advance_windows: int,
) -> ScaffoldController:
    """Map a condition name to a ScaffoldController (true-start = fixed level 0)."""
    # trajectory_<mode>: identical controller to <mode>; the env gets scaffold_source="trajectory" in main().
    if condition.startswith(TRAJECTORY_PREFIX):
        condition = condition[len(TRAJECTORY_PREFIX) :]
    if condition in (Method.UNASSISTED, "fixed0"):
        cfg = ScaffoldConfig(mode=Method.FIXED, fixed_level=0)
    elif condition.startswith(Method.FIXED) and condition[len(Method.FIXED) :].isdigit():
        cfg = ScaffoldConfig(mode=Method.FIXED, fixed_level=int(condition[len(Method.FIXED) :]))
    elif condition == Method.RANDOM_SCAFFOLD:
        cfg = ScaffoldConfig(mode=Method.RANDOM_SCAFFOLD)
    elif condition.startswith(Method.GLOBAL_ANNEAL) and condition[len(Method.GLOBAL_ANNEAL) :].isdigit():
        cfg = ScaffoldConfig(
            mode=Method.GLOBAL_ANNEAL, anneal_iters=int(condition[len(Method.GLOBAL_ANNEAL) :])
        )
    elif condition == Method.GLOBAL_ANNEAL:
        cfg = ScaffoldConfig(mode=Method.GLOBAL_ANNEAL, anneal_iters=anneal_iters)
    elif condition == Method.PER_CONTEXT_SUCCESS:
        cfg = ScaffoldConfig(mode=Method.PER_CONTEXT_SUCCESS, stale_advance_windows=stale_advance_windows)
    elif condition == Method.RFCL_STYLE:
        # RFCL-style: per-context frontier, uniform sampling, NO anti-stall (Test Campaign 4 Stage B).
        cfg = ScaffoldConfig(mode=Method.RFCL_STYLE, stale_advance_windows=0)
    elif condition in (Method.PER_CONTEXT_WILSON_CONFIDENCE, Method.PER_CONTEXT_BETA_POSTERIOR, Method.PER_CONTEXT_SUCCESS_TREND):
        # RLC-upgrade evidence-gated controllers (Wilson-CI / Beta-posterior transitions); same
        # anti-stall budget as the incumbent per_context_success.
        cfg = ScaffoldConfig(mode=condition, stale_advance_windows=stale_advance_windows)
    elif condition == Method.GLOBAL_MIXTURE:
        # Asynchronous population-level baseline (time-varying global level mix, no per-context
        # frontiers) — the decisive non-frontier control from the PnP conflict, now for MiniGrid.
        cfg = ScaffoldConfig(mode=Method.GLOBAL_MIXTURE)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return ScaffoldController(n_train, n_levels, level_fractions, cfg=cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="minigrid_ppo")
    ap.add_argument("--config", type=Path, default=Path("config/minigrid_doorkey.yaml"))
    ap.add_argument("--exp-name", type=str, required=True)
    ap.add_argument("--results-root", type=Path, default=None)
    ap.add_argument("--conditions", type=str, default=DEFAULT_METHOD, help=condition_help("minigrid"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-iterations", type=int, default=None)
    ap.add_argument(
        "--n-eval-episodes", type=int, default=None,
        help="held-out contexts per eval (0 = ALL held-out; default = config). The held-out list is "
        "group-ordered in conflict banks, so truncated eval can silently drop a whole group.",
    )
    ap.add_argument(
        "--ladder-path", type=Path, default=None, help="ScaffoldLadder JSON for trajectory_* conditions"
    )
    ap.add_argument(
        "--bank-path", type=Path, default=None,
        help="load a saved MiniGridTaskBank (e.g. per-group conflict sub-bank) instead of building from config",
    )
    ap.add_argument(
        "--harvest", action="store_true",
        help="Stage A: after training, harvest the policy's own ℓ=0 first-success rollout per train "
        "context to <out_dir>/harvest_ell0.json (self-generated scaffold source)",
    )
    ap.add_argument("--harvest-k", type=int, default=5, help="stochastic ℓ=0 rollouts/context; keep shortest success")
    ap.add_argument("--harvest-seed", type=int, default=99, help="dedicated harvest seed (disjoint from eval seeds)")
    ap.add_argument("--self-online", action="store_true",
                    help="Pillar 2: online self-generation — refresh the scaffold axis from the policy's own "
                    "ℓ=0 successes DURING training (trajectory_ condition bootstraps from --ladder-path, then self-shapes)")
    ap.add_argument("--self-refresh-interval", type=int, default=200, help="online self-gen: re-harvest every N iters")
    ap.add_argument("--self-refresh-k", type=int, default=3, help="online self-gen: stochastic rollouts/context per refresh")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(a.config.read_text())
    d, lp, sc = cfg["defaults"], cfg["loop"], cfg["scaffold"]
    seed = a.seed if a.seed is not None else int(lp.get("seed", 0))
    n_iters = a.n_iterations if a.n_iterations is not None else int(lp["n_iterations"])
    results = a.results_root or RESULTS_ROOT

    if a.bank_path is not None:
        bank = MiniGridTaskBank.load(a.bank_path)  # custom bank (e.g. per-group conflict sub-bank)
    else:
        bank = MiniGridTaskBank.build(
            d["env_id"], n_train=int(d["n_train"]), n_held=int(d["n_held"]), seed=int(d["bank_seed"])
        )
    conditions = parse_conditions(a.conditions, "minigrid")
    print(
        f"[minigrid] env={d['env_id']} bank={len(bank)} (train {len(bank.train_ids())}, "
        f"held {len(bank.held_out_ids())}) sha={bank.sha256()[:12]} seed={seed} "
        f"iters={n_iters} conditions={conditions}",
        flush=True,
    )

    # trajectory_* conditions restore an extracted ladder (trajectory source) for the TRAIN env; eval is
    # always level-0 true start, so the eval env stays manual.
    ladder = ScaffoldLadder.load(a.ladder_path) if a.ladder_path is not None else None

    # Fail fast if the ladder does not cover the loaded bank's train contexts (e.g. the generic default
    # paired with a per-group sub-bank): trajectory restoration would otherwise mis-restore foreign states.
    if ladder is not None and any(c.startswith(TRAJECTORY_PREFIX) for c in conditions):
        missing = [c for c in bank.train_ids() if (int(c), 1) not in ladder.cells]
        if missing:
            raise ValueError(
                f"--ladder-path {a.ladder_path} is missing restore cells for {len(missing)} train "
                f"contexts (e.g. {missing[:5]}); it does not match --bank-path {a.bank_path}. "
                f"Pass the matching per-group ladder."
            )

    for condition in conditions:
        is_auto = condition.startswith(TRAJECTORY_PREFIX)
        if is_auto and ladder is None:
            raise ValueError(f"condition {condition!r} needs --ladder-path (trajectory scaffold)")
        env = MiniGridTaskEnvironment(
            bank,
            d["env_id"],
            max_steps=int(lp["max_steps"]),
            scaffold_source=ScaffoldSource.TRAJECTORY if is_auto else ScaffoldSource.MANUAL,
            scaffold_ladder=ladder if is_auto else None,
        )
        eval_env = MiniGridTaskEnvironment(bank, d["env_id"], max_steps=int(lp["max_steps"]))
        # For trajectory (trajectory_) conditions the LADDER is the source of truth for the scaffold levels —
        # derive n_levels/level_fractions from it so the controller matches ladders of any level count
        # (Stage-B long-horizon 5/9/13 sweep). Non-auto conditions keep the config's levels. When the
        # ladder's quantiles equal the config (the default 5-level case) this is a no-op.
        if is_auto and ladder is not None:
            n_levels, level_fractions = ladder.n_levels, list(ladder.quantiles)
        else:
            n_levels, level_fractions = int(sc["n_levels"]), list(sc["level_fractions"])
        controller = make_controller(
            condition,
            len(bank.train_ids()),
            n_levels,
            level_fractions,
            int(sc["anneal_iters"]),
            int(sc["stale_advance_windows"]),
        )
        loop_cfg = MiniGridPPOConfig(
            n_iterations=n_iters,
            episodes_per_iter=int(lp["episodes_per_iter"]),
            max_steps=int(lp["max_steps"]),
            gamma=float(lp["gamma"]),
            gae_lambda=float(lp["gae_lambda"]),
            clip=float(lp["clip"]),
            epochs=int(lp["epochs"]),
            minibatch=int(lp["minibatch"]),
            lr=float(lp["lr"]),
            ent_coef=float(lp["ent_coef"]),
            encoder=str(lp["encoder"]),
            eval_interval=int(lp["eval_interval"]),
            n_eval_episodes=(
                (None if a.n_eval_episodes <= 0 else a.n_eval_episodes)
                if a.n_eval_episodes is not None
                else (None if lp["n_eval_episodes"] is None else int(lp["n_eval_episodes"]))
            ),
            seed=seed,
            harvest_path=str(results / a.exp_name / condition / "harvest_ell0.json") if a.harvest else None,
            harvest_k=a.harvest_k,
            harvest_seed=a.harvest_seed,
            self_online=bool(a.self_online) and is_auto,  # online self-gen only meaningful with a trajectory ladder
            self_refresh_interval=a.self_refresh_interval,
            self_refresh_k=a.self_refresh_k,
        )
        out_dir = results / a.exp_name / condition
        print(f"[minigrid] running condition={condition} -> {out_dir}", flush=True)
        res = run_minigrid_ppo_loop(env, eval_env, bank, controller, loop_cfg, out_dir)
        env.close()
        eval_env.close()
        print(f"[minigrid] condition={condition} done: {res}", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
