"""Collect scripted FetchPickAndPlace demonstrations (Nair 2018 demo source).

Rolls the four-phase scripted pick-place controller
(:class:`pcrc.algorithms.pnp_demos.ScriptedPnPController`) over the train
contexts of a calibration bank and saves the successful episodes to a flat
``.npz`` for the demos + behavior-cloning lever.

Usage::

    uv run python scripts/collect_pnp_demos.py \
        --bank-path results/priority_1a_pnp_sparse_calibration/calibration/bank.json \
        --out results/pnp_demos/demos.npz \
        --n-episodes 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pcrc.algorithms.pnp_demos import collect_scripted_demos, save_demos
from pcrc.envs.fetch_task_bank import FetchTaskBank


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collect_pnp_demos")
    parser.add_argument(
        "--bank-path",
        type=Path,
        default=Path("results/priority_1a_pnp_sparse_calibration/calibration/bank.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/pnp_demos/demos.npz"))
    parser.add_argument("--env-id", type=str, default="FetchPickAndPlace-v4")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=0,
        help="number of demo episodes (0 = all train contexts).",
    )
    parser.add_argument("--success-distance-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)

    # Deferred import so the env (mujoco) is only loaded when running.
    from pcrc.envs.fetch_environment import FetchTaskEnvironment  # noqa: PLC0415

    bank = FetchTaskBank.load(args.bank_path)
    train_ids = bank.train_ids()
    context_ids = train_ids if args.n_episodes <= 0 else train_ids[: args.n_episodes]

    env = FetchTaskEnvironment(bank, args.env_id, max_steps=args.max_steps)
    try:
        episodes, collect = collect_scripted_demos(
            env,
            context_ids,
            max_steps=args.max_steps,
            success_distance_threshold=args.success_distance_threshold,
            only_successful=True,
        )
    finally:
        env.close()

    saved = save_demos(episodes, args.out)
    print(
        f"[collect-demos] attempted={collect.n_episodes_attempted} "
        f"successful={collect.n_episodes_successful} "
        f"saved_episodes={saved.n_episodes_attempted} "
        f"transitions={saved.n_transitions} -> {saved.path}",
        flush=True,
    )
    if collect.n_episodes_successful == 0:
        print("[collect-demos] WARNING: zero successful demos collected", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
