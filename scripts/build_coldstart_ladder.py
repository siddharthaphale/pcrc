"""Build the C5.13 COLD-START bootstrap ladder for MiniGrid DoorKey (campaign 5).

Unlike `build_doorkey_traj_scaffold.py` (BFS reference solutions), every context's axis here is
the DEGENERATE single-frame trajectory = its own ℓ=0 reset state, so every assisted level
collapses to the true start (`ReferenceTrajectoryScaffoldBuilder` documents this case). Training
that starts from this ladder is therefore plain target-start training — no planner, no BFS, no
scripted fallback, no prior scaffold — until `--self-online` refresh harvests a context's first
unassisted success and overwrites its axis in place. That is the cold-start self-generation test
(review Critical #6 / Question 9).

Usage::

    uv run --no-sync python scripts/build_coldstart_ladder.py \
        --out results/banks/scaffold_ladders/doorkey_8x8_coldstart.json
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  -- registers MiniGrid-* envs

from pcrc.envs.minigrid_env import GridStateCodec, MiniGridTaskBank
from pcrc.envs.trajectory_scaffold import (
    ReferenceTrajectoryScaffoldBuilder,
    ScaffoldLadder,
    TrajectoryFrame,
)

QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)  # matches config/minigrid_doorkey.yaml level_fractions


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_coldstart_ladder")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-held", type=int, default=100)
    p.add_argument("--bank-seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=160)
    a = p.parse_args(argv)

    bank = MiniGridTaskBank.build(a.env_id, n_train=a.n_train, n_held=a.n_held, seed=a.bank_seed)
    builder = ReferenceTrajectoryScaffoldBuilder(QUANTILES)
    codec = GridStateCodec()
    env = gym.make(bank.env_id, max_steps=int(a.max_steps))
    u = env.unwrapped
    ladder = ScaffoldLadder(env_id=bank.env_id, quantiles=QUANTILES)
    try:
        for cid in bank.train_ids():
            env.reset(seed=bank.contexts[cid].seed)
            frames = [TrajectoryFrame(codec.serialize(u), 0.0)]  # degenerate: s_0 only
            ladder.add_context(builder.build_context(cid, frames))
    finally:
        env.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ladder.save(a.out)
    sha = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(
        f"[build_coldstart_ladder] env={a.env_id} bank_sha={bank.sha256()[:12]} "
        f"train={len(bank.train_ids())} -> {a.out} sha256={sha[:12]} "
        f"(all {len(bank.train_ids())} contexts degenerate = true start)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
