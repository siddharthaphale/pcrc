"""Build a trajectory-derived scaffold ladder for in-air FetchPickAndPlace (Stage 3).

Reference trajectory = one frozen scripted expert per train context — the four-phase
:class:`~pcrc.algorithms.pnp_demos.ScriptedPnPController` (hover → descend → grasp → carry-to-goal),
the same demo source used for BC. At each step we serialize the **full MuJoCo state** via
:class:`~pcrc.envs.fetch_environment.FetchStateCodec` (qpos/qvel/**ctrl**/mocap — ctrl holds the
grasp, so a restored in-air carry state keeps the block pinched) and record cumulative **object**
(achieved_goal) arc length as progress. States are selected at quantiles {0, 0.25, 0.5, 0.75, 0.9}
via :class:`ReferenceTrajectoryScaffoldBuilder`.

Because the object is stationary during approach/grasp and only moves during the carry, the assisted
quantiles all land in the carry phase: level 0 is the true start (object on the table, not grasped),
and levels 1..4 are the object *grasped and lifted* at increasing fractions toward the in-air goal —
the physical analog of the manual reverse curriculum (which *synthesizes* a grasped state at the
interpolated point). Stage-3 expectation: automatic ~ manual PCRC (lower retention bar, [0.75, 0.85],
since exact physical restoration is harder than analytic synthesis).

One successful trajectory per context; the expert is NOT selected on the downstream PCRC result. A
context whose expert fails is logged and excluded (writes an exclusion list; ≤10% cap — the scripted
controller reaches 1.00 on the in-air calibration bank, so in practice nothing is excluded). Validate
with ``scripts/validate_traj_scaffold.py --task fetch --ladder <out> --bank <bank>``.

Usage::

    uv run --no-sync python scripts/build_fetch_traj_scaffold.py \
        --bank results/priority_1a_pnp_sparse_calibration/calibration/bank.json \
        --out results/banks/scaffold_ladders/fetch_inair_pnp_s0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pcrc.algorithms.pnp_demos import ScriptedPnPController
from pcrc.envs.fetch_environment import FetchStateCodec, FetchTaskEnvironment
from pcrc.envs.fetch_task_bank import FetchTaskBank
from pcrc.envs.trajectory_scaffold import (
    ReferenceTrajectoryScaffoldBuilder,
    ScaffoldLadder,
    TrajectoryFrame,
)

QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9)  # continuous convention (matches PointMaze + validate_fetch)


def roll_expert(
    env: FetchTaskEnvironment,
    codec: FetchStateCodec,
    cid: int,
    max_steps: int,
    noise_std: float = 0.0,
    rng=None,
):
    """Roll the scripted expert once; return (frames, success, end_obj_z). frames[0] is true start.

    Progress = cumulative arc length of the object (achieved_goal); serialized states are the full
    MuJoCo snapshot AFTER each step. ``noise_std``>0 adds Gaussian action noise (Stage-4 ablation:
    distinct successful expert trajectories per context); ``noise_std==0`` is the clean expert.
    """
    od = env.reset(int(cid))
    ctl = ScriptedPnPController()
    u = env._env.unwrapped
    frames = [TrajectoryFrame(codec.serialize(u), 0.0)]
    prev = np.asarray(od["achieved_goal"], dtype=np.float64)
    cum = 0.0
    success = False
    for _ in range(max_steps):
        action = ctl.act(od["observation"], od["desired_goal"])
        if noise_std > 0.0 and rng is not None:
            action = np.clip(action + rng.normal(0.0, noise_std, size=action.shape), -1.0, 1.0)
            action = action.astype(np.float32)
        od2, _r, term, trunc, info = env.step(action)
        ag = np.asarray(od2["achieved_goal"], dtype=np.float64)
        cum += float(np.linalg.norm(ag - prev))
        prev = ag
        frames.append(TrajectoryFrame(codec.serialize(u), cum))
        if bool(info.get("is_success", False)):
            success = True
        od = od2
        if term or trunc:
            break
    end_obj_z = float(prev[2])  # object z at the end of the carry (in-air => grasped lift captured)
    return frames, success, end_obj_z


def build_ladder(
    bank: FetchTaskBank,
    env_id: str,
    max_steps: int,
    quantiles=QUANTILES,
    n_traj: int = 1,
    noise_std: float = 0.05,
    seed: int = 0,
):
    """Build a ladder over ``bank``'s train contexts from scripted-expert trajectories.

    ``n_traj``>1 (Stage-4 ablation): per context, collect up to ``n_traj`` SUCCESSFUL expert rollouts
    — trajectory 0 is the clean expert, the rest add Gaussian action noise (``noise_std``) — and merge
    their quantile states into one multi-trajectory cell (env samples 1-of-N per reset). Noised rolls
    that fail are retried a few times, then skipped (a context keeps however many succeeded; >=1 always,
    since the clean expert reaches 1.00 on this bank).
    """
    builder = ReferenceTrajectoryScaffoldBuilder(quantiles)
    codec = FetchStateCodec()
    env = FetchTaskEnvironment(bank, env_id, max_steps=max_steps)
    ladder = ScaffoldLadder(env_id=env_id, quantiles=tuple(quantiles))
    rng = np.random.default_rng(seed)
    excluded: list[int] = []
    end_obj_z: list[float] = []  # object z at the end of the carry (in-air => grasped lift captured)
    arclens: list[float] = []
    traj_per_ctx: list[int] = []
    try:
        for cid in bank.train_ids():
            frames_list = []
            for k in range(n_traj):
                std = 0.0 if k == 0 else noise_std
                got = None
                for _retry in range(4):  # noised rolls occasionally miss; retry a few times
                    frames, success, oz = roll_expert(env, codec, cid, max_steps, std, rng)
                    if success:
                        got = (frames, oz)
                        break
                    if std == 0.0:
                        break  # clean expert deterministic; no point retrying
                if got is not None:
                    frames_list.append(got[0])
                    if k == 0:
                        arclens.append(got[0][-1].cumulative_progress)
                        end_obj_z.append(got[1])
            if not frames_list:
                excluded.append(int(cid))
                continue
            traj_per_ctx.append(len(frames_list))
            cells = (
                builder.build_context(cid, frames_list[0])
                if len(frames_list) == 1
                else builder.build_context_multi(cid, frames_list)
            )
            ladder.add_context(cells)
    finally:
        env.close()
    stats = {
        "n_train": len(bank.train_ids()),
        "n_excluded": len(excluded),
        "excluded_ids": excluded,
        "frac_excluded": len(excluded) / max(1, len(bank.train_ids())),
        "arc_min": float(min(arclens)) if arclens else 0.0,
        "arc_median": float(np.median(arclens)) if arclens else 0.0,
        "arc_max": float(max(arclens)) if arclens else 0.0,
        "end_obj_z_median": float(np.median(end_obj_z)) if end_obj_z else 0.0,
        "traj_per_ctx_median": float(np.median(traj_per_ctx)) if traj_per_ctx else 0.0,
        "traj_per_ctx_min": int(min(traj_per_ctx)) if traj_per_ctx else 0,
    }
    return ladder, stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_fetch_traj_scaffold")
    p.add_argument("--bank", type=Path, required=True, help="in-air FetchPickAndPlace bank")
    p.add_argument("--out", type=Path, required=True, help="output ScaffoldLadder JSON")
    p.add_argument("--max-steps", type=int, default=50, help="expert rollout horizon (config max_steps)")
    p.add_argument(
        "--n-traj",
        type=int,
        default=1,
        help="reference trajectories per context (Stage-4 ablation: 1 vs 3, env samples 1-of-N)",
    )
    p.add_argument("--noise-std", type=float, default=0.05, help="action-noise std for trajs > 0")
    p.add_argument("--traj-seed", type=int, default=0, help="seed for expert action noise")
    a = p.parse_args(argv)

    bank = FetchTaskBank.load(a.bank)
    ladder, stats = build_ladder(
        bank, bank.env_id, a.max_steps, n_traj=a.n_traj, noise_std=a.noise_std, seed=a.traj_seed
    )
    if stats["frac_excluded"] > 0.10:
        print(
            f"[build_fetch_traj_scaffold] ABORT: {stats['n_excluded']}/{stats['n_train']} "
            f"({stats['frac_excluded']:.1%}) contexts have no successful expert (> 10% cap). "
            f"excluded={stats['excluded_ids']}",
            flush=True,
        )
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ladder.save(a.out)
    if stats["n_excluded"]:
        excl_path = a.out.with_suffix(".excluded.json")
        excl_path.write_text(json.dumps(stats["excluded_ids"]))
        print(f"  excluded contexts -> {excl_path} (exclude from ALL conditions before training)")
    sha = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(
        f"[build_fetch_traj_scaffold] env={bank.env_id} train={stats['n_train']} "
        f"levels={ladder.n_levels} excluded={stats['n_excluded']} ({stats['frac_excluded']:.1%})",
        flush=True,
    )
    print(
        f"  object arc length: min={stats['arc_min']:.3f} median={stats['arc_median']:.3f} "
        f"max={stats['arc_max']:.3f}; end object-z median={stats['end_obj_z_median']:.3f} "
        f"(in-air => grasped lift captured)",
        flush=True,
    )
    print(
        f"  trajectories/context: median={stats['traj_per_ctx_median']} min={stats['traj_per_ctx_min']} "
        f"(n_traj requested={a.n_traj})",
        flush=True,
    )
    print(f"  -> {a.out}  sha256={sha[:16]}…", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
