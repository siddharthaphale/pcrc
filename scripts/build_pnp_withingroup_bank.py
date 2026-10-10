"""Build the C5.10 WITHIN-GROUP pacing-conflict bank (campaign 5, review Critical #2 second half).

Every context is nominally IN-AIR (goal-z >= 0.45 — one oracle group under the paper's
on-table/in-air split), but goal heights are sampled BIMODALLY with the middle removed:
low-air z in [0.45, 0.55) vs high-air z in [0.75, +), mirroring the constructed conflict's
removed-middle design *inside* a single nominal group. Group labels therefore cannot help a
group frontier here — it degenerates to one synchronized frontier — while per-context pacing
can still separate the sub-bands. Sub-band identity (low/high) is an ANALYSIS-side label only.

Layout mirrors pnp_conflict_bank: train ids 0-49 low-air, 50-99 high-air; held-out 100-124
low-air, 125-149 high-air.

Usage::

    uv run --no-sync python scripts/build_pnp_withingroup_bank.py \
        --out results/banks/pnp_withingroup_bank.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from pcrc.envs.fetch_calibration import _assign_quantile_bins, _direction_bin, _probe_pool
from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext

ENV_ID = "FetchPickAndPlace-v4"
Z_LOW = (0.45, 0.55)   # low-air band
Z_HIGH = (0.75, 10.0)  # high-air band (pool max ~0.87)
N_TRAIN_BAND, N_HELD_BAND = 50, 25
N_POOL = 3000
BASE_SEED = 20260717


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_pnp_withingroup_bank")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    pool = _probe_pool(ENV_ID, N_POOL, BASE_SEED)
    low = [r for r in pool if Z_LOW[0] <= float(r.desired_goal[2]) < Z_LOW[1]]
    high = [r for r in pool if Z_HIGH[0] <= float(r.desired_goal[2]) < Z_HIGH[1]]
    need = N_TRAIN_BAND + N_HELD_BAND
    if len(low) < need or len(high) < need:
        print(f"!! bands too small: low {len(low)}, high {len(high)}; need {need} each")
        return 1

    # train ids 0-49 low, 50-99 high; held-out 100-124 low, 125-149 high
    sel = low[:N_TRAIN_BAND] + high[:N_TRAIN_BAND] + low[N_TRAIN_BAND:need] + high[N_TRAIN_BAND:need]
    splits = ["train"] * (2 * N_TRAIN_BAND) + ["held_out"] * (2 * N_HELD_BAND)

    dists = np.array([float(np.linalg.norm(r.desired_goal - r.initial_achieved_goal)) for r in sel])
    distance_bins = _assign_quantile_bins(dists, 5)
    contexts: list[FetchTaskContext] = []
    initial_achieved_goals: dict[int, np.ndarray] = {}
    for i, (r, split) in enumerate(zip(sel, splits, strict=True)):
        contexts.append(
            FetchTaskContext(
                context_id=i,
                env_id=ENV_ID,
                reset_seed=r.reset_seed,
                desired_goal=r.desired_goal,
                reset_payload=r.reset_payload,
                initial_object_pose=r.initial_object_pose,
                distance_bin=int(distance_bins[i]),
                direction_bin=_direction_bin(r.desired_goal, r.initial_achieved_goal, 5),
                pose_bin=0,
                split=split,
            )
        )
        initial_achieved_goals[i] = r.initial_achieved_goal

    bank = FetchTaskBank.from_contexts(contexts, initial_achieved_goals)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bank.save(a.out)
    z = np.array([float(c.desired_goal[2]) for c in contexts])
    print(
        f"[build-withingroup] saved {len(contexts)} contexts -> {a.out}\n"
        f"  low-air train z [{z[:50].min():.3f}, {z[:50].max():.3f}], "
        f"high-air train z [{z[50:100].min():.3f}, {z[50:100].max():.3f}]; "
        f"middle ({Z_LOW[1]:.2f}-{Z_HIGH[0]:.2f}) removed; all >= 0.45 (one nominal group)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
