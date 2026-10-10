"""Build an IN-AIR FetchPickAndPlace bank for PCRC-RC (Stage 3).

The standard calibration (priority_1a) selects a medium-difficulty keep-set by
STARTER success rate, which needs a trained PPO warmup. PCRC-RC trains from
scratch with scaffolding and evaluates true-start held-out success, so it does
NOT need starter-success bins — it needs in-air contexts + a train/held split +
(geometric) difficulty bins. This builder reuses the calibration's pose sampler
(``_probe_pool``, which rolls no policy) and bins by geodesic object→goal
distance, so no starter checkpoint is required.

Writes to the path config/fetch_pick_and_place.yaml expects:
    results/priority_1a_pnp_sparse_calibration/calibration/bank.json

    uv run python scripts/build_inair_pnp_bank.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from pcrc.envs.fetch_calibration import _assign_quantile_bins, _direction_bin, _probe_pool
from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext

ENV_ID = "FetchPickAndPlace-v4"
Z_IN_AIR = 0.45  # desired_goal[2] >= this ⇒ in-air goal (needs grasp+lift)
N_TRAIN, N_HELD = 100, 50
N_POOL = 1500  # ~50% of PnP goals are in-air, so this comfortably yields 150
BASE_SEED = 20260617
OUT = Path("results/priority_1a_pnp_sparse_calibration/calibration/bank.json")


def main() -> int:
    """Sample, filter to in-air, bin, split, and save the bank."""
    pool = _probe_pool(ENV_ID, N_POOL, BASE_SEED)
    in_air = [r for r in pool if float(r.desired_goal[2]) >= Z_IN_AIR]
    need = N_TRAIN + N_HELD
    if len(in_air) < need:
        print(f"!! only {len(in_air)} in-air contexts from {N_POOL} pool; need {need}", flush=True)
        return 1
    sel = in_air[:need]

    dists = np.array([float(np.linalg.norm(r.desired_goal - r.initial_achieved_goal)) for r in sel])
    distance_bins = _assign_quantile_bins(dists, 5)

    contexts: list[FetchTaskContext] = []
    initial_achieved_goals: dict[int, np.ndarray] = {}
    for i, r in enumerate(sel):
        split = "train" if i < N_TRAIN else "held_out"
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
                pose_bin=0,  # geometric build: no pose stratification
                split=split,
            )
        )
        initial_achieved_goals[i] = r.initial_achieved_goal

    bank = FetchTaskBank.from_contexts(contexts, initial_achieved_goals)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    bank.save(OUT)
    z = np.array([float(c.desired_goal[2]) for c in contexts])
    print(
        f"[build-inair] saved {len(contexts)} contexts ({N_TRAIN} train / {N_HELD} held) -> {OUT}\n"
        f"[build-inair] goal-z range [{z.min():.3f}, {z.max():.3f}] (all >= {Z_IN_AIR}); "
        f"dist range [{dists.min():.3f}, {dists.max():.3f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
