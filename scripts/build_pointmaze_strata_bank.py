"""Build a bimodal route-group PointMaze bank (direct vs bypass) for the Nav-bypass
conflict experiment (autonomous plan Gate D / breadth).

Labels (start, goal) pairs by detour ratio = geodesic/euclidean: direct (< --direct-max,
off-path scaffold preserves the route) vs bypass (>= --bypass-min, off-path scaffold skips
a bottleneck → decreasing transfer). Middle dropped → bimodal conflict. Use the bank with
`scaffold_geometry="offpath"`. Group recoverable via route_group (0=direct, 1=bypass).

    uv run python scripts/build_pointmaze_strata_bank.py \
        --out results/pointmaze_strata_calibration/calibration/bank.json
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
import sys

from pcrc.envs.pointmaze_env import PointMazeTaskBank


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_pointmaze_strata_bank")
    p.add_argument("--env-id", default="PointMaze_Large-v3")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-per-group", type=int, default=75)
    p.add_argument("--n-held-per-group", type=int, default=25)
    p.add_argument("--direct-max", type=float, default=1.3)
    p.add_argument("--bypass-min", type=float, default=2.0)
    p.add_argument("--min-distance", type=int, default=2)
    p.add_argument("--direct-min-distance", type=int, default=1)
    p.add_argument("--bypass-max-distance", type=int, default=None)
    p.add_argument(
        "--bypass-sharp-alpha",
        type=float,
        default=None,
        help="require bypass off-path at this alpha already bypassed (no-compromise gate)",
    )
    p.add_argument("--bypass-sharp-max-frac", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    bank = PointMazeTaskBank.build_strata(
        a.env_id,
        n_per_group=a.n_per_group,
        n_held_per_group=a.n_held_per_group,
        direct_max=a.direct_max,
        bypass_min=a.bypass_min,
        min_distance=a.min_distance,
        direct_min_distance=a.direct_min_distance,
        bypass_max_distance=a.bypass_max_distance,
        bypass_sharp_alpha=a.bypass_sharp_alpha,
        bypass_sharp_max_frac=a.bypass_sharp_max_frac,
        seed=a.seed,
    )
    bank.save(a.out)
    names = {0: "direct", 1: "bypass"}
    tr = collections.Counter(bank[c].route_group for c in bank.train_ids())
    hd = collections.Counter(bank[c].route_group for c in bank.held_out_ids())
    print(
        f"[build-pm] {a.env_id} -> {a.out}\n"
        f"  train {len(bank.train_ids())}: "
        + ", ".join(f"{names[g]}={tr[g]}" for g in (0, 1))
        + f"   held {len(bank.held_out_ids())}: "
        + ", ".join(f"{names[g]}={hd[g]}" for g in (0, 1))
        + f"   cuts direct<{a.direct_max} / bypass>={a.bypass_min}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
