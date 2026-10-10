"""Build a FetchPickAndPlace bank by goal-height filter (in-air / on-table / mixed / sharp).

Reuses the calibration pose sampler (`_probe_pool`, no starter policy) + geometric
distance bins. Supports the heterogeneity experiment: on-table goals (z<0.45) are
Push-like (no grasp), in-air goals (z>=0.45) need grasp+lift, and `mixed` packs both
into one bank so a single global scaffold level cannot be optimal for every context.

`sharp` (autonomous plan Gate C) builds a *sharper* 3-strata heterogeneous bank by
goal-z cuts (default on-table z<0.45 / moderate 0.45<=z<0.65 / hard z>=0.65; the hard
band is pushed up to z>=0.65 so its best fixed level lands sub-ceiling — goal-z and
grasp distance are corr~0.81, so a pure-z cut already buys "high AND long-distance").
It writes the combined bank (group recoverable from goal-z) PLUS one single-stratum
bank per group (for the per-stratum fixed-level heterogeneity sweep, C1).

    uv run python scripts/build_pnp_bank.py --filter ontable \
        --out results/pnp_ontable_calibration/calibration/bank.json
    uv run python scripts/build_pnp_bank.py --filter mixed \
        --out results/pnp_mixed_calibration/calibration/bank.json
    uv run python scripts/build_pnp_bank.py --filter sharp \
        --out results/pnp_sharp_calibration/calibration/bank.json \
        --sharp-cuts 0.45,0.65 --sharp-train 50 --sharp-held 25 --n-pool 4000
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from pcrc.envs.fetch_calibration import _assign_quantile_bins, _direction_bin, _probe_pool
from pcrc.envs.fetch_task_bank import FetchTaskBank, FetchTaskContext

ENV_ID = "FetchPickAndPlace-v4"
Z_SPLIT = 0.45  # desired_goal[2] >= this ⇒ in-air (grasp+lift); else on-table (Push-like)


def _make_contexts(records, n_train, n_held, base_id=0):
    """Build FetchTaskContexts (+ initial_achieved_goals) from probe records."""
    sel = records[: n_train + n_held]
    dists = np.array([float(np.linalg.norm(r.desired_goal - r.initial_achieved_goal)) for r in sel])
    dbins = _assign_quantile_bins(dists, 5)
    contexts, iag = [], {}
    for i, r in enumerate(sel):
        cid = base_id + i
        split = "train" if i < n_train else "held_out"
        contexts.append(
            FetchTaskContext(
                context_id=cid,
                env_id=ENV_ID,
                reset_seed=r.reset_seed,
                desired_goal=r.desired_goal,
                reset_payload=r.reset_payload,
                initial_object_pose=r.initial_object_pose,
                distance_bin=int(dbins[i]),
                direction_bin=_direction_bin(r.desired_goal, r.initial_achieved_goal, 5),
                pose_bin=0,
                split=split,
            )
        )
        iag[cid] = r.initial_achieved_goal
    return contexts, iag


def _split_by_cuts(records, cuts):
    """Partition probe records into len(cuts)+1 strata by desired_goal[2] (ascending cuts).

    Stratum index = number of cut thresholds the goal-z meets/exceeds (0 = lowest band).
    Pure (no env) so the stratification is unit-testable.
    """
    cuts = sorted(float(c) for c in cuts)
    groups = [[] for _ in range(len(cuts) + 1)]
    for r in records:
        z = float(r.desired_goal[2])
        gi = sum(1 for c in cuts if z >= c)
        groups[gi].append(r)
    return groups


def _build_sharp(records, cuts, n_train_s, n_held_s, names, keep=None):
    """Build a combined multi-strata bank + one single-stratum bank per group.

    Returns (combined_contexts, combined_iag, strata) where strata is a list of
    (name, contexts, iag) with per-stratum-local context ids (0..n_train_s-1 train,
    then held). The combined bank reindexes to a contiguous train-block then held-block
    (so train ids 0..N_train-1 = controller indices) with group recoverable from goal-z.
    ``keep`` (default = all names) selects which strata go into the COMBINED bank — used to
    drop a middle stratum and build a bimodal max-conflict bank; all stratum files are still
    returned regardless.
    """
    groups = _split_by_cuts(records, cuts)
    if len(groups) != len(names):
        raise ValueError(f"{len(names)} names but cuts give {len(groups)} strata")
    keep = list(names) if keep is None else list(keep)
    unknown = [k for k in keep if k not in names]
    if unknown:
        raise ValueError(f"--sharp-keep names {unknown} not in strata {names}")
    strata = []
    for name, recs in zip(names, groups):
        if len(recs) < n_train_s + n_held_s:
            raise ValueError(
                f"stratum {name!r}: only {len(recs)} records; need {n_train_s + n_held_s} "
                "(raise --n-pool or widen the cut)"
            )
        c, i = _make_contexts(recs, n_train_s, n_held_s, base_id=0)
        strata.append((name, c, i))

    comb_c, comb_i, nid = [], {}, 0
    for want in ("train", "held_out"):  # train-block first, then held-block
        for name, c, i in strata:
            if name not in keep:
                continue
            for ctx in c:
                if ctx.split == want:
                    comb_c.append(replace(ctx, context_id=nid))
                    comb_i[nid] = i[ctx.context_id]
                    nid += 1
    return comb_c, comb_i, strata


def main(argv: list[str] | None = None) -> int:
    """Sample, filter by goal height, split, save."""
    p = argparse.ArgumentParser(prog="build_pnp_bank")
    p.add_argument("--filter", choices=("inair", "ontable", "mixed", "sharp"), default="inair")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-held", type=int, default=50)
    p.add_argument("--n-pool", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260618)
    # sharp-only knobs (Gate C):
    p.add_argument("--sharp-cuts", type=str, default="0.45,0.65", help="ascending goal-z cuts")
    p.add_argument("--sharp-train", type=int, default=50, help="train contexts per stratum")
    p.add_argument("--sharp-held", type=int, default=25, help="held-out contexts per stratum")
    p.add_argument("--sharp-names", type=str, default="ontable,moderate,hard")
    p.add_argument(
        "--sharp-keep", type=str, default="", help="strata to put in combined bank (default all)"
    )
    args = p.parse_args(argv)

    pool = _probe_pool(ENV_ID, args.n_pool, args.seed)

    if args.filter == "sharp":
        cuts = [float(c) for c in args.sharp_cuts.split(",") if c.strip()]
        names = [n.strip() for n in args.sharp_names.split(",") if n.strip()]
        keep = [k.strip() for k in args.sharp_keep.split(",") if k.strip()] or None
        contexts, iag, strata = _build_sharp(
            pool, cuts, args.sharp_train, args.sharp_held, names, keep=keep
        )
        bank = FetchTaskBank.from_contexts(contexts, iag)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        bank.save(args.out)
        z = np.array([float(c.desired_goal[2]) for c in contexts])
        n_kept = len(contexts) // (args.sharp_train + args.sharp_held)
        print(
            f"[build-pnp] filter=sharp combined {len(contexts)} ctx "
            f"({n_kept} strata x {args.sharp_train} train/{args.sharp_held} held; "
            f"keep={keep or names}) cuts={cuts} goal-z [{z.min():.3f},{z.max():.3f}] -> {args.out}",
            flush=True,
        )
        for name, c, i in strata:
            sbank = FetchTaskBank.from_contexts(c, i)
            spath = args.out.parent / f"stratum_{name}.json"
            sbank.save(spath)
            sz = np.array([float(ctx.desired_goal[2]) for ctx in c])
            sd = np.array(
                [float(np.linalg.norm(ctx.desired_goal - i[ctx.context_id])) for ctx in c]
            )
            print(
                f"[build-pnp]   stratum {name:9s} {len(c)} ctx "
                f"({args.sharp_train} train/{args.sharp_held} held) "
                f"z[{sz.min():.3f},{sz.max():.3f}] dist_med={np.median(sd):.3f} -> {spath}",
                flush=True,
            )
        return 0

    inair = [r for r in pool if float(r.desired_goal[2]) >= Z_SPLIT]
    ontable = [r for r in pool if float(r.desired_goal[2]) < Z_SPLIT]

    if args.filter == "mixed":
        # Half on-table, half in-air; held-out matched. Group is recoverable from goal z.
        nt2, nh2 = args.n_train // 2, args.n_held // 2
        if len(ontable) < nt2 + nh2 or len(inair) < nt2 + nh2:
            print(f"!! pool too small: ontable {len(ontable)}, inair {len(inair)}", flush=True)
            return 1
        ot_c, ot_i = _make_contexts(ontable, nt2, nh2, base_id=0)
        ia_c, ia_i = _make_contexts(inair, args.n_train - nt2, args.n_held - nh2, base_id=10_000)
        # reindex train ids 0..n_train-1 and held ids n_train..; interleave groups
        contexts = ot_c + ia_c
        iag = {**ot_i, **ia_i}
        # reassign contiguous ids preserving split
        trains = [c for c in contexts if c.split == "train"]
        helds = [c for c in contexts if c.split == "held_out"]

        new_c, new_i = [], {}
        for i, c in enumerate(trains):
            new_c.append(replace(c, context_id=i))
            new_i[i] = iag[c.context_id]
        for j, c in enumerate(helds):
            nid = args.n_train + j
            new_c.append(replace(c, context_id=nid))
            new_i[nid] = iag[c.context_id]
        contexts, iag = new_c, new_i
    else:
        recs = inair if args.filter == "inair" else ontable
        if len(recs) < args.n_train + args.n_held:
            print(f"!! only {len(recs)} {args.filter} contexts; need {args.n_train + args.n_held}")
            return 1
        contexts, iag = _make_contexts(recs, args.n_train, args.n_held)

    bank = FetchTaskBank.from_contexts(contexts, iag)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bank.save(args.out)
    z = np.array([float(c.desired_goal[2]) for c in contexts])
    n_air = int((z >= Z_SPLIT).sum())
    print(
        f"[build-pnp] filter={args.filter} saved {len(contexts)} ctx "
        f"({args.n_train} train/{args.n_held} held; {n_air} in-air, {len(contexts) - n_air} on-table) "
        f"goal-z [{z.min():.3f},{z.max():.3f}] -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
