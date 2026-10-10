"""Build a SELF-generated scaffold ladder from the policy's own ℓ=0 first-success rollouts (Stage A).

This is the Campaign-4 flagship: swap the *reference source* of the (unchanged)
:class:`ReferenceTrajectoryScaffoldBuilder` from a planner/BFS/scripted-expert to the **policy's own
first successful rollout** per training context. The rollout frames are produced by the in-loop harvest
(``--harvest`` on the minigrid / fetch drivers) as ``harvest_ell0.json`` files:

    {"env_id", "source":"policy_rollout", "n_train", "n_solved",
     "frames": {"<cid>": [[serialized_state, progress], ...]}}   # shortest ℓ=0 success per solved ctx

Each frame sequence already starts at the true start (progress 0) and drops the goal-reaching step, so it
feeds :meth:`ReferenceTrajectoryScaffoldBuilder.build_context` directly.

**Tiered provenance (A.2 bootstrap).** Harvest files are consumed in priority order — pass the unassisted
(true-start) harvest first (Tier-1, *pure self*), then the scaffold-trained (auto) harvest (Tier-2,
*bootstrapped self*). A context solved by neither falls back to the scripted ``--fallback-ladder`` cells
(Tier-3, *scripted*). The output ``bootstrap_fraction`` = (|Tier-2| + |Tier-3|) / n_train bounds how
"self-generated" the ladder is; Tier-3 is reported separately as the hardest bound. Train ids and quantiles
are taken from the fallback ladder (it covers every train context), so no bank is needed.

Usage::

    uv run --no-sync python scripts/build_self_traj_scaffold.py \
        --harvest .../unassisted/harvest_ell0.json .../trajectory_per_context_success/harvest_ell0.json \
        --fallback-ladder results/banks/scaffold_ladders/doorkey_8x8_s0.json \
        --out results/banks/scaffold_ladders/doorkey_8x8_self.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pcrc.envs.trajectory_scaffold import (
    ReferenceTrajectoryScaffoldBuilder,
    ScaffoldLadder,
    TrajectoryFrame,
)


def _load_harvest(path: Path) -> tuple[str, dict[int, list[TrajectoryFrame]]]:
    """Load a harvest_ell0.json → (env_id, {cid: [TrajectoryFrame,...]}) for solved contexts."""
    d = json.loads(Path(path).read_text())
    frames = {
        int(cid): [TrajectoryFrame(f[0], float(f[1])) for f in seq]
        for cid, seq in d.get("frames", {}).items()
    }
    return str(d.get("env_id", "")), frames


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_self_traj_scaffold")
    p.add_argument(
        "--harvest", type=Path, nargs="+", required=True,
        help="harvest_ell0.json files in PRIORITY order (Tier-1 true-start first, then Tier-2 auto)",
    )
    p.add_argument(
        "--fallback-ladder", type=Path, required=True,
        help="scripted ScaffoldLadder — supplies train ids, quantiles, and Tier-3 fallback cells",
    )
    p.add_argument("--out", type=Path, required=True, help="output self ScaffoldLadder JSON")
    a = p.parse_args(argv)

    fallback = ScaffoldLadder.load(a.fallback_ladder)
    train_ids = fallback.context_ids()
    quantiles = fallback.quantiles
    builder = ReferenceTrajectoryScaffoldBuilder(quantiles)

    # Harvest sources in priority order; label each by its parent-dir condition name for the tier report.
    sources: list[tuple[str, dict[int, list[TrajectoryFrame]]]] = []
    for hp in a.harvest:
        env_id, frames = _load_harvest(hp)
        sources.append((hp.parent.name, frames))
        print(f"[self] harvest {hp.parent.name}: {len(frames)} solved contexts (env {env_id})", flush=True)

    ladder = ScaffoldLadder(env_id=fallback.env_id, quantiles=quantiles)
    tier_of: dict[int, str] = {}
    n_pure = n_boot = n_scripted = n_excluded = 0
    for cid in train_ids:
        chosen = None
        for tier_idx, (name, frames) in enumerate(sources):
            if cid in frames and len(frames[cid]) >= 2:
                chosen = (tier_idx, name, frames[cid])
                break
        if chosen is not None:
            tier_idx, name, seq = chosen
            ladder.add_context(builder.build_context(cid, seq))
            tier_of[cid] = f"self:{name}"
            if tier_idx == 0:
                n_pure += 1
            else:
                n_boot += 1
        else:
            # Tier-3: neither tier's policy solved ℓ=0 → scripted fallback cells (fully non-self).
            cells = [fallback.get(cid, lvl) for lvl in range(fallback.n_levels)]
            ladder.add_context(cells)
            tier_of[cid] = "scripted_fallback"
            n_scripted += 1

    n_train = len(train_ids)
    bootstrap_fraction = (n_boot + n_scripted) / max(1, n_train)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ladder.save(a.out)
    # Sidecar provenance: per-context tier + the three fractions that bound the self-generation claim.
    prov = {
        "source": "policy_rollout",
        "n_train": n_train,
        "n_pure_self": n_pure,
        "n_bootstrapped_self": n_boot,
        "n_scripted_fallback": n_scripted,
        "n_excluded": n_excluded,
        "pure_self_fraction": n_pure / max(1, n_train),
        "bootstrap_fraction": bootstrap_fraction,
        "scripted_fallback_fraction": n_scripted / max(1, n_train),
        "tier_of_context": {str(c): t for c, t in tier_of.items()},
    }
    a.out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2))
    sha = hashlib.sha256(a.out.read_bytes()).hexdigest()

    print(
        f"[self] SELF ladder env={ladder.env_id} train={n_train} levels={ladder.n_levels}\n"
        f"  pure-self (Tier-1)        = {n_pure:3d}  ({n_pure / max(1, n_train):.1%})\n"
        f"  bootstrapped-self (Tier-2)= {n_boot:3d}  ({n_boot / max(1, n_train):.1%})\n"
        f"  scripted-fallback (Tier-3)= {n_scripted:3d}  ({n_scripted / max(1, n_train):.1%})\n"
        f"  => bootstrap_fraction     = {bootstrap_fraction:.3f}  (non-pure-self share)\n"
        f"  -> {a.out}  sha256={sha[:16]}…",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
