"""Build an ON-TABLE FetchPickAndPlace bank from a calibrated pool.

Keeps only contexts whose desired_goal is on the table (z < 0.45) — i.e. a
PUSH-like task (slide the object to an on-table goal, no grasp/lift required),
which should be learnable from scratch AND have Push's large heterogeneous
low-success band. Tests whether PnP-given-a-band shows the value-uncertainty
win (vs the in-air PnP bank, which lacked the band).

Reindexes so train ids are 0..n_train-1 (loop invariant). Saves to
results/pnp_ontable_calibration/calibration/bank.json.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pcrc.envs.fetch_task_bank import FetchTaskBank

POOL = "results/priority_1a_pnp_ontable_pool/calibration/bank.json"
OUT = Path("results/pnp_ontable_calibration/calibration/bank.json")
N_TRAIN, N_HELD = 100, 50
Z_ONTABLE = 0.45


def main() -> int:
    bank = FetchTaskBank.load(POOL)
    ctxs = list(bank.contexts.values())
    ontable = [c for c in ctxs if float(c.desired_goal[2]) < Z_ONTABLE]
    train = [c for c in ontable if c.split == "train"][:N_TRAIN]
    held = [c for c in ontable if c.split == "held_out"][:N_HELD]
    if len(train) < N_TRAIN or len(held) < N_HELD:
        print(
            f"!! not enough on-table contexts: train {len(train)}/{N_TRAIN}, held {len(held)}/{N_HELD}"
        )
        return 1
    new_ctxs = []
    new_iag = {}
    for i, c in enumerate(train):
        new_ctxs.append(replace(c, context_id=i, split="train"))
        new_iag[i] = bank.initial_achieved_goals[c.context_id]
    for j, c in enumerate(held):
        nid = N_TRAIN + j
        new_ctxs.append(replace(c, context_id=nid, split="held_out"))
        new_iag[nid] = bank.initial_achieved_goals[c.context_id]
    newbank = FetchTaskBank.from_contexts(new_ctxs, new_iag)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    newbank.save(OUT)
    zs = [float(c.desired_goal[2]) for c in new_ctxs]
    print(f"wrote on-table PnP bank: {N_TRAIN} train / {N_HELD} held_out -> {OUT}")
    print(
        f"  goal z range [{min(zs):.3f}, {max(zs):.3f}] (all on-table); train_ids==range: "
        f"{newbank.train_ids() == list(range(N_TRAIN))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
