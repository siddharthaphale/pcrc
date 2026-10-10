"""Build + save a deterministic Meta-World PickPlace context bank and print its SHA-256.

Matches the bank the PCRC driver builds in-memory from ``bank_seed`` (config
``metaworld_pickplace.yaml``: n_train=100, n_held=50, seed=0), so the printed hash is the bank hash
the dev/final runs use. Usage:

    uv run python scripts/build_mw_pickplace_bank.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pcrc.envs.metaworld_env import MetaWorldTaskBank
from pcrc.utils.paths import RESULTS_ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_mw_pickplace_bank")
    ap.add_argument("--task", default="pick-place-v3")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-held", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(RESULTS_ROOT) / "pcrc_new_tasks" / "metaworld_pickplace" / "bank_seed0.json",
    )
    a = ap.parse_args(argv)
    bank = MetaWorldTaskBank.build(a.task, n_train=a.n_train, n_held=a.n_held, seed=a.seed)
    path = bank.save(a.out)
    train, held = bank.train_ids(), bank.held_out_ids()
    assert set(train).isdisjoint(set(held)), "train/held overlap"
    print(f"task={a.task} seed={a.seed} n_train={len(train)} n_held={len(held)}")
    print(f"saved: {path}")
    print(f"sha256: {bank.sha256()}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
