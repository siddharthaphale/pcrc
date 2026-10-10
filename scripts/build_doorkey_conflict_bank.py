"""DoorKey natural pacing-conflict bank (Test Campaign 4 Stage C, C.2.2/C.2.3).

Partition the candidate pool (build_doorkey_conflict_pool.py) into natural easy/hard pacing groups using
geometry/BFS features only (no RL), then build:
  - the full conflict bank (easy+hard train / easy+hard held) + a group-map sidecar;
  - the automatic trajectory-derived ladder over its train contexts;
  - per-group sub-banks (easy / hard), each train+held of one group, for the Gate-C1 fixed-level sweep
    (the ladder is shared — context_ids are stable across the full and per-group banks).

Groups (campaign C.2.2): easy = short solution + early key (T<=p35 and t_key/T<=0.30); hard = long
solution + late key/door (T>=p65 and (t_key/T>=0.35 or t_door/T>=0.70)).

Usage: uv run python scripts/build_doorkey_conflict_bank.py \
    --pool results/banks/doorkey_conflict_pool.json \
    --out-prefix results/banks/doorkey_conflict
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from pcrc.envs.minigrid_env import MiniGridContext, MiniGridTaskBank
from pcrc.envs.trajectory_scaffold import ScaffoldLadder

_DKB = importlib.util.spec_from_file_location(
    "_dkb", Path(__file__).with_name("build_doorkey_traj_scaffold.py")
)
_dk = importlib.util.module_from_spec(_DKB)
_DKB.loader.exec_module(_dk)


def partition(pool: list[dict]) -> tuple[list[dict], list[dict]]:
    T = np.array([c["T"] for c in pool])
    tkr = np.array([c["t_key"] / max(1, c["T"]) for c in pool])
    tdr = np.array([c["t_door"] / max(1, c["T"]) for c in pool])
    t35, t65 = np.percentile(T, [35, 65])
    easy_m = (t35 >= T) & (tkr <= 0.30)
    hard_m = (t65 <= T) & ((tkr >= 0.35) | (tdr >= 0.70))
    easy = [pool[i] for i in np.where(easy_m)[0]]
    hard = [pool[i] for i in np.where(hard_m)[0]]
    return easy, hard


def partition_length(
    pool: list[dict], short_hi: float, long_lo: float, long_hi: float
) -> tuple[list[dict], list[dict]]:
    """Bimodal solution-length split (constructed DK16 conflict, run_log 2026-07-10).

    SHORT = bottom ``short_hi`` percentile of BFS solution length T (target-start learnable;
    hypothesis: high-assist near-goal starts do not transfer, so prolonged assistance wastes its
    budget). LONG = [``long_lo``, ``long_hi``] percentile band (target-start unlearnable at the
    16x16 horizon per the long-horizon appendix; needs sustained scaffold — capped at long_hi to
    exclude layouts that may be unlearnable even with the scaffold). The middle band is dropped,
    mirroring the PnP conflict's bimodal construction.
    """
    T = np.array([c["T"] for c in pool])
    s_hi, l_lo, l_hi = np.percentile(T, [short_hi, long_lo, long_hi])
    easy = [pool[i] for i in np.where(T <= s_hi)[0]]
    hard = [pool[i] for i in np.where((T >= l_lo) & (T <= l_hi))[0]]
    print(f"[length-split] T cuts: short<= {s_hi:.0f}, long in [{l_lo:.0f}, {l_hi:.0f}]")
    return easy, hard


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_doorkey_conflict_bank")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--split", choices=("geometry", "length"), default="geometry",
                   help="group partition: geometry (C1 natural split) or length (bimodal T quantiles)")
    p.add_argument("--short-pct", type=float, default=20.0)
    p.add_argument("--long-lo-pct", type=float, default=60.0)
    p.add_argument("--long-hi-pct", type=float, default=90.0)
    # imbalanced composition (minority-worst-group construction): per-group train sizes; held-out
    # stays balanced at --n-held per group so the worst-group metric keeps equal measurement power.
    p.add_argument("--n-train-easy", type=int, default=None)
    p.add_argument("--n-train-hard", type=int, default=None)
    p.add_argument("--n-held", type=int, default=None, help="held-out contexts PER GROUP")
    a = p.parse_args(argv)

    pool_d = json.loads(a.pool.read_text())
    if a.split == "length":
        easy, hard = partition_length(
            pool_d["contexts"], a.short_pct, a.long_lo_pct, a.long_hi_pct
        )
    else:
        easy, hard = partition(pool_d["contexts"])
    print(f"pool={len(pool_d['contexts'])} -> easy={len(easy)} hard={len(hard)}")

    if a.n_train_easy is not None or a.n_train_hard is not None:
        # imbalanced composition: explicit per-group sizes
        n_tr_e, n_tr_h = int(a.n_train_easy), int(a.n_train_hard)
        n_hd = int(a.n_held if a.n_held is not None else 50)
        need_e, need_h = n_tr_e + n_hd, n_tr_h + n_hd
        if len(easy) < need_e or len(hard) < need_h:
            print(f"!! not enough contexts: need easy={need_e} hard={need_h}, "
                  f"have easy={len(easy)} hard={len(hard)}")
            return 1
        print(f"using train {n_tr_e}+{n_tr_h} (easy+hard), held {n_hd}+{n_hd}")
    else:
        # target 100+100 train / 50+50 held; fallback 75+75 / 25+25
        n_tr, n_hd = (100, 50) if min(len(easy), len(hard)) >= 150 else (75, 25)
        n_tr_e = n_tr_h = n_tr
        need_e = need_h = n_tr + n_hd
        if min(len(easy), len(hard)) < need_e:
            print(f"!! not enough contexts: need {need_e}/group, have easy={len(easy)} hard={len(hard)}")
            return 1
        print(f"using {n_tr} train + {n_hd} held per group")
    easy, hard = easy[:need_e], hard[:need_h]

    # context ids: easy train, hard train, easy held, hard held (stable order)
    ctxs, group = [], {}
    cid = 0

    def add(items, split, grp):
        nonlocal cid
        ids = []
        for c in items:
            ctxs.append(MiniGridContext(context_id=cid, seed=int(c["context_seed"]), split=split))
            group[cid] = grp
            ids.append(cid)
            cid += 1
        return ids

    add(easy[:n_tr_e], "train", "easy")
    add(hard[:n_tr_h], "train", "hard")
    add(easy[n_tr_e:need_e], "held_out", "easy")
    add(hard[n_tr_h:need_h], "held_out", "hard")

    full = MiniGridTaskBank(ctxs, a.env_id)
    bank_path = Path(f"{a.out_prefix}_bank_s0.json")
    full.save(bank_path)
    gmap_path = Path(f"{a.out_prefix}_groups_s0.json")
    gmap_path.write_text(json.dumps({str(k): v for k, v in group.items()}, indent=2))
    print(f"[full bank] {bank_path} sha={full.sha256()[:12]} (train {len(full.train_ids())} held {len(full.held_out_ids())})")

    # per-group sub-banks: RE-INDEX context_ids to 0..N-1 (train contexts first) — the MiniGrid loop uses
    # the controller index directly as the context_id, so a sub-bank must have train ids 0..n_train-1.
    # Build a per-group ladder over each (the full bank's shared ladder is keyed by full-bank ids and would
    # only align for whichever group occupies ids 0..n_train-1).
    for grp in ["easy", "hard"]:
        train_items = [c for c in ctxs if group[c.context_id] == grp and c.split == "train"]
        held_items = [c for c in ctxs if group[c.context_id] == grp and c.split == "held_out"]
        reixed = [
            MiniGridContext(context_id=i, seed=c.seed, split=c.split)
            for i, c in enumerate(train_items + held_items)
        ]
        sub = MiniGridTaskBank(reixed, a.env_id)
        sp = Path(f"{a.out_prefix}_{grp}_s0.json")
        sub.save(sp)
        sub_ladder, _ = _dk.build_ladder(sub, a.max_steps)
        slp = Path(
            str(a.out_prefix).replace("banks/", "banks/scaffold_ladders/") + f"_{grp}_auto_s0.json"
        )
        slp.parent.mkdir(parents=True, exist_ok=True)
        sub_ladder.save(slp)
        print(f"[{grp} sub-bank] {sp} train {len(sub.train_ids())} held {len(sub.held_out_ids())}; "
              f"ladder {slp.name} sha={sub_ladder.sha256()[:12]}")

    # automatic ladder over the FULL bank's train contexts
    ladder, stats = _dk.build_ladder(full, a.max_steps)
    lp = Path(str(a.out_prefix).replace("banks/", "banks/scaffold_ladders/") + "_auto_s0.json")
    lp.parent.mkdir(parents=True, exist_ok=True)
    ladder.save(lp)
    # verify the ladder is loadable and covers all train ids
    rl = ScaffoldLadder.load(lp)
    miss = [c for c in full.train_ids() if (c, 1) not in rl.cells]
    print(f"[ladder] {lp} sha={ladder.sha256()[:12]} levels={ladder.n_levels} "
          f"unsolved={stats['n_unsolved']} stage_span>=3 frac={stats['frac_ctx_span_ge3']:.2f} "
          f"missing_train_cells={len(miss)}")

    # feature means by group (provenance)
    for grp, items in [("easy", easy), ("hard", hard)]:
        T = np.mean([c["T"] for c in items])
        tk = np.mean([c["t_key"] / max(1, c["T"]) for c in items])
        print(f"  {grp}: mean T={T:.1f}  mean t_key/T={tk:.2f}  n={len(items)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
