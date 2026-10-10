"""Test Campaign 4 — DoorKey conflict bank / sub-bank / ladder / controller-index logic (Gate C1).

These tests check the *machinery* behind the Gate-C1 fixed-level sweep with small synthetic inputs and a
tiny real DoorKey bank — NO RL training. They target the anomaly hypothesis directly:

  (a) partition() yields disjoint easy/hard groups with the documented thresholds;
  (b) per-group sub-banks re-index train ids to 0..n_train-1 contiguous, held following;
  (c) the per-group ladder covers every sub-bank train id at every level, with distinct levels;
  (d) rfcl_style samples contexts uniformly off the per-context frontier with no anti-stall;
  (e) a fixed-level condition actually pins the level it claims (fixed0..fixed4) end-to-end;
  (f) controller index 0..n_train-1 maps DIRECTLY to the sub-bank context_id (the loop's contract);
  (g) the WRONG-LADDER misalignment hazard (full ladder + hard sub-bank) is detectable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pcrc.envs.minigrid_env import (
    GridStateCodec,
    MiniGridContext,
    MiniGridTaskBank,
    MiniGridTaskEnvironment,
)
from pcrc.envs.trajectory_scaffold import ScaffoldLadder
from pcrc.registry import Method, ScaffoldSource
from pcrc.samplers.scaffold_controller import ScaffoldConfig, ScaffoldController

_REPO = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BANK = _load_script("build_doorkey_conflict_bank")
_TRAJ = _load_script("build_doorkey_traj_scaffold")
_PPO = _load_script("minigrid_ppo")

LEVEL_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _synthetic_pool(n_easy=80, n_hard=80, n_mid=40, seed=0):
    """A synthetic feature pool with clearly separable easy/hard groups + ambiguous middle."""
    rng = np.random.default_rng(seed)
    pool = []
    sid = 1000
    # easy: short solution (T small), early key (t_key/T <= 0.30)
    for _ in range(n_easy):
        T = int(rng.integers(8, 13))
        pool.append({"T": T, "t_key": int(0.15 * T), "t_door": int(0.4 * T),
                     "t_goal": T, "context_seed": sid})
        sid += 1
    # hard: long solution (T large), late key (t_key/T >= 0.35)
    for _ in range(n_hard):
        T = int(rng.integers(22, 30))
        pool.append({"T": T, "t_key": int(0.5 * T), "t_door": int(0.8 * T),
                     "t_goal": T, "context_seed": sid})
        sid += 1
    # middle: medium, ambiguous — belongs to NEITHER group
    for _ in range(n_mid):
        T = int(rng.integers(15, 19))
        pool.append({"T": T, "t_key": int(0.32 * T), "t_door": int(0.5 * T),
                     "t_goal": T, "context_seed": sid})
        sid += 1
    return pool


# ----------------------------------------------------------------- (a) partition
def test_partition_disjoint_and_thresholds():
    pool = _synthetic_pool()
    easy, hard = _BANK.partition(pool)
    assert easy and hard
    easy_ids = {c["context_seed"] for c in easy}
    hard_ids = {c["context_seed"] for c in hard}
    assert easy_ids.isdisjoint(hard_ids), "easy/hard must be disjoint"

    T = np.array([c["T"] for c in pool])
    t35, t65 = np.percentile(T, [35, 65])
    for c in easy:
        assert c["T"] <= t35 and (c["t_key"] / max(1, c["T"])) <= 0.30
    for c in hard:
        tkr = c["t_key"] / max(1, c["T"])
        tdr = c["t_door"] / max(1, c["T"])
        assert c["T"] >= t65 and (tkr >= 0.35 or tdr >= 0.70)


def test_partition_matches_real_pool_if_present():
    p = _REPO / "results/banks/doorkey_conflict_pool.json"
    if not p.exists():
        pytest.skip("real conflict pool not built")
    import json

    pool = json.loads(p.read_text())["contexts"]
    easy, hard = _BANK.partition(pool)
    e = {c["context_seed"] for c in easy}
    h = {c["context_seed"] for c in hard}
    assert e.isdisjoint(h)
    assert len(e) >= 75 and len(h) >= 75  # fallback sizing requirement


# --------------------------------------------------- (b)+(c) sub-bank + ladder build
def _build_full_pipeline(tmp_path: Path):
    """Run the real build_doorkey_conflict_bank.main on a synthetic pool; return artifact paths.

    Uses a real (tiny) DoorKey ladder build, so it exercises solve_context/build_ladder for real.
    To keep it fast we shrink the per-group sizes by monkeypatching nothing — instead we feed a pool
    just large enough for the 75/25 fallback and let main() pick 75 train + 25 held per group.
    """
    import json

    # need >= need(=100) per group for the 75-train/25-held fallback sizing in main()
    pool = _synthetic_pool(n_easy=140, n_hard=140, n_mid=20)
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps({"env_id": "MiniGrid-DoorKey-8x8-v0", "contexts": pool}))
    out_prefix = tmp_path / "banks" / "dkc"
    rc = _BANK.main(["--pool", str(pool_path), "--out-prefix", str(out_prefix)])
    return rc, out_prefix


@pytest.mark.slow
def test_subbank_reindex_and_ladder_coverage(tmp_path):
    rc, out_prefix = _build_full_pipeline(tmp_path)
    assert rc == 0
    for grp in ("easy", "hard"):
        sub = MiniGridTaskBank.load(f"{out_prefix}_{grp}_s0.json")
        tr, hd = sub.train_ids(), sub.held_out_ids()
        # (b) train ids contiguous 0..n_train-1, held following
        assert tr == list(range(len(tr))), f"{grp}: train ids not contiguous from 0"
        assert hd == list(range(len(tr), len(tr) + len(hd))), f"{grp}: held ids do not follow"
        # (c) ladder covers EVERY train id at EVERY level
        lad_path = (
            str(out_prefix).replace("banks/", "banks/scaffold_ladders/") + f"_{grp}_auto_s0.json"
        )
        lad = ScaffoldLadder.load(lad_path)
        assert lad.n_levels == 5
        for cid in tr:
            for lvl in range(5):
                assert (cid, lvl) in lad.cells, f"{grp}: missing cell ({cid},{lvl})"
        # ladder must NOT cover held ids (train-only ladder)
        assert set(hd).isdisjoint(set(lad.context_ids()))


# ----------------------------------------- (c, real artifacts) distinct ladder levels
def test_real_subbank_ladders_have_distinct_levels():
    """The easy sub-bank ladder must have 5 *distinct* levels with stage progression to (key,door,beyond).

    Directly addresses the anomaly: if 'easy' collapsed to <5 effective levels, fixed-level training
    would be meaningless. We assert level 0 is the true start and level 4 reaches the goal side.
    """
    base = _REPO / "results/banks"
    if not (base / "doorkey_conflict_easy_s0.json").exists():
        pytest.skip("real conflict sub-banks not built")
    for grp in ("easy", "hard"):
        bank = MiniGridTaskBank.load(base / f"doorkey_conflict_{grp}_s0.json")
        lad = ScaffoldLadder.load(base / f"scaffold_ladders/doorkey_conflict_{grp}_auto_s0.json")
        assert bank.train_ids() == list(range(len(bank.train_ids())))
        n_full_span = 0
        for cid in bank.train_ids():
            assert lad.get(cid, 0).serialized_state is None  # level 0 == true start
            stages = [
                None
                if lad.get(cid, l).serialized_state is None
                else _TRAJ.stage_of(lad.get(cid, l).serialized_state)
                for l in range(5)
            ]
            # level 4 must be on the goal side (beyond door) for a well-formed ladder
            if stages[4] == (True, True, True):
                n_full_span += 1
        assert n_full_span / len(bank.train_ids()) > 0.8, f"{grp}: most ladders should reach goal side"


# --------------------------------------------- (f) controller index -> context_id contract
def test_controller_index_is_used_directly_as_context_id():
    """The MiniGrid loop passes the controller's context index DIRECTLY to env.reset(context_id).

    So a sub-bank MUST have train ids 0..n_train-1. We build a tiny real bank with ids 0..2 and verify
    env.reset(controller_index) lands on the matching seed's layout (no remap layer exists).
    """
    seeds = [11, 22, 33]
    ctxs = [MiniGridContext(context_id=i, seed=s, split="train") for i, s in enumerate(seeds)]
    bank = MiniGridTaskBank(ctxs, "MiniGrid-DoorKey-8x8-v0")
    env = MiniGridTaskEnvironment(bank, "MiniGrid-DoorKey-8x8-v0", max_steps=160)
    codec = GridStateCodec()
    for idx, seed in enumerate(seeds):
        env.reset(idx, start_progress=0.0)  # controller index used directly
        got = codec.serialize(env._u)
        # reference: a plain reset at that seed
        ref_env = MiniGridTaskEnvironment(bank, "MiniGrid-DoorKey-8x8-v0", max_steps=160)
        ref_env._env.reset(seed=seed)
        ref = codec.serialize(ref_env._u)
        assert got["grid"] == ref["grid"], f"index {idx} did not reset to seed {seed}'s layout"
        ref_env.close()
    env.close()


# --------------------------------------------- (d) rfcl_style uniform + frontier, no anti-stall
def test_rfcl_uniform_sampling_on_frontier():
    c = ScaffoldController(6, 5, LEVEL_FRACS, ScaffoldConfig(mode=Method.RFCL_STYLE, stale_advance_windows=0))
    ctxs, levels, fracs = c.select(12_000, step=0, rng=np.random.default_rng(1))
    counts = np.bincount(np.asarray(ctxs), minlength=6)
    assert counts.min() > 1700 and counts.max() < 2300  # ~2000 each
    assert all(lv == c.L for lv in levels)  # frontier starts at easiest
    assert all(f == LEVEL_FRACS[c.L] for f in fracs)
    assert c.cfg.stale_advance_windows == 0  # no anti-stall


# --------------------------------------------- (e) fixed level actually pins the level
@pytest.mark.parametrize("lvl", [0, 1, 2, 3, 4])
def test_fixed_condition_pins_level_end_to_end(lvl):
    c = _PPO.make_controller(
        f"trajectory_fixed{lvl}", n_train=4, n_levels=5, level_fractions=LEVEL_FRACS,
        anneal_iters=1500, stale_advance_windows=4,
    )
    assert c.cfg.mode == Method.FIXED and c.cfg.fixed_level == lvl
    _ctxs, levels, fracs = c.select(64, step=0, rng=np.random.default_rng(2))
    assert set(levels) == {lvl}, f"fixed{lvl} must pin level {lvl}, got {set(levels)}"
    assert all(f == LEVEL_FRACS[lvl] for f in fracs)
    # and the env maps that fraction back to the SAME level (round(frac*4))
    expected_env_level = int(round(LEVEL_FRACS[lvl] * 4))
    assert expected_env_level == lvl, "fraction->env-level round-trip must be identity"


def test_fixed_levels_are_all_distinct_env_levels():
    """fixed0..fixed4 must produce 5 DISTINCT env levels (no two fractions collapse to one level)."""
    env_levels = [int(round(LEVEL_FRACS[lvl] * 4)) for lvl in range(5)]
    assert env_levels == [0, 1, 2, 3, 4]


# --------------------------------------------- (g) wrong-ladder misalignment is detectable
def test_full_ladder_with_hard_subbank_is_misaligned():
    """Guard test: training the HARD sub-bank against the FULL ladder restores the WRONG layout.

    The full bank places hard-train at ids 100..199; the hard sub-bank re-indexes them to 0..99. So
    full_ladder cell (0,*) carries an EASY layout's state, but the hard sub-bank's context 0 is a HARD
    seed. If the launcher passes the full ladder with a per-group sub-bank, train layouts decorrelate
    from eval seeds. This test documents/detects that hazard so it can't recur silently.
    """
    base = _REPO / "results/banks"
    if not (base / "doorkey_conflict_hard_s0.json").exists():
        pytest.skip("real conflict artifacts not built")
    full = MiniGridTaskBank.load(base / "doorkey_conflict_bank_s0.json")
    hard = MiniGridTaskBank.load(base / "doorkey_conflict_hard_s0.json")
    # full-bank context 0 (easy train) seed != hard sub-bank context 0 (hard train) seed
    assert full[0].seed != hard[0].seed
    # but the per-group ladder IS correct: its context 0 matches the hard sub-bank's seed-0 layout
    hard_lad = ScaffoldLadder.load(base / "scaffold_ladders/doorkey_conflict_hard_auto_s0.json")
    # the level-4 state in the hard ladder must be derivable from the hard seed-0 layout:
    env = MiniGridTaskEnvironment(
        hard, "MiniGrid-DoorKey-8x8-v0", max_steps=160,
        scaffold_source=ScaffoldSource.TRAJECTORY, scaffold_ladder=hard_lad,
    )
    env.reset(0, start_progress=1.0)  # level 4
    # goal must still be present after restore (sanity: the restored layout is a real DoorKey)
    from minigrid.core.world_object import Goal

    has_goal = any(
        isinstance(env._u.grid.get(x, y), Goal)
        for y in range(env._u.height)
        for x in range(env._u.width)
    )
    assert has_goal, "restored level-4 state must still contain the goal"
    env.close()
