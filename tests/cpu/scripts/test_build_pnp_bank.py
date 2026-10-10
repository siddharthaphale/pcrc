"""Tests for the `sharp` 3-strata bank builder (autonomous plan Gate C).

Covers the pure stratification logic in `scripts/build_pnp_bank.py`:
`_split_by_cuts` (goal-z partitioning) and `_build_sharp` (combined bank with
group recoverable from goal-z + one single-stratum bank per group, with the
contiguous train-block/held-block id layout the controller relies on).

Uses synthetic probe-record stand-ins (SimpleNamespace) so no live MuJoCo env
is needed — the geometry/stratification is what we want to lock down.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "build_pnp_bank", _REPO / "scripts" / "build_pnp_bank.py"
)
bpb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpb)

CUTS = [0.45, 0.65]
NAMES = ["ontable", "moderate", "hard"]


def _rec(z: float, seed: int) -> SimpleNamespace:
    """Minimal probe-record stand-in with the attrs `_make_contexts` reads."""
    return SimpleNamespace(
        desired_goal=np.array([1.33, 0.70, z], dtype=np.float64),
        initial_achieved_goal=np.array([1.33 + 0.01 * (seed % 5), 0.70, 0.425], dtype=np.float64),
        reset_seed=seed,
        reset_payload={"k": seed},
        initial_object_pose=np.zeros(7, dtype=np.float64),
    )


def _records(n_ot=12, n_mod=10, n_hard=10) -> list[SimpleNamespace]:
    """Records spanning the three bands, deliberately interleaved (not z-sorted)."""
    recs = []
    s = 0
    for z in np.linspace(0.420, 0.445, n_ot):
        recs.append(_rec(float(z), s))
        s += 1
    for z in np.linspace(0.460, 0.630, n_mod):
        recs.append(_rec(float(z), s))
        s += 1
    for z in np.linspace(0.660, 0.850, n_hard):
        recs.append(_rec(float(z), s))
        s += 1
    rng = np.random.default_rng(0)
    rng.shuffle(recs)
    return recs


def _group_of(z: float) -> int:
    return sum(1 for c in CUTS if z >= c)


def test_split_by_cuts_partitions_by_goal_z():
    groups = bpb._split_by_cuts(_records(12, 10, 10), CUTS)
    assert [len(g) for g in groups] == [12, 10, 10]
    # every record landed in the band its goal-z dictates
    for gi, g in enumerate(groups):
        for r in g:
            assert _group_of(float(r.desired_goal[2])) == gi


def test_split_by_cuts_is_value_not_order_dependent():
    recs = _records(5, 5, 5)
    a = [len(g) for g in bpb._split_by_cuts(recs, CUTS)]
    b = [len(g) for g in bpb._split_by_cuts(list(reversed(recs)), CUTS)]
    assert a == b == [5, 5, 5]


def test_build_sharp_strata_banks_are_within_band_and_sized():
    _comb, _iag, strata = bpb._build_sharp(_records(), CUTS, n_train_s=6, n_held_s=3, names=NAMES)
    assert [name for name, _, _ in strata] == NAMES
    for gi, (name, ctxs, iag) in enumerate(strata):
        assert len(ctxs) == 9  # 6 train + 3 held
        # local id layout: 0..5 train, 6..8 held
        trains = [c for c in ctxs if c.split == "train"]
        helds = [c for c in ctxs if c.split == "held_out"]
        assert sorted(c.context_id for c in trains) == [0, 1, 2, 3, 4, 5]
        assert sorted(c.context_id for c in helds) == [6, 7, 8]
        # every context is inside this stratum's goal-z band
        for c in ctxs:
            assert _group_of(float(c.desired_goal[2])) == gi
            assert c.context_id in iag  # iag keyed by local id


def test_build_sharp_combined_layout_and_group_recoverable():
    comb, iag, _strata = bpb._build_sharp(_records(), CUTS, n_train_s=6, n_held_s=3, names=NAMES)
    trains = [c for c in comb if c.split == "train"]
    helds = [c for c in comb if c.split == "held_out"]
    assert len(trains) == 18 and len(helds) == 9 and len(comb) == 27
    # contiguous train-block (0..17) then held-block (18..26) — controller idx = train id
    assert sorted(c.context_id for c in trains) == list(range(18))
    assert sorted(c.context_id for c in helds) == list(range(18, 27))
    assert set(iag) == set(range(27))
    # group recoverable from goal-z: 9 contexts per band (6 train + 3 held)
    per_group = [0, 0, 0]
    for c in comb:
        per_group[_group_of(float(c.desired_goal[2]))] += 1
    assert per_group == [9, 9, 9]


def test_build_sharp_keep_drops_middle_stratum_from_combined():
    # keep only the two ends -> bimodal max-conflict combined bank; all 3 stratum files still built
    comb, iag, strata = bpb._build_sharp(
        _records(), CUTS, n_train_s=6, n_held_s=3, names=NAMES, keep=["ontable", "hard"]
    )
    assert [name for name, _, _ in strata] == NAMES  # all strata still returned
    assert len(comb) == 18 and set(iag) == set(range(18))  # only 2 kept x (6+3)
    per_group = [0, 0, 0]
    for c in comb:
        per_group[_group_of(float(c.desired_goal[2]))] += 1
    assert per_group == [9, 0, 9]  # middle (moderate) excluded from combined


def test_build_sharp_keep_rejects_unknown_name():
    with pytest.raises(ValueError, match="not in strata"):
        bpb._build_sharp(_records(), CUTS, n_train_s=6, n_held_s=3, names=NAMES, keep=["nope"])


def test_build_sharp_raises_when_stratum_too_small():
    with pytest.raises(ValueError, match="moderate"):
        bpb._build_sharp(_records(n_mod=4), CUTS, n_train_s=6, n_held_s=3, names=NAMES)


def test_build_sharp_name_count_must_match_strata():
    with pytest.raises(ValueError, match="names"):
        bpb._build_sharp(_records(), CUTS, n_train_s=6, n_held_s=3, names=["a", "b"])
