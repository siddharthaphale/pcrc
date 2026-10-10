"""CPU smoke for :func:`pcrc.envs.fetch_calibration.calibrate_fetch_bank`.

Exercises the full pipeline (probe → roll → keep/drop → bin assign →
bank construction → JSON write) end-to-end on a tiny pool with a
zero-action policy. Fast (≤ 5 s) so it stays in ``make test-cpu-fetch``.

Does NOT validate that the starter policy produces a useful success-band
distribution (that is the calibration empirics, owned by the Stage A1
mid-run on the GPU node, not a unit test). What this test asserts:

    - The runner returns a valid :class:`FetchTaskBank`.
    - ``per_context.jsonl`` + ``summary.json`` + ``bank.json`` are
      written and round-trip-loadable.
    - The kept context count equals ``num_train + num_held_out``
      whenever the band leaves enough survivors (with a zero policy
      and tight band, dense-fallback kicks in and the pool survives
      whole).
    - The dense-fallback path is exercised (zero policy → near-zero
      success variance).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium_robotics")

from pcrc.envs.fetch_calibration import (  # noqa: E402
    calibrate_fetch_bank,
    write_calibration_outputs,
)
from pcrc.envs.fetch_task_bank import FetchTaskBank  # noqa: E402

ENV_ID_REACH = "FetchReachDense-v4"


def _zero_policy(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Always-zero action — the simplest valid starter for smoke tests."""
    del obs
    return np.zeros(4, dtype=np.float32)


def test_calibrate_reach_runs_end_to_end(tmp_path: Path) -> None:
    # Permissive band so the full pipeline runs without depending on
    # starter-induced filtering. The Stage A1 mid-run on the GPU node
    # uses the real PPO warmup starter and the realistic 0.05–0.70
    # band; those mechanics are validated by
    # ``test_calibrate_filters_under_strict_band`` below, not by this
    # end-to-end smoke.
    outcome = calibrate_fetch_bank(
        env_id=ENV_ID_REACH,
        n_pool=12,
        num_train=6,
        num_held_out=3,
        policy_fn=_zero_policy,
        episodes_per_context=2,
        max_steps_per_episode=10,
        distance_bins=3,
        direction_bins=3,
        pose_bins=1,
        target_success_band=(0.05, 0.70),
        # Wide fallback threshold so any non-zero variance routes
        # through the dense-fallback path (which accepts the whole
        # pool). Keeps this test deterministic regardless of how many
        # seed-sampled goals happen to land within 0.05 m of the EE
        # start under the zero-action policy.
        dense_fallback_variance_threshold=1.0,
        pool_base_seed=10000,
        split_seed=42,
    )
    assert isinstance(outcome.bank, FetchTaskBank)
    assert len(outcome.bank.train_ids()) == 6
    assert len(outcome.bank.held_out_ids()) == 3
    # Train contexts MUST be reindexed to ids 0..n_train-1 — the invariant
    # build_sampler_for_condition asserts for every Stage D loop. Regression
    # guard for the calibration reindex (a calibrated bank fed straight into
    # run_fetch_ppo_loop would otherwise raise on scattered train ids).
    assert outcome.bank.train_ids() == list(range(6))
    assert outcome.bank.held_out_ids() == list(range(6, 9))
    assert outcome.used_dense_fallback is True
    # Whole pool accepted under dense fallback.
    assert outcome.summary["dropped_saturated"] == 0
    assert outcome.summary["dropped_dead"] == 0
    # 12 - (6 train + 3 held_out) = 3 unused.
    assert outcome.summary["dropped_unused"] == 3

    # Bin indices respect the configured widths.
    for ctx in outcome.bank.contexts.values():
        assert 0 <= ctx.distance_bin < 3
        assert 0 <= ctx.direction_bin < 3
        assert ctx.pose_bin == 0  # Reach has no object


def test_calibrate_raises_when_real_signal_drops_everything() -> None:
    """A starter that happens to land everything outside the band must
    raise rather than silently produce an empty bank.

    Zero-action on Reach for 20 steps: some seed-sampled goals land
    within 0.05 m of EE start (trivial saturation) while the rest stay
    far (zero success). Strict band (0.05, 0.70) drops everything as
    saturated or dead → not enough survivors → raises.
    """
    with pytest.raises(ValueError, match="not enough surviving"):
        calibrate_fetch_bank(
            env_id=ENV_ID_REACH,
            n_pool=12,
            num_train=6,
            num_held_out=3,
            policy_fn=_zero_policy,
            episodes_per_context=2,
            max_steps_per_episode=20,
            distance_bins=3,
            direction_bins=3,
            pose_bins=1,
            target_success_band=(0.05, 0.70),
            dense_fallback_variance_threshold=0.0,  # disable fallback
            pool_base_seed=10000,
            split_seed=42,
        )


def test_calibrate_dense_fallback_keeps_whole_pool() -> None:
    """Zero variance → dense-fallback path → whole pool accepted.

    With ``max_steps_per_episode=1`` no successes can be earned (one
    step is not enough to move from any non-saturated start), so
    success variance is exactly zero across non-trivial seeds.
    Combined with a wide ``dense_fallback_variance_threshold``, this
    forces the fallback branch deterministically.
    """
    outcome = calibrate_fetch_bank(
        env_id=ENV_ID_REACH,
        n_pool=8,
        num_train=4,
        num_held_out=2,
        policy_fn=_zero_policy,
        episodes_per_context=2,
        max_steps_per_episode=1,
        distance_bins=2,
        direction_bins=2,
        pose_bins=1,
        target_success_band=(0.05, 0.70),
        dense_fallback_variance_threshold=1.0,  # any variance triggers fallback
        pool_base_seed=20100,
        split_seed=11,
    )
    assert outcome.used_dense_fallback is True
    assert outcome.summary["dropped_saturated"] == 0
    assert outcome.summary["dropped_dead"] == 0


def test_calibration_outputs_written_and_loadable(tmp_path: Path) -> None:
    outcome = calibrate_fetch_bank(
        env_id=ENV_ID_REACH,
        n_pool=8,
        num_train=4,
        num_held_out=2,
        policy_fn=_zero_policy,
        episodes_per_context=2,
        max_steps_per_episode=15,
        distance_bins=2,
        direction_bins=2,
        pose_bins=1,
        target_success_band=(0.05, 0.70),
        dense_fallback_variance_threshold=0.01,
        pool_base_seed=20000,
        split_seed=7,
    )
    out_root = tmp_path / "exp_smoke"
    paths = write_calibration_outputs(outcome, out_root)
    assert paths["bank"].exists()
    assert paths["per_context"].exists()
    assert paths["summary"].exists()

    # Bank round-trips.
    bank = FetchTaskBank.load(paths["bank"])
    assert sorted(bank.contexts) == sorted(outcome.bank.contexts)

    # summary.json has the expected fields.
    summary = json.loads(paths["summary"].read_text())
    for k in (
        "env_id",
        "n_pool",
        "num_train",
        "num_held_out",
        "kept",
        "dropped_saturated",
        "dropped_dead",
        "dropped_unused",
        "used_dense_fallback",
        "bank_path",
    ):
        assert k in summary, k

    # per_context.jsonl has one row per pool entry with required cols.
    rows = [json.loads(line) for line in paths["per_context"].read_text().splitlines()]
    assert len(rows) == outcome.summary["n_pool"]
    cols = {
        "context_id",
        "env_id",
        "reset_seed",
        "success_rate",
        "mean_dense_return",
        "mean_final_goal_distance",
        "kept",
        "drop_reason",
        "split",
        "distance_bin",
        "direction_bin",
        "pose_bin",
    }
    for r in rows:
        missing = cols - set(r.keys())
        assert not missing, missing


def test_calibrate_rejects_invalid_band() -> None:
    with pytest.raises(ValueError, match="target_success_band"):
        calibrate_fetch_bank(
            env_id=ENV_ID_REACH,
            n_pool=4,
            num_train=2,
            num_held_out=1,
            policy_fn=_zero_policy,
            episodes_per_context=1,
            max_steps_per_episode=5,
            distance_bins=2,
            direction_bins=2,
            pose_bins=1,
            target_success_band=(0.7, 0.7),  # upper not > lower
            dense_fallback_variance_threshold=0.01,
            pool_base_seed=0,
            split_seed=0,
        )


def test_calibrate_raises_when_pool_too_small_for_band() -> None:
    # Force band such that even the dense-fallback survives → require
    # too many contexts.
    with pytest.raises(ValueError, match="not enough surviving"):
        calibrate_fetch_bank(
            env_id=ENV_ID_REACH,
            n_pool=4,
            num_train=10,  # bigger than pool
            num_held_out=2,
            policy_fn=_zero_policy,
            episodes_per_context=1,
            max_steps_per_episode=5,
            distance_bins=2,
            direction_bins=2,
            pose_bins=1,
            target_success_band=(0.05, 0.70),
            dense_fallback_variance_threshold=0.01,
            pool_base_seed=0,
            split_seed=0,
        )
