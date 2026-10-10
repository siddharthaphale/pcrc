"""Tests for ``pcrc.samplers.base.SamplerStats``.

§7.1 row: ``observed_mask[x] = False`` → EMA fields unchanged after
``Sampler.update(stats)``; programmer-error checks fire on inconsistent
masks; ``samples > 0, success_sum = 0`` is a real all-fail observation
(not rejected); ``samples > 0, success_sum = samples`` is real
saturation (not rejected).
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.samplers.base import SamplerStats


def _zeros(n: int) -> SamplerStats:
    return SamplerStats.zeros(n)


def test_zeros_constructor_returns_all_unobserved() -> None:
    stats = _zeros(8)
    assert stats.observed_mask.shape == (8,)
    assert not stats.observed_mask.any()
    assert (stats.samples_this_step == 0).all()
    assert (stats.success_sum_this_step == 0).all()


def test_shape_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        SamplerStats(
            observed_mask=np.zeros(4, dtype=np.bool_),
            samples_this_step=np.zeros(5, dtype=np.int64),  # mismatched
            success_sum_this_step=np.zeros(4),
            return_sum_this_step=np.zeros(4),
            utility_sum_this_step=np.zeros(4),
            utility_sq_sum_this_step=np.zeros(4),
            success_ema=np.zeros(4),
            return_ema=np.zeros(4),
            utility_ema=np.zeros(4),
            n_visits_total=np.zeros(4, dtype=np.int64),
        )


def test_observed_mask_dtype_must_be_bool() -> None:
    with pytest.raises(TypeError, match="must be bool"):
        SamplerStats(
            observed_mask=np.zeros(4, dtype=np.int64),  # wrong dtype
            samples_this_step=np.zeros(4, dtype=np.int64),
            success_sum_this_step=np.zeros(4),
            return_sum_this_step=np.zeros(4),
            utility_sum_this_step=np.zeros(4),
            utility_sq_sum_this_step=np.zeros(4),
            success_ema=np.zeros(4),
            return_ema=np.zeros(4),
            utility_ema=np.zeros(4),
            n_visits_total=np.zeros(4, dtype=np.int64),
        )


def test_observed_with_zero_samples_is_programmer_error() -> None:
    n = 4
    base = _zeros(n)
    obs = np.zeros(n, dtype=np.bool_)
    obs[0] = True  # observed but no samples
    with pytest.raises(ValueError, match="observed_mask=True but samples"):
        SamplerStats(
            observed_mask=obs,
            samples_this_step=base.samples_this_step,
            success_sum_this_step=base.success_sum_this_step,
            return_sum_this_step=base.return_sum_this_step,
            utility_sum_this_step=base.utility_sum_this_step,
            utility_sq_sum_this_step=base.utility_sq_sum_this_step,
            success_ema=base.success_ema,
            return_ema=base.return_ema,
            utility_ema=base.utility_ema,
            n_visits_total=base.n_visits_total,
        )


def test_negative_samples_rejected() -> None:
    n = 3
    samples = np.zeros(n, dtype=np.int64)
    samples[1] = -1
    with pytest.raises(ValueError, match="negative"):
        SamplerStats(
            observed_mask=np.zeros(n, dtype=np.bool_),
            samples_this_step=samples,
            success_sum_this_step=np.zeros(n),
            return_sum_this_step=np.zeros(n),
            utility_sum_this_step=np.zeros(n),
            utility_sq_sum_this_step=np.zeros(n),
            success_ema=np.zeros(n),
            return_ema=np.zeros(n),
            utility_ema=np.zeros(n),
            n_visits_total=np.zeros(n, dtype=np.int64),
        )


def test_real_all_fail_observation_is_valid() -> None:
    # samples > 0, success_sum = 0 — perfectly valid "context was rolled
    # 4 times, all failed" observation.
    n = 3
    obs = np.zeros(n, dtype=np.bool_)
    obs[0] = True
    samples = np.zeros(n, dtype=np.int64)
    samples[0] = 4
    stats = SamplerStats(
        observed_mask=obs,
        samples_this_step=samples,
        success_sum_this_step=np.zeros(n),
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=np.zeros(n),
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=np.zeros(n, dtype=np.int64),
    )
    assert stats.observed_mask[0]
    assert stats.samples_this_step[0] == 4
    assert stats.success_sum_this_step[0] == 0


def test_real_saturated_observation_is_valid() -> None:
    n = 3
    obs = np.zeros(n, dtype=np.bool_)
    obs[2] = True
    samples = np.zeros(n, dtype=np.int64)
    samples[2] = 4
    success = np.zeros(n)
    success[2] = 4.0  # all 4 succeeded
    stats = SamplerStats(
        observed_mask=obs,
        samples_this_step=samples,
        success_sum_this_step=success,
        return_sum_this_step=np.zeros(n),
        utility_sum_this_step=np.zeros(n),
        utility_sq_sum_this_step=np.zeros(n),
        success_ema=np.zeros(n),
        return_ema=np.zeros(n),
        utility_ema=np.zeros(n),
        n_visits_total=np.zeros(n, dtype=np.int64),
    )
    assert stats.success_sum_this_step[2] == 4
