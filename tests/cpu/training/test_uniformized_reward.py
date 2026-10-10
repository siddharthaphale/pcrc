"""Tests for ``pcrc.training.fetch_uniformized_reward``.

§7.1 row: math matches hand-computed; selection-vs-learning is zero
when sampler is uniform; positive when sampler concentrates on
above-average contexts.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.training.fetch_uniformized_reward import (
    compute_sampler_weighted_reward,
    compute_selection_vs_learning_decomposition,
    compute_uniformized_reward,
)


def test_uniformized_reward_matches_mean_over_train_split() -> None:
    success_ema = np.array([0.2, 0.4, 0.6, 0.8, 0.0])
    train_mask = np.array([True, True, True, True, False])
    # Mean of first 4 = 0.5.
    val = compute_uniformized_reward(success_ema, train_mask)
    assert np.isclose(val, 0.5)


def test_uniformized_reward_raises_on_empty_train_split() -> None:
    with pytest.raises(ValueError, match="all False"):
        compute_uniformized_reward(np.zeros(4), np.zeros(4, dtype=np.bool_))


def test_sampler_weighted_reward_matches_dot_product() -> None:
    s = np.array([0.2, 0.4, 0.6, 0.8])
    q = np.array([0.1, 0.2, 0.3, 0.4])
    val = compute_sampler_weighted_reward(s, q)
    assert np.isclose(val, float((s * q).sum()))


def test_selection_signal_zero_under_uniform_sampler_over_train_split() -> None:
    s = np.array([0.2, 0.4, 0.6, 0.8])
    q = np.full(4, 0.25)
    train_mask = np.ones(4, dtype=np.bool_)
    decomp = compute_selection_vs_learning_decomposition(s, q, train_mask)
    assert np.isclose(decomp["selection_signal"], 0.0, atol=1e-9)


def test_selection_signal_positive_when_sampler_picks_above_average() -> None:
    s = np.array([0.0, 0.0, 0.0, 1.0])  # mean = 0.25
    q = np.array([0.0, 0.0, 0.0, 1.0])  # all mass on the high context
    train_mask = np.ones(4, dtype=np.bool_)
    decomp = compute_selection_vs_learning_decomposition(s, q, train_mask)
    assert decomp["sampler_weighted_reward"] == 1.0
    assert np.isclose(decomp["uniformized_reward"], 0.25)
    assert np.isclose(decomp["selection_signal"], 0.75)
