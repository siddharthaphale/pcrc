"""Tests for ``pcrc.samplers.exploration_floor.apply_floor_and_cap``.

§7.1 row: floor math matches expected mass to 1e-9 on a fixture;
``q_max`` cap enforced; renormalization preserves sum-to-1; independent
of ``TwoStageSampler`` (no import of that module); validity checks
fire loudly on infeasible parameter combinations.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.samplers import exploration_floor
from pcrc.samplers.exploration_floor import apply_floor_and_cap

# ---------------------------------------------------------------------------
# Independence from TwoStageSampler
# ---------------------------------------------------------------------------


def test_module_does_not_import_two_stage_sampler() -> None:
    """The plan's separation of concerns: this module is the canonical
    home for floor + cap, distinct from candidate-batch warmup blending.

    Checked at the import-graph level (not via string match) because the
    docstring deliberately references ``two_stage_sampler`` to point
    readers at the distinction.
    """
    import ast  # noqa: PLC0415

    src = exploration_floor.__file__
    with open(src) as f:
        tree = ast.parse(f.read())
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
    assert not any("two_stage_sampler" in m for m in imported_modules), imported_modules


# ---------------------------------------------------------------------------
# Floor math
# ---------------------------------------------------------------------------


def test_floor_zero_epsilon_passes_raw_through() -> None:
    q = np.array([0.5, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
    dist = apply_floor_and_cap(q, epsilon=0.0, q_max=None)
    assert np.allclose(dist.raw, q)
    assert np.allclose(dist.floored, q)
    assert np.allclose(dist.capped, q)
    assert np.allclose(dist.final, q)


def test_floor_one_epsilon_returns_uniform() -> None:
    n = 5
    q = np.array([0.5, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
    dist = apply_floor_and_cap(q, epsilon=1.0, q_max=None)
    assert np.allclose(dist.final, np.full(n, 1.0 / n))


def test_floor_half_epsilon_mixes_correctly() -> None:
    n = 4
    q = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float64)
    eps = 0.5
    expected = (1 - eps) * q + eps * (1.0 / n)
    dist = apply_floor_and_cap(q, epsilon=eps, q_max=None)
    np.testing.assert_allclose(dist.floored, expected, atol=1e-9)
    np.testing.assert_allclose(dist.final, expected, atol=1e-9)


def test_floor_handles_one_hot_distribution() -> None:
    n = 6
    q = np.zeros(n, dtype=np.float64)
    q[2] = 1.0
    dist = apply_floor_and_cap(q, epsilon=0.1, q_max=None)
    # Expected: arm 2 → 0.9 + 0.1/6; others → 0.1/6.
    expected = np.full(n, 0.1 / n)
    expected[2] = 0.9 + 0.1 / n
    np.testing.assert_allclose(dist.final, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Cap math
# ---------------------------------------------------------------------------


def test_cap_below_max_is_noop() -> None:
    n = 5
    q = np.full(n, 1.0 / n)
    dist = apply_floor_and_cap(q, epsilon=0.0, q_max=0.5)
    np.testing.assert_allclose(dist.final, q)


def test_cap_enforces_max_probability() -> None:
    q = np.array([0.7, 0.1, 0.1, 0.05, 0.05], dtype=np.float64)
    q_max = 0.4
    dist = apply_floor_and_cap(q, epsilon=0.0, q_max=q_max)
    assert dist.final.max() <= q_max + 1e-9
    np.testing.assert_allclose(dist.final.sum(), 1.0, atol=1e-9)


def test_cap_equal_to_uniform_returns_uniform() -> None:
    n = 4
    q = np.array([0.5, 0.3, 0.1, 0.1], dtype=np.float64)
    dist = apply_floor_and_cap(q, epsilon=0.0, q_max=1.0 / n)
    # The cap forces every entry to exactly 1/n.
    np.testing.assert_allclose(dist.final, np.full(n, 1.0 / n), atol=1e-9)


def test_cap_redistributes_proportionally_to_under_cap_mass() -> None:
    q = np.array([0.7, 0.2, 0.05, 0.05], dtype=np.float64)
    q_max = 0.4
    dist = apply_floor_and_cap(q, epsilon=0.0, q_max=q_max)
    # arm 0 capped to 0.4 → excess 0.3. Under-cap entries are
    # 0.2 / 0.05 / 0.05 = 0.3 total. Redistribute proportionally:
    # 0.2 → 0.2 + 0.3 · (0.2 / 0.3) = 0.4
    # 0.05 → 0.05 + 0.3 · (0.05 / 0.3) = 0.10
    # arm 1 = 0.4 hits cap exactly; arms 2/3 = 0.10 each. Sum = 1.0.
    # After one iteration, arm 1 == cap (within tolerance) so no second
    # iteration triggers.
    assert dist.final[0] <= q_max + 1e-9
    assert dist.final[1] <= q_max + 1e-9
    np.testing.assert_allclose(dist.final.sum(), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Validity checks
# ---------------------------------------------------------------------------


def test_q_max_below_uniform_raises() -> None:
    n = 5
    q = np.full(n, 1.0 / n)
    with pytest.raises(ValueError, match="below 1/"):
        apply_floor_and_cap(q, epsilon=0.0, q_max=0.1)  # 0.1 < 1/5 = 0.2


def test_epsilon_out_of_range_raises() -> None:
    n = 4
    q = np.full(n, 1.0 / n)
    with pytest.raises(ValueError, match=r"epsilon must be in \[0, 1\]"):
        apply_floor_and_cap(q, epsilon=-0.1, q_max=None)
    with pytest.raises(ValueError, match=r"epsilon must be in \[0, 1\]"):
        apply_floor_and_cap(q, epsilon=1.5, q_max=None)


def test_q_raw_not_summing_to_one_raises() -> None:
    n = 4
    q = np.full(n, 0.4)  # sums to 1.6
    with pytest.raises(ValueError, match="must sum to 1"):
        apply_floor_and_cap(q, epsilon=0.0, q_max=None)


def test_q_raw_with_negative_entries_raises() -> None:
    q = np.array([0.6, -0.1, 0.3, 0.2], dtype=np.float64)
    with pytest.raises(ValueError, match="negative"):
        apply_floor_and_cap(q, epsilon=0.0, q_max=None)


def test_empty_q_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        apply_floor_and_cap(np.array([], dtype=np.float64), epsilon=0.0, q_max=None)


# ---------------------------------------------------------------------------
# Bundle fields
# ---------------------------------------------------------------------------


def test_returns_sampler_distribution_with_entropy_and_max() -> None:
    q = np.array([0.5, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
    dist = apply_floor_and_cap(q, epsilon=0.1, q_max=0.4)
    # entropy in nats matches hand computation
    expected_entropy = float(-np.sum(dist.final * np.log(dist.final)))
    assert np.isclose(dist.entropy, expected_entropy, atol=1e-9)
    assert np.isclose(dist.max_probability, dist.final.max(), atol=1e-9)
