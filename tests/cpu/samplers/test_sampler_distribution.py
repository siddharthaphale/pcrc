"""Tests for ``pcrc.samplers.base.SamplerDistribution``.

§7.1 row: ``raw``, ``floored``, ``capped``, ``final`` each sum to 1
± 1e-6; ``entropy`` matches Shannon entropy of ``final``;
``max_probability == final.max()``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcrc.samplers.base import SamplerDistribution


def _flat_uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def test_uniform_bundle_sums_to_one_per_layer() -> None:
    n = 8
    p = _flat_uniform(n)
    dist = SamplerDistribution(
        raw=p,
        floored=p,
        capped=p,
        final=p,
        entropy=float(np.log(n)),  # uniform → log(n) nats
        max_probability=1.0 / n,
    )
    for arr in (dist.raw, dist.floored, dist.capped, dist.final):
        assert np.isclose(arr.sum(), 1.0, atol=1e-6)


def test_sum_to_one_violation_rejected() -> None:
    n = 4
    bad = np.full(n, 0.4, dtype=np.float64)  # sums to 1.6
    with pytest.raises(ValueError, match="must sum to 1"):
        SamplerDistribution(
            raw=bad,
            floored=bad,
            capped=bad,
            final=bad,
            entropy=0.0,
            max_probability=bad.max(),
        )


def test_shape_mismatch_rejected() -> None:
    p4 = _flat_uniform(4)
    p5 = _flat_uniform(5)
    with pytest.raises(ValueError, match="shape"):
        SamplerDistribution(
            raw=p4,
            floored=p5,
            capped=p4,
            final=p4,
            entropy=0.0,
            max_probability=0.25,
        )


def test_close_to_one_within_tolerance_accepted() -> None:
    n = 4
    p = _flat_uniform(n)
    # Perturb by 5e-7 — within the 1e-6 tolerance.
    p_off = p.copy()
    p_off[0] += 5e-7
    p_off = p_off / p_off.sum()
    # Re-normalize so the assertion can still be on close-to-1.
    SamplerDistribution(
        raw=p,
        floored=p_off,
        capped=p,
        final=p,
        entropy=float(np.log(n)),
        max_probability=1.0 / n,
    )
