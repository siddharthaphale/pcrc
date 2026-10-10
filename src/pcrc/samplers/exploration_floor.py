"""Canonical exploration-floor + cap primitive.

Distinct from :class:`pcrc.algorithms.two_stage_sampler.TwoStageSampler`
— that primitive handles candidate-batch warmup blending and is only
used where curator-scored logits exist for a sub-bank. For the pre-VLA
setting (``|X| ≤ 1000``), full-bank scoring is preferred and the
exploration floor + cap is applied here.

Math:
    ``q_floored = (1 - ε) · q_raw + ε / |X|``
    then cap entries to ``q_max`` via iterative water-filling, then
    renormalize. The water-filling step is needed because a single
    clip pass may push redistributed mass past the cap on previously
    under-cap entries; the loop terminates because each iteration
    strictly increases the under-cap mass until the constraint is
    satisfied to the requested tolerance.

Validity (raised as ``ValueError`` at call time):
    - ``0 <= epsilon <= 1``
    - ``q_max is None or q_max >= 1 / n`` (otherwise no distribution
      on ``n`` items can have ``max(q) <= q_max < 1/n``)

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``exploration_floor.py`` row), §3.1 (``SamplerDistribution`` bundle
fields), §7.1 ``test_exploration_floor.py`` (full validity test list).
"""

from __future__ import annotations

import numpy as np

from pcrc.samplers.base import SamplerDistribution


def _shannon_entropy_nats(p: np.ndarray) -> float:
    """Shannon entropy in nats; ``0 · log 0 = 0`` by convention."""
    p = np.asarray(p, dtype=np.float64)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))


def apply_floor_and_cap(
    q_raw: np.ndarray,
    epsilon: float,
    q_max: float | None,
    *,
    cap_tolerance: float = 1e-9,
    max_iterations: int = 64,
) -> SamplerDistribution:
    """Apply exploration floor + optional max-probability cap.

    Args:
        q_raw: shape ``(|X|,)`` raw sampler distribution. Must sum to 1
            within ``1e-6``.
        epsilon: exploration floor weight. ``0`` disables the floor;
            ``1`` collapses to uniform. Must satisfy ``0 <= ε <= 1``.
        q_max: optional per-entry maximum probability. ``None`` skips
            the cap. Otherwise must satisfy ``q_max >= 1 / |X|``.
        cap_tolerance: iteration stops when ``q.max() <= q_max +
            cap_tolerance``.
        max_iterations: safety cap on the water-filling loop; raises
            ``RuntimeError`` if not converged.

    Returns:
        :class:`SamplerDistribution` with the raw / floored / capped /
        final layers populated.

    Raises:
        ValueError: epsilon out of range, q_max below ``1/|X|``,
            ``q_raw`` does not sum to 1, or
            ``(epsilon, q_max)`` is infeasible.
    """
    q_raw = np.asarray(q_raw, dtype=np.float64)
    if q_raw.ndim != 1:
        raise ValueError(f"q_raw must be 1-D; got shape {q_raw.shape}")
    n = q_raw.shape[0]
    if n == 0:
        raise ValueError("q_raw is empty")
    s = float(q_raw.sum())
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"q_raw must sum to 1 within 1e-6; got {s}")
    if (q_raw < -1e-12).any():
        raise ValueError("q_raw contains negative entries")
    # Clamp tiny negatives (numerical noise from upstream) to zero so
    # downstream arithmetic stays in [0, 1].
    q_raw = np.maximum(q_raw, 0.0)

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1]; got {epsilon}")
    uniform_p = 1.0 / n
    if q_max is not None and q_max < uniform_p - 1e-12:
        raise ValueError(
            f"q_max={q_max} is below 1/|X|={uniform_p}; no distribution "
            f"on {n} items can have max <= q_max < 1/|X|."
        )

    # Step 1 — floor.
    floored = (1.0 - epsilon) * q_raw + epsilon * uniform_p
    # Normalize defensively (the affine combination is sum-preserving
    # up to floating-point noise).
    floored = floored / floored.sum()

    # Step 2 — cap.
    if q_max is None:
        capped = floored.copy()
    else:
        capped = floored.copy()
        for _it in range(max_iterations):
            over = capped > q_max + cap_tolerance
            if not over.any():
                break
            # Excess mass collectively held above the cap.
            excess = float(capped[over].sum() - q_max * over.sum())
            capped[over] = q_max
            under = ~over
            under_mass = float(capped[under].sum())
            if under_mass <= 0:
                # No room to redistribute — only possible at the
                # boundary q_max == 1/n where the cap forces uniform.
                # Fall through to renormalization below; further
                # iterations cannot help.
                break
            # Water-fill proportional to current under-cap mass. Pure
            # uniform redistribution among under-cap entries is the
            # other natural choice and produces the same fixed point
            # in the limit; proportional is gentler under noise.
            capped[under] += excess * capped[under] / under_mass
        else:
            raise RuntimeError(
                f"apply_floor_and_cap: cap iteration did not converge in "
                f"{max_iterations} steps (q_max={q_max}, epsilon={epsilon}, "
                f"n={n}). This is likely an infeasible "
                f"(epsilon, q_max) combination."
            )
        # Final renormalization to absorb floating-point drift from the
        # water-filling loop.
        capped = capped / capped.sum()

    final = capped
    return SamplerDistribution(
        raw=q_raw.copy(),
        floored=floored,
        capped=capped,
        final=final,
        entropy=_shannon_entropy_nats(final),
        max_probability=float(final.max()),
    )
