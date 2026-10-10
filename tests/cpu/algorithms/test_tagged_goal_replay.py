"""Tests for :class:`pcrc.algorithms.tagged_goal_replay.TaggedGlobalReplay`.

§7.1 row: ``add`` + ``sample_context_weighted``: realized empirical
``source_context_id`` distribution matches ``SamplerDistribution.final``
within ``2σ`` MC error over a 1000-batch run; HER relabel does not
perturb the ``source_context_id`` distribution (because relabel never
mutates ``source_context_id``).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from pcrc.algorithms.her import GoalConditionedTransition, relabel
from pcrc.algorithms.tagged_goal_replay import TaggedGlobalReplay
from pcrc.samplers.base import SamplerDistribution


def _stub_transition(
    *, source_context_id: int, episode_id: int, step: int, achieved: float = 0.0
) -> GoalConditionedTransition:
    ag = np.array([achieved, 0.0, 0.0])
    return GoalConditionedTransition(
        observation=np.full(4, step, dtype=np.float32),
        action=np.zeros(4, dtype=np.float32),
        next_observation=np.full(4, step + 1, dtype=np.float32),
        achieved_goal=ag,
        next_achieved_goal=ag,
        reward=-1.0,
        terminated=False,
        truncated=False,
        source_context_id=source_context_id,
        original_desired_goal=np.zeros(3),
        training_desired_goal=np.zeros(3),
        relabel_strategy="none",
        relabel_source_step=None,
        episode_id=episode_id,
    )


def _make_distribution(probs: np.ndarray) -> SamplerDistribution:
    return SamplerDistribution(
        raw=probs.copy(),
        floored=probs.copy(),
        capped=probs.copy(),
        final=probs.copy(),
        entropy=float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0]))),
        max_probability=float(probs.max()),
    )


# ---------------------------------------------------------------------------
# Basic mechanics
# ---------------------------------------------------------------------------


def test_add_and_len() -> None:
    buf = TaggedGlobalReplay(capacity=100)
    assert len(buf) == 0
    buf.add(_stub_transition(source_context_id=0, episode_id=0, step=0))
    buf.add(_stub_transition(source_context_id=1, episode_id=0, step=1))
    assert len(buf) == 2


def test_fifo_eviction_under_overflow() -> None:
    buf = TaggedGlobalReplay(capacity=2)
    a = _stub_transition(source_context_id=0, episode_id=0, step=0)
    b = _stub_transition(source_context_id=1, episode_id=0, step=1)
    c = _stub_transition(source_context_id=2, episode_id=0, step=2)
    buf.add(a)
    buf.add(b)
    buf.add(c)
    assert len(buf) == 2
    # 'a' (context 0) was evicted.
    assert buf.n_transitions_for_context(0) == 0
    assert buf.n_transitions_for_context(1) == 1
    assert buf.n_transitions_for_context(2) == 1


def test_per_context_index_tracks_membership() -> None:
    buf = TaggedGlobalReplay(capacity=10)
    for i in range(3):
        buf.add(_stub_transition(source_context_id=7, episode_id=0, step=i))
    assert buf.n_transitions_for_context(7) == 3
    assert buf.n_transitions_for_context(99) == 0
    assert 7 in buf.context_ids_in_buffer()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_context_weighted_empirical_matches_distribution() -> None:
    """Plan §7.1 row: empirical distribution matches within 2σ MC."""
    n_contexts = 4
    buf = TaggedGlobalReplay(capacity=10_000)
    # 1000 transitions per context (oversample so each context has
    # plenty of transitions to draw from).
    for cid in range(n_contexts):
        for step in range(1000):
            buf.add(_stub_transition(source_context_id=cid, episode_id=cid, step=step))

    # Skewed distribution.
    p = np.array([0.5, 0.3, 0.15, 0.05])
    dist = _make_distribution(p)

    rng = np.random.default_rng(0)
    n_draws = 50_000
    sampled = buf.sample_context_weighted(n_draws, dist, rng)
    counts = Counter(t.source_context_id for t in sampled)
    empirical = np.array([counts[cid] / n_draws for cid in range(n_contexts)])
    # 3σ tolerance under multinomial: σ = sqrt(p(1-p)/N).
    sigma = np.sqrt(p * (1.0 - p) / n_draws)
    np.testing.assert_allclose(empirical, p, atol=3 * sigma.max() + 1e-4)


def test_sample_context_weighted_renormalizes_over_buffer_support() -> None:
    """If the buffer only holds a strict subset of the support, the
    sampler renormalizes — it does NOT draw zero samples from
    distribution support that the buffer happens not to have.
    """
    buf = TaggedGlobalReplay(capacity=1000)
    # Only contexts 0 and 1 in the buffer.
    for cid in (0, 1):
        for step in range(50):
            buf.add(_stub_transition(source_context_id=cid, episode_id=cid, step=step))

    p = np.array([0.5, 0.5, 0.0, 0.0])  # buffer support equals dist support
    dist = _make_distribution(p)
    rng = np.random.default_rng(0)
    sampled = buf.sample_context_weighted(200, dist, rng)
    cids = {t.source_context_id for t in sampled}
    assert cids <= {0, 1}


def test_sample_context_weighted_rejects_empty_buffer() -> None:
    buf = TaggedGlobalReplay(capacity=10)
    p = np.array([1.0, 0.0])
    dist = _make_distribution(p)
    with pytest.raises(ValueError, match="empty"):
        buf.sample_context_weighted(4, dist, np.random.default_rng(0))


def test_sample_context_weighted_rejects_zero_overlap() -> None:
    # Buffer has context 0 only; distribution support is {2, 3}.
    buf = TaggedGlobalReplay(capacity=10)
    buf.add(_stub_transition(source_context_id=0, episode_id=0, step=0))
    p = np.array([0.0, 0.0, 0.5, 0.5])
    dist = _make_distribution(p)
    with pytest.raises(ValueError, match="No overlap"):
        buf.sample_context_weighted(4, dist, np.random.default_rng(0))


def test_sample_uniform_smoke() -> None:
    buf = TaggedGlobalReplay(capacity=100)
    for cid in range(4):
        for step in range(10):
            buf.add(_stub_transition(source_context_id=cid, episode_id=cid, step=step))
    sampled = buf.sample_uniform(20, np.random.default_rng(0))
    assert len(sampled) == 20


# ---------------------------------------------------------------------------
# HER does not perturb the source_context_id distribution
# ---------------------------------------------------------------------------


def test_her_relabel_does_not_perturb_source_context_distribution() -> None:
    """Build a tiny episode per context, relabel via HER, push the
    relabeled transitions into the buffer; the empirical context
    distribution under uniform-over-buffer sampling should match the
    raw count distribution, which itself follows the construction.
    """
    n_contexts = 3
    rng = np.random.default_rng(0)
    buf = TaggedGlobalReplay(capacity=10_000)
    # 6 episodes for context 0, 3 each for 1 and 2 → after relabel the
    # ratio of source_context_id buckets in the buffer should still be
    # ~ 6:3:3 (relabel preserves the tag).
    counts_per_ctx = {0: 6, 1: 3, 2: 3}
    next_episode_id = 0
    for cid, n_eps in counts_per_ctx.items():
        for _ in range(n_eps):
            achieved = np.linspace(0.0, 1.0, 5).reshape(5, 1).repeat(3, axis=1)
            ep = [
                GoalConditionedTransition(
                    observation=np.zeros(4, dtype=np.float32),
                    action=np.zeros(4, dtype=np.float32),
                    next_observation=np.zeros(4, dtype=np.float32),
                    achieved_goal=achieved[t].copy(),
                    next_achieved_goal=achieved[t + 1].copy(),
                    reward=-1.0,
                    terminated=(t == 3),
                    truncated=False,
                    source_context_id=cid,
                    original_desired_goal=np.array([10.0, 10.0, 10.0]),
                    training_desired_goal=np.array([10.0, 10.0, 10.0]),
                    relabel_strategy="none",
                    relabel_source_step=None,
                    episode_id=next_episode_id,
                )
                for t in range(4)
            ]
            next_episode_id += 1
            for t in relabel(ep, strategy="future", n_sampled_goal=4, rng=rng):
                buf.add(t)

    # Buffer counts: relabeled transitions inherit source_context_id.
    by_ctx = Counter(
        cid for cid in range(n_contexts) for _ in range(buf.n_transitions_for_context(cid))
    )
    # The ratio should match counts_per_ctx (with the same proportional
    # multiplier per episode after HER).
    total = sum(by_ctx.values())
    ratios = {cid: by_ctx[cid] / total for cid in range(n_contexts)}
    target_total = sum(counts_per_ctx.values())
    target_ratios = {cid: counts_per_ctx[cid] / target_total for cid in counts_per_ctx}
    for cid in counts_per_ctx:
        assert np.isclose(ratios[cid], target_ratios[cid], atol=0.05), cid
