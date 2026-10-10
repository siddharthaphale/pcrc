"""TwoStageSampler tests, ramp-agnostic.

Spec: docs/phase_2_correctness_tests_v2.md Section E.
"""

import torch

from pcrc.algorithms.two_stage_sampler import TwoStageSampler


def test_dormant_phase_returns_uniform():
    """Dormant phase ignores curator entirely."""
    n_candidates = 100
    sampler = TwoStageSampler(
        n_candidates=n_candidates,
        dormant_until=10,
        warmup_until=20,
    )
    curator_logits = torch.randn(n_candidates) * 100  # extreme — should be ignored
    p = sampler.get_distribution(step=5, curator_logits=curator_logits)

    expected_uniform = torch.ones(n_candidates) / n_candidates
    assert torch.allclose(p, expected_uniform, atol=1e-6)


def test_warmup_phase_is_between_uniform_and_curator():
    """Warmup p is some mixture of uniform and curator."""
    n_candidates = 100
    sampler = TwoStageSampler(
        n_candidates=n_candidates,
        dormant_until=10,
        warmup_until=20,
    )
    curator_logits = torch.zeros(n_candidates)
    curator_logits[0] = 10.0  # strongly prefer arm 0

    softmax_curator = torch.softmax(curator_logits, dim=0)
    uniform = torch.ones(n_candidates) / n_candidates

    p = sampler.get_distribution(step=15, curator_logits=curator_logits)

    assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-6)
    assert (p >= 0).all()
    assert not torch.allclose(p, uniform, atol=1e-3), (
        "Warmup distribution shouldn't equal pure uniform"
    )
    assert not torch.allclose(p, softmax_curator, atol=1e-3), (
        "Warmup distribution shouldn't equal pure curator softmax"
    )
    assert 0.01 < p[0] < 0.99, (
        f"Warmup arm 0 prob should be between uniform and curator, got {p[0]:.4f}"
    )


def test_warmup_progresses_toward_curator():
    """Later warmup steps weight curator at least as much as earlier warmup steps."""
    n_candidates = 100
    sampler = TwoStageSampler(
        n_candidates=n_candidates,
        dormant_until=10,
        warmup_until=30,
    )
    curator_logits = torch.zeros(n_candidates)
    curator_logits[0] = 10.0

    p_early = sampler.get_distribution(step=11, curator_logits=curator_logits)
    p_late = sampler.get_distribution(step=29, curator_logits=curator_logits)

    assert p_late[0] >= p_early[0] - 1e-6, (
        f"Late warmup should weight curator more (or equally). "
        f"Early p[0] = {p_early[0]:.4f}, Late p[0] = {p_late[0]:.4f}"
    )


def test_active_phase_uses_only_curator():
    """After warmup_until, p = softmax(curator_logits)."""
    n_candidates = 100
    sampler = TwoStageSampler(
        n_candidates=n_candidates,
        dormant_until=10,
        warmup_until=20,
    )
    curator_logits = torch.randn(n_candidates)
    p = sampler.get_distribution(step=50, curator_logits=curator_logits)
    expected = torch.softmax(curator_logits, dim=0)
    assert torch.allclose(p, expected, atol=1e-6)


def test_distribution_sums_to_one_in_all_phases():
    """Distribution always normalizes."""
    sampler = TwoStageSampler(n_candidates=100, dormant_until=10, warmup_until=20)
    curator_logits = torch.randn(100)
    for step in [0, 5, 10, 11, 15, 19, 20, 50, 100]:
        p = sampler.get_distribution(step=step, curator_logits=curator_logits)
        assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-6), (
            f"Sum != 1 at step {step}: sum={p.sum()}"
        )
        assert (p >= 0).all(), f"Negative probabilities at step {step}"
