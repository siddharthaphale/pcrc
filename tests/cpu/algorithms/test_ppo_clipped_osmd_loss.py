"""PPO-clipped OSMD surrogate tests across all 4 quadrants of (sign(A), ratio).

Spec: docs/phase_2_correctness_tests_v2.md Section B.2.

Spec note: the spec uses rounded literals like 1.65 / 1.105 with atol=1e-3.
Actual values are e^0.5 ≈ 1.6487 and e^0.1 ≈ 1.1052; |1.6487 - 1.65| = 0.0013
exceeds 1e-3. We compare against math.exp(...) directly to keep atol=1e-3 honest.
"""

import math

import torch

from pcrc.algorithms.osmd import compute_ppo_clipped_osmd_loss


def test_ppo_clip_loss_shape_and_dtype():
    log_p_old = torch.randn(8)
    log_p_new = torch.randn(8, requires_grad=True)
    advantages = torch.randn(8)
    loss = compute_ppo_clipped_osmd_loss(log_p_old, log_p_new, advantages, eps=0.2)
    assert loss.dim() == 0
    assert loss.requires_grad


def test_ppo_clip_matches_unclipped_when_ratio_in_range():
    """ratio = 1 (all log-probs equal) → loss = -mean(advantages)."""
    log_p_old = torch.zeros(8)
    log_p_new = torch.zeros(8, requires_grad=True)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 0.9, -0.9])

    loss = compute_ppo_clipped_osmd_loss(log_p_old, log_p_new, advantages, eps=0.2)
    expected = -advantages.mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_ppo_clip_q1_positive_advantage_high_ratio_gradient_zero():
    """Q1: A > 0, ratio > 1+ε → clipped term selected, gradient zero."""
    log_p_old = torch.zeros(1)
    log_p_new = torch.tensor([0.5], requires_grad=True)  # ratio = e^0.5 ≈ 1.65
    advantages = torch.tensor([1.0])

    loss = compute_ppo_clipped_osmd_loss(log_p_old, log_p_new, advantages, eps=0.2)
    loss.backward()

    # min(1.65 · 1, 1.2 · 1) = 1.2; loss = -1.2
    assert torch.allclose(loss, torch.tensor(-1.2), atol=1e-3)
    # PPO's intent: don't push policy further when already moved in good
    # direction
    assert torch.allclose(log_p_new.grad, torch.zeros(1), atol=1e-6)


def test_ppo_clip_q2_positive_advantage_in_range_gradient_nonzero():
    """Q2: A > 0, ratio in clip range → unclipped, gradient = -A·ratio."""
    log_p_old = torch.zeros(1)
    log_p_new = torch.tensor([0.1], requires_grad=True)  # ratio = e^0.1 ≈ 1.105
    advantages = torch.tensor([1.0])

    loss = compute_ppo_clipped_osmd_loss(log_p_old, log_p_new, advantages, eps=0.2)
    loss.backward()

    r = math.exp(0.1)  # ≈ 1.1052
    # Both terms: r · 1 = r; min = r; loss = -r
    assert torch.allclose(loss, torch.tensor(-r), atol=1e-3)
    # Gradient: ∂(-r·A)/∂log_p_new = -r·A = -r
    assert torch.allclose(log_p_new.grad, torch.tensor([-r]), atol=1e-3)


def test_ppo_clip_q3_negative_advantage_high_ratio_gradient_nonzero():
    """Q3: A < 0, ratio > 1+ε → unclipped term wins, gradient nonzero.

    Critical PPO behavior: when policy moved in wrong direction (high ratio
    but negative advantage), give full gradient to push back.
    """
    log_p_old = torch.zeros(1)
    log_p_new = torch.tensor([0.5], requires_grad=True)  # ratio = e^0.5 ≈ 1.65
    advantages = torch.tensor([-1.0])

    loss = compute_ppo_clipped_osmd_loss(log_p_old, log_p_new, advantages, eps=0.2)
    loss.backward()

    r = math.exp(0.5)  # ≈ 1.6487
    # Unclipped: r · (-1) = -r
    # Clipped:   1.2 · (-1) = -1.2
    # min(-r, -1.2) = -r (since r > 1.2); loss = -(-r) = r
    assert torch.allclose(loss, torch.tensor(r), atol=1e-3)
    # ∂(-r·A)/∂log_p_new = -r·A = r
    assert torch.allclose(log_p_new.grad, torch.tensor([r]), atol=1e-3)


def test_ppo_clip_q4_negative_advantage_low_ratio_gradient_zero():
    """Q4: A < 0, ratio < 1-ε → clipped term selected, gradient zero."""
    log_p_old = torch.zeros(1)
    log_p_new = torch.tensor([-0.5], requires_grad=True)  # ratio = e^-0.5 ≈ 0.607
    advantages = torch.tensor([-1.0])

    loss = compute_ppo_clipped_osmd_loss(log_p_old, log_p_new, advantages, eps=0.2)
    loss.backward()

    # Unclipped: 0.607 · (-1) = -0.607
    # Clipped: 0.8 · (-1) = -0.8
    # min(-0.607, -0.8) = -0.8; loss = 0.8
    assert torch.allclose(loss, torch.tensor(0.8), atol=1e-3)
    assert torch.allclose(log_p_new.grad, torch.zeros(1), atol=1e-6)
