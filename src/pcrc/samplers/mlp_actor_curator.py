"""MLP curator sampler — Priority 2 main result.

Trains a tiny MLP on :meth:`FetchTaskBank.features` to map every
context's feature vector to a softmax logit; the resulting distribution
over the bank is updated via the PPO-clipped surrogate
:func:`pcrc.algorithms.osmd.compute_ppo_clipped_osmd_loss` with the
per-context policy-improvement utility as the advantage.

The curator's parameter update is *separate* from the actor PPO
update — tensors flowing into the curator loss are detached from the
actor graph (the utility comes in as a Python float dict from
:func:`ppo_policy_improvement_utility`), and the curator's optimizer
sees only curator params. The two loops therefore can't accidentally
share gradient signal.

Feature perturbation supports the §4 Priority 2 controls:

    - ``"none"``: features straight from the bank.
    - ``"shuffled"``: at construction, build a seed-stable permutation
      ``π`` over context ids; the feature row for context ``cid`` is
      whatever the bank produces for ``π(cid)``. Same permutation is
      used across every update.
    - ``"random"``: at construction, replace every feature row with an
      iid sample of the same dim and rough variance — a true
      destruction control.

NEVER receives ``context_id`` / ``reset_seed`` / unique IDs as a
feature for the main result (§9 risk #13). The MLP curator becomes
load-bearing only at Priority 2 scale; for Priorities 1b–1c the
tabular AC sampler is the canonical AC condition.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``mlp_actor_curator.py`` row), §4 Priority 2 (controls + scale),
§7.1 ``test_mlp_actor_curator_controls.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pcrc.algorithms.osmd import compute_ppo_clipped_osmd_loss
from pcrc.envs.fetch_task_bank import FetchTaskBank
from pcrc.registry import SamplerName
from pcrc.samplers.base import Sampler, SamplerDistribution, SamplerStats
from pcrc.samplers.exploration_floor import apply_floor_and_cap


def _build_curator_mlp(in_dim: int, hidden_sizes: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.Tanh())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class MLPActorCuratorSampler(Sampler):
    """MLP-conditioned PPO-clipped curator over the bank.

    Attributes:
        alpha: softmax temperature on the MLP logits.
        clip_eps: PPO clip for the curator parameter update.
        n_epochs: number of gradient steps per :meth:`update` call.
        feature_noise: stddev of Gaussian noise added to features
            on every forward (regularization).
    """

    name = SamplerName.MLP_ACTOR_CURATOR

    def __init__(
        self,
        bank: FetchTaskBank,
        *,
        alpha: float = 1.0,
        learning_rate: float = 1.0e-3,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        hidden_sizes: tuple[int, ...] = (64, 64),
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        feature_noise: float = 0.0,
        feature_perturbation: str = "none",
        feature_perturbation_seed: int = 0,
        epsilon: float = 0.0,
        q_max: float | None = None,
        device: str = "cpu",
    ):
        """Validate hyperparameters; build the (perturbed) feature matrix + MLP."""
        train_ids = bank.train_ids()
        if list(range(len(train_ids))) != train_ids:
            raise ValueError(
                "MLPActorCuratorSampler requires bank.train_ids() == "
                "range(0, n_train); got "
                f"min={train_ids[0]} max={train_ids[-1]} "
                f"len={len(train_ids)}. Reorder the bank so train "
                "contexts have ids 0..n_train-1."
            )
        # Operate over the train split only — consistent with the loop's
        # build_sampler_for_condition convention.
        super().__init__(len(train_ids))
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0; got {alpha}")
        if clip_eps <= 0:
            raise ValueError(f"clip_eps must be > 0; got {clip_eps}")
        if n_epochs <= 0:
            raise ValueError(f"n_epochs must be > 0; got {n_epochs}")
        if feature_perturbation not in ("none", "shuffled", "random"):
            raise ValueError(
                f"feature_perturbation must be 'none' / 'shuffled' / "
                f"'random'; got {feature_perturbation!r}"
            )
        self.alpha = float(alpha)
        self.clip_eps = float(clip_eps)
        self.n_epochs = int(n_epochs)
        self.feature_noise = float(feature_noise)
        self.feature_perturbation = feature_perturbation
        self.epsilon = float(epsilon)
        self.q_max = q_max
        self.device = torch.device(device)

        # Build the feature matrix once. Row order matches train-bank
        # context_id 0..n_train-1.
        feature_matrix = np.stack([bank.features(cid) for cid in train_ids])
        if feature_perturbation == "shuffled":
            perm = np.random.default_rng(feature_perturbation_seed).permutation(len(train_ids))
            feature_matrix = feature_matrix[perm]
        elif feature_perturbation == "random":
            rng = np.random.default_rng(feature_perturbation_seed)
            mu = feature_matrix.mean(axis=0)
            sigma = np.maximum(feature_matrix.std(axis=0), 1e-3)
            feature_matrix = rng.normal(loc=mu, scale=sigma, size=feature_matrix.shape)
        self._features = torch.as_tensor(feature_matrix, dtype=torch.float32, device=self.device)

        self._mlp = _build_curator_mlp(
            in_dim=self._features.shape[1],
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(self.device)
        self._optimizer = torch.optim.AdamW(
            self._mlp.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    # ----- internal helpers -----

    def _forward_log_probs(self, *, noisy: bool) -> torch.Tensor:
        """Return ``log_softmax`` over the bank with optional input noise."""
        f = self._features
        if noisy and self.feature_noise > 0:
            f = f + torch.randn_like(f) * self.feature_noise
        logits = self._mlp(f).squeeze(-1) * self.alpha
        return torch.log_softmax(logits, dim=0)

    # ----- Sampler API -----

    def update(self, stats: SamplerStats, step: int) -> None:
        """``n_epochs`` PPO-clipped gradient steps on the curator MLP.

        Observed contexts only. Unobserved contexts contribute neither
        to the loss nor to the gradient — the curator only learns from
        contexts the actor actually rolled this iteration.
        """
        del step
        if stats.observed_mask.shape != (self.n_contexts,):
            raise ValueError(
                f"stats shape mismatch: expected ({self.n_contexts},), got "
                f"{stats.observed_mask.shape}"
            )
        obs_mask = stats.observed_mask
        if not obs_mask.any():
            return

        obs_idx = torch.as_tensor(np.where(obs_mask)[0], dtype=torch.long, device=self.device)
        # Per-rollout mean utility on observed contexts.
        u_obs = stats.utility_sum_this_step[obs_mask] / stats.samples_this_step[obs_mask]
        u_t = torch.as_tensor(u_obs, dtype=torch.float32, device=self.device).detach()

        # OLD log probs from the policy AT THE START of this update.
        with torch.no_grad():
            old_log_probs = self._forward_log_probs(noisy=False).detach()
        old_logp_obs = old_log_probs[obs_idx]

        self._mlp.train()
        for _ in range(self.n_epochs):
            new_log_probs = self._forward_log_probs(noisy=True)
            new_logp_obs = new_log_probs[obs_idx]
            loss = compute_ppo_clipped_osmd_loss(old_logp_obs, new_logp_obs, u_t, eps=self.clip_eps)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def distribution(self, step: int) -> SamplerDistribution:
        """Softmax over MLP logits → floor + cap."""
        del step
        self._mlp.eval()
        with torch.no_grad():
            log_probs = self._forward_log_probs(noisy=False)
            raw = log_probs.exp().cpu().numpy().astype(np.float64)
        # Numerical safety: rebuild a clean simplex point.
        raw = np.maximum(raw, 0.0)
        raw = raw / raw.sum()
        return apply_floor_and_cap(raw, epsilon=self.epsilon, q_max=self.q_max)

    # ----- diagnostics -----

    @torch.no_grad()
    def get_softmax_probs(self) -> np.ndarray:
        """Raw softmax probability per context (no floor / cap applied)."""
        self._mlp.eval()
        return self._forward_log_probs(noisy=False).exp().cpu().numpy().astype(np.float64)
