"""Contrastive RL (CRL) agent — goal-conditioned critic trained by InfoNCE, no TD learning.

Implements Eysenbach et al. 2022 ("Contrastive Learning as Goal-Conditioned RL") on top of the
existing SAC plumbing: the actor is the unchanged :class:`SACActor` (so rollout-worker actor
state-dict sync keeps working), the running state normalizer is shared, and the critic is replaced
by a contrastive pair of encoders

    f(s, a, g) = phi(s, a) . psi(g)

trained with the InfoNCE classification loss over in-batch negatives, where the positive goal g_i
is an ACHIEVED goal from a future step of the same episode. In this codebase those positives come
for free: HER's "future" relabels store exactly that in ``training_desired_goal``, so the loop
feeds CRL minibatches of future-relabeled transitions (no reward, done, or bootstrapping used).

The actor maximizes f(s, pi(s,g), g) with SAC-style automatic entropy temperature. B2 / M4 of
run_log.md; docs/rlc_maintrack_updgrade_experiments.md (T2.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from pcrc.algorithms.sac_continuous import RunningMeanStd, SACActor, SACContinuousConfig


@dataclass
class CRLContinuousConfig:
    """CRL hyperparameters; ``sac`` supplies dims, action bounds, lr, and actor arch."""

    sac: SACContinuousConfig
    goal_dim: int = 3
    repr_dim: int = 64  # phi/psi embedding size (CRL paper uses 64)
    logsumexp_penalty: float = 0.1  # regularizer on logits magnitude (stabilizes InfoNCE)

    def __post_init__(self) -> None:
        if self.goal_dim <= 0 or self.goal_dim >= self.sac.state_dim:
            raise ValueError(
                f"goal_dim must be in (0, state_dim={self.sac.state_dim}); got {self.goal_dim}"
            )
        if self.repr_dim <= 0:
            raise ValueError(f"repr_dim must be > 0; got {self.repr_dim}")


def _mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class CRLAgent:
    """Duck-types the subset of :class:`SACAgent` the SAC+HER loop uses.

    Provided: ``act`` / ``actor`` / ``obs_rms`` / ``device`` / ``update_crl``. TD-error paths are
    unsupported (``supports_td = False``); the loop guards them. ``update`` raises — the loop must
    route CRL minibatches (obs, action, future-goal) through :meth:`update_crl`.
    """

    supports_td = False

    def __init__(self, cfg: CRLContinuousConfig):
        self.cfg = cfg
        sac = cfg.sac
        self.device = torch.device(sac.device)
        self.obs_dim = sac.state_dim - cfg.goal_dim
        self.obs_rms = RunningMeanStd(sac.state_dim, self.device)
        self.actor = SACActor(sac).to(self.device)
        self.phi = _mlp(self.obs_dim + sac.action_dim, cfg.repr_dim, sac.hidden_sizes).to(
            self.device
        )
        self.psi = _mlp(cfg.goal_dim, cfg.repr_dim, sac.hidden_sizes).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=sac.learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.phi.parameters()) + list(self.psi.parameters()), lr=sac.learning_rate
        )
        self.log_alpha = torch.tensor(sac.init_log_alpha, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=sac.learning_rate)
        self.target_entropy = (
            -float(sac.action_dim) if sac.target_entropy is None else float(sac.target_entropy)
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(
        self, state: np.ndarray, *, deterministic: bool = False, update_rms: bool = False
    ) -> np.ndarray:
        """Identical contract to SACAgent.act (state = concat(obs, goal))."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if update_rms:
            self.obs_rms.update(s)
        s = self.obs_rms.normalize(s)
        if deterministic:
            mean, _ = self.actor(s)
            action = torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
        else:
            action, _, _ = self.actor.sample(s)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, *args, **kwargs):  # pragma: no cover - defensive
        raise RuntimeError("CRLAgent has no TD update; the loop must call update_crl")

    def update_crl(
        self, obs: np.ndarray, actions: np.ndarray, goals: np.ndarray
    ) -> dict[str, float]:
        """One contrastive critic + actor step.

        Args:
            obs: raw observations ``(B, obs_dim)`` (no goal concat).
            actions: executed actions ``(B, action_dim)``.
            goals: FUTURE achieved goals of the same episode ``(B, goal_dim)`` (HER "future"
                relabels) — the InfoNCE positives; other rows serve as in-batch negatives.
        """
        b = obs.shape[0]
        s_full = np.concatenate([obs, goals], axis=1).astype(np.float32)
        s = self.obs_rms.normalize(
            torch.as_tensor(s_full, dtype=torch.float32, device=self.device)
        )
        obs_n = s[:, : self.obs_dim]
        g_n = s[:, self.obs_dim :]
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

        # ------------------------------------------------------------- critic (InfoNCE)
        za = self.phi(torch.cat([obs_n, a], dim=1))  # (B, d)
        zg = self.psi(g_n)  # (B, d)
        logits = za @ zg.T  # (B, B); diagonal = positives
        labels = torch.arange(b, device=self.device)
        critic_loss = F.cross_entropy(logits, labels)
        if self.cfg.logsumexp_penalty > 0:
            critic_loss = critic_loss + self.cfg.logsumexp_penalty * (
                torch.logsumexp(logits, dim=1) ** 2
            ).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------------------------------------------------------ actor
        a_pi, logp, _ = self.actor.sample(s)
        with torch.no_grad():
            zg_d = self.psi(g_n)
        f_pi = (self.phi(torch.cat([obs_n, a_pi], dim=1)) * zg_d).sum(dim=1)
        actor_loss = (self.alpha.detach() * logp - f_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "contrastive_acc": float(acc.item()),
        }
