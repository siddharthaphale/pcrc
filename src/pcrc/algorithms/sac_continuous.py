"""Continuous-action SAC for goal-conditioned Fetch.

Adapted from CleanRL ``sac_continuous_action.py`` (MIT). Original
copyright Costa Huang et al. The goal is part of the state — callers
concatenate ``observation`` and ``desired_goal`` and feed the
resulting ``state`` vector to :meth:`SACAgent.act` / :meth:`update`,
so the SAC layer is goal-agnostic and only sees a continuous state.

Shape mirrors the standard SAC formulation (Haarnoja et al. 2018):
double-Q critic with EMA target networks, tanh-squashed Gaussian
actor with reparameterized sampling, auto-tuned entropy temperature.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§5 (CleanRL SAC port — chosen
deliberately here for the cleaner adapter path), §4 Priority 3 (SAC
+ HER driver for sparse Fetch).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812 — standard PyTorch alias

from pcrc.algorithms.ppo_continuous import RunningMeanStd
from pcrc.utils.atomic_io import atomic_write_torch

_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0
_EPS_TANH = 1e-6


@dataclass
class SACContinuousConfig:
    """SAC + double-Q + auto-α hyperparameters."""

    state_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: float | None = None  # default -action_dim
    init_log_alpha: float = float(np.log(0.2))
    # Floor on the entropy temperature. Auto-alpha collapses to ~0 on hard
    # sparse tasks once logp_pi rises, killing exploration so a drifting policy
    # can never recover (observed on Meta-World push demos+BC: alpha->0 then the
    # critic diverges). alpha_min > 0 clamps it so exploration persists. 0 = OFF.
    alpha_min: float = 0.0
    action_low: float = -1.0
    action_high: float = 1.0
    device: str = "cpu"
    # PCGrad gradient surgery across groups (Yu et al. 2020) — mitigates shared-net negative
    # transfer between heterogeneous context groups (PCRC generalization probe: the §3.8a washout
    # is interference). When True AND update() is given group_ids, per-group critic/actor gradients
    # are projected to remove conflicting components before the optimizer step. 0/False = vanilla.
    pcgrad: bool = False
    # RLPD/DroQ recipe (Ball et al. 2023): LayerNorm in the critic MLP stabilizes value learning at a high
    # update-to-data ratio — the key lever for sparse-reward continuous control. Critic-only; 0/False = OFF.
    layer_norm: bool = False


def _pcgrad_project(per_task_grads: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """PCGrad (Yu et al. 2020): for each task gradient, project out the component conflicting with
    every other task (where the dot product is negative), then sum. ``per_task_grads[t][p]`` is the
    gradient of task ``t`` for parameter ``p``; returns the projected sum per parameter.
    """
    n = len(per_task_grads)
    if n == 1:
        return per_task_grads[0]
    flat = [torch.cat([g.reshape(-1) for g in task]) for task in per_task_grads]
    proj = [f.clone() for f in flat]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot = torch.dot(proj[i], flat[j])
            if dot < 0:
                proj[i] = proj[i] - (dot / (torch.dot(flat[j], flat[j]) + 1e-12)) * flat[j]
    summed = torch.stack(proj).sum(dim=0)
    out, idx = [], 0
    for g in per_task_grads[0]:
        k = g.numel()
        out.append(summed[idx : idx + k].view_as(g))
        idx += k
    return out


def _mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...], layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        if layer_norm:  # RLPD: Linear -> LayerNorm -> activation
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SACActor(nn.Module):
    """Tanh-squashed Gaussian policy."""

    def __init__(self, cfg: SACContinuousConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = _mlp(cfg.state_dim, cfg.hidden_sizes[-1], cfg.hidden_sizes[:-1])
        self.mean_head = nn.Linear(cfg.hidden_sizes[-1], cfg.action_dim)
        self.log_std_head = nn.Linear(cfg.hidden_sizes[-1], cfg.action_dim)
        # Action scaling so tanh output maps to [low, high].
        self.register_buffer(
            "action_scale",
            torch.tensor((cfg.action_high - cfg.action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((cfg.action_high + cfg.action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) tensors per state."""
        h = F.relu(self.trunk(state))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterized sample. Returns (action, log_prob, tanh_mean).

        ``log_prob`` is the squashed log-prob (CleanRL Eq. 21):
        ``log π(a|s) = log N(z|μ, σ) - sum log(1 - tanh(z)^2 + ε)``.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        y = torch.tanh(z)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1 - y**2) + _EPS_TANH)
        log_prob = log_prob.sum(-1)
        tanh_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, tanh_mean

    def log_prob_of_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-density of a GIVEN env-space action under this policy.

        Inverts the tanh squash (``y = (a - bias)/scale``, ``z = atanh(y)``)
        and evaluates the same squashed log-prob as :meth:`sample`. Used to
        form the importance ratio ``π_new(a|s)/π_old(a|s)`` for the
        Actor-Curator IS-corrected advantage utility (the actions are the ones
        actually executed during rollout, not freshly sampled).
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        y = (action - self.action_bias) / self.action_scale
        y = y.clamp(-1.0 + _EPS_TANH, 1.0 - _EPS_TANH)
        z = torch.atanh(y)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1 - y**2) + _EPS_TANH)
        return log_prob.sum(-1)


class SACQNetwork(nn.Module):
    """Q(s, a) → scalar."""

    def __init__(self, cfg: SACContinuousConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(cfg.state_dim + cfg.action_dim, 1, cfg.hidden_sizes, layer_norm=cfg.layer_norm)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class SACAgent:
    """SAC double-Q + auto-α + EMA target nets, on CPU by default.

    Stateless w.r.t. replay — the training loop supplies minibatches
    to :meth:`update` and is responsible for replay buffer management.
    """

    def __init__(self, cfg: SACContinuousConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # Running state normalizer — same role as in PPOAgent. Raw Fetch
        # states (positions ~1.3 m) are hard to learn from unnormalized,
        # even with HER providing reward signal. Updated only during
        # rollout collection (act with update_rms=True).
        self.obs_rms = RunningMeanStd(cfg.state_dim, self.device)
        self.actor = SACActor(cfg).to(self.device)
        self.q1 = SACQNetwork(cfg).to(self.device)
        self.q2 = SACQNetwork(cfg).to(self.device)
        self.target_q1 = SACQNetwork(cfg).to(self.device)
        self.target_q2 = SACQNetwork(cfg).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        for p in self.target_q1.parameters():
            p.requires_grad = False
        for p in self.target_q2.parameters():
            p.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.learning_rate
        )
        self.log_alpha = torch.tensor(cfg.init_log_alpha, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.learning_rate)
        self.target_entropy = (
            -float(cfg.action_dim) if cfg.target_entropy is None else float(cfg.target_entropy)
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(
        self, state: np.ndarray, *, deterministic: bool = False, update_rms: bool = False
    ) -> np.ndarray:
        """One-shot action selection. ``state`` shape ``(state_dim,)``.

        ``update_rms`` folds this state into the running normalizer
        (set during rollout collection; left False for evaluation).
        """
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

    @torch.no_grad()
    def action_log_prob(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Per-sample log π(a|s) for executed actions (obs-normalized).

        ``states`` (N, state_dim), ``actions`` (N, action_dim) → (N,) log-probs.
        Used to build the IS ratio π_new/π_old for the AC IS-advantage utility.
        """
        s = self.obs_rms.normalize(torch.as_tensor(states, dtype=torch.float32, device=self.device))
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        return self.actor.log_prob_of_action(s, a).cpu().numpy().astype(np.float64)

    def _pcgrad_optimizer_step(self, optimizer, params, per_group_losses) -> None:
        """One optimizer step using PCGrad-projected per-group gradients."""
        grads = []
        for loss in per_group_losses:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads.append(
                [
                    (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                    for p in params
                ]
            )
        proj = _pcgrad_project(grads)
        optimizer.zero_grad()
        for p, g in zip(params, proj, strict=True):
            p.grad = g
        optimizer.step()

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        *,
        demo_states: np.ndarray | None = None,
        demo_actions: np.ndarray | None = None,
        bc_lambda: float = 0.0,
        use_q_filter: bool = True,
        group_ids: np.ndarray | None = None,
    ) -> dict[str, float]:
        """One SAC gradient step over the supplied minibatch.

        Args:
            states: shape ``(B, state_dim)``.
            actions: shape ``(B, action_dim)``.
            rewards: shape ``(B,)``.
            next_states: shape ``(B, state_dim)``.
            dones: shape ``(B,)`` 0/1.
            demo_states: optional demo states ``(D, state_dim)`` for the
                behavior-cloning auxiliary loss (Nair et al. 2018). Pre-
                normalization is done internally with the agent's obs_rms,
                exactly like ``states`` (callers pass the raw
                ``[observation, desired_goal]`` demo state).
            demo_actions: optional demo actions ``(D, action_dim)``.
            bc_lambda: weight of the BC auxiliary loss added to the SAC
                actor loss. ``0.0`` (default) makes the actor update bit-
                for-bit identical to vanilla SAC — the BC branch is skipped
                entirely so the default training path is unchanged.
            use_q_filter: if True, apply the Nair 2018 Q-filter — only
                demo samples where ``Q(s, a_demo) > Q(s, π(s))`` contribute
                to the BC loss (clone the demo only where it beats the
                current policy under the critic).

        Returns:
            dict of finite scalar metrics: ``q_loss``, ``actor_loss``,
            ``alpha_loss``, ``alpha``, ``q1_mean``, ``q2_mean``. When the BC
            branch is active it also returns ``bc_loss`` and ``bc_frac``
            (fraction of demo samples passing the Q-filter).
        """
        # Normalize with the current running stats (updated during
        # rollout collection); states / next_states share the same
        # frozen stats so the Bellman target stays consistent.
        s = self.obs_rms.normalize(torch.as_tensor(states, dtype=torch.float32, device=self.device))
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        s_next = self.obs_rms.normalize(
            torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        )
        d = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # PCGrad group masks (gradient surgery across context groups). Off unless cfg.pcgrad and
        # group_ids supplied with >1 distinct group present in this minibatch.
        pc_masks: list[torch.Tensor] = []
        if self.cfg.pcgrad and group_ids is not None:
            gid = torch.as_tensor(group_ids, device=self.device).reshape(-1)
            pc_masks = [(gid == g) for g in torch.unique(gid) if int((gid == g).sum()) > 0]
        do_pcgrad = len(pc_masks) > 1

        # Critic update.
        with torch.no_grad():
            a_next, logp_next, _ = self.actor.sample(s_next)
            target_q1 = self.target_q1(s_next, a_next)
            target_q2 = self.target_q2(s_next, a_next)
            target_q = torch.min(target_q1, target_q2) - self.alpha * logp_next
            y = r + (1.0 - d) * self.cfg.gamma * target_q
        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        q_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
        if do_pcgrad:
            q_params = list(self.q1.parameters()) + list(self.q2.parameters())
            per_group = [
                F.mse_loss(q1_pred[m], y[m]) + F.mse_loss(q2_pred[m], y[m]) for m in pc_masks
            ]
            self._pcgrad_optimizer_step(self.q_optimizer, q_params, per_group)
        else:
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

        # Actor update.
        a_pi, logp_pi, _ = self.actor.sample(s)
        q1_pi = self.q1(s, a_pi)
        q2_pi = self.q2(s, a_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        sac_actor_loss = (self.alpha.detach() * logp_pi - min_q_pi).mean()

        # Behavior-cloning auxiliary loss with Q-filter (Nair et al. 2018).
        # L_actor = L_SAC + λ_bc · Σ_demo ||π(s) − a_demo||² · 1[Q(s, a_demo) > Q(s, π(s))].
        # bc_lambda == 0 (or no demos) → skip entirely; the default training
        # path is then bit-for-bit vanilla SAC.
        bc_loss_val = 0.0
        bc_frac = 0.0
        actor_loss = sac_actor_loss
        if bc_lambda > 0.0 and demo_states is not None and demo_actions is not None:
            ds = self.obs_rms.normalize(
                torch.as_tensor(demo_states, dtype=torch.float32, device=self.device)
            )
            da = torch.as_tensor(demo_actions, dtype=torch.float32, device=self.device)
            # Mean policy action (deterministic, env-space) at the demo states.
            mean_d, _ = self.actor(ds)
            pi_d = torch.tanh(mean_d) * self.actor.action_scale + self.actor.action_bias
            per_sample = ((pi_d - da) ** 2).sum(dim=-1)  # ||π(s) − a_demo||²
            if use_q_filter:
                with torch.no_grad():
                    q_demo = torch.min(self.q1(ds, da), self.q2(ds, da))
                    q_pi_d = torch.min(self.q1(ds, pi_d), self.q2(ds, pi_d))
                    mask = (q_demo > q_pi_d).float()
                bc_loss = (per_sample * mask).sum() / da.shape[0]
                bc_frac = float(mask.mean().item())
            else:
                bc_loss = per_sample.mean()
                bc_frac = 1.0
            actor_loss = sac_actor_loss + bc_lambda * bc_loss
            bc_loss_val = float(bc_loss.item())

        if do_pcgrad and bc_lambda <= 0.0:
            actor_params = list(self.actor.parameters())
            per_group_a = [
                (self.alpha.detach() * logp_pi[m] - min_q_pi[m]).mean() for m in pc_masks
            ]
            self._pcgrad_optimizer_step(self.actor_optimizer, actor_params, per_group_a)
        else:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Temperature update.
        alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # Floor alpha so exploration never fully dies on hard sparse tasks.
        if self.cfg.alpha_min > 0.0:
            with torch.no_grad():
                self.log_alpha.clamp_(min=float(np.log(self.cfg.alpha_min)))

        # Soft target update.
        with torch.no_grad():
            for src, dst in (
                (self.q1, self.target_q1),
                (self.q2, self.target_q2),
            ):
                for p_src, p_dst in zip(src.parameters(), dst.parameters(), strict=True):
                    p_dst.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p_src.data)

        metrics = {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q1_mean": float(q1_pred.mean().item()),
            "q2_mean": float(q2_pred.mean().item()),
        }
        if bc_lambda > 0.0 and demo_states is not None and demo_actions is not None:
            metrics["bc_loss"] = bc_loss_val
            metrics["bc_frac"] = bc_frac
        return metrics

    def save(self, path: Path) -> Path:
        return atomic_write_torch(
            Path(path),
            {
                "config": self.cfg,
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "target_q1": self.target_q1.state_dict(),
                "target_q2": self.target_q2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "obs_rms": self.obs_rms.state_dict(),
            },
        )

    def load(self, path: Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        self.q1.load_state_dict(payload["q1"])
        self.q2.load_state_dict(payload["q2"])
        self.target_q1.load_state_dict(payload["target_q1"])
        self.target_q2.load_state_dict(payload["target_q2"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.q_optimizer.load_state_dict(payload["q_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        if "obs_rms" in payload:  # back-compat with pre-normalization checkpoints
            self.obs_rms.load_state_dict(payload["obs_rms"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
