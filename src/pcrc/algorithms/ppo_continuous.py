"""Continuous-action PPO building blocks for pre-VLA Fetch loops.

Adapted from CleanRL ``ppo_continuous_action.py`` (MIT). Original
copyright Costa Huang et al. The single-file CleanRL reference is the
canonical port target; we keep the same algorithmic shape (MLP-Gaussian
actor with state-independent log_std, MLP value critic, GAE,
PPO-clipped surrogate with optional entropy + value-loss coefficients)
but drop the gym/tensorboard scaffolding because the Fetch loops use
gymnasium + :class:`pcrc.utils.RunLogger`.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§2 (``ppo_continuous.py`` row), §5 OSS references (CleanRL port path),
§4 Priority 1b (PPO driver for dense Fetch).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from pcrc.utils.atomic_io import atomic_write_torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PPOContinuousConfig:
    """PPO + MLP-Gaussian actor + MLP value critic hyperparameters."""

    observation_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 3.0e-4
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_epochs: int = 10
    minibatch_size: int = 64
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_std_init: float = 0.0
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Actor / Critic
# ---------------------------------------------------------------------------


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias: float = 0.0) -> nn.Linear:
    """Orthogonal init + constant bias — standard CleanRL convention.

    Stabilizes early training on tiny networks; the small-Fetch
    settings we run here are otherwise sensitive to default init.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def _build_mlp(in_dim: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    """MLP trunk with tanh activations + orthogonal init."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(_layer_init(nn.Linear(prev, h)))
        layers.append(nn.Tanh())
        prev = h
    return nn.Sequential(*layers)


class PPOActor(nn.Module):
    """MLP-Gaussian policy: state-dependent mean, state-independent log_std.

    The state-independent ``log_std`` parameter matches CleanRL and
    SB3 conventions and is what makes continuous-PPO numerically
    well-behaved on tiny networks like the ones we use here.
    """

    def __init__(self, cfg: PPOContinuousConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = _build_mlp(cfg.observation_dim, cfg.hidden_sizes)
        self.mean_head = _layer_init(nn.Linear(cfg.hidden_sizes[-1], cfg.action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), float(cfg.log_std_init)))

    def distribution(self, obs: torch.Tensor) -> Normal:
        """Return the Normal action distribution for a batch of obs."""
        h = self.trunk(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor) -> Normal:
        """Alias for :meth:`distribution` so the module is callable."""
        return self.distribution(obs)

    def act(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or take mean) action; return action, logp, entropy.

        Args:
            obs: shape ``(B, observation_dim)``.
            deterministic: if True, return the distribution mean
                instead of a sample (used for greedy evaluation).

        Returns:
            ``(action, log_prob, entropy)`` each tensors with batch
            dim ``B``. ``log_prob`` and ``entropy`` are summed over
            the action dim so the result is shape ``(B,)``.
        """
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.rsample()
        logp = dist.log_prob(action).sum(-1)
        ent = dist.entropy().sum(-1)
        return action, logp, ent

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-prob of ``action`` under the policy at ``obs``, summed over action dim."""
        return self.distribution(obs).log_prob(action).sum(-1)


class PPOCritic(nn.Module):
    """MLP value critic: ``V(s) ∈ ℝ``."""

    def __init__(self, cfg: PPOContinuousConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = _build_mlp(cfg.observation_dim, cfg.hidden_sizes)
        self.value_head = _layer_init(nn.Linear(cfg.hidden_sizes[-1], 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return ``V(obs)`` with shape ``(B,)`` (squeezed value head output)."""
        return self.value_head(self.trunk(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over a single trajectory.

    Args:
        rewards: shape ``(T,)`` rewards per step.
        values: shape ``(T,)`` value estimates ``V(s_t)``.
        dones: shape ``(T,)`` 0/1 per step. ``1`` means the episode
            terminated AT step t (so the bootstrap target at t+1 is
            zero).
        next_value: bootstrap value ``V(s_T)`` for the step after the
            last one in the trajectory. ``0.0`` if the trajectory
            ends in a terminal state.
        gamma, gae_lambda: GAE hyperparameters.

    Returns:
        ``(advantages, returns)`` each shape ``(T,)``.
            ``returns = advantages + values``.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    T = rewards.shape[0]
    if values.shape != (T,) or dones.shape != (T,):
        raise ValueError(
            f"compute_gae shape mismatch: rewards={rewards.shape} "
            f"values={values.shape} dones={dones.shape}"
        )
    advantages = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_v = next_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_v = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Clipped surrogate
# ---------------------------------------------------------------------------


def ppo_clipped_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO-clipped policy surrogate.

    Args:
        new_logp: shape ``(B,)`` log-prob under the current policy.
        old_logp: shape ``(B,)`` log-prob under the pre-update policy.
            Detached internally.
        advantages: shape ``(B,)`` advantage estimates. Detached.
        clip_eps: clip width.

    Returns:
        ``(policy_loss, clip_fraction)`` where ``policy_loss`` is the
        scalar PPO surrogate (negative of the maximin objective) and
        ``clip_fraction`` is the fraction of samples whose ratio fell
        outside ``[1 - clip_eps, 1 + clip_eps]`` (a diagnostic, not a
        gradient signal).
    """
    old_logp = old_logp.detach()
    advantages = advantages.detach()
    ratio = (new_logp - old_logp).exp()
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    policy_loss = -surrogate.mean()
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
    return policy_loss, clip_fraction


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------


class RunningMeanStd:
    """Welford running mean/variance for observation normalization.

    Standard for MuJoCo / Fetch control: the raw observation carries
    positions (~1.3 m), velocities (~0), and gripper state on very
    different scales, and an unnormalized MLP learns coarse direction
    but never refines to the 0.05 m success band. Mirrors the
    ``RunningMeanStd`` in OpenAI baselines / SB3 ``VecNormalize``.

    Stats live on ``device`` as float64 tensors; :meth:`normalize`
    returns ``clip((x - mean) / sqrt(var + eps), -clip, clip)``.
    """

    def __init__(self, dim: int, device: torch.device, epsilon: float = 1e-8):
        self.mean = torch.zeros(dim, dtype=torch.float64, device=device)
        self.var = torch.ones(dim, dtype=torch.float64, device=device)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x: torch.Tensor) -> None:
        """Update stats from a batch ``x`` of shape ``(N, dim)``."""
        x = x.to(self.mean.dtype)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        """Return the clipped, standardized tensor (same dtype as ``x``)."""
        norm = (x.to(self.mean.dtype) - self.mean) / torch.sqrt(self.var + self.epsilon)
        return norm.clamp(-clip, clip).to(x.dtype)

    def state_dict(self) -> dict[str, Any]:
        """CPU-numpy snapshot for checkpointing."""
        return {
            "mean": self.mean.detach().cpu().numpy(),
            "var": self.var.detach().cpu().numpy(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`state_dict`."""
        self.mean = torch.as_tensor(state["mean"], dtype=torch.float64, device=self.mean.device)
        self.var = torch.as_tensor(state["var"], dtype=torch.float64, device=self.var.device)
        self.count = float(state["count"])


# ---------------------------------------------------------------------------
# Agent (thin wiring + save/load)
# ---------------------------------------------------------------------------


class PPOAgent:
    """Wires actor + critic + Adam optimizers + save/load.

    Stateless w.r.t. rollouts — the training loop is responsible for
    collecting transitions and calling :meth:`update`. The agent
    exposes minimal hooks (:meth:`act`, :meth:`update`, :meth:`save`,
    :meth:`load`) so the Fetch loops can drive it without leaking
    PPO internals.
    """

    def __init__(self, cfg: PPOContinuousConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.actor = PPOActor(cfg).to(self.device)
        self.critic = PPOCritic(cfg).to(self.device)
        # Running observation normalizer — applied in every forward
        # (act / value / evaluate_actions / update) so train, bootstrap,
        # utility, and eval all see identically-normalized inputs. Stats
        # update only during rollout collection (act with update_rms=True).
        self.obs_rms = RunningMeanStd(cfg.observation_dim, self.device)
        # Single Adam with two parameter groups so the loop only ever
        # sees a single optimizer.
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.learning_rate,
        )

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, *, deterministic: bool = False, update_rms: bool = False
    ) -> tuple[np.ndarray, float, float]:
        """Pick an action for one observation.

        Args:
            obs: shape ``(observation_dim,)`` single-step obs.
            deterministic: greedy decoding for evaluation.
            update_rms: if True, fold this obs into the running
                normalizer stats. Set during rollout collection; left
                False for evaluation so eval uses frozen training stats.

        Returns:
            ``(action, log_prob, value)``. ``action`` is shape
            ``(action_dim,)``; ``log_prob`` and ``value`` are floats.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if update_rms:
            self.obs_rms.update(obs_t)
        obs_n = self.obs_rms.normalize(obs_t)
        action, logp, _ = self.actor.act(obs_n, deterministic=deterministic)
        value = self.critic(obs_n)
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(logp.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        """Critic value at ``obs`` (normalized, no rms update). Used for
        GAE bootstrap on a truncated episode's final observation.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(self.obs_rms.normalize(obs_t)).item())

    @torch.no_grad()
    def evaluate_actions(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Recompute log p(a|s) at stored (s, a) pairs under the current
        actor, with the same obs normalization (no rms update). Used by
        the PPO policy-improvement utility.
        """
        obs_t = self.obs_rms.normalize(
            torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        )
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        return self.actor.log_prob(obs_t, act_t).cpu().numpy()

    def update(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        """One PPO update pass over the supplied rollout.

        Runs ``cfg.n_epochs`` epochs over minibatches of
        ``cfg.minibatch_size`` drawn without replacement each epoch.
        Advantages are standardized within the rollout for variance
        reduction.

        Returns:
            dict with averaged ``policy_loss``, ``value_loss``,
            ``entropy``, ``clip_fraction``, ``approx_kl``,
            ``explained_variance``.
        """
        # Normalize with the current rms (already updated during this
        # iteration's rollout). No rms update here, so old_log_probs
        # (collected under the same final stats) stay consistent.
        obs_t = self.obs_rms.normalize(
            torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        )
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        n = obs_t.shape[0]
        mb_size = min(self.cfg.minibatch_size, n)
        idx = np.arange(n)
        metrics_accum = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
        }
        n_minibatches = 0
        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb_size):
                mb = idx[start : start + mb_size]
                mb_obs = obs_t[mb]
                mb_act = act_t[mb]
                mb_old_logp = old_logp_t[mb]
                mb_adv = adv_t[mb]
                mb_ret = ret_t[mb]

                dist = self.actor.distribution(mb_obs)
                new_logp = dist.log_prob(mb_act).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                policy_loss, clip_fraction = ppo_clipped_loss(
                    new_logp, mb_old_logp, mb_adv, self.cfg.clip_eps
                )
                value = self.critic(mb_obs)
                value_loss = 0.5 * (value - mb_ret).pow(2).mean()
                loss = (
                    policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.cfg.max_grad_norm,
                    )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_logp - new_logp).mean()
                metrics_accum["policy_loss"] += float(policy_loss.item())
                metrics_accum["value_loss"] += float(value_loss.item())
                metrics_accum["entropy"] += float(entropy.item())
                metrics_accum["clip_fraction"] += float(clip_fraction.item())
                metrics_accum["approx_kl"] += float(approx_kl.item())
                n_minibatches += 1

        averaged = {k: v / max(n_minibatches, 1) for k, v in metrics_accum.items()}
        # Explained variance computed once over the full rollout.
        with torch.no_grad():
            var_y = float(ret_t.var().item())
            var_pred = float((ret_t - self.critic(obs_t)).var().item())
            averaged["explained_variance"] = float("nan") if var_y == 0 else 1.0 - var_pred / var_y
        return averaged

    def save(self, path: Path) -> Path:
        """Atomic torch.save of actor + critic + optimizer + config."""
        return atomic_write_torch(
            Path(path),
            {
                "config": self.cfg,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_rms": self.obs_rms.state_dict(),
            },
        )

    def load(self, path: Path) -> None:
        """Restore from a :meth:`save` checkpoint."""
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if "obs_rms" in payload:  # back-compat with pre-normalization checkpoints
            self.obs_rms.load_state_dict(payload["obs_rms"])
