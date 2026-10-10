"""Discrete PPO loop for MiniGrid with PCRC scaffold control.

On-policy PPO over a flat MiniGrid observation, integrated with the (loop-agnostic) ScaffoldController:
each iteration asks the controller for context-scaffold cells, rolls one episode per cell from the
scaffolded start, reports binary success back, then does a clipped PPO update. Evaluation is always on
held-out contexts at level 0 (the true start). Success-only: no TD/uncertainty signal is needed
(td_errors passed as zeros), matching the per_context_success controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from pcrc.registry import SamplerName


@dataclass
class MiniGridPPOConfig:
    n_iterations: int = 600
    episodes_per_iter: int = 16
    max_steps: int = 160
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 512
    lr: float = 2.5e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    hidden: tuple[int, ...] = (128, 128)
    encoder: str = "cnn"  # "cnn" (over the 7x7 image) or "mlp" (flat obs)
    eval_interval: int = 20
    n_eval_episodes: int | None = None  # None = all held-out contexts
    eval_start_progress: float = 0.0  # stage fraction for held-out eval (0 = true start)
    seed: int = 0
    device: str = "cpu"
    # Stage A (self-generated scaffold): after training, harvest the policy's own first ℓ=0-success
    # rollout per train context (serialized states) for the trajectory extractor. None = no harvest.
    harvest_path: str | None = None
    harvest_k: int = 5  # stochastic rollouts per context; keep the shortest success (capture rule)
    harvest_seed: int = 99  # dedicated harvest seed, disjoint from eval seeds
    # Online self-generation (Pillar 2): refresh the scaffold AXIS from the policy's own successes DURING
    # training instead of extract-once. Bootstraps from the initial trajectory ladder (trajectory_ condition +
    # --ladder-path); every ``self_refresh_interval`` iters we re-harvest ℓ=0 successes and OVERWRITE each
    # solved context's axis, so it migrates from the bootstrap source to policy-generated ("self-shaping").
    self_online: bool = False
    self_refresh_interval: int = 200
    self_refresh_k: int = 3


class ActorCritic(nn.Module):
    """MLP actor (categorical logits) + value critic over the flat MiniGrid obs."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.body = nn.Sequential(*layers)
        self.pi = nn.Linear(d, n_actions)
        self.v = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.body(x)
        return self.pi(z), self.v(z).squeeze(-1)


class CNNActorCritic(nn.Module):
    """Small CNN over the 7x7x3 MiniGrid image + extra features (direction, carrying).

    The flat obs packs ``image.flatten()`` (H*W*C) then the extras; this reshapes the image block and
    convolves it, which generalizes across layouts far better than an MLP over flattened pixels.
    """

    def __init__(
        self,
        img_shape: tuple[int, int, int],
        extra_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (128,),
    ):
        super().__init__()
        h, w, c = img_shape
        self.h, self.w, self.c, self.img_n = h, w, c, h * w * c
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, 2), nn.ReLU(), nn.Conv2d(16, 32, 2), nn.ReLU(), nn.Flatten()
        )
        conv_out = 32 * (h - 2) * (w - 2)
        layers: list[nn.Module] = []
        d = conv_out + extra_dim
        for hid in hidden:
            layers += [nn.Linear(d, hid), nn.Tanh()]
            d = hid
        self.body = nn.Sequential(*layers)
        self.pi = nn.Linear(d, n_actions)
        self.v = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img = x[:, : self.img_n].reshape(-1, self.h, self.w, self.c).permute(0, 3, 1, 2)
        extra = x[:, self.img_n :]
        z = torch.cat([self.conv(img), extra], dim=1)
        z = self.body(z)
        return self.pi(z), self.v(z).squeeze(-1)


def _append(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _evaluate(
    ac,
    env,
    held_ids: list[int],
    max_steps: int,
    device: str,
    n: int | None,
    start_progress: float = 0.0,
) -> tuple[float, dict[int, int]]:
    ids = held_ids if n is None else held_ids[:n]
    per_ctx: dict[int, int] = {}
    for cid in ids:
        o = env.reset(int(cid), start_progress=start_progress)
        per_ctx[int(cid)] = 0
        for _ in range(max_steps):
            with torch.no_grad():
                logits, _ = ac(torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0))
            o, _r, term, trunc, info = env.step(int(torch.argmax(logits, dim=-1).item()))
            if term or trunc:
                per_ctx[int(cid)] = int(bool(info.get("is_success", False)))
                break
    return sum(per_ctx.values()) / max(1, len(ids)), per_ctx


def _harvest_ell0(ac, env, train_ids, max_steps, device, k, seed):
    """Stage A harvest: roll the trained policy STOCHASTICALLY at ℓ=0 (true start) on each train context
    up to ``k`` times; keep the *shortest* successful rollout's serialized states (capture rule). The
    goal-reaching step is dropped so the most-assisted frame sits adjacent to the goal (mirrors the BFS
    builder). Returns ``{cid: [[state, progress], ...]}`` for solved contexts only. Progress = step index.
    """
    from pcrc.envs.minigrid_env import GridStateCodec  # noqa: PLC0415

    codec = GridStateCodec()
    torch.manual_seed(seed)
    frames_by_ctx: dict[int, list] = {}
    solved = 0
    for cid in train_ids:
        best = None
        for _ in range(int(k)):
            o = env.reset(int(cid), start_progress=0.0)
            frames = [[codec.serialize(env._u), 0.0]]
            success = False
            for step in range(1, int(max_steps) + 1):
                with torch.no_grad():
                    logits, _ = ac(
                        torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    )
                    a = int(torch.distributions.Categorical(logits=logits).sample().item())
                o, _r, term, trunc, info = env.step(a)
                if term and bool(info.get("is_success", False)):
                    success = True  # drop this goal-reaching frame
                    break
                if term or trunc:
                    break
                frames.append([codec.serialize(env._u), float(step)])
            if success and len(frames) >= 2 and (best is None or len(frames) < len(best)):
                best = frames
        if best is not None:
            frames_by_ctx[int(cid)] = best
            solved += 1
    return frames_by_ctx


def _online_self_refresh(ac, env, train_ids, max_steps, device, k, seed):
    """Online self-generation: re-harvest the policy's ℓ=0 successes and OVERWRITE each solved context's
    axis in the env's ladder *in place*, so the scaffold axis self-shapes during training. Returns the set
    of contexts whose axis is now policy-generated (harvested this call)."""
    from pcrc.envs.trajectory_scaffold import (  # noqa: PLC0415
        ReferenceTrajectoryScaffoldBuilder,
        TrajectoryFrame,
    )

    ladder = getattr(env, "_ladder", None)
    if ladder is None:
        return set()
    builder = ReferenceTrajectoryScaffoldBuilder(ladder.quantiles)
    frames_by_ctx = _harvest_ell0(ac, env, train_ids, max_steps, device, k, seed)
    for cid, frames in frames_by_ctx.items():
        traj = [TrajectoryFrame(s, float(p)) for s, p in frames]
        ladder.add_context(builder.build_context(int(cid), traj))
    return {int(c) for c in frames_by_ctx}


def run_minigrid_ppo_loop(
    env, eval_env, bank, controller, cfg: MiniGridPPOConfig, out_dir: str | Path
) -> dict[str, Any]:
    """Train; return a summary. ``controller`` None means true-start (always level 0)."""
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    if cfg.encoder == "cnn":
        ac = CNNActorCritic(env.img_shape, env.extra_dim, env.n_actions, cfg.hidden).to(dev)
    else:
        ac = ActorCritic(env.obs_dim, env.n_actions, cfg.hidden).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    train_ids = bank.train_ids()
    held = bank.held_out_ids()
    eval_path = Path(out_dir) / "eval" / SamplerName.UNIFORM / "held_out.jsonl"
    scaf_path = Path(out_dir) / "training" / SamplerName.UNIFORM / "scaffold.jsonl"
    # Race-safe recursive mkdir: concurrent array tasks (esp. the CPU-partition DoorKey screen, where all
    # tasks of a seed start at once) share parent dirs, and Python <3.12 `mkdir(parents=True, exist_ok=True)`
    # is not atomic — a losing racer can raise EITHER FileExistsError (a parent another task just created)
    # or FileNotFoundError (a parent mid-creation) during the recursive walk. Retry until the dir exists.
    for _p in (eval_path.parent, scaf_path.parent):
        for _attempt in range(8):
            try:
                _p.mkdir(parents=True, exist_ok=True)
            except (FileExistsError, FileNotFoundError):
                pass  # transient race on a shared parent; re-check and retry
            if _p.is_dir():
                break
        else:
            _p.mkdir(parents=True, exist_ok=True)  # retries exhausted → surface the real error
    total_steps = 0
    self_sourced: set[int] = set()  # contexts whose axis is now policy-generated (online self-gen)
    self_log_path = Path(out_dir) / "training" / SamplerName.UNIFORM / "self_online.jsonl"

    for it in range(cfg.n_iterations):
        if cfg.self_online and it > 0 and it % cfg.self_refresh_interval == 0:
            self_sourced |= _online_self_refresh(
                ac, env, train_ids, cfg.max_steps, dev, cfg.self_refresh_k, cfg.harvest_seed + it
            )
            with open(self_log_path, "a") as _f:
                _f.write(json.dumps({
                    "step": it, "n_self_sourced": len(self_sourced),
                    "self_sourced_frac": len(self_sourced) / max(1, len(train_ids)),
                }) + "\n")
        if controller is not None:
            ctxs, levels, fracs = controller.select(cfg.episodes_per_iter, step=it, rng=rng)
        else:
            ctxs = [int(rng.choice(train_ids)) for _ in range(cfg.episodes_per_iter)]
            levels = [0] * cfg.episodes_per_iter
            fracs = [0.0] * cfg.episodes_per_iter

        b_obs, b_act, b_logp, b_ret, b_adv = [], [], [], [], []
        successes: list[bool] = []
        for ctx, frac in zip(ctxs, fracs):
            o = env.reset(int(ctx), start_progress=float(frac))
            o_l, a_l, lp_l, v_l, r_l = [], [], [], [], []
            terminal = False
            for _ in range(cfg.max_steps):
                ot = torch.as_tensor(o, dtype=torch.float32, device=dev).unsqueeze(0)
                with torch.no_grad():
                    logits, val = ac(ot)
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    lp = dist.log_prob(a)
                no, r, term, trunc, info = env.step(int(a.item()))
                o_l.append(o)
                a_l.append(int(a.item()))
                lp_l.append(float(lp.item()))
                v_l.append(float(val.item()))
                r_l.append(float(r))
                o = no
                total_steps += 1
                if term or trunc:
                    terminal = bool(term)
                    successes.append(bool(info.get("is_success", False)))
                    break
            else:
                successes.append(False)
            # bootstrap value (0 if true terminal, else value of the last obs)
            last_v = 0.0
            if not terminal:
                with torch.no_grad():
                    _, bv = ac(torch.as_tensor(o, dtype=torch.float32, device=dev).unsqueeze(0))
                    last_v = float(bv.item())
            n = len(r_l)
            adv = np.zeros(n, dtype=np.float32)
            lastgae = 0.0
            for i in reversed(range(n)):
                nextv = last_v if i == n - 1 else v_l[i + 1]
                nonterm = 0.0 if (i == n - 1 and terminal) else 1.0
                delta = r_l[i] + cfg.gamma * nextv * nonterm - v_l[i]
                lastgae = delta + cfg.gamma * cfg.gae_lambda * nonterm * lastgae
                adv[i] = lastgae
            ret = adv + np.asarray(v_l, dtype=np.float32)
            b_obs += o_l
            b_act += a_l
            b_logp += lp_l
            b_ret += ret.tolist()
            b_adv += adv.tolist()

        if controller is not None:
            controller.update(
                [int(c) for c in ctxs],
                [int(l) for l in levels],
                successes,
                [0.0] * len(successes),
                it,
            )
            _append(scaf_path, {**controller.snapshot(), "step": it, "env_steps": total_steps})

        # PPO update
        obs_t = torch.as_tensor(np.asarray(b_obs, dtype=np.float32), device=dev)
        act_t = torch.as_tensor(b_act, dtype=torch.int64, device=dev)
        oldlp = torch.as_tensor(b_logp, dtype=torch.float32, device=dev)
        ret_t = torch.as_tensor(b_ret, dtype=torch.float32, device=dev)
        adv_t = torch.as_tensor(b_adv, dtype=torch.float32, device=dev)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        N = obs_t.shape[0]
        idx = np.arange(N)
        for _ in range(cfg.epochs):
            rng.shuffle(idx)
            for s in range(0, N, cfg.minibatch):
                mb = idx[s : s + cfg.minibatch]
                logits, v = ac(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(act_t[mb])
                ratio = torch.exp(lp - oldlp[mb])
                a1 = ratio * adv_t[mb]
                a2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_t[mb]
                loss = (
                    -torch.min(a1, a2).mean()
                    + cfg.vf_coef * ((v - ret_t[mb]) ** 2).mean()
                    - cfg.ent_coef * dist.entropy().mean()
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()

        if it % cfg.eval_interval == 0 or it == cfg.n_iterations - 1:
            sr, per_ctx = _evaluate(
                ac, eval_env, held, cfg.max_steps, dev, cfg.n_eval_episodes, cfg.eval_start_progress
            )
            _append(
                eval_path,
                {"step": it, "env_steps": total_steps, "success_rate": sr, "per_context": per_ctx},
            )

    summary = {"total_steps": total_steps, "n_iterations": cfg.n_iterations}
    # Final policy checkpoint (parity with the Fetch/SAC loop's checkpoint_final.pt): the eval
    # and scaffold logs carry every paper number, but the policy is needed for post-hoc rollouts,
    # videos, and re-evaluation.
    torch.save(
        {"actor_critic": ac.state_dict(), "cfg": {"encoder": cfg.encoder, "hidden": cfg.hidden},
         "n_iterations": cfg.n_iterations, "seed": cfg.seed},
        Path(out_dir) / "checkpoint_final.pt",
    )
    if cfg.harvest_path:
        frames = _harvest_ell0(
            ac, eval_env, train_ids, cfg.max_steps, dev, cfg.harvest_k, cfg.harvest_seed
        )
        out = {
            "env_id": bank.env_id,
            "source": "policy_rollout",
            "n_train": len(train_ids),
            "n_solved": len(frames),
            "harvest_k": cfg.harvest_k,
            "harvest_seed": cfg.harvest_seed,
            "frames": {str(c): f for c, f in frames.items()},
        }
        hp = Path(cfg.harvest_path)
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(json.dumps(out))
        print(
            f"[minigrid harvest] ℓ=0 self-success on {len(frames)}/{len(train_ids)} train contexts "
            f"(k={cfg.harvest_k}) -> {hp}",
            flush=True,
        )
        summary["harvest_solved"] = len(frames)
        summary["harvest_n_train"] = len(train_ids)
    return summary
