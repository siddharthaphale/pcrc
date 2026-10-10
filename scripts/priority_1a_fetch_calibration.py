"""Priority 1a launcher — Fetch task-bank calibration.

Reads ``config/fetch_priority_1a.yaml`` (or a custom path), runs the
calibration pipeline against the configured Fetch env, and writes the
resulting bank + per-context table + summary under
``<RESULTS_ROOT>/<exp_name>/calibration/``.

Plan reference: ``docs/actor_curator_gymrobotics_implementation_plan.md``
§Priority 1a, §10 Stage A1 mid-run.

Starter policy resolution:
    - ``--starter-checkpoint PATH``: load a PPO warmup checkpoint
      saved by :meth:`pcrc.algorithms.ppo_continuous.PPOAgent.save`
      and roll it greedily for calibration. Greedy decoding strips
      exploration noise so the success-rate measurement is repeatable.
    - ``--use-stub-policy``: use a deterministic zero-action policy.
      Suitable only for Stage A1 wiring smoke; produces nonsense
      starter success rates. The runner writes the result anyway with
      ``"starter": "stub-zero-action"`` recorded in summary.json so
      readers cannot mistake it for a real calibration.

Typical usage on the GPU node (Stage A1 mid-run):

    uv run python scripts/priority_1a_fetch_calibration.py \\
        --config config/fetch_priority_1a.yaml \\
        --exp-name priority_1a_fetch_reach \\
        --starter-checkpoint <CHECKPOINTS_ROOT>/fetch_starter_FetchReachDense-v4.pt

Typical usage during Stage A1 wiring smoke (any host, no GPU):

    uv run python scripts/priority_1a_fetch_calibration.py \\
        --config config/fetch_priority_1a.yaml \\
        --exp-name priority_1a_wiring_smoke \\
        --use-stub-policy \\
        --n-pool 16 --num-train 8 --num-held-out 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

from pcrc.algorithms.ppo_continuous import PPOAgent
from pcrc.envs.fetch_calibration import calibrate_fetch_bank, write_calibration_outputs
from pcrc.utils.paths import RESULTS_ROOT


def _zero_action_stub(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Deterministic zero action — Stage A1 wiring smoke only.

    Returns the appropriate-shape zero action for any Fetch env
    (action_space is always Box(-1, 1, (4,))).
    """
    del obs
    return np.zeros(4, dtype=np.float32)


def _load_starter_policy(checkpoint_path: Path):
    """Load a PPO warmup checkpoint and return a greedy policy callable.

    The checkpoint format matches :meth:`pcrc.algorithms.ppo_continuous.PPOAgent.save`
    — a torch-pickled dict with ``config`` (a
    :class:`PPOContinuousConfig`), plus actor / critic / optimizer
    state dicts. We rehydrate the agent from the embedded config, then
    return a closure that calls :meth:`PPOAgent.act` with
    ``deterministic=True`` so calibration runs against the warmup
    policy's mean action (no exploration noise contaminating the
    success-rate measurement).
    """
    import torch  # noqa: PLC0415 — deferred so --help stays cheap

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"starter checkpoint not found: {checkpoint_path}. Train one via "
            "the PPO warmup before running Priority 1a calibration."
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in payload:
        raise ValueError(
            f"checkpoint at {checkpoint_path} is missing the 'config' key — "
            "it does not look like a PPOAgent.save() artifact."
        )
    agent = PPOAgent(payload["config"])
    agent.load(checkpoint_path)

    def _policy_fn(obs: dict[str, np.ndarray]) -> np.ndarray:
        obs_vec = np.asarray(obs["observation"], dtype=np.float32)
        action, _, _ = agent.act(obs_vec, deterministic=True)
        return action

    return _policy_fn


def _resolve_env_id(cfg: dict[str, Any], cli_env_key: str | None) -> str:
    """Pick the env id from config; CLI override wins."""
    env_ids = cfg.get("env_ids", {})
    key = cli_env_key or cfg.get("default_env")
    if key is None:
        raise KeyError("config has no `default_env` and --env not passed")
    if key not in env_ids:
        raise KeyError(f"env key {key!r} not in config.env_ids (known: {sorted(env_ids)})")
    return env_ids[key]


def main(argv: list[str] | None = None) -> int:
    """Priority 1a entry point. Returns shell-style exit code."""
    parser = argparse.ArgumentParser(prog="priority_1a_fetch_calibration")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/fetch_priority_1a.yaml"),
        help="Path to the priority-1a YAML config.",
    )
    parser.add_argument(
        "--exp-name",
        required=True,
        type=str,
        help="Experiment identifier (results subdir name).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Override config.default_env. Must be a key under config.env_ids.",
    )
    starter = parser.add_mutually_exclusive_group(required=True)
    starter.add_argument(
        "--starter-checkpoint",
        type=Path,
        default=None,
        help="Path to a PPO warmup checkpoint (Stage A2 dependency).",
    )
    starter.add_argument(
        "--use-stub-policy",
        action="store_true",
        help="Use a zero-action stub policy (Stage A1 wiring smoke only).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=f"Override results root (default: {RESULTS_ROOT}).",
    )
    parser.add_argument("--n-pool", type=int, default=None)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-held-out", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--episodes-per-context",
        type=int,
        default=None,
        help="Override calibration.episodes_per_context.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="Override calibration.max_steps_per_episode.",
    )
    parser.add_argument(
        "--dense-fallback-variance-threshold",
        type=float,
        default=None,
        help="Override calibration.dense_fallback_variance_threshold. "
        "Set to 1.0 to force the dense-return / final-goal-distance "
        "binning path — required when a short warmup yields degenerate "
        "(bimodal all-0 / all-1) binary success so the strict band would "
        "drop every context (plan §9 risk #1 step 7).",
    )
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    env_id = _resolve_env_id(cfg, args.env)
    bank_cfg = cfg["bank"]
    cal_cfg = cfg["calibration"]

    n_pool = args.n_pool or (
        bank_cfg["num_train_contexts"]
        + bank_cfg["num_held_out_contexts"]
        + 32  # headroom for saturated / dead drops
    )
    num_train = args.num_train or bank_cfg["num_train_contexts"]
    num_held_out = args.num_held_out or bank_cfg["num_held_out_contexts"]
    seed = args.seed or bank_cfg["seed"]
    episodes_per_context = args.episodes_per_context or cal_cfg["episodes_per_context"]
    max_steps_per_episode = args.max_steps_per_episode or cal_cfg["max_steps_per_episode"]
    dense_fallback_variance_threshold = (
        args.dense_fallback_variance_threshold
        if args.dense_fallback_variance_threshold is not None
        else cal_cfg["dense_fallback_variance_threshold"]
    )

    if args.use_stub_policy:
        print(
            "[priority-1a] WARNING: using zero-action stub policy — Stage A1 "
            "wiring smoke only. Starter success rates are meaningless under "
            "this policy. Use --starter-checkpoint for production calibration.",
            file=sys.stderr,
            flush=True,
        )
        policy_fn = _zero_action_stub
        starter_label = "stub-zero-action"
    else:
        policy_fn = _load_starter_policy(args.starter_checkpoint)
        starter_label = str(args.starter_checkpoint)

    print(
        f"[priority-1a] env_id={env_id} n_pool={n_pool} "
        f"num_train={num_train} num_held_out={num_held_out} "
        f"episodes_per_context={episodes_per_context} "
        f"max_steps_per_episode={max_steps_per_episode}",
        flush=True,
    )

    outcome = calibrate_fetch_bank(
        env_id=env_id,
        n_pool=n_pool,
        num_train=num_train,
        num_held_out=num_held_out,
        policy_fn=policy_fn,
        episodes_per_context=episodes_per_context,
        max_steps_per_episode=max_steps_per_episode,
        distance_bins=bank_cfg["distance_bins"],
        direction_bins=bank_cfg["direction_bins"],
        pose_bins=bank_cfg["pose_bins"],
        target_success_band=tuple(cal_cfg["target_success_band"]),
        dense_fallback_variance_threshold=dense_fallback_variance_threshold,
        pool_base_seed=seed,
        split_seed=seed + 1,
    )

    results_root = args.results_root or RESULTS_ROOT
    out_dir = Path(results_root) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_calibration_outputs(outcome, out_dir)

    # Record metadata alongside the calibration outputs so the
    # provenance of the starter policy is recoverable from disk alone.
    metadata = {
        "config_path": str(args.config),
        "env_id": env_id,
        "starter": starter_label,
        "args": {
            "n_pool": n_pool,
            "num_train": num_train,
            "num_held_out": num_held_out,
            "seed": seed,
            "episodes_per_context": episodes_per_context,
            "max_steps_per_episode": max_steps_per_episode,
        },
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(f"[priority-1a] wrote bank to {paths['bank']}", flush=True)
    print(f"[priority-1a] summary: {outcome.summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
