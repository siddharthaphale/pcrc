"""Collect Meta-World oracle demonstrations (Nair 2018 demo source).

Sparse Meta-World ``push-v3`` is oracle-verified solvable but does NOT learn from scratch under
SAC+HER (the arm starts away from the object -> a reach-then-push wall with no gradient for the
reaching phase; see docs/gymrobotics_results/metaworld_push_gate1_results.md). Nair et al. 2018's
fix is to seed the replay buffer + a BC auxiliary loss with a few SUCCESSFUL demos.

Meta-World ships a scripted oracle (``SawyerPushV3Policy``) that solves push-v3 (verified 10/10
through our wrapper). We roll it over the train contexts of the SAME bank used for training
(``bank_seed`` must match) and save successful episodes to the flat ``.npz`` that the SAC+HER loop
already consumes via ``demos_path`` + ``bc_lambda`` (schema: pcrc.algorithms.pnp_demos.save_demos).

The oracle consumes the FULL goal-observable Meta-World obs (``env._env._get_obs()``), but the
saved demo ``state`` is the SAC train state ``[wrapper observation, desired_goal]`` (goal-stripped
obs + goal), so it drops straight into the actor / Q nets and the BC loss like the Fetch demos.

Usage::

    uv run python scripts/collect_metaworld_demos.py --env-id push-v3 \
        --out results/mw_demos/push_demos.npz --n-episodes 100 --bank-seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from pcrc.algorithms.her import GoalConditionedTransition
from pcrc.algorithms.pnp_demos import save_demos
from pcrc.envs.metaworld_env import MetaWorldTaskBank, MetaWorldTaskEnvironment

# env-id -> oracle policy class (extend as more Meta-World tasks are screened).
_ORACLE = {
    "push-v3": "SawyerPushV3Policy",
    "push-wall-v3": "SawyerPushWallV3Policy",
    "coffee-push-v3": "SawyerCoffeePushV3Policy",
}


def _make_oracle(env_id: str):
    import metaworld.policies as policies  # noqa: PLC0415

    name = _ORACLE.get(env_id)
    if name is None:
        raise ValueError(f"no oracle registered for {env_id!r}; have {sorted(_ORACLE)}")
    return getattr(policies, name)()


def main(argv: list[str] | None = None) -> int:
    """Roll the Meta-World oracle over train contexts and save successful demos."""
    parser = argparse.ArgumentParser(prog="collect_metaworld_demos")
    parser.add_argument("--env-id", type=str, default="push-v3")
    parser.add_argument("--out", type=Path, default=Path("results/mw_demos/push_demos.npz"))
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-held", type=int, default=50)
    parser.add_argument("--bank-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=0,
        help="number of demo episodes (0 = all train contexts).",
    )
    parser.add_argument("--success-distance-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)

    bank = MetaWorldTaskBank.build(
        env_id=args.env_id, n_train=args.n_train, n_held=args.n_held, seed=args.bank_seed
    )
    train_ids = bank.train_ids()
    context_ids = train_ids if args.n_episodes <= 0 else train_ids[: args.n_episodes]

    env = MetaWorldTaskEnvironment(bank, args.env_id, max_steps=args.max_steps)
    oracle = _make_oracle(args.env_id)
    thr = float(args.success_distance_threshold)

    episodes: list[list[GoalConditionedTransition]] = []
    n_success = 0
    try:
        for episode_id, cid in enumerate(context_ids):
            obs_dict = env.reset(int(cid))
            transitions: list[GoalConditionedTransition] = []
            success = False
            for _step in range(args.max_steps):
                obs = obs_dict["observation"]
                goal = obs_dict["desired_goal"]
                full_obs = env._env._get_obs()  # oracle needs the full goal-observable obs
                action = np.asarray(oracle.get_action(full_obs), dtype=np.float32)
                action = np.clip(action, -1.0, 1.0)
                next_obs, reward, terminated, truncated, info = env.step(action)
                d = float(np.linalg.norm(next_obs["achieved_goal"] - next_obs["desired_goal"]))
                if bool(info.get("is_success", d < thr)) or d < thr:
                    success = True
                transitions.append(
                    GoalConditionedTransition(
                        observation=np.asarray(obs, dtype=np.float32),
                        action=action,
                        next_observation=np.asarray(next_obs["observation"], dtype=np.float32),
                        achieved_goal=np.asarray(obs_dict["achieved_goal"], dtype=np.float64),
                        next_achieved_goal=np.asarray(next_obs["achieved_goal"], dtype=np.float64),
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        source_context_id=int(cid),
                        original_desired_goal=np.asarray(goal, dtype=np.float64),
                        training_desired_goal=np.asarray(goal, dtype=np.float64),
                        relabel_strategy="none",
                        relabel_source_step=None,
                        episode_id=int(episode_id),
                    )
                )
                obs_dict = next_obs
                if terminated or truncated:
                    break
            if success:
                n_success += 1
                episodes.append(transitions)  # keep only successful demos (Nair 2018)
    finally:
        env.close()

    saved = save_demos(episodes, args.out)
    print(
        f"[mw-demos] env={args.env_id} attempted={len(context_ids)} successful={n_success} "
        f"saved_episodes={len(episodes)} transitions={saved.n_transitions} -> {saved.path}",
        flush=True,
    )
    if n_success == 0:
        print("[mw-demos] WARNING: zero successful demos collected", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
