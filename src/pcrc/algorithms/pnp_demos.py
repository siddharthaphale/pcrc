"""Scripted FetchPickAndPlace demonstration generator (Nair 2018 demo source).

Nair et al. 2018 ("Overcoming Exploration in RL with Demonstrations") seeds
the replay buffer + a behavior-cloning auxiliary loss with a small set of
SUCCESSFUL demonstrations. For in-air FetchPickAndPlace — which does not
learn from scratch because of the grasp wall — the cleanest reliable demo
source is a scripted pick-place controller (the reference repos
jangirrishabh/Overcoming-exploration-from-demos and
TianhongDai/hindsight-experience-replay both script their Fetch demos this
way), rather than rolling a partially-trained ``grasp_only`` policy.

The controller is a four-phase finite-state machine over the standard
gymnasium-robotics Fetch observation layout:

    obs[0:3]   gripper (grip-site) position
    obs[3:6]   object position  (== achieved_goal)
    obs[9:11]  the two gripper finger joint positions (width proxy)

Phases: hover above the object (gripper open) -> descend onto it (open) ->
close the gripper for a few steps -> carry the object to the desired goal
(closed). On the calibration bank this reaches 1.00 success across train
contexts, so every collected episode is a genuine success demonstration.

Demos are saved as a flat ``.npz`` of per-transition arrays so they load
cheaply into both the SAC replay buffer (so the critic sees them) and the
BC auxiliary loss (state, action pairs). The saved state is the SAC train
state ``[observation, desired_goal]`` so it can be fed to the actor / Q
networks directly (the loop's ``_concat_state``).

Plan reference: demonstrations + behavior-cloning lever (Nair et al. 2018),
in-air FetchPickAndPlace grasp-wall fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcrc.algorithms.her import GoalConditionedTransition

# Controller gains / thresholds. Action is [dx, dy, dz, gripper] in the env
# action space [-1, 1]; gripper > 0 opens, < 0 closes.
_POS_GAIN = 6.0
_HOVER_DZ = 0.03
_HOVER_XY_TOL = 0.005
_HOVER_Z_TOL = 0.01
_GRASP_TOL = 0.015
_CLOSE_STEPS = 3


class ScriptedPnPController:
    """Stateful four-phase scripted FetchPickAndPlace controller.

    One instance per episode (it carries phase state); call :meth:`act`
    each step with the raw 25-dim Fetch ``observation`` and the episode's
    ``desired_goal``.
    """

    def __init__(self) -> None:
        self._phase = 0
        self._close_count = 0

    def act(self, observation: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        grip = np.asarray(observation[:3], dtype=np.float64)
        obj = np.asarray(observation[3:6], dtype=np.float64)
        goal = np.asarray(desired_goal, dtype=np.float64)
        action = np.zeros(4, dtype=np.float32)
        obj_rel = obj - grip
        dist_xy = float(np.linalg.norm(obj_rel[:2]))

        if self._phase == 0:  # hover above object, gripper open
            target = obj + np.array([0.0, 0.0, _HOVER_DZ])
            action[:3] = (target - grip) * _POS_GAIN
            action[3] = 1.0
            if dist_xy < _HOVER_XY_TOL and abs(grip[2] - (obj[2] + _HOVER_DZ)) < _HOVER_Z_TOL:
                self._phase = 1
        elif self._phase == 1:  # descend onto object, gripper open
            action[:3] = obj_rel * _POS_GAIN
            action[3] = 1.0
            if float(np.linalg.norm(obj_rel)) < _GRASP_TOL:
                self._phase = 2
                self._close_count = 0
        elif self._phase == 2:  # close gripper for a few steps
            action[:3] = obj_rel * _POS_GAIN
            action[3] = -1.0
            self._close_count += 1
            if self._close_count >= _CLOSE_STEPS:
                self._phase = 3
        else:  # carry object to goal, gripper closed
            action[:3] = (goal - obj) * _POS_GAIN
            action[3] = -1.0

        return np.clip(action, -1.0, 1.0).astype(np.float32)


@dataclass
class DemoCollectionResult:
    """Summary of a scripted demo-collection run."""

    n_episodes_attempted: int
    n_episodes_successful: int
    n_transitions: int
    path: Path


def collect_scripted_demos(
    env,
    context_ids: list[int],
    *,
    max_steps: int = 50,
    success_distance_threshold: float = 0.05,
    only_successful: bool = True,
) -> tuple[list[list[GoalConditionedTransition]], DemoCollectionResult]:
    """Roll the scripted controller over ``context_ids`` and return episodes.

    Each episode is a list of :class:`GoalConditionedTransition` (tagged with
    its ``source_context_id``), so the caller can both HER-relabel + push them
    into the replay buffer and extract (state, action) pairs for the BC loss.

    Args:
        env: a ``FetchTaskEnvironment`` bound to FetchPickAndPlace.
        context_ids: contexts to roll (one demo episode each).
        max_steps: per-episode horizon.
        success_distance_threshold: success gate (matches the loop config).
        only_successful: keep only episodes that reached ``is_success``.

    Returns:
        ``(episodes, summary)`` — ``summary.path`` is empty; it is filled in
        by :func:`save_demos`. This function does not write to disk.
    """
    episodes: list[list[GoalConditionedTransition]] = []
    n_success = 0
    for episode_id, cid in enumerate(context_ids):
        obs_dict = env.reset(int(cid))
        controller = ScriptedPnPController()
        transitions: list[GoalConditionedTransition] = []
        success = False
        for _step in range(max_steps):
            obs = obs_dict["observation"]
            goal = obs_dict["desired_goal"]
            action = controller.act(obs, goal)
            next_obs, reward, terminated, truncated, info = env.step(action)
            d = float(np.linalg.norm(next_obs["achieved_goal"] - next_obs["desired_goal"]))
            if bool(info.get("is_success", d < success_distance_threshold)):
                success = True
            transitions.append(
                GoalConditionedTransition(
                    observation=np.asarray(obs, dtype=np.float32),
                    action=np.asarray(action, dtype=np.float32),
                    next_observation=np.asarray(next_obs["observation"], dtype=np.float32),
                    achieved_goal=np.asarray(obs_dict["achieved_goal"], dtype=np.float64),
                    next_achieved_goal=np.asarray(next_obs["achieved_goal"], dtype=np.float64),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    source_context_id=int(cid),
                    original_desired_goal=np.asarray(obs_dict["desired_goal"], dtype=np.float64),
                    training_desired_goal=np.asarray(obs_dict["desired_goal"], dtype=np.float64),
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
        if success or not only_successful:
            episodes.append(transitions)
    result = DemoCollectionResult(
        n_episodes_attempted=len(context_ids),
        n_episodes_successful=n_success,
        n_transitions=int(sum(len(ep) for ep in episodes)),
        path=Path(),
    )
    return episodes, result


def _concat_state(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(obs, dtype=np.float32), np.asarray(goal, dtype=np.float32)])


def save_demos(episodes: list[list[GoalConditionedTransition]], path: Path) -> DemoCollectionResult:
    """Save demo episodes to a flat ``.npz`` of per-transition arrays.

    Stored arrays (all keyed on the ORIGINAL desired goal — these are
    successful demos, so no HER relabel is applied at save time):

        state              (N, state_dim)  = [observation, desired_goal]
        action             (N, action_dim)
        next_state         (N, state_dim)
        reward             (N,)
        terminated         (N,)
        source_context_id  (N,)
        achieved_goal      (N, goal_dim)   = object pos at the transition
        next_achieved_goal (N, goal_dim)   = object pos after the step

    The state composition matches the SAC loop's ``_concat_state`` so the
    arrays drop straight into the actor / Q networks and the BC loss.
    ``achieved_goal`` / ``next_achieved_goal`` are stored EXPLICITLY (rather
    than re-derived from an obs slice) so HER relabeling of demos is correct
    on any env layout — Fetch's object is at obs[3:6] but Meta-World's is at
    obs[4:7]; the loop reads these columns instead of assuming a slice.
    """
    states, actions, next_states, rewards, terminated, ctx = [], [], [], [], [], []
    ach, next_ach = [], []
    for ep in episodes:
        for t in ep:
            states.append(_concat_state(t.observation, t.original_desired_goal))
            next_states.append(_concat_state(t.next_observation, t.original_desired_goal))
            actions.append(np.asarray(t.action, dtype=np.float32))
            rewards.append(float(t.reward))
            terminated.append(1.0 if t.terminated else 0.0)
            ctx.append(int(t.source_context_id))
            ach.append(np.asarray(t.achieved_goal, dtype=np.float32))
            next_ach.append(np.asarray(t.next_achieved_goal, dtype=np.float32))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        next_state=np.asarray(next_states, dtype=np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated, dtype=np.float32),
        source_context_id=np.asarray(ctx, dtype=np.int64),
        achieved_goal=np.asarray(ach, dtype=np.float32),
        next_achieved_goal=np.asarray(next_ach, dtype=np.float32),
    )
    return DemoCollectionResult(
        n_episodes_attempted=len(episodes),
        n_episodes_successful=len(episodes),
        n_transitions=len(states),
        path=path,
    )


def load_demos(path: Path) -> dict[str, np.ndarray]:
    """Load a demos ``.npz`` saved by :func:`save_demos` into a dict of arrays."""
    data = np.load(Path(path))
    return {k: data[k] for k in data.files}
