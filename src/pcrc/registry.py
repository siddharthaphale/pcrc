"""Canonical project, method, sampler, and publication names for PCRC.

Configs and artifacts store the IDs defined here. Code uses the enums; CLIs and
publication exporters consume the same registry. There are no legacy aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

PROJECT_NAME = "pcrc"
PROJECT_LABEL = "PCRC"
PROJECT_TITLE = "Per-Context Reverse Curricula for Sparse-Reward Reinforcement Learning"
REPO_ROOT_ENV = "PCRC_REPO_ROOT"
SCRATCH_ROOT_ENV = "PCRC_SCRATCH_ROOT"
WORKSPACE_ROOT_ENV = "PCRC_WORKSPACE_ROOT"
FIGURE_TARGET_ENV = "PCRC_FIG_TARGET"
FIGURE_NO_TEX_ENV = "PCRC_FIG_NO_TEX"
FIGURE_FROM_SCRATCH_ENV = "PCRC_FIG_FROM_SCRATCH"
TRAJECTORY_PREFIX = "trajectory_"
CONTINUOUS_TASKS = frozenset({"fetch", "pointmaze", "metaworld"})
ALL_TASKS = CONTINUOUS_TASKS | {"minigrid"}


class Name(str, Enum):
    """String-valued names serialize directly in JSON and configurations."""

    def __str__(self) -> str:
        return self.value


class Method(Name):
    PER_CONTEXT_SUCCESS = "per_context_success"
    PER_CONTEXT_UNCERTAINTY_GATED = "per_context_uncertainty_gated"
    PER_CONTEXT_UNCERTAINTY_PRIORITY = "per_context_uncertainty_priority"
    PER_CONTEXT_UNCERTAINTY_TIEBREAK = "per_context_uncertainty_tiebreak"
    PER_CONTEXT_LEARNING_PROGRESS = "per_context_learning_progress"
    PER_CONTEXT_WILSON_CONFIDENCE = "per_context_wilson_confidence"
    PER_CONTEXT_BETA_POSTERIOR = "per_context_beta_posterior"
    PER_CONTEXT_SUCCESS_TREND = "per_context_success_trend"
    PER_CONTEXT_SUCCESS_DWELL = "per_context_success_dwell"
    PER_GROUP_SUCCESS = "per_group_success"
    UNASSISTED = "unassisted"
    GLOBAL_ANNEAL = "global_anneal"
    GLOBAL_LINEAR_CONTINUOUS = "global_linear_continuous"
    GLOBAL_MIXTURE = "global_mixture"
    RANDOM_SCAFFOLD = "random_scaffold"
    FIXED = "fixed"
    MIXTURE = "mixture"
    ORACLE = "oracle"
    REPLAY_AGGREGATE = "replay_aggregate"
    REPLAY_PERMUTED = "replay_permuted"
    REPLAY_OPENLOOP = "replay_openloop"
    RFCL_STYLE = "rfcl_style"
    RFCL_FAITHFUL = "rfcl_faithful"
    ACED_STYLE = "aced_style"
    PLR_CELLS = "plr_cells"
    ALP_CELLS = "alp_cells"
    OFFICIAL_RFCL = "official_rfcl"
    GLOBAL_ANNEAL_BALANCED_REPLAY = "global_anneal_balanced_replay"


class SamplerName(Name):
    UNIFORM = "uniform"
    HEADROOM = "headroom"
    UNCERTAINTY = "uncertainty"
    HEADROOM_ACTOR = "headroom_actor"
    LEARNING_PROGRESS = "learning_progress"
    REWARD_GREEDY = "reward_greedy"
    PLR = "plr"
    TABULAR_ACTOR_CURATOR = "tabular_actor_curator"
    MLP_ACTOR_CURATOR = "mlp_actor_curator"
    ACDC_LITE = "acdc_lite"


class ScaffoldSource(Name):
    MANUAL = "manual"
    TRAJECTORY = "trajectory"


@dataclass(frozen=True)
class MethodSpec:
    """Presentation and task support for a canonical method."""

    label: str
    description: str
    tasks: frozenset[str] = CONTINUOUS_TASKS


METHODS = {
    Method.PER_CONTEXT_SUCCESS: MethodSpec(
        "Per-context success",
        "Success-driven per-context advance, retreat, and anti-stall.",
        ALL_TASKS,
    ),
    Method.PER_CONTEXT_UNCERTAINTY_GATED: MethodSpec(
        "Uncertainty-gated", "Critic-uncertainty scoring and transition gates."
    ),
    Method.PER_CONTEXT_UNCERTAINTY_PRIORITY: MethodSpec(
        "Uncertainty priority",
        "Uncertainty, progress, and recency scoring without success-headroom scoring.",
    ),
    Method.PER_CONTEXT_UNCERTAINTY_TIEBREAK: MethodSpec(
        "Uncertainty tie-break", "Success transitions with bounded uncertainty tie-breaking."
    ),
    Method.PER_CONTEXT_LEARNING_PROGRESS: MethodSpec(
        "Learning progress", "Absolute learning-progress scoring on the per-context frontier."
    ),
    Method.PER_CONTEXT_WILSON_CONFIDENCE: MethodSpec(
        "Wilson confidence", "Discounted-success Wilson confidence gates.", ALL_TASKS
    ),
    Method.PER_CONTEXT_BETA_POSTERIOR: MethodSpec(
        "Beta posterior", "Discounted Beta-posterior transition gates.", ALL_TASKS
    ),
    Method.PER_CONTEXT_SUCCESS_TREND: MethodSpec(
        "Success trend", "Predictive success-trend transition gates.", ALL_TASKS
    ),
    Method.PER_CONTEXT_SUCCESS_DWELL: MethodSpec(
        "Success with dwell", "Success-driven frontiers with post-solution dwell."
    ),
    Method.PER_GROUP_SUCCESS: MethodSpec(
        "Per-group success",
        "Success pooled across contexts sharing an oracle group.",
        frozenset({"fetch", "metaworld"}),
    ),
    Method.UNASSISTED: MethodSpec(
        "Unassisted", "Uniform contexts at true target starts.", ALL_TASKS
    ),
    Method.GLOBAL_ANNEAL: MethodSpec(
        "Global anneal", "A shared discrete scaffold schedule.", ALL_TASKS
    ),
    Method.GLOBAL_LINEAR_CONTINUOUS: MethodSpec(
        "Global continuous RC", "A shared continuous linear reverse curriculum."
    ),
    Method.GLOBAL_MIXTURE: MethodSpec(
        "Global mixture", "Success-adaptive global level mixture.", ALL_TASKS
    ),
    Method.RANDOM_SCAFFOLD: MethodSpec(
        "Random scaffold", "Uniform random scaffold levels.", ALL_TASKS
    ),
    Method.FIXED: MethodSpec("Fixed scaffold", "A fixed scaffold level.", ALL_TASKS),
    Method.MIXTURE: MethodSpec(
        "Static mixture", "A fixed categorical mixture over scaffold levels."
    ),
    Method.ORACLE: MethodSpec("Oracle scaffold", "A supplied per-context fixed level."),
    Method.REPLAY_AGGREGATE: MethodSpec(
        "Aggregate replay", "Replay the aggregate level histogram.", frozenset({"fetch"})
    ),
    Method.REPLAY_PERMUTED: MethodSpec(
        "Permuted replay", "Replay context-permuted frontier trajectories.", frozenset({"fetch"})
    ),
    Method.REPLAY_OPENLOOP: MethodSpec(
        "Open-loop replay",
        "Replay another seed's per-context frontier trajectory.",
        frozenset({"fetch"}),
    ),
    Method.RFCL_STYLE: MethodSpec(
        "RFCL style", "Local frontiers with uniform sampling and no anti-stall.", ALL_TASKS
    ),
    Method.RFCL_FAITHFUL: MethodSpec(
        "RFCL advance rule", "Windowed RFCL-style advance and struggling-demo sampling."
    ),
    Method.ACED_STYLE: MethodSpec(
        "ACED-style advance", "Success-threshold local advance without retreat."
    ),
    Method.PLR_CELLS: MethodSpec(
        "PLR cells", "TD-error prioritization over all context-level cells."
    ),
    Method.ALP_CELLS: MethodSpec(
        "ALP cells", "Absolute learning-progress prioritization over all cells."
    ),
    Method.OFFICIAL_RFCL: MethodSpec(
        "Official RFCL", "External official RFCL reference.", frozenset()
    ),
    Method.GLOBAL_ANNEAL_BALANCED_REPLAY: MethodSpec(
        "Anneal + balanced replay",
        "Global annealing with group-balanced replay.",
        frozenset({"metaworld"}),
    ),
}
DEFAULT_METHOD = Method.PER_CONTEXT_SUCCESS
ARTIFACT_ONLY_METHODS = {Method.OFFICIAL_RFCL, Method.GLOBAL_ANNEAL_BALANCED_REPLAY}
SCAFFOLD_METHODS = (
    frozenset(Method) - ARTIFACT_ONLY_METHODS - {Method.UNASSISTED, Method.GLOBAL_LINEAR_CONTINUOUS}
)


def base_method(condition: str) -> Method:
    """Identify a canonical condition, including trajectory and numeric variants."""
    value = condition.removeprefix(TRAJECTORY_PREFIX)
    if value.startswith("mixture_"):
        return Method.MIXTURE
    for method in (Method.FIXED, Method.GLOBAL_ANNEAL, Method.PER_CONTEXT_SUCCESS_DWELL):
        suffix = value[len(method) :] if value.startswith(method) else ""
        if suffix.isdigit():
            return method
    return Method(value)


def method_label(condition: str) -> str:
    """Return one publication label for a condition ID."""
    value = str(condition)
    base = value.removeprefix(TRAJECTORY_PREFIX)
    method = base_method(value)
    label = METHODS[method].label
    if method == Method.FIXED and base != method:
        label = f"Fixed level {int(base[len(method) :])}"
    elif method == Method.GLOBAL_ANNEAL and base != method:
        label += f" ({base[len(method) :]})"
    elif method == Method.PER_CONTEXT_SUCCESS_DWELL and base != method:
        label += f" ({base[len(method) :]} windows)"
    if value.startswith(TRAJECTORY_PREFIX):
        label = f"{label} (trajectory)"
    return label


def parse_conditions(value: str, task: str) -> list[str]:
    """Validate user-facing conditions before launching an experiment."""
    if task not in ALL_TASKS:
        raise ValueError(f"Unknown task {task!r}")
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("At least one condition is required")
    for name in names:
        try:
            method = base_method(name)
        except ValueError as exc:
            if (
                name in {s.value for s in SamplerName if s != SamplerName.UNIFORM}
                and task in CONTINUOUS_TASKS
            ):
                continue
            raise ValueError(
                f"Unknown condition {name!r}; use {Method.UNASSISTED} for target starts"
            ) from exc
        if task not in METHODS[method].tasks:
            raise ValueError(f"Condition {name!r} is not supported for {task}")
        if name.startswith(TRAJECTORY_PREFIX) and method in {
            Method.UNASSISTED,
            Method.GLOBAL_LINEAR_CONTINUOUS,
        }:
            raise ValueError(f"Condition {name!r} does not use a scaffold ladder")
        if (
            task == "minigrid"
            and method == Method.FIXED
            and name.removeprefix(TRAJECTORY_PREFIX) == Method.FIXED
        ):
            raise ValueError("MiniGrid fixed scaffold conditions require a level, e.g. fixed2")
    return names


def condition_help(task: str) -> str:
    """CLI help generated from the authoritative registry."""
    return (
        "Comma-separated conditions: "
        + ", ".join(
            "fixed<level>" if m == Method.FIXED and task == "minigrid" else str(m)
            for m, spec in METHODS.items()
            if task in spec.tasks
        )
        + f"; prefix {TRAJECTORY_PREFIX} for trajectory scaffolds"
    )
