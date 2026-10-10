"""Priority 2 launcher — Large dense bank, tabular vs MLP curator.

Reuses ``priority_1b_fetch_reach_smoke.main`` under the hood. The only
Priority-2-specific machinery is the pseudo-condition mapping that
translates ``mlp_actor_curator_shuffled`` / ``mlp_actor_curator_random``
into the underlying ``mlp_actor_curator`` condition with the right
feature-perturbation kwarg (the loop's
:func:`build_sampler_for_condition` does not know about the pseudo
condition names).

Plan reference: docs/actor_curator_gymrobotics_implementation_plan.md
§Priority 2, §9 risk #13 (MLP memorization controls).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from pcrc.registry import SamplerName

sys.path.insert(0, str(Path(__file__).resolve().parent))
import priority_1b_fetch_reach_smoke as _1b  # noqa: E402, PLC0415

# Pseudo-condition → (real condition, extra sampler_kwargs key).
_PSEUDO_CONDITION_MAP: dict[str, str] = {
    "mlp_actor_curator_shuffled": SamplerName.MLP_ACTOR_CURATOR,
    "mlp_actor_curator_random": SamplerName.MLP_ACTOR_CURATOR,
}


def _resolve_conditions(raw: str) -> list[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def _patch_config_for_pseudo_conditions(config_path: Path, conditions: list[str]) -> Path:
    """Rewrite the YAML so pseudo conditions are exposed as real ones.

    The config provides ``sampler_kwargs.mlp_actor_curator_shuffled``
    etc. The 1b launcher resolves sampler_kwargs by the literal
    condition name passed via ``--conditions``. We patch the YAML so
    each pseudo condition is also addressable under its real condition
    name, then rewrite ``--conditions`` to use the real names while
    looking up the pseudo entry's kwargs.

    Concretely: we duplicate the pseudo entry under a unique sentinel
    name, write a temporary YAML, and rewrite ``--conditions`` to use
    the sentinel. The 1b launcher then sees three different
    "mlp_actor_curator" runs (one per pseudo), each with different
    feature_perturbation kwargs.

    Implementation: write to a sibling temp file in the same dir so
    relative bank paths resolve.
    """
    cfg = yaml.safe_load(config_path.read_text())
    samplers = cfg.setdefault("sampler_kwargs", {})

    # Build the rewritten condition list while keeping a parallel
    # samplers dict where each entry resolves to the right real
    # sampler kwargs.
    rewritten_conditions: list[str] = []
    for cond in conditions:
        if cond in _PSEUDO_CONDITION_MAP:
            sentinel = cond  # use the pseudo name itself as the sentinel
            kwargs = dict(samplers.get(cond, {}))
            # 1b looks up samplers[condition_passed_to_loop]; tell it
            # the condition passed to the loop is the *real* one and
            # carry the kwargs via a per-condition entry.
            samplers[sentinel] = kwargs
            rewritten_conditions.append(sentinel)
        else:
            rewritten_conditions.append(cond)
    cfg["sampler_kwargs"] = samplers
    out = config_path.with_suffix(".resolved.yaml")
    out.write_text(yaml.safe_dump(cfg))
    return out


def main(argv: list[str] | None = None) -> int:
    """Priority 2 entry point."""
    parser = argparse.ArgumentParser(prog="priority_2_fetch_large_bank")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/fetch_priority_2.yaml"),
    )
    parser.add_argument("--exp-name", required=True, type=str)
    parser.add_argument("--bank-path", type=Path, default=None)
    parser.add_argument(
        "--conditions",
        type=str,
        default=(
            "uniform,tabular_actor_curator,mlp_actor_curator,"
            "mlp_actor_curator_shuffled,mlp_actor_curator_random"
        ),
        help=(
            "Comma-separated list of conditions. Pseudo conditions "
            "mlp_actor_curator_{shuffled,random} resolve to mlp_actor_curator "
            "with the matching feature_perturbation kwarg from the YAML."
        ),
    )
    parser.add_argument("--results-root", type=Path, default=None)
    args = parser.parse_args(argv)
    conditions = _resolve_conditions(args.conditions)
    resolved_cfg = _patch_config_for_pseudo_conditions(args.config, conditions)

    # When we surface a pseudo condition to 1b, the 1b loop config
    # builder receives that pseudo name as ``condition``. The sampler
    # factory then sees an unknown condition and raises. To avoid
    # that, we monkey-patch the factory's mapping at runtime by
    # supplying a sampler_kwargs entry under the pseudo name and a
    # condition-rewrite lookup table.
    #
    # Simpler approach: import the loop factory and extend it inline.
    from pcrc.training import fetch_ppo_loop as _loop  # noqa: PLC0415

    _orig = _loop.build_sampler_for_condition

    def _build_with_pseudo(*, condition: str, bank, epsilon, q_max, sampler_kwargs):  # type: ignore[no-untyped-def]
        if condition in _PSEUDO_CONDITION_MAP:
            real = _PSEUDO_CONDITION_MAP[condition]
            return _orig(
                condition=real,
                bank=bank,
                epsilon=epsilon,
                q_max=q_max,
                sampler_kwargs=sampler_kwargs,
            )
        return _orig(
            condition=condition,
            bank=bank,
            epsilon=epsilon,
            q_max=q_max,
            sampler_kwargs=sampler_kwargs,
        )

    _loop.build_sampler_for_condition = _build_with_pseudo
    try:
        forwarded = [
            "--config",
            str(resolved_cfg),
            "--exp-name",
            args.exp_name,
            "--conditions",
            ",".join(conditions),
        ]
        if args.bank_path is not None:
            forwarded.extend(["--bank-path", str(args.bank_path)])
        if args.results_root is not None:
            forwarded.extend(["--results-root", str(args.results_root)])
        return _1b.main(forwarded)
    finally:
        _loop.build_sampler_for_condition = _orig


if __name__ == "__main__":
    sys.exit(main())
