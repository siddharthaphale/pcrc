"""Priority 1c launcher — Dense PPO on FetchPushDense-v4.

Thin variant of ``priority_1b_fetch_reach_smoke.py``. Shares the same
loop builder; only the config + default conditions differ.

Plan reference: docs/actor_curator_gymrobotics_implementation_plan.md
§Priority 1c, §10 Stage D step 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Reuse the Priority 1b machinery via direct script import — both
# launchers are thin wrappers around the same loop driver.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import priority_1b_fetch_reach_smoke as _1b  # noqa: E402, PLC0415


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="priority_1c_fetch_push_dense")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/fetch_priority_1c.yaml"),
    )
    parser.add_argument("--exp-name", required=True, type=str)
    parser.add_argument("--bank-path", type=Path, default=None)
    parser.add_argument(
        "--conditions",
        type=str,
        default=(
            "uniform,reward_greedy,headroom,learning_progress,"
            "tabular_actor_curator,mlp_actor_curator"
        ),
    )
    parser.add_argument("--results-root", type=Path, default=None)
    args = parser.parse_args(argv)
    # Forward to the 1b machinery; the only difference is the default
    # config path and default condition list, both already overridden.
    forwarded = [
        "--config",
        str(args.config),
        "--exp-name",
        args.exp_name,
        "--conditions",
        args.conditions,
    ]
    if args.bank_path is not None:
        forwarded.extend(["--bank-path", str(args.bank_path)])
    if args.results_root is not None:
        forwarded.extend(["--results-root", str(args.results_root)])
    return _1b.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
