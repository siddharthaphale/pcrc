"""Priority 4 launcher — ACDC-lite + full proxy + AC comparison.

Thin variant of priority_3_fetch_sparse_her — same SAC+HER driver,
different default config (priority_4 enables all 7 conditions
including ACDC-lite). Importing priority_3.main keeps the SAC+HER
setup code in one place.

Plan reference: docs/actor_curator_gymrobotics_implementation_plan.md
§Priority 4, §10 Stage D step 5b.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import priority_3_fetch_sparse_her as _p3  # noqa: E402, PLC0415


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="priority_4_fetch_acdc_lite")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/fetch_priority_4.yaml"),
    )
    parser.add_argument("--exp-name", required=True, type=str)
    parser.add_argument("--bank-path", type=Path, default=None)
    parser.add_argument(
        "--conditions",
        type=str,
        default=(
            "uniform,reward_greedy,headroom,learning_progress,"
            "acdc_lite,tabular_actor_curator,mlp_actor_curator"
        ),
    )
    parser.add_argument("--results-root", type=Path, default=None)
    args = parser.parse_args(argv)
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
    return _p3.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
