"""Portable input and output roots with optional scratch-directory overrides.

The PCRC_* environment variables configure repository and scratch paths for
launchers. Editable checkouts default to the repository root; installed wheels
default to the current working directory.
"""

import os
from pathlib import Path

from pcrc.registry import REPO_ROOT_ENV, SCRATCH_ROOT_ENV

_checkout = Path(__file__).resolve().parents[3]
_default_root = _checkout if (_checkout / "pyproject.toml").is_file() else Path.cwd()
REPO_ROOT = Path(os.environ.get(REPO_ROOT_ENV, str(_default_root))).expanduser()
SCRATCH_ROOT = Path(os.environ.get(SCRATCH_ROOT_ENV, str(REPO_ROOT))).expanduser()

DATA_ROOT = REPO_ROOT / "data"
CHECKPOINTS_ROOT = SCRATCH_ROOT / "checkpoints"
RESULTS_ROOT = SCRATCH_ROOT / "results"
