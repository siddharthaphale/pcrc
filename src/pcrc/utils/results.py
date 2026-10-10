"""Lightweight result logger for PCRC.

Writes JSON (one-shot, atomic) and JSONL (append-flush, crash-safe) files into
``<root>/<exp_name>/``. Default root is :data:`pcrc.utils.paths.RESULTS_ROOT` if it exists, else
``./results`` locally (so CI / unsourced runs work without the scratch env var set).

Conventions used across the PCRC loops:

- ``log_event("summary.json", {...})``                  -- one-shot run metadata / summaries
- ``log_step("training/<cond>/metrics.jsonl", {...})``  -- per-step training metrics
- ``log_step("training/<cond>/scaffold.jsonl", {...})`` -- per-step frontier snapshots
- ``log_step("eval/<cond>/held_out.jsonl", {...})``     -- held-out eval curve

Every record is stamped with ``_ts`` (wall-clock seconds). Scratch is ephemeral, so the caller
syncs ``<root>`` to durable storage (the GCS bucket) after the run; RunLogger only writes.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import IO, Any

from pcrc.utils.atomic_io import atomic_write_json
from pcrc.utils.paths import RESULTS_ROOT


def _results_root_present() -> bool:
    """Whether ``RESULTS_ROOT`` is reachable from this process.

    ``Path.exists()`` propagates ``PermissionError`` (e.g. a root-owned scratch / container path the
    current user can't stat), so treat any ``OSError`` as "absent → fall back to local ./results".
    """
    try:
        return RESULTS_ROOT.exists()
    except OSError:
        return False


class RunLogger:
    """Append-flush JSON/JSONL writer scoped to one experiment run."""

    def __init__(self, exp_name: str, root: str | None = None):
        """Open the run directory ``<root>/<exp_name>`` (created on demand)."""
        if root is None:
            # Default: RESULTS_ROOT if it's reachable, else local ./results.
            root = str(RESULTS_ROOT) if _results_root_present() else "./results"
        self.dir = Path(root) / exp_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._streams: dict[str, IO[str]] = {}

    def log_event(self, path: str, payload: dict[str, Any]) -> Path:
        """Atomic single-shot JSON write (e.g., gating, eval results).

        Uses ``atomic_write_json`` so a crash mid-write leaves no
        partial file at the target path — the rename happens only
        after fsync. Existing callers relied on the previous
        non-atomic ``open("w")`` semantics for the docstring's promise;
        this just makes the impl match the contract.
        """
        full = self.dir / path
        return atomic_write_json(full, {"_ts": time.time(), **payload})

    def log_step(self, stream: str, payload: dict[str, Any]) -> None:
        """Append one JSONL line to a long-running stream (flushed)."""
        if stream not in self._streams:
            full = self.dir / stream
            full.parent.mkdir(parents=True, exist_ok=True)
            self._streams[stream] = full.open("a")
        self._streams[stream].write(json.dumps({"_ts": time.time(), **payload}) + "\n")
        self._streams[stream].flush()

    def close(self) -> None:
        """Flush and close every open append-flush stream."""
        for f in self._streams.values():
            f.close()
        self._streams.clear()

    def __enter__(self) -> RunLogger:
        """Enter as a context manager; returns ``self`` unchanged."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close all open streams on context exit."""
        self.close()
