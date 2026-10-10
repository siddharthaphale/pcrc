"""Atomic file-write helpers — fsync + os.replace pattern in one place.

Used by the run logger (:class:`pcrc.utils.results.RunLogger`), ``write_kill_reason``, and the
bank / checkpoint writers. Crash-safe: readers never see a partial file because the rename is atomic
on POSIX, and the fsync before rename ensures the data is durably on disk before the inode pointer
moves.

Failure mode: every helper raises ``OSError`` on disk / permission failures. Callers wrap with
``contextlib.suppress`` where best-effort semantics are desired.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically.

    Parent dirs are created on demand. Pattern: write to ``path.tmp``,
    fsync the file descriptor, then ``os.replace`` to the final path.
    Crash between fsync and replace leaves a stray ``.tmp`` but no
    partial target file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_json(path: Path, payload: dict[str, Any], *, indent: int = 2) -> Path:
    """JSON-serialize ``payload`` and atomic-write via :func:`atomic_write_bytes`."""
    return atomic_write_bytes(path, json.dumps(payload, indent=indent).encode("utf-8"))


def atomic_write_torch(path: Path, obj: Any) -> Path:
    """``torch.save`` with the same atomicity guarantee as :func:`atomic_write_bytes`.

    Separate from the bytes path because ``torch.save`` writes its own
    pickle-stream binary format directly to the file. We still get
    fsync + atomic-rename around it.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    with tmp.open("rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path
