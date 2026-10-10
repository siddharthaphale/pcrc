"""Shared utilities for the PCRC package.

Output paths, atomic file writes, the JSONL run logger, and a mixed-precision fp32 zone. Ported
faithfully from the upstream ``probe-curator`` ``acvla.utils`` (those submodules were not carried
over in the ``acvla`` -> ``pcrc`` package rename). See AGENTS.md for the results/checkpoints
location rules these paths honor.
"""
