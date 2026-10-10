#!/usr/bin/env bash
# Source from a lambda checkout before training or evaluation.
_PCRC_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PCRC_REPO_ROOT="$(cd "$_PCRC_ENV_DIR/.." && pwd)"
export PCRC_WORKSPACE_ROOT="$(dirname "$PCRC_REPO_ROOT")"
export PCRC_SCRATCH_ROOT="$PCRC_REPO_ROOT"
export MUJOCO_GL="egl"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
source "$PCRC_REPO_ROOT/scripts/method_names.sh"
unset _PCRC_ENV_DIR

if [[ -d "$HOME/.local/bin" && ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Optional local configuration; .env is never committed.
if [[ -f "$PCRC_REPO_ROOT/.env" ]]; then
    set -a
    source "$PCRC_REPO_ROOT/.env"
    set +a
fi
