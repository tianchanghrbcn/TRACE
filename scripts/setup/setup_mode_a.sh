#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"

if [[ -f "$PROJECT_ROOT/configs/runtime/system.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/configs/runtime/system.local.env"
fi

CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda.sh not found: $CONDA_SH"
  echo "Install Miniconda or set CONDA_DIR."
  exit 1
fi

source "$CONDA_SH"

ENV_FILE="$PROJECT_ROOT/envs/mode_a_trace_runner.yml"
ENV_NAME="$(awk -F': *' '/^name:/{print $2; exit}' "$ENV_FILE")"

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$ENV_NAME"; then
  echo "[TRACE] Updating environment: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "[TRACE] Creating environment: $ENV_NAME"
  conda env create -f "$ENV_FILE"
fi

echo "[TRACE] Mode A environment is ready: $ENV_NAME"