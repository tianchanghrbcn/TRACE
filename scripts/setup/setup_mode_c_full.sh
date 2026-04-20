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

conda_env_exists() {
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$1"
}

echo "[TRACE] Project root: $PROJECT_ROOT"

if [[ "${TRACE_INSTALL_SYSTEM_PACKAGES:-0}" == "1" ]]; then
  echo "[TRACE] Installing system packages. This may require sudo."
  if command -v apt >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y software-properties-common libatlas-base-dev libblas-dev liblapack-dev gfortran curl postgresql postgresql-contrib
  else
    echo "[WARN] Automatic system package installation currently supports apt only."
  fi
else
  echo "[TRACE] Skip system packages. Set TRACE_INSTALL_SYSTEM_PACKAGES=1 to enable."
fi

ENV_FILE="$PROJECT_ROOT/envs/mode_c_pipeline_original.yml"
ENV_NAME="$(awk -F': *' '/^name:/{print $2; exit}' "$ENV_FILE")"
ENV_NAME="${ENV_NAME:-torch110}"

echo "[TRACE] Mode C primary env file: $ENV_FILE"
echo "[TRACE] Mode C primary env name: $ENV_NAME"

TMP_ENV="$(mktemp --suffix=.yml 2>/dev/null || mktemp /tmp/trace_mode_c_env.XXXXXX.yml)"
grep -vE '^\s*prefix:\s*' "$ENV_FILE" > "$TMP_ENV"

if conda_env_exists "$ENV_NAME"; then
  echo "[TRACE] Updating environment: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$TMP_ENV" --prune
else
  echo "[TRACE] Creating environment: $ENV_NAME"
  conda env create -n "$ENV_NAME" -f "$TMP_ENV"
fi

rm -f "$TMP_ENV"

echo "[TRACE] Ensuring auxiliary env: hc37"
if conda_env_exists "hc37"; then
  echo "[TRACE] hc37 already exists."
else
  conda create -y -n hc37 python=3.7
fi

echo "[TRACE] Ensuring auxiliary env: activedetect"
if conda_env_exists "activedetect"; then
  echo "[TRACE] activedetect already exists."
else
  if ! conda create -y -n activedetect python=2.7; then
    echo "[ERROR] Failed to create activedetect with python=2.7."
    echo "Try: conda create -y -n activedetect python=2.7 -c conda-forge"
    exit 1
  fi
fi

if [[ "${TRACE_INSTALL_HOLOCLEAN:-1}" == "1" ]]; then
  HC_DIR="$PROJECT_ROOT/src/cleaning/holoclean-master"
  if [[ -d "$HC_DIR" ]]; then
    echo "[TRACE] Installing HoloClean requirements in hc37."
    conda run -n hc37 python -m pip install -r "$HC_DIR/requirements.txt"
  else
    echo "[WARN] HoloClean directory not found: $HC_DIR"
  fi
fi

if [[ "${TRACE_INSTALL_BOOSTCLEAN:-1}" == "1" ]]; then
  BC_DIR="$PROJECT_ROOT/src/cleaning/BoostClean"
  if [[ -d "$BC_DIR" ]]; then
    echo "[TRACE] Installing BoostClean in activedetect."
    conda run -n activedetect python -m pip install -e "$BC_DIR"
  else
    echo "[WARN] BoostClean directory not found: $BC_DIR"
  fi
fi

echo "[TRACE] Mode C full environment setup completed."
echo "[TRACE] Activate primary env with:"
echo "  conda activate $ENV_NAME"