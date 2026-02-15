#!/bin/bash
#SBATCH --job-name=gsrd_prep
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

CONFIG=${1:-configs/engaging_low_compute.yaml}
DETECTOR=${2:-grounding_dino}
DATASETS=${3:-coco_val2017}
REPO_ROOT=${4:-$PWD}
CONDA_ENV_NAME=${5:-gsrd}

cd "$REPO_ROOT"

ensure_modules() {
  if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/Modules/init/bash ]]; then
      # shellcheck disable=SC1091
      source /usr/share/Modules/init/bash
    fi
  fi
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: 'module' command is unavailable in this batch environment." >&2
    return 2
  fi
}

bootstrap_env() {
  ensure_modules
  unset PYTHONPATH
  unset PYTHONHOME
  export PYTHONNOUSERSITE=1

  if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
    module purge
    module load deprecated-modules
    module load python/3.10.8-x86_64
    source "${VENV_PATH}/bin/activate"
    return 0
  fi

  module purge
  module load deprecated-modules
  module load anaconda3/2022.05-x86_64
  if command -v conda >/dev/null 2>&1; then
    local conda_exe conda_base
    conda_exe="$(command -v conda)"
    conda_base="$(dirname "$(dirname "$(readlink -f "${conda_exe}")")")"
    if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      source "${conda_base}/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_NAME}"
      return 0
    fi
  fi

  echo "ERROR: Failed to activate environment. Set VENV_PATH or fix conda env ${CONDA_ENV_NAME}." >&2
  return 2
}

bootstrap_env

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
DATASET_ARGS=()
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
for ds in "${DATASET_LIST[@]}"; do
  ds_trimmed="$(echo "$ds" | xargs)"
  if [[ -n "$ds_trimmed" ]]; then
    DATASET_ARGS+=(--dataset "$ds_trimmed")
  fi
done

echo "[prep] Running data preparation..."
python -m gsrd.cli data prepare --config "$CONFIG"

echo "[prep] Generating vocabularies..."
python -m gsrd.cli vocab generate --config "$CONFIG" "${DATASET_ARGS[@]}"

echo "[prep] Completed."
