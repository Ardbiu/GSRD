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

module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

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
