#!/bin/bash
#SBATCH --job-name=gsrd_post
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

CONFIG=${1:-configs/engaging_low_compute.yaml}
DETECTOR=${2:-grounding_dino}
DATASETS=${3:-coco_val2017}
REPO_ROOT=${4:-$PWD}
CONDA_ENV_NAME=${5:-gsrd}
NUM_SHARDS=${6:-2}

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

echo "[post] Running merge/eval/risk/report..."
python -m gsrd.cli run \
  --config "$CONFIG" \
  --stage merge,eval,risk,report \
  "${DATASET_ARGS[@]}" \
  --detector "$DETECTOR" \
  --set inference.num_shards="$NUM_SHARDS"

echo "[post] Completed."
