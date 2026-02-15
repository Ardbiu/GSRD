#!/bin/bash
#SBATCH --job-name=gsrd_inf_h200
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

CONFIG=${1:-configs/engaging_low_compute.yaml}
DETECTOR=${2:-grounding_dino}
DATASETS=${3:-coco_val2017}
REPO_ROOT=${4:-$PWD}
CONDA_ENV_NAME=${5:-gsrd}
NUM_SHARDS=${6:-2}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: This script must be launched as a Slurm array job." >&2
  exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  export SLURM_ARRAY_TASK_COUNT="$NUM_SHARDS"
fi

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

echo "[inference] shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on detector=${DETECTOR}"
python -m gsrd.cli inference run \
  --config "$CONFIG" \
  "${DATASET_ARGS[@]}" \
  --detector "$DETECTOR" \
  --set inference.num_shards="$NUM_SHARDS" \
  --set inference.split_from_env=true \
  --set inference.skip_existing=true

echo "[inference] shard ${SLURM_ARRAY_TASK_ID} complete."
