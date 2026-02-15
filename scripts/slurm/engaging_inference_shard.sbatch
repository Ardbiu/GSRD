#!/bin/bash
#SBATCH --job-name=gsrd_inf_h200
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

CONFIG=${1:-configs/engaging_cluster.yaml}
DETECTOR=${2:-grounding_dino}
REPO_ROOT=${3:-$PWD}
CONDA_ENV_NAME=${4:-gsrd}
NUM_SHARDS=${5:-2}

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

echo "[inference] shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on detector=${DETECTOR}"
gsrd inference run \
  --config "$CONFIG" \
  --dataset coco_val2017 \
  --dataset bdd100k_val \
  --detector "$DETECTOR" \
  --set inference.num_shards="$NUM_SHARDS" \
  --set inference.split_from_env=true \
  --set inference.skip_existing=true

echo "[inference] shard ${SLURM_ARRAY_TASK_ID} complete."
