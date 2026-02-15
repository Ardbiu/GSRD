#!/bin/bash
#SBATCH --job-name=gsrd_post
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

CONFIG=${1:-configs/engaging_cluster.yaml}
DETECTOR=${2:-grounding_dino}
REPO_ROOT=${3:-$PWD}
CONDA_ENV_NAME=${4:-gsrd}
NUM_SHARDS=${5:-2}

cd "$REPO_ROOT"

module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "[post] Running merge/eval/risk/report..."
gsrd run \
  --config "$CONFIG" \
  --stage merge,eval,risk,report \
  --dataset coco_val2017 \
  --dataset bdd100k_val \
  --detector "$DETECTOR" \
  --set inference.num_shards="$NUM_SHARDS"

echo "[post] Completed."
