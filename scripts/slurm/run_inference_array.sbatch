#!/bin/bash
#SBATCH --job-name=gsrd_inf
#SBATCH --output=logs/gsrd_inf_%A_%a.out
#SBATCH --error=logs/gsrd_inf_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-63

set -euo pipefail

CONFIG=${1:-configs/engaging_cluster.yaml}
DATASETS=${2:-coco_val2017}
DETECTOR=${3:-grounding_dino}

module load anaconda/2023a || true
source ~/.bashrc || true
conda activate gsrd || true

export SLURM_ARRAY_TASK_COUNT=64

gsrd inference run \
  --config "$CONFIG" \
  --dataset "$DATASETS" \
  --detector "$DETECTOR" \
  --set inference.num_shards=64 \
  --set inference.split_from_env=true
