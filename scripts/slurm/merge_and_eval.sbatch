#!/bin/bash
#SBATCH --job-name=gsrd_post
#SBATCH --output=logs/gsrd_post_%j.out
#SBATCH --error=logs/gsrd_post_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

CONFIG=${1:-configs/engaging_cluster.yaml}
DATASETS=${2:-coco_val2017}
DETECTOR=${3:-grounding_dino}

module load anaconda/2023a || true
source ~/.bashrc || true
conda activate gsrd || true

gsrd inference merge --config "$CONFIG" --dataset "$DATASETS" --detector "$DETECTOR" --set inference.num_shards=64
gsrd eval run --config "$CONFIG" --dataset "$DATASETS" --detector "$DETECTOR"
gsrd risk run --config "$CONFIG" --dataset "$DATASETS" --detector "$DETECTOR"
gsrd report generate --config "$CONFIG" --dataset "$DATASETS" --detector "$DETECTOR"
