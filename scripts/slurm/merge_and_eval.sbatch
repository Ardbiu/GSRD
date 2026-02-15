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

module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gsrd

python -m gsrd.cli inference merge --config "$CONFIG" --dataset "$DATASETS" --detector "$DETECTOR" --set inference.num_shards=64
python -m gsrd.cli eval run --config "$CONFIG" --dataset "$DATASETS" --detector "$DETECTOR"
python -m gsrd.cli risk run --config "$CONFIG" --dataset "$DATASETS" --detector "$DETECTOR"
python -m gsrd.cli report generate --config "$CONFIG" --dataset "$DATASETS" --detector "$DETECTOR"
