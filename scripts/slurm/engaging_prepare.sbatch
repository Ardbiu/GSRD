#!/bin/bash
#SBATCH --job-name=gsrd_prep
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

CONFIG=${1:-configs/engaging_cluster.yaml}
DETECTOR=${2:-grounding_dino}
REPO_ROOT=${3:-$PWD}
CONDA_ENV_NAME=${4:-gsrd}

cd "$REPO_ROOT"

module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "[prep] Running data preparation..."
gsrd data prepare --config "$CONFIG"

echo "[prep] Generating vocabularies..."
gsrd vocab generate --config "$CONFIG" --dataset coco_val2017 --dataset bdd100k_val

echo "[prep] Completed."
