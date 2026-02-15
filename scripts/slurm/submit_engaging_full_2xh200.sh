#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/slurm/submit_engaging_full_2xh200.sh [config] [detector] [conda_env]

CONFIG=${1:-configs/engaging_cluster.yaml}
DETECTOR=${2:-grounding_dino}
CONDA_ENV_NAME=${3:-gsrd}

GPU_PARTITION=${GPU_PARTITION:-mit_normal_gpu}
CPU_PARTITION=${CPU_PARTITION:-mit_normal}
GPU_TYPE=${GPU_TYPE:-h200}
NUM_SHARDS=${NUM_SHARDS:-2}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${REPO_ROOT}/artifacts/slurm_logs}"
mkdir -p "${SLURM_LOG_DIR}"

if [[ "${NUM_SHARDS}" -ne 2 ]]; then
  echo "INFO: NUM_SHARDS is ${NUM_SHARDS}; set NUM_SHARDS=2 to use exactly two H200s." >&2
fi

echo "Submitting GSRD full Engaging pipeline..."
echo "  repo: ${REPO_ROOT}"
echo "  config: ${CONFIG}"
echo "  detector: ${DETECTOR}"
echo "  conda env: ${CONDA_ENV_NAME}"
echo "  gpu partition: ${GPU_PARTITION}"
echo "  cpu partition: ${CPU_PARTITION}"
echo "  gpu type: ${GPU_TYPE}"
echo "  shards: ${NUM_SHARDS}"

PREP_JOB_ID=$(
  sbatch --parsable \
    --partition="${CPU_PARTITION}" \
    --chdir="${REPO_ROOT}" \
    --output="${SLURM_LOG_DIR}/%x_%j.out" \
    --error="${SLURM_LOG_DIR}/%x_%j.err" \
    "${REPO_ROOT}/scripts/slurm/engaging_prepare.sbatch" \
    "${CONFIG}" "${DETECTOR}" "${REPO_ROOT}" "${CONDA_ENV_NAME}"
)

INF_JOB_ID=$(
  sbatch --parsable \
    --dependency="afterok:${PREP_JOB_ID}" \
    --partition="${GPU_PARTITION}" \
    --array="0-$((NUM_SHARDS - 1))" \
    -G "${GPU_TYPE}:1" \
    --chdir="${REPO_ROOT}" \
    --output="${SLURM_LOG_DIR}/%x_%A_%a.out" \
    --error="${SLURM_LOG_DIR}/%x_%A_%a.err" \
    "${REPO_ROOT}/scripts/slurm/engaging_inference_shard.sbatch" \
    "${CONFIG}" "${DETECTOR}" "${REPO_ROOT}" "${CONDA_ENV_NAME}" "${NUM_SHARDS}"
)

POST_JOB_ID=$(
  sbatch --parsable \
    --dependency="afterok:${INF_JOB_ID}" \
    --partition="${CPU_PARTITION}" \
    --chdir="${REPO_ROOT}" \
    --output="${SLURM_LOG_DIR}/%x_%j.out" \
    --error="${SLURM_LOG_DIR}/%x_%j.err" \
    "${REPO_ROOT}/scripts/slurm/engaging_postprocess.sbatch" \
    "${CONFIG}" "${DETECTOR}" "${REPO_ROOT}" "${CONDA_ENV_NAME}" "${NUM_SHARDS}"
)

cat <<EOF
Submitted successfully.
  prep job:      ${PREP_JOB_ID}
  inference job: ${INF_JOB_ID}  (array of ${NUM_SHARDS}, one ${GPU_TYPE} per task)
  post job:      ${POST_JOB_ID}

Monitor:
  squeue -j ${PREP_JOB_ID},${INF_JOB_ID},${POST_JOB_ID}
  tail -f ${SLURM_LOG_DIR}/gsrd_inf_h200_${INF_JOB_ID}_0.out
EOF
