#!/bin/bash
#SBATCH --account=macs30123
#SBATCH --partition=gpu
#SBATCH --job-name=govsim_test
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs manifests analysis_outputs

module load python
module load cuda

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${GOVSIM_CONDA_ENV:-GovComVLLMv2}"
else
    echo "conda is not available on PATH; load the correct Python module/env first" >&2
    exit 1
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-${SLURM_SUBMIT_DIR}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export WANDB_MODE=disabled

python scripts/midway_preflight.py

python scripts/generate_manifest.py \
    --preset smoke \
    --trials 1 \
    --output manifests/govsim_test.csv

python scripts/run_manifest.py \
    --manifest manifests/govsim_test.csv \
    --index "${GOVSIM_TEST_INDEX:-0}" \
    --force
