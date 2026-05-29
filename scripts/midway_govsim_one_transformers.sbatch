#!/bin/bash
#SBATCH --account=ssd-stu
#SBATCH --qos=ssd-stu
#SBATCH --partition=ssd-gpu
#SBATCH --job-name=govsim_one_tf
#SBATCH --output=logs/one-tf-%j.out
#SBATCH --error=logs/one-tf-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs manifests analysis_outputs

module load python
module load cuda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${GOVSIM_CONDA_ENV:-GovComVLLMv2}"

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/generate_manifest.py \
    --preset quick \
    --backend transformers \
    --output manifests/govsim_quick_transformers.csv

python scripts/run_manifest.py \
    --manifest manifests/govsim_quick_transformers.csv \
    --index "${GOVSIM_TEST_INDEX:-0}" \
    --force
