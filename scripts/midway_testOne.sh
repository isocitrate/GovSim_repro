#!/bin/bash
set -euo pipefail

cd ~/GovSim_repro
mkdir -p logs manifests analysis_outputs

module load python
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate GovComVLLMv2

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

python scripts/generate_manifest.py --preset quick --output manifests/govsim_quick.csv
python scripts/run_manifest.py --manifest manifests/govsim_quick.csv --index 0
EOF
