#!/bin/bash
set -euo pipefail

INDEX="${1:?usage: bash scripts/run_one_3round.sh <manifest_index>}"

cd ~/GovSim_repro
mkdir -p analysis_outputs /scratch/midway3/$USER/govsim_logs /scratch/midway3/$USER/huggingface

module load python
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate GovComVLLMv2

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HOME=/scratch/midway3/$USER/huggingface
export TRANSFORMERS_CACHE=/scratch/midway3/$USER/huggingface/transformers
export HF_HUB_DISABLE_XET=1
export HF_HUB_VERBOSITY=error
export TRANSFORMERS_VERBOSITY=error
export PATHFINDER_DISABLE_TORCH_COMPILE=1

case "$INDEX" in
1)
    EXPERIMENT=fish_free_communication
    GROUP=fishing/free_communication/qwen2_5_7b_3round
    ;;
2)
    EXPERIMENT=fish_costly_punishment
    GROUP=fishing/costly_punishment/qwen2_5_7b_3round
    ;;
*)
    echo "Only indexes 1 and 2 are supported here" >&2
    exit 2
    ;;
esac

python -m simulation.main \
experiment="$EXPERIMENT" \
llm.path=Qwen/Qwen2.5-7B-Instruct \
llm.backend=transformers \
llm.is_api=false \
seed=10000 \
group_name="$GROUP" \
output_run_name=trial_0_seed_10000 \
experiment.env.max_num_rounds=3 \
debug=true