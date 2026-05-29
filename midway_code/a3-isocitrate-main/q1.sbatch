#!/bin/bash
#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --job-name=macs30123_a3_q1
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load python
source activate a3_env

python3 q1_figs.py
