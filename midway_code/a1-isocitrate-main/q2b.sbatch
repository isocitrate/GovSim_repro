#!/bin/bash
#SBATCH --account=macs30123
#SBATCH --partition=amd
#SBATCH --job-name=macs30123_q2a
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1          
#SBATCH --ntasks=1

module load python
source activate a1_env


python3 q2b.py