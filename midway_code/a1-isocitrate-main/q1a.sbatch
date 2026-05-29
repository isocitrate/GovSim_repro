#!/bin/bash
#SBATCH --job-name=q1a_serial
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=caslake
#SBATCH --account=macs30123

module load python
source activate a1_env

python3 q1a.py