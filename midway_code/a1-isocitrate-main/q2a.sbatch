#!/bin/bash

#SBATCH --account=macs30123
#SBATCH --partition=amd
#SBATCH --job-name=macs30123_q2a
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1          
#SBATCH --ntasks=10

module load python
source activate a1_env

rm -f q2a.out

mpirun -n 10 python3 q2a.py >> q2a.out