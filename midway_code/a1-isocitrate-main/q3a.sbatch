#!/bin/bash

#SBATCH --account=macs30123
#SBATCH --partition=gpu
#SBATCH --job-name=macs30123_q3a
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1          
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G

module load python
module load cuda
source activate a1_env

rm -f q3a.out

python3 q3a.py