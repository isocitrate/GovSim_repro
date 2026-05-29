#!/bin/bash
#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --job-name=macs30123_a3_q2b
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load python
module load java
source activate a3_env

python3 q2b.py
