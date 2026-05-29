#!/bin/bash

#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --job-name=macs30123_a3_q3
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00

module load python
module load java
source $(conda info --base)/etc/profile.d/conda.sh
conda activate a3_env

python3 q2a.py
python3 q3.py