#!/bin/bash

#SBATCH --account=macs30123
#SBATCH --partition=amd
#SBATCH --job-name=macs30123_q1b
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1          
#SBATCH --ntasks=20

module load python
source activate a1_env

rm -f q1b.out

for i in {1..20}
do
    mpirun -n $i python3 q1b.py >> q1b.out
done

python3 q1b_plot.py