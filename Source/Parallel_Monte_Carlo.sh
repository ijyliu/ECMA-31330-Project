#!/bin/bash

#SBATCH --account=pi-cpatt
#SBATCH --partition=standard
#SBATCH --job-name=Parallel_Monte_Carlo

#SBATCH --array=0-100

# Suppress output
#SBATCH --output=/dev/null

# Run array jobs
module load python/booth/3.6/3.6.12
python3
python3 Run_Simulations.py $SLURM_ARRAY_TASK_ID
