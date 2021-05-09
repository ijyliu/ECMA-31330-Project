#!/bin/bash

#SBATCH --account=pi-cpatt
#SBATCH --partition=standard
#SBATCH --job-name=Parallel_Run_Simulations

# Suppress output
#SBATCH --output=/dev/null

# Run array job given the task id
module load python/booth/3.6/3.6.12
python3 Run_Simulations.py $SLURM_ARRAY_TASK_ID
