#!/bin/bash

#SBATCH --account=pi-cpatt

#SBATCH --partition=standard       # standard (default), long, gpu, mpi, highmem
#SBATCH --cpus-per-task=1          # number of CPUs requested (for parallel tasks)
#SBATCH --mem=1G           # requested memory
#SBATCH --time=2-00:00:00          # wall clock limit (d-hh:mm:ss)

#SBATCH --job-name=Parallel_Run_Simulations
#SBATCH --output=%a_%x.out

# Run array job given the task id
module load python/booth/3.6/3.6.12
python3 Parallel_Run_Simulations.py $SLURM_ARRAY_TASK_ID
