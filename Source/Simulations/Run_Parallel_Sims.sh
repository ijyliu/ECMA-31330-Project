#!/bin/bash

#SBATCH --partition=broadwl       # standard (default), long, gpu, mpi, highmem
#SBATCH --cpus-per-task=1          # number of CPUs requested (for parallel tasks)
#SBATCH --mem=1G           # requested memory
#SBATCH --time=06:00:00          # wall clock limit (d-hh:mm:ss)
#SBATCH --array=0-21

#SBATCH --job-name=Run_Parallel_Sims   # user-defined job name

#SBATCH --output=%a_%x.out # output file name

echo "started job"
module load python
echo "loaded python"
python Run_Parallel_Sims.py $SLURM_ARRAY_TASK_ID
echo "completed script"