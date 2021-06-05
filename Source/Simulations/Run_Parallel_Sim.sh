#!/bin/bash

#SBATCH --partition=broadwl       # cluster partition
#SBATCH --cpus-per-task=1          # number of CPUs requested (for parallel tasks)
#SBATCH --mem=1G           # requested memory
#SBATCH --time=06:00:00          # wall clock limit (d-hh:mm:ss)

#SBATCH --job-name=Run_Parallel_Sim   # user-defined job name

#SBATCH --output=%a_%x.out # output file name

module load python
python Run_Parallel_Sim.py $SLURM_ARRAY_TASK_ID