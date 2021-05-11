#!/bin/bash

#SBATCH --partition=standard       # standard (default), long, gpu, mpi, highmem
#SBATCH --cpus-per-task=1          # number of CPUs requested (for parallel tasks)
#SBATCH --mem=4G           # requested memory
#SBATCH --time=2-00:00:00          # wall clock limit (d-hh:mm:ss)

#SBATCH --job-name=Parallel_Compile_Results
#SBATCH --output=%x.out

module load python/booth/3.6/3.6.12
python3 Parallel_Compile_Results.py
