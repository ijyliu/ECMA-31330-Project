#!/bin/bash

#SBATCH --account=pi-cpatt

#SBATCH --partition=standard       # standard (default), long, gpu, mpi, highmem
#SBATCH --cpus-per-task=1          # number of CPUs requested (for parallel tasks)
#SBATCH --mem=16G           # requested memory
#SBATCH --time=2-00:00:00          # wall clock limit (d-hh:mm:ss)

#SBATCH --job-name=Preliminary_Analysis    # user-defined job name

#SBATCH --output=%x.out # output file name

module load R
Rscript Preliminary_Analysis.R
