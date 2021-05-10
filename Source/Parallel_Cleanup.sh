#!/bin/bash

#SBATCH --job-name=Parallel_Cleanup

# Suppress output
#SBATCH --output=/dev/null

# Concatenate all appropriate .out files into one
cat *_Parallel_Run_Simulations.out > Parallel_Run_Simulations.out
rm *_Parallel_Run_Simulations.out