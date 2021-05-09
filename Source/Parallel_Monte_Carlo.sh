#!/bin/bash

#SBATCH --job-name=Parallel_Monte_Carlo   # user-defined job name

# Suppress output
#SBATCH --output=/dev/null

# Get the array end
readarray -d '' filename < <(find ~/Box/ECMA-31330-Project -name "*_parameters.csv" -printf '%f\n')
let arrayend=${filename//[^0-9]/}-1

module load python/booth/3.6/3.6.12
FIRSTJOBID=$(sbatch --parsable Setup_Simulations.sh)
SECONDJOBID=$(sbatch --parsable --dependency=afterany:$FIRSTJOBID --array=0-$((arrayend)) Parallel_Run_Simulations.sh)
sbatch --parsable --dependency=afterok:$SECONDJOBID Parallel_Compile_Results.sh
