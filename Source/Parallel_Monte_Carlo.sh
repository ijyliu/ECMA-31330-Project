#!/bin/bash

#SBATCH --job-name=Parallel_Monte_Carlo   # user-defined job name

# Suppress output
#SBATCH --output=/dev/null

# Get the array end
readarray -d '' filename < <(find ~/Box/ECMA-31330-Project -name "*_parameter_combos.csv" -printf '%f\n')
let arrayend=${filename//[^0-9]/}-1

module load python/booth/3.6/3.6.12
FIRSTJOBID=$(sbatch --parsable Parallel_Setup_Simulations.sh)
SECONDJOBID=$(sbatch --parsable --dependency=afterany:$FIRSTJOBID --array=0-$((arrayend)) Parallel_Run_Simulations.sh)
THIRDJOBID=$(sbatch --parsable --dependency=afterok:$SECONDJOBID Parallel_Compile_Results.sh)
sbatch --parsable --dependency=afterok:$THIRDJOBID Parallel_Cleanup.sh
