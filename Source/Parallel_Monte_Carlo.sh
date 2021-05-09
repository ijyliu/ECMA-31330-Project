#!/bin/bash

#SBATCH --job-name=Parallel_Monte_Carlo   # user-defined job name

# Suppress output
#SBATCH --output=/dev/null

# Get the array end
readarray -d '' filenames < <(find /project/burning_glass/Input/'Burning Glass Data'/Main -name "*.txt" -print0)
let arrayend=${#filenames[@]}-1

ls | sed -e s/[^0-9]//g

module load python/booth/3.6/3.6.12
FIRSTJOBID=$(sbatch --parsable Setup_Simulations.sh)
SECONDJOBID=$(sbatch --parsable --dependency=afterany:$FIRSTJOBID --array=0-$((arrayend)) Parallel_Run_Simulations.sh)
sbatch --parsable --dependency=afterok:$SECONDJOBID Cl_File_Cleanup.sh
