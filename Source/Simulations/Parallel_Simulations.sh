#!/bin/bash
#SBATCH --job-name=Parallel_Simulations
# Suppress output
#SBATCH --output=/dev/null

# Get the array end
readarray -d '' filename < <(find ~/repo/ECMA-31330-Project/Output/Sim_Results -name "*_parameter_combos_to_run.csv" -printf '%f\n')
let arrayend=${filename//[^0-9]/}-1

FIRSTJOBID=$(sbatch --parsable Setup_Parallel_Sims.sh)
SECONDJOBID=$(sbatch --parsable --dependency=afterany:$FIRSTJOBID --array=0-$((arrayend)) Run_Parallel_Sim.sh)
sbatch --parsable --dependency=afterok:$SECONDJOBID Compile_Parallel_Sims.sh
