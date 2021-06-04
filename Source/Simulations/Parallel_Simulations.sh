#!/bin/bash
#SBATCH --partition=broadwl       # standard (default), long, gpu, mpi, highmem
#SBATCH --cpus-per-task=1          # number of CPUs requested (for parallel tasks)
#SBATCH --mem=1G           # requested memory
#SBATCH --time=06:00:00          # wall clock limit (d-hh:mm:ss)
#SBATCH --job-name=Parallel_Simulations   # user-defined job name
#SBATCH --output=%x.out

# Get the array end
readarray filename < <(find ~/repo/ECMA-31330-Project/Output/Sim_Results -name "*_parameter_combos_to_run.csv" -printf '%f\n')
echo $filename
let arrayend=${filename//[^0-9]/}-1
echo $arrayend

FIRSTJOBID=$(sbatch --parsable Setup_Parallel_Sims.sh)
SECONDJOBID=$(sbatch --parsable --dependency=afterany:$FIRSTJOBID --array=0-$((arrayend)) Run_Parallel_Sim.sh)
sbatch --parsable --dependency=afterok:$SECONDJOBID Compile_Parallel_Sims.sh
