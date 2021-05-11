# Run_Simulations.py
# Read in a row of parameter values, and run a number of simulations using them

# Import objects from the setup file
import os
import sys
sys.path.append(os.path.expanduser('~/repo/ECMA-31330-Project/Source'))
from ME_Setup import *

# Packages
from os import sys
import pandas as pd

# Get slurm array number which will be used to get a row of the parameter combinations csv
slurm_number = int(sys.argv[1])
print('slurm job array number: ' + str(slurm_number))

# Get the name of the parameters file for read in
param_file_name = [f for f in os.listdir(os.path.expanduser(parameters_dir)) if '_parameter_combos' in f][0]

# Read in the parameters
simulations = (pd.read_csv(parameters_dir + "/" + param_file_name)
                 .iloc[slurm_number, :])

# Number of simulations to run
num_sims = simulations['num_sims'][0]

# Make a row for each simulation
simulations = (pd.concat([simulations] * num_sims, axis = 1)
                 .reset_index()
                 .transpose()
                 .drop(columns = 'Unnamed: 0'))

# Clean up the column names
simulations.columns = simulations.iloc[0]
simulations = simulations[1:]

# Apply the DGP function scenario parameters to get the results
simulations[['ols_true', 'ols_mismeasured', 'pcr', 'iv']] = simulations.apply(lambda x: pd.Series(get_estimators(x['N'], x['beta'], x['me_means'], x['me_cov'], x['kappa'])), axis = 1)

# Save the results
simulations.to_csv(scenario_files_dir + "/sim_results_" + str(sys.argv[1]) + ".csv")

print('completed sims')