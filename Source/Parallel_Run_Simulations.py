# Run_Simulations.py
# Read in a row of parameter values, and run a number of simulations using them

# Packages
from os import sys
import pandas as pd

# Functions and objects
from ME_Setup import *

slurm_number = int(sys.argv[1])
print('slurm job array number: ' + str(slurm_number))

# Get the name of the parameters file for read in
param_file_name = [f for f in os.listdir(os.path.expanduser(data_dir)) if '_parameter_combos' in f][0]
print(param_file_name)

# Number of simulations to run
num_sims = 1

# Read in the parameters
simulations = (pd.read_csv(data_dir + "/" + param_file_name)
                 .iloc[slurm_number, :])

print(simulations)

# Make a row for each simulation
simulations = pd.concat([simulations] * num_sims, axis = 1).reset_index().transpose()

print(simulations)
print(simulations.columns)

simulations.columns = simulations.iloc[0]
simulations = simulations[1:]

print(simulations)
print(simulations.columns)

# Apply the DGP function scenario parameters to get the results
simulations[['ols_true', 'ols_mismeasured', 'pcr', 'iv']] = simulations.apply(lambda x: pd.Series(get_estimators(x['N'], x['rho'], x['p'], x['kappa'], x['beta_list'], x['me_list'])), axis = 1)

print(simulations)

# Save the results
simulations.to_csv(data_dir + "/sim_results_" + str(sys.argv[1]) + ".csv")

print('completed sims')
