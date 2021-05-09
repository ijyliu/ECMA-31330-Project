# Run_Simulations.py
# Read in a row of parameter values, and run a number of simulations using them

from os import sys
import pandas as pd

from ME_Setup import *

slurm_number = sys.argv[1]
print('slurm job array number: ' + str(slurm_number))

# Get the name of the parameters file for read in
param_file_name = 

# Number of simulations to run
num_sims = 10

# Read in the parameters
simulations = (pd.read_csv(param_file_name)
                 .iloc[slurm_number, :])

# Make a row for each simulation
simulations = pd.concat([simulations] * num_sims).reset_index()

# Apply the DGP function scenario parameters to get the results
simulations[['OLS_true', 'OLS_mismeasured', 'PCR', 'IV']] = simulations.apply(lambda x: get_estimators(x.N, x.rho, x.p, x.kappa, x.beta, x.me), axis = 1)

# Save the results
simulations.to_csv(data_dir + "/sim_results_" + str(sys.argv[1]) + ".csv")
