# Benchmark_Estimator.py
# This file benchmarks PCA for dealing with measurement error

# Import objects from the setup file
import os
import sys
sys.path.append(os.path.expanduser('~/repo/ECMA-31330-Project/Source'))
from ME_Setup import *

# Packages
import pandas as pd

# Simulations dataframe with variations of parameter values
# For local runs such as this file, we limit the number of simulations and scenarios considered
num_sims = 100
Ns = [1000]
betas = [1]
# The numpy arrays of measurement error values have to be written out as strings for storage and converted later
me_means = ['[0,0]']
me_covs = ['[[100, 0], [0, 1]]']
kappas = [1]

# Scenarios dataframe
scenarios = produce_scenarios_cartesian(Ns, betas, me_means, me_covs, kappas)

# Make a row for each simulation
scenarios = pd.concat([scenarios] * num_sims).sort_index()

# Apply the DGP function scenario parameters to get the results
scenarios[['ols_true', 'ols_mismeasured', 'pcr', 'iv']] = scenarios.apply(lambda x: pd.Series(get_estimators(x['N'], x['beta'], x['me_means'], x['me_cov'], x['kappa'])), axis = 1)

# Write a csv of local results for easy perusal
scenarios.to_csv(sim_results_dir + "/local_estimator_results.csv")

# Perform the main analysis
perform_analysis(scenarios, "local")