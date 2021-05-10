# Benchmark_Estimator.py
# This file benchmarks PCA against IV for dealing with measurment error

# Import objects from the setup file
from ME_Setup import *

# Packages
import numpy as np
import pandas as pd

# Simulations dataframe with variations of parameter values
# For local runs such as this file, we limit the number of scenarios considered
num_sims = 100
Ns = [1000]
betas = [1]
# The numpy arrays of measurement error values have to be written out as strings for storage and converted later
me_means = ['[0,0]']
me_covs = ['[[100, 0], [0, 1]]']
kappas = [1]

# Cartesian product of scenarios
index = pd.MultiIndex.from_product([Ns, betas, me_means, me_covs, kappas], names = ["N", "beta", "me_means", "me_cov", "kappa"])

# Scenarios dataframe
scenarios = pd.DataFrame(index = index).reset_index()

# Make a row for each simulation
scenarios = pd.concat([scenarios] * num_sims).sort_index()

# Apply the DGP function scenario parameters to get the results
scenarios[['ols_true', 'ols_mismeasured', 'pcr', 'iv']] = scenarios.apply(lambda x: pd.Series(get_estimators(x['N'], x['beta'], x['me_means'], x['me_cov'], x['kappa'])), axis = 1)

scenarios['sim_num'] = np.tile(range(num_sims), int(len(scenarios) / num_sims))

# Write a csv of local results for easy perusal
scenarios.to_csv(data_dir + "/estimator_results.csv")

# Perform the main analysis
perform_analysis(scenarios, "local")
