# Benchmark_Estimator.py
# This file benchmarks PCA against IV for dealing with measurment error

# Import objects from the setup file
from ME_Setup import *

# Packages
import numpy as np
import pandas as pd

# Simulations dataframe with variations of parameter values
# For local runs such as this file, we limit the number of scenarios considered
num_sims = 1000
Ns = [100, 1000]
rhos = [0.8, 0.9]
ps = [3]
kappas = [0.1, 0.9]

# This is an awkward/bad way of making the combos of lists of coefficients
betas = ['beta_combo_1', 'beta_combo_2']
mes = ['me_combo_1', 'me_combo_2']

def assign_beta(beta_combo):
    if beta_combo == 'beta_combo_1':
        return([1, 1, 1])
    if beta_combo == 'beta_combo_2':
        return([1, 0, 0])

def assign_me(me_combo):
    if me_combo == 'me_combo_1':
        return([10, 0, 0])
    if me_combo == 'me_combo_2':
        return([100, 0, 0])

# Cartesian product of scenarios
index = pd.MultiIndex.from_product([Ns, rhos, ps, kappas, betas, mes], names = ["N", "rho", "p", "kappa", "beta", "me"])

# Scenarios dataframe
scenarios = pd.DataFrame(index = index).reset_index()

# Convert the combo strings into lists
scenarios['beta_list'] = scenarios.apply(lambda x: assign_beta(x.beta), axis = 1)
scenarios['me_list'] = scenarios.apply(lambda x: assign_me(x.me), axis = 1)

# Make a row for each simulation
scenarios = pd.concat([scenarios] * num_sims).sort_index()

# Apply the DGP function scenario parameters to get the results
scenarios[['OLS_true', 'OLS_mismeasured', 'PCR', 'IV']] = scenarios.apply(lambda x: get_estimators(x.N, x.rho, x.p, x.kappa, x.beta, x.me), axis = 1)

scenarios['sim_num'] = np.tile(range(num_sims), int(len(scenarios) / num_sims))

# Write a csv of local results for easy perusal
scenarios.to_csv(data_dir + "/estimator_results.csv")

# Perform the main analysis
perform_analysis(scenarios, "local")
