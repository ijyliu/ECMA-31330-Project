# Setup_Simulations.py

import pandas as pd

from Prelim import *

# Simulations dataframe with variations of parameter values
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
        return([2, 1, 1])

def assign_me(me_combo):
    if me_combo == 'me_combo_1':
        return([10, 0, 0])
    if me_combo == 'me_combo_2':
        return([2, 0, 0])

# Cartesian product of scenarios
index = pd.MultiIndex.from_product([Ns, rhos, ps, kappas, betas, mes], names = ["N", "rho", "p", "kappa", "beta", "me"])

# Produce scenarios dataframe
scenarios = pd.DataFrame(index = index).reset_index()

pd.write_csv(data_dir + "/" + str(len(scenarios)) + "_parameters.csv")
