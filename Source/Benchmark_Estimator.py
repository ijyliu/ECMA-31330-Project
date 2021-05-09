# Benchmark_Estimator.py
# This file benchmarks PCA against IV for dealing with measurment error

# Import objects from the setup file
from Prelim import *

# Packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stargazer.stargazer import Stargazer
from pandas.plotting import scatter_matrix
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from linearmodels import IV2SLS

# Load in the DGP function
from Simulate_DGP import *

# Load in the PCR/Factor estimator
from Factor_ME_Estimator import *

# Given some simulation parameters, run both the PCA regression and the IV regression for a simulation
def get_estimators(N, rho, p, kappa, beta, x_measurement_errors):

    # Run the DGP
    Y, true_X, mismeasured_X, Z = DGP(N, rho, p, kappa, beta, x_measurement_errors)

    # Standardize
    Y = StandardScaler().fit_transform(Y)
    true_X = StandardScaler().fit_transform(true_X)
    mismeasured_X = StandardScaler().fit_transform(mismeasured_X)
    Z = StandardScaler().fit_transform(Z)

    # Calculate estimators
    beta_OLS_true = sm.OLS(Y, true_X[:, 0]).fit().params[0]
    beta_OLS_mismeasured = sm.OLS(Y, mismeasured_X[:, 0]).fit().params[0]
    beta_PCR = PCR_coeffs(Y, mismeasured_X)[0]

    # Sadly I have to the IV estimation by hand, because the packages I tried required exogenous control variables
    beta_IV = np.cov(Y, Z)[0,1] / np.cov(mismeasured_X[:, 0].reshape(N, 1), Z)[0,1]

    return(beta_OLS_true, beta_OLS_mismeasured, beta_PCR, beta_IV)

# Simulations dataframe with variations of parameter values
num_sims = 2
Ns = [100, 1000]
rhos = [0.1, 0.9]
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
        return([1, 1, 1])
    if me_combo == 'me_combo_2':
        return([1, 0, 0])

# Cartesian product of scenarios
index = pd.MultiIndex.from_product([Ns, rhos, ps, kappas, betas, mes], names = ["N", "rho", "p", "kappa", "beta", "me"])

# Scenarios dataframe
scenarios = pd.DataFrame(index = index).reset_index()

# Convert the combo strings into lists
scenarios['beta'] = scenarios.apply(lambda x: assign_beta(x.beta), axis = 1)
scenarios['me'] = scenarios.apply(lambda x: assign_me(x.me), axis = 1)

# Make a row for each simulation
scenarios = pd.concat([scenarios] * num_sims).sort_index()

# Apply the DGP function scenario parameters to get the results
scenarios['results'] = scenarios.apply(lambda x: get_estimators(x.N, x.rho, x.p, x.kappa, x.beta, x.me), axis = 1)

scenarios['sim_num'] = np.tile(range(num_sims), int(len(scenarios) / num_sims))

# Mergesort ensures stability
scenarios = scenarios.sort_values('sim_num', kind='mergesort')

scenarios['ols_true'] = scenarios.explode('results').reset_index().iloc[::4].reset_index()['results']
scenarios['ols_mismeasured'] = scenarios.explode('results').reset_index().iloc[1::4].reset_index()['results']
scenarios['pcr'] = scenarios.explode('results').reset_index().iloc[2::4].reset_index()['results']
scenarios['iv'] = scenarios.explode('results').reset_index().iloc[3::4].reset_index()['results']

scenarios.to_csv(data_dir + "/estimator_results.csv")

print(scenarios)
