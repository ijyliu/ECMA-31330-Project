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
    beta_OLS_true = sm.OLS(Y, true_X[:, 0]).fit().params
    beta_OLS_mismeasured = sm.OLS(Y, mismeasured_X[:, 0]).fit().params
    beta_PCR = PCR_coeffs(Y, mismeasured_X)
    beta_IV = IV2SLS(dependent = Y, endog = mismeasured_X[:, 0], instruments = Z).fit().params

    return(beta_OLS_true, beta_OLS_mismeasured, beta_PCR, beta_IV)
