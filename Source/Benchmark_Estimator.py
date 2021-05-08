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

# Load in the DGP function
from Simulate_DGP import *

DGP(100, 0.9, 3, 0.9, [1,0,0], [1,1,1])

data_dir = "~/Box/ECMA-31330-Project"

sim_data = pd.read_csv(data_dir + "/ME_Sim.csv")

# Standardize
std_sim_data = pd.DataFrame(StandardScaler().fit_transform(sim_data), index=sim_data.index, columns=sim_data.columns)

# Given some simulation parameters, run both the PCA regression and the IV regression for a simulation
def benchmark_estimator(N, rho, p, kappa, beta, x_measurement_errors, num_sims):

    Y, true_X, mismeasured_X, Z = DGP(N, rho, p, kappa, beta, x_measurement_errors)
