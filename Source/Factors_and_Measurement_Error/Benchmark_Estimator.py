
# Packages
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stargazer.stargazer import Stargazer
from pandas.plotting import scatter_matrix
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = "~/Box/ECMA-31330-Project"

sim_data = pd.read_csv(data_dir + "/ME_Sim.csv")

# Standardize
std_sim_data = pd.DataFrame(StandardScaler().fit_transform(sim_data), index=sim_data.index, columns=sim_data.columns)
