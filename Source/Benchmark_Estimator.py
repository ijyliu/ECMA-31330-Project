# Benchmark_Estimator.py
# This file benchmarks PCA against IV for dealing with measurment error

# Import objects from the setup file
from ME_Setup import *

# Packages
import numpy as np
import pandas as pd
from stargazer.stargazer import Stargazer
from pandas.plotting import scatter_matrix
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

# Simulations dataframe with variations of parameter values
num_sims = 1
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
scenarios['results'] = scenarios.apply(lambda x: get_estimators(x.N, x.rho, x.p, x.kappa, x.beta_list, x.me_list), axis = 1)

scenarios['sim_num'] = np.tile(range(num_sims), int(len(scenarios) / num_sims))

# Mergesort ensures stability
scenarios = scenarios.sort_values('sim_num', kind='mergesort')

scenarios['ols_true'] = scenarios.explode('results').reset_index().iloc[::4].reset_index()['results']
scenarios['ols_mismeasured'] = scenarios.explode('results').reset_index().iloc[1::4].reset_index()['results']
scenarios['pcr'] = scenarios.explode('results').reset_index().iloc[2::4].reset_index()['results']
scenarios['iv'] = scenarios.explode('results').reset_index().iloc[3::4].reset_index()['results']

scenarios.to_csv(data_dir + "/estimator_results.csv")

#print(scenarios)

# Plot some results
scenarios_for_plot = (scenarios.melt(id_vars=['N', 'rho', 'p', 'kappa', 'beta', 'me'], value_vars=['ols_true', 'ols_mismeasured', 'pcr', 'iv'], var_name='estimator', value_name='coeff')
                               .query('N == 1000' and 'rho == 0.1' and 'kappa == 0.9' and 'beta == "beta_combo_1"' and 'me == "me_combo_2"'))

grid = sns.FacetGrid(scenarios_for_plot, col='beta', row='me', hue='estimator')
grid.map_dataframe(sns.histplot, x='coeff')
#grid.fig.subplots_adjust(top=0.95)
#grid.fig.suptitle('Coefficients Across Simulations for _', size = 16, y = 0.99)
grid.fig.suptitle('Coefficients Across Simulations for _')
grid.add_legend()
plt.savefig(figures_dir + "/Simulation_Results_Grid.pdf")
plt.close()

# Mean coefficient values

scenarios_for_plot['coeff'] = pd.to_numeric(scenarios_for_plot['coeff'])

# Overall mean results
(scenarios_for_plot.filter(['estimator', 'coeff'])
                   .groupby('estimator')
                   .mean()
                   .to_latex(tables_dir + "/mean_estimator_results.tex"))

print(scenarios_for_plot)

# Results by ME levels
(scenarios_for_plot.filter(['estimator', 'coeff', 'me'])
                   .groupby(['estimator', 'me'])
                   .mean()
                   .to_latex(tables_dir + "/mean_me_estimator_results.tex"))
