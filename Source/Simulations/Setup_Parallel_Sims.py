# Directory structure
import os
repo_dir = os.path.join(os.path.abspath(''), '../..')
input_dir = repo_dir + "/Input"
output_dir = repo_dir + "/Output"
figures_dir = output_dir + "/Figures"
tables_dir = output_dir + "/Tables"
sim_results_dir = output_dir + "/Sim_Results"

# Packages
import pandas as pd

# Create a csv of parameter combos
beta1s = [0.1, 1, 10]
beta2s = [0.1, 1, 10]
covariances = [-0.9, -0.5, 0, 0.5, 0.9]
ps = [5, 20, 50]
me_covs = [0, 0.5]

# Only run certain sims
def make_counter(beta1, beta2, covariance, p):
    counter = 0
    if beta1 == 1:
        counter += 1
    if beta2 == 1:
        counter += 1
    if covariance == 0.5:
        counter += 1
    if p == 5:
        counter+=1
    return(counter)

# Scenarios to run
parameter_combos = (pd.MultiIndex.from_product([beta1s, beta2s, covariances, ps, me_covs])
                                 .to_frame()
                                 .reset_index(drop = True))

parameter_combos.columns = ['beta1', 'beta2', 'covariance', 'p', 'me_cov']

print(parameter_combos)

parameter_combos['counter'] = parameter_combos.apply(lambda x: make_counter(x.beta1, x.beta2, x.covariance, x.p), axis = 1)

parameter_combos_to_run = (parameter_combos.query('counter >= 3')
                                           .drop(columns = 'counter'))

parameter_combos_to_run.to_csv(sim_results_dir + '/' + str(len(parameter_combos_to_run)) + '_parameter_combos_to_run.csv', index = False)
