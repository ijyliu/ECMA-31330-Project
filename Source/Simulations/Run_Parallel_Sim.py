# Directory structure
import os
repo_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')
input_dir = repo_dir + "/Input"
output_dir = repo_dir + "/Output"
figures_dir = output_dir + "/Figures"
tables_dir = output_dir + "/Tables"
sim_results_dir = output_dir + "/Sim_Results"

# Packages
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS 

# Supressing Output
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# Read in the appropriate parameters
# Get the slurm array number
slurm_number = int(sys.argv[1])
print('slurm job array number: ' + str(slurm_number))
# Select the parameters
param_combo_to_run = (pd.read_csv(sim_results_dir + '/22_parameter_combos_to_run.csv')
                        .iloc[slurm_number, :]
                        .to_dict())

beta1 = param_combo_to_run['beta1']
beta2 = param_combo_to_run['beta2']
covariance = param_combo_to_run['covariance']
p = int(param_combo_to_run['p'])
me_cov = param_combo_to_run['me_cov']

N = 3000

output = pd.DataFrame()

# 1000 simulations
for k in range(1000):
    # Initialize Lists to store coef values for all five methods and the true coef
    pca_coef = []
    mismeasured_coef = []
    mismeasured_allvar_coef = []
    mismeasured_avg_coef = []
    iv_coef = []
    true_val_coef =[]

    # Create variables
    vars_mean = [0,0,0]
    vars_cov = np.array([[1,covariance,0],
                            [covariance,1,0],
                            [0,0,1]])
    # Producing 3 variables: x for the variable of interest, the true Z covariate, the random error
    vars_ = pd.DataFrame(np.random.multivariate_normal(vars_mean, vars_cov, N), columns = ['x','true_z','u'])
    vars_['y'] = beta1 * vars_['x'] + beta2 * vars_['true_z'] + vars_['u']

    # Create measurement errors for each of the p measurements of the covariates- mean zero and variance one
    errors_mean = np.zeros(p)
    errors_cov = np.zeros((p,p))
    if me_cov != 0:
        errors_cov[:] = me_cov
    for i in range(p):
        for j in range(p):
            if i == j:
                errors_cov[i,j] = 1
            

    errors = np.random.multivariate_normal(errors_mean, errors_cov, N)
    # Column labels for Z variables (covariates variables mismeasured)
    z_vars = []
    for i in range(p):
        z_vars.append('z'+str(i+1))
    # Add errors to the true_z to get mismeasured values
    mismeasured_z = pd.DataFrame(errors, columns = z_vars)
    for i in mismeasured_z.columns:
        mismeasured_z[i] = mismeasured_z[i] + vars_['true_z']

    # Do feature scaling (normalize to mean 0 and variance 1) for the mismeasured z
    # Note that x and y are already normalized by construction
    scaled_mismeasured_z = mismeasured_z.copy()
    for i in mismeasured_z.columns:
        scaled_mismeasured_z[i] = (mismeasured_z[i] - mismeasured_z[i].mean()) / mismeasured_z[i].std()

    # Suppress output
    with suppress_stdout():
        # Use PCA on the mismeasured values
        pca_model = PCA()
        pca_results = pca_model.fit_transform(scaled_mismeasured_z)
        pca_z = pca_results[:, 0]

    # NOTE: in non-pca cases, no need to rescale or normalize since mismeasured variables and x and y have mean 0 and sd 1

    # Average mismeasured variables:
    vars_['avg_mismeasured_z'] = mismeasured_z[z_vars].mean(axis=1)

    # Add relevant variables to vars_ dataframe
    vars_[mismeasured_z.columns] = mismeasured_z
    vars_['pca_z'] = pca_z

    # Single mismeasured covariate results
    X_mismeasured = vars_[['x','z1']]
    X_mismeasured = sm.add_constant(X_mismeasured)
    model_mismeasured = sm.OLS(vars_['y'],X_mismeasured)
    results_mismeasured = model_mismeasured.fit()
    mismeasured_coef.append(results_mismeasured.params[1])

    # All Variables Mismeasured Results
    # Create full list of items to include in regression
    tot_vars = ['x']
    tot_vars.extend(z_vars)
    X_allvar = vars_[tot_vars]
    X_allvar = sm.add_constant(X_allvar)
    model_mismeasured_allvar = sm.OLS(vars_['y'],X_allvar)
    results_mismeasured_allvar = model_mismeasured_allvar.fit()
    mismeasured_allvar_coef.append(results_mismeasured_allvar.params[1])

    # Average Mismeasured Variables Results
    X_mismeasured_avg = vars_[['x','avg_mismeasured_z']]
    X_mismeasured_avg = sm.add_constant(X_mismeasured_avg)
    model_mismeasured_avg = sm.OLS(vars_['y'],X_mismeasured_avg)
    results_mismeasured_avg = model_mismeasured_avg.fit()
    mismeasured_avg_coef.append(results_mismeasured_avg.params[1])

    # PCA Results
    X_pca = vars_[['x','pca_z']]
    X_pca = sm.add_constant(X_pca)
    model_pca = sm.OLS(vars_['y'],X_pca)
    results_pca = model_pca.fit()
    pca_coef.append(results_pca.params[1])

    # Instrumental Variables Results
    # Instrument z1 on the other items in the mismeasured df
    vars_ = sm.add_constant(vars_, has_constant='add')
    iv_results = IV2SLS(endog = vars_['y'], exog = vars_[['const','x', 'z1']], instrument = pd.concat([vars_['x'], mismeasured_z.iloc[:, 1:]], axis = 1)).fit()
    iv_coef.append(iv_results.params[1])

    # True Results
    X_true = vars_[['x','true_z']]
    X_true = sm.add_constant(X_true)
    model_true = sm.OLS(vars_['y'],X_true)
    results_true = model_true.fit()
    true_val_coef.append(results_true.params[1])

    # Output Findings
    new_output = pd.DataFrame()
    new_output['mismeasured_coef'] = mismeasured_coef
    new_output['mismeasured_allvar_coef'] = mismeasured_allvar_coef
    new_output['mismeasured_avg_coef'] = mismeasured_avg_coef
    new_output['pca_coef'] = pca_coef
    new_output['true_val_coef'] = true_val_coef
    new_output['iv_coef'] = iv_coef
    new_output['covariance'] = vars_cov[0][1]
    new_output['beta1'] = beta1
    new_output['beta2'] = beta2
    new_output['p'] = p
    new_output['me_cov'] = me_cov
    output = output.append(new_output)

# Output the dataframe of results
output.to_csv(sim_results_dir + '/' + str(slurm_number) + '_Parallel_Sim_Results.csv')
