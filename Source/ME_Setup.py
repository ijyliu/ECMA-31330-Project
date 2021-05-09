# ME_Setup.py
# Defines the project directory structure as well as functions to define the estimator and  

# Packages
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from pca import pca
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

# Directory structure
data_dir = "~/Box/ECMA-31330-Project"
repo_dir = os.path.join(os.path.dirname( __file__ ), '..')
output_dir = repo_dir + "/Output"
figures_dir = output_dir + "/Figures"
regressions_dir = output_dir + "/Regressions"
tables_dir = output_dir + "/Tables"

# Write the DGP as a function
# The inputs are sample size, correlation between the non-mismeasured covariates, the number of covariates p, the true beta, and a vector specifying the variance of the classical measurement error for each covariate
def DGP(N, rho, p, kappa, beta, x_measurement_errors):

    # Convert the beta and measurment error inputs to arrays if they are not already
    if not isinstance(beta, np.ndarray):
        beta = np.array(beta)
    if not isinstance(x_measurement_errors, np.ndarray):
        x_measurement_errors = np.array(x_measurement_errors)

    # Get beta into an array of convenient dimensions
    beta = beta.reshape(p, 1)

    # Random normal error
    u = np.random.normal(size = (N, 1))

    # Specified variance-covariance matrix
    # First fill in rho for everything, then subtract off rho on the diagonal and add a one back in
    cov = np.ones((p, p)) * rho - np.eye(p) * rho + np.eye(p)

    # Random normal draw for base values of X
    true_X = np.random.multivariate_normal(mean = np.zeros(p), cov = cov, size = N)

    # Produce an instrument for x_1
    # This Z will be x_1 times the kappa coefficient plus some random noise
    Z = true_X[:, 0].reshape(N, 1) * kappa + np.random.normal(size = (N, 1))

    # Vectors of measurement error for each component
    me_vectors = []
    for i in range(p):
        # Note we need to convert the variance into a standard deviation
        me_vectors.append(np.random.normal(size = (N, 1), scale = np.sqrt(x_measurement_errors[i])))

    # Add the true x to the list of ME vectors to get mismeasured x
    mismeasured_X = true_X + np.concatenate(me_vectors, axis=1)

    # Simulate the Y using the true X values
    Y = true_X@beta + u

    return(Y, true_X, mismeasured_X, Z)

# Function for the PCR estimator
# Please standardize the y and X beforehand
def PCR_coeffs(y, X):

    # Perform the factor analysis
    pca_model = pca()
    pca_results = pca_model.fit_transform(X)

    # Regress on the first principal component
    factor_regression = sm.OLS(y, pca_results['PC'].iloc[:, 0].reset_index(drop = True)).fit()

    # Return the parameter values
    return(factor_regression.params)

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

# Given a dataframe and an indicator for whether the run is local or on a computing cluster, perform some analysis
def perform_analysis(dataframe, local_or_cluster):

    # Plot some results
    scenarios_for_plot = (dataframe.melt(id_vars=['N', 'rho', 'p', 'kappa', 'beta', 'me'], value_vars=['ols_true', 'ols_mismeasured', 'pcr', 'iv'], var_name='estimator', value_name='coeff')
                                   .query('N == 1000' and 'rho == 0.1' and 'kappa == 0.9' and 'beta == "beta_combo_1"' and 'me == "me_combo_2"'))

    grid = sns.FacetGrid(scenarios_for_plot, col='beta', row='me', hue='estimator')
    grid.map_dataframe(sns.histplot, x='coeff')
    grid.fig.suptitle('Coefficients Across Simulations for _')
    grid.add_legend()
    plt.savefig(figures_dir + "/Simulation_Results_Grid_" + local_or_cluster + ".pdf")
    plt.close()

    # Mean coefficient values
    scenarios_for_plot['coeff'] = pd.to_numeric(scenarios_for_plot['coeff'])

    # Overall mean results
    (scenarios_for_plot.filter(['estimator', 'coeff'])
                       .groupby('estimator')
                       .mean()
                       .to_latex(tables_dir + "/mean_estimator_results_" + local_or_cluster + ".tex"))

    # Results by ME levels
    (scenarios_for_plot.filter(['estimator', 'coeff', 'me'])
                       .groupby(['estimator', 'me'])
                       .mean()
                       .to_latex(tables_dir + "/mean_me_estimator_results_" + local_or_cluster + ".tex"))
