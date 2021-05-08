# Simulate_DGP.py
# Simulate basic data to test the performance of the factor approach to measurement error
# We will calculate the estimator in a separate file

import numpy as np
import pandas as pd

# Write the DGP as a function
# The defaults are 1000 observations, very correlated true covariates (0.9)
def DGP(N, rho, p, beta, measurement_error):

    # Random normal error
    u = np.random.normal(size = (N, 1))

    # Specified variance-covariance matrix
    cov = np.ones((p, p)) * rho - np.eye(p) * rho + np.eye(p)

    # Random normal draw for base values of X
    true_X = np.random.multivariate_normal(mean = np.zeros(p), cov = cov, size = N)

    # Add in some measurement error
    mismeasured_X = true_X + measurement_error

    # Simulate the Y using the true X values
    Y = true_X@(np.transpose(beta)) + u
    
    data = pd.DataFrame(np.concatenate([Y, mismeasured_X], axis=1))

    return data

# Run the DGP an appropriate number of times and save the data
