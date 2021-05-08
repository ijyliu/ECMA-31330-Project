# Simulate_DGP.py
# Simulate basic data to test the performance of the factor approach to measurement error
# We will calculate the estimator in a separate file

import numpy as np
import pandas as pd

# Write the DGP as a function
# The inputs are sample size, correlation between the non-mismeasured covariates, the number of covariates p, the true beta, and a vector specifying the variance of the classical measurement error for each covariate
def DGP(N, rho, p, beta, x_measurement_errors):

    beta = beta.reshape(p, 1)

    # Random normal error
    u = np.random.normal(size = (N, 1))
    #print(u)

    # Specified variance-covariance matrix
    # First fill in rho for everything, then subtract off rho on the diagonal and add a one back in
    cov = np.ones((p, p)) * rho - np.eye(p) * rho + np.eye(p)

    # Random normal draw for base values of X
    true_X = np.random.multivariate_normal(mean = np.zeros(p), cov = cov, size = N)

    # Vectors of measurement error for each component
    me_vectors = []
    for i in range(p):
        # Note we need to convert the variance into a standard deviation
        me_vectors.append(np.random.normal(size = (N, 1), scale = np.sqrt(x_measurement_errors[i])))

    # Add the true x to the list of ME vectors to get mismeasured x
    mismeasured_X = true_X + np.concatenate(me_vectors, axis=1)

    #print(true_X)
    # print(np.shape(true_X))
    # print(np.shape(beta.T))
    # print(np.shape(u))
    # print(np.shape(true_X@(beta.T)))
    # print(true_X@(beta.T))
    #print(true_X@np.transpose(beta) + u)

    # Simulate the Y usin#g the true X values
    Y = true_X@beta + u
    
    #print(Y)

    # The data we observe are the true Y, but the mismeasured X
    true_data = pd.DataFrame(np.concatenate([Y, true_X], axis=1))
    data = pd.DataFrame(np.concatenate([Y, mismeasured_X], axis=1))

    return(data, true_data)

#DGP(100, 0.9, 3, np.array([1, 0, 0]), np.array([1, 1, 1]))
print(DGP(100, 0.9, 3, np.array([1, 0, 0]), np.array([1, 1, 1])))

# Run the DGP an appropriate number of times and save the data
