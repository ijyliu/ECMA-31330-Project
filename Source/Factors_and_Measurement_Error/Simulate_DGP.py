# Simulate_DGP.py
# Simulate basic data to test the performance of the factor approach to measurement error
# We will calculate the estimator in a separate file

# Packages
import numpy as np
import pandas as pd

# Location to store data
data_dir = "~/Box/ECMA-31330-Project"

# Write the DGP as a function
# The inputs are sample size, correlation between the non-mismeasured covariates, the number of covariates p, the true beta, and a vector specifying the variance of the classical measurement error for each covariate
def DGP(N, rho, p, beta, x_measurement_errors):

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

    # Vectors of measurement error for each component
    me_vectors = []
    for i in range(p):
        # Note we need to convert the variance into a standard deviation
        me_vectors.append(np.random.normal(size = (N, 1), scale = np.sqrt(x_measurement_errors[i])))

    # Add the true x to the list of ME vectors to get mismeasured x
    mismeasured_X = true_X + np.concatenate(me_vectors, axis=1)

    # Simulate the Y using the true X values
    Y = true_X@beta + u

    # Return an appropriately labelled dataframe
    Y = (pd.DataFrame(Y)
           .rename(columns={0:"y"}))
    mismeasured_X = (pd.DataFrame(mismeasured_X)
                       .add_prefix('mis_x_'))
    true_X = (pd.DataFrame(true_X)
                .add_prefix('true_x_'))
    data = pd.concat([Y, mismeasured_X, true_X], axis=1)

    return(data)

# Run the DGP an appropriate number of times and save the data
num_sims = 2

# List of the data. This will be super-fast to concatenate into a big dataframe
data_list = []
for i in range(num_sims):
    data_list.append(DGP(100, 0.9, 3, [1,0,0], [1,1,1]))

simulation_data = pd.concat(data_list, keys=range(num_sims))

# Export the CSV
simulation_data.to_csv(data_dir + "/ME_Sim.csv")
