# Simulate_DGP.py
# Simulate basic data to test the performance of the factor approach to measurement error
# We will calculate the estimator in a separate file

# Write the DGP as a function
def DGP(N = 1000, rho = 0, betas):
    
    # Make sure N is an integer
    N = int(N)

    # Variance-covariance matrix
    Sigma = [[1, rho], [rho, 1]]

    # Produce joint normal x vectors, mean 0, variance-covariance Sigma, of length N
    x_1, x_2 = np.random.multivariate_normal(mean=[0,0], cov=Sigma, size=N).T

    # Standard normal error vector of size N
    e = np.random.normal(size=N)

    # Compute y using formula
    y = alpha*np.ones(N) + beta_1*x_1 + beta_2*x_2 + e

    return(y, x_1, x_2)

def genData(N, beta, p, rho):
    u = np.random.normal(size = (N, 1))
    cov = np.ones((p, p)) * rho - np.eye(p) * rho + np.eye(p)
    X = np.random.multivariate_normal(mean = np.zeros(p), cov = cov, size = N)
    truep = len(np.transpose(beta))
    Y = X[:,:truep]@(np.transpose(beta)) + u
    data = pd.DataFrame(np.concatenate([Y, X], axis=1))
    return data
