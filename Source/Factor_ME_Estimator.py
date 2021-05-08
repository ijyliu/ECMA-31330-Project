# Factor_ME_Estimator.py
# Function for the PCR estimator
# Please standardize the y and X beforehand

# Import objects from the setup file
from Prelim import *

# Packages
from pca import pca
import statsmodels.api as sm

def PCR_coeffs(y, X):

    # Perform the factor analysis
    pca_model = pca()
    pca_results = pca_model.fit_transform(X)

    # Regress on the first principal component
    factor_regression = sm.OLS(y, pca_results['PC'].iloc[:, 0].reset_index(drop = True)).fit()

    # Return the parameter values
    return(factor_regression.params)
