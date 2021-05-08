# Factor_ME_Estimator.py
# Function for the PCR estimator
# Please standardize the y and X beforehand

from pca import pca
import statsmodels.api as sm

def PCR_coeffs(y, X):

    # Perform the factor analysis
    pca_model = pca()
    pca_results = pca_model.fit_transform(X)

    # Regress on the first principal component and output the results
    factor_regression = sm.OLS(y, pca_results['PC'].iloc[:, 0].reset_index(drop = True)).fit()

    return(factor_regression.params)
