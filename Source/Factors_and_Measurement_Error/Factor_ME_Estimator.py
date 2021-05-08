# Factor_ME_Estimator.py
# Calculate the factor estimator and benchmark it against IV

from sklearn.preprocessing import StandardScaler
from pca import pca
import statsmodels.api as sm

def PCR_coeffs(y, X):

    # Standardize
    y_std = StandardScaler().fit_transform(y)
    X_std = StandardScaler().fit_transform(X)

    # Perform the factor analysis
    pca_model = pca()
    pca_results = pca_model.fit_transform(X_std)

    # Regress y, life_expectancy, on the first principal component and output the results
    factor_regression = sm.OLS(y, pca_results['PC'].iloc[:, 0].reset_index(drop = True)).fit()

    return(factor_regression.params)
