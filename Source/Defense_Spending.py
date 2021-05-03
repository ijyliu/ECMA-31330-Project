# Defense_Spending.py
## Exploratory Analysis of SIPRI Defense Spending Data

# Packages
import pandas as pd 
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from pca import pca
import seaborn as sns
import os

# Defining directory locations for easy reference
data_dir = "~/Box/ECMA-31330-Project"
repo_dir = os.path.join( os.path.dirname( __file__ ), '..')
output_dir = repo_dir + "/Output"
figures_dir = output_dir + "/Figures"

# We can look at spending per GDP or spending as a share of government spending.
sipri_milex_per_gdp = (pd.read_excel(data_dir + "/SIPRI-Milex-data-1949-2020_0.xlsx", sheet_name="Share of GDP", header=5, engine='openpyxl')
                         .drop(columns='Notes')
                         # Threshold requires a non-NaN country column and at least one observation
                         .dropna(thresh=2)
                         .replace({"xxx":np.NaN, ". .":np.NaN})
                         # Remove "Unnamed" columns
                         .filter(regex='^((?!Unnamed).)*$')
                         .set_index('Country')
                         .transpose())

# Plot a time series of expenditure for countries
plt.figure(figsize=(15,15))
plt.plot(sipri_milex_per_gdp);
plt.legend(sipri_milex_per_gdp.columns);
plt.savefig(figures_dir + "/Milex_GDP_Time_Series.pdf")

# Exploring correlations 
sns.heatmap(sipri_milex_per_gdp.corr())
plt.savefig(figures_dir + "/Milex_correlations.pdf")

# Do the PCA
deMeansipri_milex_per_gdp = sipri_milex_per_gdp - np.mean(sipri_milex_per_gdp, axis=0) 
modelsipri_milex_per_gdp = pca(n_components=deMeansipri_milex_per_gdp.shape[1])
resultssipri_milex_per_gdp = modelsipri_milex_per_gdp.fit_transform(deMeansipri_milex_per_gdp)
sns.heatmap(resultssipri_milex_per_gdp['loadings'],cmap='YlGnBu');

# Plot
modelsipri_milex_per_gdp.plot();

# Predict values
K = 3
Fhat = resultssipri_milex_per_gdp['PC'].iloc[:,0:K].to_numpy()
Mus = resultssipri_milex_per_gdp['loadings'].iloc[0:K].to_numpy()
Yhat = Fhat@Mus

# Scatterplots and other plots of predicted and actual values
plt.scatter(deMeansipri_milex_per_gdp.iloc[:,0],Yhat[:,0])
plt.show()

plt.plot(deMeansipri_milex_per_gdp.index, deMeansipri_milex_per_gdp.iloc[:,0])
plt.plot(deMeansipri_milex_per_gdp.index, Yhat[:,0])
plt.show()

plt.plot(deMeansipri_milex_per_gdp.index, deMeansipri_milex_per_gdp.iloc[:,1])
plt.plot(deMeansipri_milex_per_gdp.index, Yhat[:,1])
plt.show()
