# Defense_Spending.py
## Exploratory Analysis of SIPRI Defense Spending Data

# Packages
import pandas as pd 
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from pca import pca
import seaborn as sns
import os

# Defining directory locations for easy reference
data_dir = "~/Box/ECMA-31330-Project"
repo_dir = os.path.join( os.path.dirname( __file__ ), '../..')
output_dir = repo_dir + "/Output"
figures_dir = output_dir + "/Figures"

# We can look at spending per GDP or spending as a share of government spending.
sipri_milex_per_gdp = (pd.read_excel(data_dir + "/SIPRI-Milex-data-1949-2020_0.xlsx", sheet_name="Share of GDP", header=5, engine='openpyxl')
                         .drop(columns='Notes')
                         .replace({"xxx":np.NaN, ". .":np.NaN})
                         # Threshold requires a non-NaN country column and at least one observation
                         .dropna(thresh=2)
                         # Remove "Unnamed" columns
                         .filter(regex='^((?!Unnamed).)*$')
                         .set_index('Country')
                         .transpose()
                         .rename_axis('Year')
                         .reset_index())

# Identify the weird spike in the time series
# print(sipri_milex_per_gdp[sipri_milex_per_gdp > 1])

# Here's a internally/interpolated version of the data
sipri_milex_per_gdp_interpolate = (sipri_milex_per_gdp.interpolate(limit_area='inside'))

sipri_milex_per_gdp_interpolate_95_on = sipri_milex_per_gdp_interpolate.query('Year >= 1995').set_index('Year')

# For every year, plot the share of missing values
sns.heatmap(sipri_milex_per_gdp.isnull(), cbar=False);
plt.savefig(figures_dir + "/SIPRI_Missing_Values.pdf")

sns.heatmap(sipri_milex_per_gdp_interpolate.isnull(), cbar=False);
plt.savefig(figures_dir + "/SIPRI_Missing_Values_Interpolate.pdf")

sns.heatmap(sipri_milex_per_gdp_interpolate_95_on.isnull(), cbar=False);
plt.savefig(figures_dir + "/SIPRI_Missing_Values_Interpolate_Post_1995.pdf")


# For the PCA matrix and LASSO, take the post 1995 interpolated data, with a lot of missing values removed
sipri_milex_inter_no_missing = (sipri_milex_per_gdp_interpolate_95_on.dropna(axis = 1))

# Plot a time series of expenditure for countries
plt.figure(figsize=(15,15))
plt.plot(sipri_milex_per_gdp.set_index('Year'));
#plt.legend(sipri_milex_per_gdp.columns);
plt.savefig(figures_dir + "/Milex_GDP_Time_Series.pdf")

# Exploring correlations 
sns.heatmap(sipri_milex_per_gdp.corr())
plt.savefig(figures_dir + "/Milex_Correlations.pdf")

# Do the PCA
demean_sipri_milex_inter_no_missing = sipri_milex_inter_no_missing - np.mean(sipri_milex_inter_no_missing, axis=0) 
model_sipri_milex_inter_no_missing = pca()
results_sipri_milex_inter_no_missing = model_sipri_milex_inter_no_missing.fit_transform(demean_sipri_milex_inter_no_missing)
sns.heatmap(results_sipri_milex_inter_no_missing['loadings'],cmap='YlGnBu');
plt.savefig(figures_dir + "/Milex_Loadings.pdf")

# Scree plot of share of variance explained
model_sipri_milex_inter_no_missing.plot();
plt.savefig(figures_dir + "/Milex_PC_Share_Explained.pdf")

# Predict values
# K = 3
# Fhat = results_sipri_milex_inter_no_missing['PC'].iloc[:,0:K].to_numpy()
# Mus = results_sipri_milex_inter_no_missing['loadings'].iloc[0:K].to_numpy()
# Yhat = Fhat@Mus

# Scatterplots and other plots of predicted and actual values
# plt.scatter(demean_sipri_milex_inter_no_missing.iloc[:,0],Yhat[:,0]);
# plt.savefig(figures_dir + "/Milex_Actual_Predicted_Scatter.pdf")

# plt.plot(demean_sipri_milex_inter_no_missing.index, demean_sipri_milex_inter_no_missing.iloc[:,0]);
# plt.plot(demean_sipri_milex_inter_no_missing.index, Yhat[:,0]);
# plt.savefig(figures_dir + "/Milex_Actual_Predicted_Line_1.pdf")

# plt.plot(demean_sipri_milex_inter_no_missing.index, demean_sipri_milex_inter_no_missing.iloc[:,1]);
# plt.plot(demean_sipri_milex_inter_no_missing.index, Yhat[:,1]);
# plt.savefig(figures_dir + "/Milex_Actual_Predicted_Line_2.pdf")

# For the lasso, the idea is to regress a country's military expenditure per GDP on that of all the other countries
# This takes a lot of manuevering to set up, but I did it
# We also have to deal with nasty missing values, for now I am using the interpolated post 1995/missing values dropped data
sipri_for_LASSO = (pd.MultiIndex.from_product([sipri_milex_inter_no_missing.reset_index()['Year'], sipri_milex_inter_no_missing.columns], names=['Year', 'Country'])
                                .to_frame()
                                .query('Country != "Year"')
                                .reset_index(drop=True))

# Longer version of sipri_milex_inter_no_missing for merging on the dependent variable
sipri_milex_inter_no_missing_long = (sipri_milex_inter_no_missing.stack()
                                                                 .reset_index())

# Merge on the dependent variable
sipri_for_LASSO = (sipri_for_LASSO.merge(sipri_milex_inter_no_missing_long)
                                  .rename(columns = {0: "Dep_Var_Spend"})
                                  .merge(sipri_milex_inter_no_missing.reset_index())
                                  .set_index(['Year', 'Country', 'Dep_Var_Spend'])
                                  .stack()
                                  .reset_index()
                                  .query('Country != level_3')
                                  .pivot(index = ['Year', 'Country', 'Dep_Var_Spend'], columns = 'level_3', values = 0)
                                  .reset_index()
                                  .rename_axis(None, axis="columns"))

# Save this dataset so we can use glmnet
sipri_for_LASSO.to_csv(data_dir + "/SIPRI_for_LASSO.csv", index=False)
