# Application_GDP_Life_Expectancy.py
# Presents and application of the factor solution to measurment error with the relationship between GDP and life expectancy

# Import objects from the setup file
import os
import sys
sys.path.append(os.path.expanduser('~/repo/ECMA-31330-Project/Source'))
from ME_Setup import *

# Packages
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from pca import pca
from stargazer.stargazer import Stargazer
from pandas.plotting import scatter_matrix
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

# Load in the data
wb_data = (pd.read_csv(data_dir + "/WB_Data.csv", index_col=['economy', 'series'])
             # Reshape and create a balanced panel
             .transpose()
             .stack(level = 'economy')
             .rename_axis(None, axis = 1)
             .rename(columns = {"SP.DYN.LE00.IN":"life_exp", "NY.GDP.PCAP.PP.CD":"gdp_pc", "NY.GNP.PCAP.PP.CD":"gnp_pc", "SI.SPR.PCAP":"survey_inc_con_pc", "SL.GDP.PCAP.EM.KD":"gdp_per_emp"})
             .rename_axis(['year','country'])
             # Sort by country for interpolation
             .sort_index(level=['country', 'year'])
             # Linear interpolation and dropping of missing values
             .interpolate(limit_area='inside')
             .dropna())

print(wb_data)

# Standardize all variables
# https://stackoverflow.com/questions/35723472/how-to-use-sklearn-fit-transform-with-pandas-and-return-dataframe-instead-of-num
std_wb_data = pd.DataFrame(StandardScaler().fit_transform(wb_data), index=wb_data.index, columns=wb_data.columns)

print(std_wb_data)

# Basic time series plot
plt.figure(figsize=(15,15))
plt.plot(std_wb_data.reset_index().set_index('year')['gdp_pc'], std_wb_data.reset_index().set_index('year')['life_exp'])
plt.savefig(figures_dir + "/GDP_PC_LE_Time_Series.pdf")
plt.close()

# Exploring correlations between the variables
sns.heatmap(std_wb_data.corr())
plt.savefig(figures_dir + "/GDP_LE_Correlations.pdf")
plt.close()

# OLS for benchmark
# I'll go with the most basic measure of GDP per capita as the independent variable here
ols_benchmark = sm.OLS(std_wb_data['life_exp'], std_wb_data['gdp_pc']).fit()

# Decompose into matrix for PCA analysis
X = std_wb_data.drop(columns = 'life_exp').to_numpy()

# Perform the factor analysis
pca_model = pca()
pca_results = pca_model.fit_transform(X)

# Plot the loadings
sns.heatmap(pca_results['loadings'], cmap='YlGnBu')
plt.savefig(figures_dir + "/GDP_LE_Loadings.pdf")
plt.close()

# Scree plot
pca_model.plot()
plt.savefig(figures_dir + "/GDP_LE_Share_Explained.pdf")
plt.close()

# Regress y, life_expectancy, on the first pca component and output the results
factor_regression = sm.OLS(std_wb_data['life_exp'].reset_index(drop = True), pca_results['PC'].iloc[:, 0].reset_index(drop = True)).fit()

# Regression table settings
reg_table = Stargazer([ols_benchmark, factor_regression])
reg_table.dependent_variable_name("Life Expectancy at Birth (Years)")
reg_table.rename_covariates({"gdp_pc":"GDP Per Capita, PPP"})
reg_table.show_degrees_of_freedom(False)
reg_table.add_custom_notes(["All variables are standardized."])

# Write regression table to LaTeX
with open(regressions_dir + "/gdp_life_exp_ols_factor.tex", "w") as f:
    f.write(reg_table.render_latex())