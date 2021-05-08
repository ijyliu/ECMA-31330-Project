
import os
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from stargazer.stargazer import Stargazer

data_dir = "~/Box/ECMA-31330-Project"
repo_dir = os.path.join(os.path.dirname( __file__ ), '../..')
output_dir = repo_dir + "/Output"
figures_dir = output_dir + "/Figures"
regressions_dir = output_dir + "/Regressions"

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
             # Linear interpolation
             .interpolate(limit_area='inside')
             .dropna())

print(wb_data)

# OLS for benchmark
# I'll go with the most basic measure of GDP per capita as the independent variable here
ols_benchmark = sm.OLS(wb_data['life_exp'], wb_data['gdp_pc']).fit()

# Decompose into matrices for PCA analysis
X = wb_data.drop(columns = 'life_exp').to_numpy()

# Perform the factor analysis
pca = PCA(n_components=1)
X = pca.fit_transform(X)

# Regress y, life_expectancy, on the single pca component and output the results
factor_regression = sm.OLS(wb_data['life_exp'], X).fit()

# Write regression table to LaTeX
with open(regressions_dir + "/gdp_life_exp_ols_factor.tex", "w") as f:
    f.write(Stargazer([ols_benchmark, factor_regression]).render_latex())
