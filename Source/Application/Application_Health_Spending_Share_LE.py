# Application_Health_Spending_Share_LE.py
# Presents and application of the PC solution to measurment error with the relationship between the Government's Share of Health Spending and life expectancy

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

# Load in the WB data
wb_data = (pd.read_csv(input_dir + "/WB_Data.csv", index_col=['economy', 'series'])
             # Reshape and create a balanced panel
             .transpose()
             .stack(level = 'economy')
             .rename_axis(None, axis = 1)
             .rename(columns = {"SP.DYN.LE00.IN":"life_exp", "NY.GDP.PCAP.PP.CD":"gdp_pc", "NY.GNP.PCAP.PP.CD":"gnp_pc", "SI.SPR.PCAP":"survey_inc_con_pc", "SL.GDP.PCAP.EM.KD":"gdp_per_emp", "SH.XPD.GHED.CH.ZS":"govt_health_share_wb"})
             .rename_axis(['year', 'country']))

oecd_data = (pd.read_csv(input_dir + "/OECD_Govt_Share_Health_Spending.csv")
               .query('INDICATOR == "HEALTHEXP' and 'SUBJECT == "COMPULSORY' and 'MEASURE == "PC_HEALTH_EXP"' and 'FREQUENCY == "A"')
               .rename(columns={"LOCATION":"country", "TIME":"year", "Value":"govt_health_share_oecd"})
               .filter(['country', 'year', 'govt_health_share_oecd']))

# This step will help fill out the oecd panel
combos_to_merge = pd.MultiIndex.from_product([oecd_data['country'].unique(), oecd_data['year'].unique()], names = ['country', 'year']).to_frame().reset_index(drop = True)

# Balance the panel and interpolate values
oecd_data = (combos_to_merge.merge(oecd_data, how = 'left')
                            .set_index(['country', 'year']))

# Merge world bank and oecd data
merged_data = (wb_data.merge(oecd_data, how='inner', left_index=True, right_index=True))

print(merged_data)

# Take mean of world bank and oecd spending percentage or use one or the other if the other is missing
merged_data = (merged_data.assign(mean_govt_health_share=merged_data.loc[:, ["govt_health_share_wb", "govt_health_share_oecd"]].mean(axis=1))
                          # Sort and interpolate
                          .sort_index(level=['country', 'year'])
                          .interpolate(limit_area='inside')
                          .drop(columns=['govt_health_share_wb', 'govt_health_share_oecd'])
                          .dropna())

print(merged_data)

exit()

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