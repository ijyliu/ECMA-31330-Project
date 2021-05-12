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
import statsmodels.formula.api as smf

# Load in the WB data
wb_data = (pd.read_csv(input_dir + "/WB_Data.csv", index_col=['economy', 'series'])
             # Reshape and create a balanced panel
             .transpose()
             .stack(level = 'economy')
             .rename_axis(None, axis = 1)
             .rename(columns = {"SP.DYN.LE00.IN":"life_exp", "NY.GDP.PCAP.PP.CD":"gdp_pc", "NY.GNP.PCAP.PP.CD":"gnp_pc", "SI.SPR.PCAP":"survey_inc_con_pc", "SL.GDP.PCAP.EM.KD":"gdp_per_emp", "SH.XPD.GHED.CH.ZS":"govt_health_share_wb"})
             .rename_axis(['year', 'country'])
             .reset_index()
             .astype({'year': 'datetime64[ns]', 'country': 'str'}))

oecd_data = (pd.read_csv(input_dir + "/OECD_Govt_Share_Health_Spending.csv")
               .query('INDICATOR == "HEALTHEXP' and 'SUBJECT == "COMPULSORY' and 'MEASURE == "PC_HEALTH_EXP"' and 'FREQUENCY == "A"')
               .rename(columns={"LOCATION":"country", "TIME":"year", "Value":"govt_health_share_oecd"})
               .filter(['country', 'year', 'govt_health_share_oecd']))

# This step will help fill out the oecd panel
combos_to_merge = pd.MultiIndex.from_product([oecd_data['country'].unique(), oecd_data['year'].unique()], names = ['country', 'year']).to_frame().reset_index(drop = True)

# Balance the panel and interpolate values
oecd_data = (combos_to_merge.merge(oecd_data, how = 'left')
                            .reset_index(drop=True)
                            .astype({'year': 'datetime64[ns]', 'country': 'str'}))

# Merge world bank and oecd data
merged_data = (wb_data.merge(oecd_data, how='outer'))

# Take mean of world bank and oecd spending percentage or use one or the other if the other is missing
merged_data = (merged_data.assign(mean_govt_health_share=merged_data.loc[:, ["govt_health_share_wb", "govt_health_share_oecd"]].mean(axis=1))
                          # Sort and interpolate
                          .sort_index(level=['country', 'year'])
                          .interpolate(limit_area='inside')
                          .drop(columns=['govt_health_share_wb', 'govt_health_share_oecd'])
                          .dropna()
                          .set_index(['year', 'country']))

# Standardize all variables
# https://stackoverflow.com/questions/35723472/how-to-use-sklearn-fit-transform-with-pandas-and-return-dataframe-instead-of-num
std_data = pd.DataFrame(StandardScaler().fit_transform(merged_data), index=merged_data.index, columns=merged_data.columns)

print(std_data)

# Basic time series plot
plt.figure(figsize=(15,15))
plt.plot(std_data.reset_index().set_index('year')['mean_govt_health_share'])
plt.savefig(figures_dir + "/Govt_Health_Share_Time_Series.pdf")
plt.close()

# Exploring correlations between the variables
sns.heatmap(std_data.corr())
plt.savefig(figures_dir + "/LE_Health_Econ_Correlations.pdf")
plt.close()

# OLS for benchmark
ols_benchmark = sm.OLS(std_data['life_exp'], std_data['mean_govt_health_share']).fit()

# Many covariate OLS
ols_many_covariates = sm.OLS(std_data['life_exp'], pd.concat([std_data['mean_govt_health_share'], std_data['gdp_pc'], std_data['gnp_pc'], std_data['survey_inc_con_pc'], std_data['gdp_per_emp']], axis = 1)).fit()

# Panel Fixed Effects Regression for Benchmark
fixed_effects_results = smf.ols("life_exp ~ mean_govt_health_share + gdp_pc + gnp_pc + survey_inc_con_pc + gdp_per_emp + C(year) + C(country)", data = std_data.reset_index()).fit(cov_type='cluster', cov_kwds={'groups': std_data.reset_index()['country']})

# Decompose into matrix for PCA analysis
# This contains only the economic covariates
X = std_data.drop(columns = ['life_exp', 'mean_govt_health_share']).to_numpy()

# Perform the factor analysis
pca_model = pca()
pca_results = pca_model.fit_transform(X)

# Add PCA results to the dataframe
std_data = pd.concat([std_data.reset_index(), pca_results['PC'].iloc[:, 0].reset_index(drop = True)], axis = 1, names = [std_data.columns, 'PC1'])

# Plot the loadings
sns.heatmap(pca_results['loadings'], cmap='YlGnBu')
plt.savefig(figures_dir + "/Econ_Indicator_Loadings.pdf")
plt.close()

# Scree plot
pca_model.plot()
plt.savefig(figures_dir + "/Econ_Indicator_Share_Explained.pdf")
plt.close()

# Main PCR spec
partial_pc_regression = smf.ols("life_exp ~ mean_govt_health_share + PC1", data = std_data).fit()

# PCR with fixed effects
pc_fixed_effects_results = smf.ols("life_exp ~ mean_govt_health_share + PC1 + C(year) + C(country)", data = std_data.reset_index()).fit(cov_type='cluster', cov_kwds={'groups': std_data.reset_index()['country']})

# Regression table settings
reg_table = Stargazer([ols_benchmark, ols_many_covariates, fixed_effects_results, partial_pc_regression, pc_fixed_effects_results])
reg_table.dependent_variable_name("Life Expectancy at Birth (Years)")
reg_table.covariate_order(['mean_govt_health_share', 'gdp_pc', 'gnp_pc', 'survey_inc_con_pc', 'gdp_per_emp', 'PC1'])
reg_table.rename_covariates({"mean_govt_health_share":"Government Share of Health Expenditure", "gdp_pc":"GDP Per Capita PPP", "gnp_pc":"GNP Per Capita PPP", "survey_inc_con_pc":"Survey Income/Consumption Per Capita", "gdp_per_emp":"GDP Per Employed Person"})
# Fixed effects indicator
reg_table.add_line('Fixed Effects', ['No', 'No', 'Yes', 'No', 'Yes'])
reg_table.show_degrees_of_freedom(False)
reg_table.add_custom_notes(["All variables are standardized."])

# Write regression table to LaTeX
with open(regressions_dir + "/LE_Health_Econ_Regressions.tex", "w") as f:
    f.write(reg_table.render_latex())