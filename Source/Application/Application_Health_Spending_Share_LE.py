# Application_Health_Spending_Share_LE.py
# Presents and application of the PC solution to measurment error with the relationship between the Government's Share of Health Spending and life expectancy

# Import objects from the setup file
import os
import sys
sys.path.append(os.path.expanduser('~/repo/ECMA-31330-Project/Source'))
from ME_Setup import *

# Packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pca import pca
from stargazer.stargazer import Stargazer
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import regex as re

# Get the list of indicators
indicators_list = get_wb_ind_list()
# Create a list of covariates
# For some reason net foreign assets pc doesn't read in correctly
covariates_list = [variable for variable in indicators_list if variable != "SP.DYN.LE00.IN" and variable != "SH.XPD.GHED.CH.ZS" and variable != "NW.NFA.PC" and variable != "SH.XPD.CHEX.GD.ZS"]

# Load in the WB data
wb_data = (pd.read_csv(apps_dir + "/WB_Data.csv", index_col=['economy', 'series'])
             # Reshape and rename a few columns
             .transpose()
             .stack(level = 'economy')
             .rename_axis(None, axis = 1)
             .rename(columns = {"SP.DYN.LE00.IN":"life_exp", "SH.XPD.GHED.CH.ZS":"govt_health_share_wb", "NY.GDP.PCAP.PP.CD":"gdp_pc_ppp"})
             # Get rid of old health GDP share measure
             .drop(columns='SH.XPD.CHEX.GD.ZS')
             .rename_axis(['year', 'country'])
             .reset_index()
             .astype({'year': 'int', 'country': 'str'}))

# Fix covariates list with renamed gdp_pc_ppp
covariates_list = ["gdp_pc_ppp" if item == "NY.GDP.PCAP.PP.CD" else item for item in covariates_list]

# Flip sign on poverty measures
cols = wb_data.columns.str.contains('POV')
wb_data.loc[:, cols] = wb_data.loc[:, cols].mul(-1)

# Remove periods from column names
wb_data.columns = wb_data.columns.str.replace(".", "_")

oecd_data = (pd.read_csv(apps_dir + "/OECD_Govt_Share_Health_Spending.csv")
               .query('INDICATOR == "HEALTHEXP' and 'SUBJECT == "COMPULSORY' and 'MEASURE == "PC_HEALTH_EXP"' and 'FREQUENCY == "A"')
               .rename(columns={"LOCATION":"country", "TIME":"year", "Value":"govt_health_share_oecd"})
               .filter(['country', 'year', 'govt_health_share_oecd']))

# This step will help fill out the oecd panel
combos_to_merge = (pd.MultiIndex.from_product([oecd_data['country'].unique(), oecd_data['year'].unique()], names = ['country', 'year'])
                                .to_frame()
                                .reset_index(drop = True))

# Balance the panel and interpolate values
oecd_data = (combos_to_merge.merge(oecd_data, how = 'left')
                            .reset_index(drop=True)
                            .astype({'year': 'int', 'country': 'str'}))

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

# Calculate the 'averaged' covariate measure, now that the standardization is done
covariates_list = [var_name.replace('.', '_') for var_name in covariates_list]
print(std_data)
std_data['covariates_mean'] =  std_data[covariates_list].mean(axis = 1)

# Basic time series plot
plt.figure(figsize=(15,15))
plt.plot(std_data.reset_index().set_index('year')['mean_govt_health_share'])
plt.savefig(figures_dir + "/Govt_Health_Share_Time_Series.pdf")
plt.close()

# Exploring correlations between the variables
sns.set(font_scale=0.25)
sns.heatmap(std_data.corr())
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.savefig(figures_dir + "/LE_Health_Econ_Correlations.pdf")
plt.close()

# OLS for benchmark
ols_benchmark = smf.ols("life_exp ~ mean_govt_health_share", data = std_data.reset_index()).fit()

# Single mismeasured covariate OLS
ols_one_covariate = smf.ols("life_exp ~ mean_govt_health_share + gdp_pc_ppp", data = std_data.reset_index()).fit()

# Many covariate OLS
# String format of covariates for patsy formulas
covariates_formula_string = covariates_list[0]
for i in range(1, len(covariates_list)):
    covariates_formula_string += " + " + covariates_list[i]
ols_many_covariates = smf.ols("life_exp ~ mean_govt_health_share + " + covariates_formula_string, data = std_data.reset_index()).fit()

# Mean of standardized covariates OLS
ols_mean_covariates = smf.ols("life_exp ~ mean_govt_health_share + covariates_mean", data = std_data.reset_index()).fit()

# Panel Fixed Effects Regression for Benchmark
fixed_effects_results = smf.ols("life_exp ~ mean_govt_health_share + " + covariates_formula_string + " + C(year) + C(country)", data = std_data.reset_index()).fit(cov_type='cluster', cov_kwds={'groups': std_data.reset_index()['country']})

# Decompose into matrix for PCA analysis
# This contains only the economic covariates
X = std_data[covariates_list].to_numpy()

# Perform the factor analysis
pca_model = pca()
pca_results = pca_model.fit_transform(X)

# Count number of PCs
num_pcs = pca_results['PC'].shape[1]
PC_names = ['PC' + str(i + 1) for i in range(num_pcs)]

# Add PCA results to the dataframe
std_data = pd.concat([std_data.reset_index(), pca_results['PC'].reset_index(drop = True)], axis = 1, names = [std_data.columns, PC_names])

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
reg_table = Stargazer([ols_benchmark, ols_one_covariate, ols_many_covariates, ols_mean_covariates, partial_pc_regression])
reg_table.dependent_variable_name("Life Expectancy at Birth (Years)")
reg_table.covariate_order(['mean_govt_health_share'])
reg_table.rename_covariates({"mean_govt_health_share":"Govt. Share of Health Exp."})
# Fixed effects indicator
reg_table.add_line('Covariates', ['None', 'GDP PC', 'Econ Indicators', 'Mean', 'PCs'])
reg_table.show_degrees_of_freedom(False)
reg_table.add_custom_notes(["All variables are standardized."])

# Write regression table to LaTeX
with open(tables_dir + "/LE_Health_Econ_Regressions.tex", "w") as f:
    corrected_table = re.sub('\\cline{[0-9\-]+}', '', reg_table.render_latex())
    f.write(corrected_table)