# Application_Gov_Health_Spending_Share_LE.py
# Presents an application of the PCA solution to measurement error with the relationship between the government's share of health spending and life expectancy

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
from statsmodels.sandbox.regression.gmm import IV2SLS 
import regex as re

# Load in the WB data
wb_data = (pd.read_csv(input_dir + "/WB_Data.csv", index_col=['economy', 'series'])
             # Reshape and rename a few columns
             .transpose()
             .stack(level = 'economy')
             .rename_axis(None, axis = 1)
             .rename(columns = {"SP.DYN.LE00.IN":"life_exp", "SH.XPD.GHED.CH.ZS":"govt_health_share", "NY.GDP.PCAP.PP.CD":"gdp_pc_ppp"})
             .rename_axis(['year', 'country'])
             .reset_index()
             .astype({'year': 'int', 'country': 'str'})
             .set_index(['year', 'country'])
             # Toss rows with any missing data
             .dropna())

# Remove periods from column names for convenience
wb_data.columns = wb_data.columns.str.replace(".", "_")

# Create a list of the key covariates for convenience
short_covariates_list = ['gdp_pc_ppp', 'NY_GDP_PCAP_CD', 'NY_GNP_PCAP_PP_CD', 'NY_GNP_PCAP_CD', 'SL_GDP_PCAP_EM_KD']
# String format of covariates for patsy formulas
covariates_formula_string = short_covariates_list[0]
for i in range(1, len(short_covariates_list)):
    covariates_formula_string += " + " + short_covariates_list[i]

# Dictionary for linking column names/variables to nice/written out version
variables_mapped_to_long = {"gdp_pc_ppp":"GDP Per Capita PPP (Current International $)", "NY_GDP_PCAP_CD":"GDP Per Capita (Current USD)", "NY_GNP_PCAP_PP_CD":"GNP Per Capita PPP (Current International $)", "NY_GNP_PCAP_CD":"GNP Per Capita (Current USD)", "SL_GDP_PCAP_EM_KD":"ILO GDP Per Person Employed", "life_exp":"Life Expectancy at Birth (All Population)", "govt_health_share":"Government Share of Health Expenditure"}
variables_mapped_to_short = {"gdp_pc_ppp":"GDP PC PPP", "NY_GDP_PCAP_CD":"GDP PC USD", "NY_GNP_PCAP_PP_CD":"GNP PC PPP", "NY_GNP_PCAP_CD":"GNP PC USD", "SL_GDP_PCAP_EM_KD":"ILO GDP Per Emp", "life_exp":"Life Expectancy", "govt_health_share":"Gov Health Share"}

# Run the empirical analysis

# Summary statistics table
sum_stats = (wb_data.describe()
                    .rename(columns = variables_mapped_to_long)
                    .transpose()
                    .reset_index()
                    .drop(columns = ['25%', '75%'])
                    .round(2)
                    #.astype({'count': 'int32'})
                    .rename(columns = {"index":"Variable", "count":"Obs", "mean":"Mean", "std":"SD", "min":"Min", "50%":"Med", "max":"Max"}))
# Ensure entire strings/columns get printed
with pd.option_context('display.max_colwidth', -1):

    # Render the sum stats table as a string
    sum_stats_latex = sum_stats.to_latex(index = False, caption = "Summary Statistics", label = "Sum_Stats", column_format = 'l' + 'c'*(len(sum_stats.columns) - 1), float_format = "{:,.10g}".format)

    # Write the a corrected (scaled down by 0.75) version to the file
    with open(tables_dir + '/sum_stats_wb_only_short.tex', "w") as f:
        corrected_table = re.sub("begin{tabular}", r"scalebox{0.75}{\\begin{tabular}", sum_stats_latex)
        corrected_table = re.sub("end{tabular}", "end{tabular}}", corrected_table)
        f.write(corrected_table)

# Standardize all variables
# https://stackoverflow.com/questions/35723472/how-to-use-sklearn-fit-transform-with-pandas-and-return-wb_dataframe-instead-of-num
std_data = pd.DataFrame(StandardScaler().fit_transform(wb_data), index=wb_data.index, columns=wb_data.columns)

# Calculate the 'averaged' covariate measure, now that the standardization is done
std_data['covariates_mean'] =  std_data[short_covariates_list].mean(axis = 1)

# Correlations map
sns.set(font_scale=0.8)
plt.subplots(figsize=(12, 10))
sns.heatmap(std_data.rename(columns = variables_mapped_to_short).drop(columns = 'covariates_mean').corr())
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.savefig(figures_dir + "/LE_Health_Econ_Correlations_wb_only_short.pdf")
plt.close()

# OLS for benchmark
ols_benchmark = smf.ols("life_exp ~ -1 + govt_health_share", data = std_data.reset_index()).fit()

# Single mismeasured covariate OLS
ols_one_covariate = smf.ols("life_exp ~ -1 + govt_health_share + gdp_pc_ppp", data = std_data.reset_index()).fit()

# Many covariate OLS
ols_many_covariates = smf.ols("life_exp ~ -1 + govt_health_share + " + covariates_formula_string, data = std_data.reset_index()).fit()

# Mean of standardized covariates OLS
ols_mean_covariates = smf.ols("life_exp ~ -1 + govt_health_share + covariates_mean", data = std_data.reset_index()).fit()

# Decompose into matrix for PCA analysis
# This contains only the economic covariates
X = std_data[short_covariates_list].to_numpy()

# Perform the factor analysis
pca_model = pca()
pca_results = pca_model.fit_transform(X)

# Count number of PCs
num_pcs = pca_results['PC'].shape[1]
PC_names = ['PC' + str(i + 1) for i in range(num_pcs)]

# Add PCA results to the dataframe
std_data = pd.concat([std_data.reset_index(), pca_results['PC'].reset_index(drop = True)], axis = 1, names = [std_data.columns, PC_names])

# Plot the loadings
sns.set(font_scale=0.85)
plt.subplots(figsize=(12, 10))
loadings = pca_results['loadings']
loadings.columns = [variables_mapped_to_short[item] for item in short_covariates_list]
sns.heatmap(loadings, cmap='YlGnBu')
plt.yticks(rotation=0)
plt.savefig(figures_dir + "/Econ_Indicator_Loadings_wb_only_short.pdf")
plt.close()

# Scree plot
pca_model.plot()
plt.rc('font', size=12)
plt.title('')
plt.ylabel('Share of Variance Explained, Cumulative Share of Variance Explained')
plt.xlabel('Principal Component')
plt.savefig(figures_dir + "/Econ_Indicator_Share_Explained_wb_only_short.pdf")
plt.close()

# Main PCR spec
partial_pc_regression = smf.ols("life_exp ~ -1 + govt_health_share + PC1", data = std_data).fit()

# Regression table settings
reg_table = Stargazer([ols_benchmark, ols_one_covariate, ols_many_covariates, ols_mean_covariates, partial_pc_regression])
reg_table.title("Regressions of Life Expectancy on Government Share of Health Spending \label{main_regs}")
reg_table.dependent_variable_name("Life Expectancy at Birth (Years)")
reg_table.covariate_order(['govt_health_share'])
reg_table.rename_covariates({"govt_health_share":"Govt. Share of Health Exp."})
reg_table.add_line('Covariates', ['None', 'Single', 'All', 'Average of', 'PCA'])
reg_table.add_line('', ['', 'Measurement', 'Measurements', 'Measurements', ''])
reg_table.add_line('', ['', '(GDP Per', '', '', ''])
reg_table.add_line('', ['', 'Capita PPP)', '', '', ''])
reg_table.show_degrees_of_freedom = False
reg_table.show_r2 = False
reg_table.show_adj_r2 = False
reg_table.show_residual_std_err = False
reg_table.show_f_statistic = False

# Write regression table to LaTeX
with open(tables_dir + "/LE_Health_Econ_Regressions_wb_only_short.tex", "w") as f:
    corrected_table = re.sub('\\cline{[0-9\-]+}', '', reg_table.render_latex())
    corrected_table = re.sub("begin{tabular}", r"scalebox{0.75}{\\begin{tabular}", corrected_table)
    corrected_table = re.sub("end{tabular}", "end{tabular}}", corrected_table)
    corrected_table = re.sub("Covariates", "\hline \\\\\\[-1.8ex]\n  Covariates", corrected_table)
    f.write(corrected_table)

# Additional results

# Fixed effects
# Panel Fixed Effects Regression for Benchmark
fixed_effects_results = smf.ols("life_exp ~ -1 + govt_health_share + " + covariates_formula_string + " + C(year) + C(country)", data = std_data.reset_index()).fit(cov_type='cluster', cov_kwds={'groups': std_data.reset_index()['country']})
# PCR with fixed effects
pc_fixed_effects_results = smf.ols("life_exp ~ -1 + govt_health_share + PC1 + C(year) + C(country)", data = std_data.reset_index()).fit(cov_type='cluster', cov_kwds={'groups': std_data.reset_index()['country']})
# Use more principal components (this gets at a large share of the variance)
more_pcs_results = smf.ols("life_exp ~ -1 + govt_health_share + PC1 + PC2", data = std_data).fit()
# Instrumental variables- instrument GDP per capita (probably mismeasured) on all the other development indicators
iv_no_gdp = [item for item in short_covariates_list if item != 'gdp_pc_ppp']
iv_no_gdp.append('govt_health_share')
# Ugh, so confusing: https://stackoverflow.com/questions/37012110/how-to-do-2sls-iv-regression-using-statsmodels-python
iv_results = IV2SLS(endog = std_data['life_exp'], exog = std_data[['govt_health_share', 'gdp_pc_ppp']], instrument = std_data[iv_no_gdp]).fit()

# Regression table settings
additional_reg_table = Stargazer([fixed_effects_results, pc_fixed_effects_results, more_pcs_results, iv_results])
additional_reg_table.title("Additional Regressions \label{additional_regs}")
additional_reg_table.dependent_variable_name("Life Expectancy at Birth (Years)")
additional_reg_table.covariate_order(['govt_health_share'])
additional_reg_table.rename_covariates({"govt_health_share":"Govt. Share of Health Exp."})
additional_reg_table.add_line('Covariates', ['None', 'PCA', 'PC 1-2', 'Instrumental Variable'])
additional_reg_table.add_line('', ['', '', '', '(GDP Per'])
additional_reg_table.add_line('', ['', '', '', 'Capita PPP)'])
additional_reg_table.add_line('Fixed Effects', ['Yes', 'Yes', 'No', 'No'])
additional_reg_table.show_degrees_of_freedom(False)
additional_reg_table.show_r2 = False 
additional_reg_table.show_adj_r2 = False
additional_reg_table.show_residual_std_err = False
additional_reg_table.show_f_statistic = False

# Write regression table to LaTeX
with open(tables_dir + "/Additional_LE_Health_Econ_Regressions_wb_only_short.tex", "w") as f:
    corrected_table = re.sub('\\cline{[0-9\-]+}', '', additional_reg_table.render_latex())
    corrected_table = re.sub("begin{tabular}", r"scalebox{0.75}{\\begin{tabular}", corrected_table)
    corrected_table = re.sub("end{tabular}", "end{tabular}}", corrected_table)
    corrected_table = re.sub("Covariates", "\hline \\\\\\[-1.8ex]\n  Covariates", corrected_table)
    f.write(corrected_table)
