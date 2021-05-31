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
from statsmodels.sandbox.regression.gmm import IV2SLS 
import regex as re

# # Font settings for plots
# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Get the list of indicators
indicators_list = get_wb_ind_list()
# Create a list of covariates
# For some reason net foreign assets pc doesn't read in correctly
covariates_list = [variable for variable in indicators_list if variable != "SP.DYN.LE00.IN" and variable != "SH.XPD.GHED.CH.ZS" and variable != "NW.NFA.PC" and variable != "SH.XPD.CHEX.GD.ZS"]
raw_short_covariates_list = ['NY.GDP.PCAP.PP.CD', 'NY.GDP.PCAP.CD', 'NY.GNP.PCAP.PP.CD', 'NY.GNP.PCAP.CD', 'SL.GDP.PCAP.EM.KD']

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
# Also get rid of problematic periods and replace with underscores
covariates_list = [var_name.replace('.', '_') for var_name in covariates_list]
short_covariates_list = ["gdp_pc_ppp" if item == "NY.GDP.PCAP.PP.CD" else item for item in raw_short_covariates_list]
short_covariates_list = [var_name.replace('.', '_') for var_name in short_covariates_list]

# Dictionary for linking column names/variables to nice/written out version
variables_mapped_to_long = {"gdp_pc_ppp":"GDP Per Capita PPP (Current International $)", "NY_GDP_PCAP_CD":"GDP Per Capita (Current USD)", "NY_GNP_PCAP_PP_CD":"GNP Per Capita PPP (Current International $)", "NY_GNP_PCAP_CD":"GNP Per Capita (Current USD)", "SL_GDP_PCAP_EM_KD":"ILO GDP Per Person Employed", "life_exp":"Life Expectancy at Birth (All Population)", "govt_health_share":"Government Share of Health Expenditure"}
variables_mapped_to_short = {"gdp_pc_ppp":"GDP PC PPP", "NY_GDP_PCAP_CD":"GDP PC USD", "NY_GNP_PCAP_PP_CD":"GNP PC PPP", "NY_GNP_PCAP_CD":"GNP PC USD", "SL_GDP_PCAP_EM_KD":"ILO GDP Per Emp", "life_exp":"Life Expectancy", "govt_health_share":"Gov Health Share"}

# Flip sign on poverty and ODA measures
cols = np.logical_or(wb_data.columns.str.contains('POV'), wb_data.columns.str.contains('ODA'))
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

# Combined data
# Take mean of world bank and oecd spending percentage or use one or the other if the other is missing
combined_data = (merged_data.assign(govt_health_share=merged_data.loc[:, ["govt_health_share_wb", "govt_health_share_oecd"]].mean(axis=1)))

# WB only data
# The obove procedure is a little weird, so we can also use just the WB govt health share measure
wb_only_data = (merged_data.assign(govt_health_share = merged_data['govt_health_share_wb']))

# OECD only data
# Also do the similar with only the OECD measure
oecd_only_data = (merged_data.assign(govt_health_share = merged_data['govt_health_share_oecd']))

# Run the empirical analysis for dataset
# This could be using both the World Bank and OECD data or just the World Bank Data
def run_empirical_analysis(data, name, covariates):

    # Exploring correlations between the variables
    if covariates == short_covariates_list:
        data = data.filter(short_covariates_list + ['country', 'year', 'life_exp', 'govt_health_share', 'govt_health_share_wb', 'govt_health_share_oecd'])

    # Sort, interpolate and fill
    data = (data.sort_index(level=['country', 'year'])
                #.interpolate(limit_area='inside')
                .drop(columns=['govt_health_share_wb', 'govt_health_share_oecd'])
                .dropna()
                .set_index(['year', 'country']))

    print(name)
    print(data.apply(lambda x: x.count()))

    # Summary statistics table
    sum_stats = (data.describe()
                     .rename(columns = variables_mapped_to_long)
                     .transpose()
                     .reset_index()
                     .drop(columns = ['25%', '75%'])
                     .round(2)
                     .astype({'count': 'int32'})
                     .rename(columns = {"index":"Variable", "count":"Obs", "mean":"Mean", "std":"SD", "min":"Min", "50%":"Med", "max":"Max"}))
    # Ensure entire strings/columns get printed
    with pd.option_context('display.max_colwidth', -1):
        sum_stats.to_latex(tables_dir + '/sum_stats_' + name + '.tex', index = False, caption = "Summary Statistics", label = "Sum_Stats", column_format = 'l' + 'c'*(len(sum_stats.columns) - 1))

    # Standardize all variables
    # https://stackoverflow.com/questions/35723472/how-to-use-sklearn-fit-transform-with-pandas-and-return-dataframe-instead-of-num
    std_data = pd.DataFrame(StandardScaler().fit_transform(data), index=data.index, columns=data.columns)

    # Calculate the 'averaged' covariate measure, now that the standardization is done
    std_data['covariates_mean'] =  std_data[covariates].mean(axis = 1)

    # Basic time series plot
    #plt.figure(figsize=(15,15))
    #plt.rc('font', size=8) 
    #plt.plot(std_data.reset_index().set_index('year')['govt_health_share'])
    #plt.savefig(figures_dir + "/Govt_Health_Share_Time_Series_" + name + ".pdf")
    #plt.close()

    # Correlations map
    sns.set(font_scale=0.8)
    plt.subplots(figsize=(12, 10))
    sns.heatmap(std_data.rename(columns = variables_mapped_to_short).drop(columns = 'covariates_mean').corr())
    #plt.rc('font', size=2) 
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.savefig(figures_dir + "/LE_Health_Econ_Correlations_" + name + ".pdf")
    plt.close()

    # OLS for benchmark
    ols_benchmark = smf.ols("life_exp ~ govt_health_share", data = std_data.reset_index()).fit()

    # Single mismeasured covariate OLS
    ols_one_covariate = smf.ols("life_exp ~ govt_health_share + gdp_pc_ppp", data = std_data.reset_index()).fit()

    # Many covariate OLS
    # String format of covariates for patsy formulas
    covariates_formula_string = covariates[0]
    for i in range(1, len(covariates)):
        covariates_formula_string += " + " + covariates[i]
    ols_many_covariates = smf.ols("life_exp ~ govt_health_share + " + covariates_formula_string, data = std_data.reset_index()).fit()

    # Mean of standardized covariates OLS
    ols_mean_covariates = smf.ols("life_exp ~ govt_health_share + covariates_mean", data = std_data.reset_index()).fit()

    # Decompose into matrix for PCA analysis
    # This contains only the economic covariates
    X = std_data[covariates].to_numpy()

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
    loadings.columns = [variables_mapped_to_short[item] for item in covariates]
    sns.heatmap(loadings, cmap='YlGnBu')
    plt.yticks(rotation=0)
    #plt.xticks(rotation=90)
    plt.savefig(figures_dir + "/Econ_Indicator_Loadings_" + name + ".pdf")
    plt.close()

    # Scree plot
    pca_model.plot()
    plt.rc('font', size=12)
    plt.title('')
    plt.ylabel('Share of Variance Explained, Cumulative Share of Variance Explained')
    plt.xlabel('Principal Component')
    plt.savefig(figures_dir + "/Econ_Indicator_Share_Explained_" + name + ".pdf")
    plt.close()

    # Main PCR spec
    partial_pc_regression = smf.ols("life_exp ~ govt_health_share + PC1", data = std_data).fit()

    # Regression table settings
    reg_table = Stargazer([ols_benchmark, ols_one_covariate, ols_many_covariates, ols_mean_covariates, partial_pc_regression])
    reg_table.title("Regressions of Life Expectancy on Government Share of Health Spending \label{main_regs}")
    reg_table.dependent_variable_name("Life Expectancy at Birth (Years)")
    reg_table.covariate_order(['govt_health_share'])
    reg_table.rename_covariates({"govt_health_share":"Govt. Share of Health Exp."})
    # Fixed effects indicator
    reg_table.add_line('Covariates', ['None', 'Single', 'All', 'Average of', 'PCA'])
    reg_table.add_line('', ['', 'Measurement', 'Measurements', 'Measurements', ''])
    reg_table.add_line('', ['', '(GDP Per', '', '', ''])
    reg_table.add_line('', ['', 'Capita PPP)', '', '', ''])
    reg_table.show_degrees_of_freedom = False
    reg_table.show_r2 = False
    reg_table.show_adj_r2 = False
    reg_table.show_residual_std_err = False
    reg_table.show_f_statistic = False
    #reg_table.add_custom_notes(["All variables are standardized. All columns make use of robust standard errors."])

    # Write regression table to LaTeX
    with open(tables_dir + "/LE_Health_Econ_Regressions_" + name + ".tex", "w") as f:
        corrected_table = re.sub('\\cline{[0-9\-]+}', '', reg_table.render_latex())
        corrected_table = re.sub("begin{tabular}", r"scalebox{0.75}{\\begin{tabular}", corrected_table)
        corrected_table = re.sub("end{tabular}", "end{tabular}}", corrected_table)
        corrected_table = re.sub("Covariates", "\hline \\\\\\[-1.8ex]\n  Covariates", corrected_table)
        f.write(corrected_table)

    # Additional results

    # Fixed effects
    # Panel Fixed Effects Regression for Benchmark
    fixed_effects_results = smf.ols("life_exp ~ govt_health_share + " + covariates_formula_string + " + C(year) + C(country)", data = std_data.reset_index()).fit(cov_type='cluster', cov_kwds={'groups': std_data.reset_index()['country']})
    # PCR with fixed effects
    pc_fixed_effects_results = smf.ols("life_exp ~ govt_health_share + PC1 + C(year) + C(country)", data = std_data.reset_index()).fit(cov_type='cluster', cov_kwds={'groups': std_data.reset_index()['country']})
    # Use more principal components (this gets at a large share of the variance)
    if covariates == covariates_list:
        more_pcs_results = smf.ols("life_exp ~ govt_health_share + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7", data = std_data).fit()
    else:
        more_pcs_results = smf.ols("life_exp ~ govt_health_share + PC1 + PC2", data = std_data).fit()
    # Instrumental variables- instrument GDP per capita (probably mismeasured) on all the other development indicators
    #iv_instruments_string = covariates_formula_string.replace('gdp_pc_ppp + ', '')
    # Create the predicted value of gdp_pc_ppp
    #std_data['pred_gdp_pc_ppp'] = smf.ols("gdp_pc_ppp ~ " + iv_instruments_string, data = std_data).fit().predict()
    #iv_results = smf.ols("life_exp ~ govt_health_share + pred_gdp_pc_ppp", data = std_data.reset_index()).fit()
    iv_no_gdp = [item for item in covariates if item != 'gdp_pc_ppp']
    iv_no_gdp.append('govt_health_share')
    #print(iv_no_gdp)
    # print(std_data.columns)
    # print(covariates)
    # print(std_data['life_exp'])
    # print(std_data[covariates])
    # print(std_data[iv_exog])
    # ugh, so confusing: https://stackoverflow.com/questions/37012110/how-to-do-2sls-iv-regression-using-statsmodels-python
    iv_results = IV2SLS(endog = std_data['life_exp'], exog = std_data[['govt_health_share', 'gdp_pc_ppp']], instrument = std_data[iv_no_gdp]).fit()

    # Regression table settings
    print(more_pcs_results.summary())
    print(iv_results.summary())
    print(iv_results.params)
    print(iv_results.cov_type)
    print(iv_results.bse)
    additional_reg_table = Stargazer([fixed_effects_results, pc_fixed_effects_results, more_pcs_results, iv_results])
    additional_reg_table.title("Additional Regressions \label{additional_regs}")
    additional_reg_table.dependent_variable_name("Life Expectancy at Birth (Years)")
    additional_reg_table.covariate_order(['govt_health_share'])
    additional_reg_table.rename_covariates({"govt_health_share":"Govt. Share of Health Exp."})
    # Fixed effects indicator
    if covariates == covariates_list:
        additional_reg_table.add_line('Covariates', ['None', 'PCA', 'PC 1-7', 'Instrumental'])
        additional_reg_table.add_line('', ['', '', '', 'Variable'])
        additional_reg_table.add_line('', ['', '', '', '(GDP Per'])
        additional_reg_table.add_line('', ['', '', '', 'Capita PPP)'])
    else:
        additional_reg_table.add_line('Covariates', ['None', 'PCA', 'PC 1-2', 'Instrumental Variable'])
        additional_reg_table.add_line('', ['', '', '', '(GDP Per'])
        additional_reg_table.add_line('', ['', '', '', 'Capita PPP)'])
    additional_reg_table.add_line('Fixed Effects', ['Yes', 'Yes', 'No', 'No'])
    additional_reg_table.show_degrees_of_freedom(False)
    additional_reg_table.show_r2 = False 
    additional_reg_table.show_adj_r2 = False
    additional_reg_table.show_residual_std_err = False
    additional_reg_table.show_f_statistic = False
    #additional_reg_table.add_custom_notes(["All variables are standardized. \nFixed effects columns make use of country clustered standard errors: \nothers use robust standard errors."])

    # Write regression table to LaTeX
    with open(tables_dir + "/Additional_LE_Health_Econ_Regressions_" + name + ".tex", "w") as f:
        corrected_table = re.sub('\\cline{[0-9\-]+}', '', additional_reg_table.render_latex())
        #corrected_table_added_iv_coeff = re.sub(pattern = r'& \\', repl = '& '+ str(iv_results.params['govt_health_share']) + r' \\', string = corrected_table, count = 0)
        #corrected_table_added_iv = re.sub(pattern = r'& \\', repl = '& ('+ str(iv_results.bse['govt_health_share']) + r') \\', string = corrected_table_added_iv_coeff, count = 0)
        corrected_table = re.sub("begin{tabular}", r"scalebox{0.75}{\\begin{tabular}", corrected_table)
        corrected_table = re.sub("end{tabular}", "end{tabular}}", corrected_table)
        corrected_table = re.sub("Covariates", "\hline \\\\\\[-1.8ex]\n  Covariates", corrected_table)
        f.write(corrected_table)

# Do the analysis on the two datasets and all the covariates combos
#run_empirical_analysis(data = combined_data, name = "combined_full", covariates=covariates_list)
#run_empirical_analysis(data = wb_only_data, name = "wb_only_full", covariates=covariates_list)
#run_empirical_analysis(data = oecd_only_data, name = "oecd_only_full", covariates=covariates_list)
run_empirical_analysis(data = combined_data, name = "combined_short", covariates=short_covariates_list)
run_empirical_analysis(data = wb_only_data, name = "wb_only_short", covariates=short_covariates_list)
run_empirical_analysis(data = oecd_only_data, name = "oecd_only_short", covariates=short_covariates_list)
