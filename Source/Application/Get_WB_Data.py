# Get_WBData.py 
# Retrives world bank data for requested indicators, countries, and time periods from the API

# Directory structure
import os
repo_dir = os.path.join(os.path.dirname( __file__ ), '../..')
input_dir = repo_dir + "/Input"
output_dir = repo_dir + "/Output"
figures_dir = output_dir + "/Figures"
tables_dir = output_dir + "/Tables"
sim_results_dir = output_dir + "/Sim_Results"

# Packages
import wbgapi as wb
import pandas as pd

# Get the list of world bank indicators from the csv file
indicators_list = (pd.read_csv(input_dir + '/wb_indicators_list.csv')
                    ['Indicator_Code']
                    .tolist())

# Load in the series specified in the file
wb_data = wb.data.DataFrame(indicators_list, skipAggs=True, numericTimeKeys=True)

wb_data.to_csv(input_dir + "/WB_Data.csv")
