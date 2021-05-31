# Get_WBData.py 
# Retrives world bank data for requested indicators, countries, and time periods from the API

# Import objects from the setup file
import os
import sys
sys.path.append(os.path.expanduser('~/repo/ECMA-31330-Project/Source'))
from ME_Setup import *

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