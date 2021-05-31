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

# Load in some series
# Life expectancy at birth (total), GDP PC international PPP, GDP PC Current USD, GNI PC PPP, GNI PC Atlas Method, Survey Mean Income/Consumption Per Capita, Survey mean inc/cons for the bottom 40% of the population, ILO GDP per person employed, Net Foreign Assets Per Capita, Total Reserves, Poverty HCR 2011 PPP percent pop, Poverty HCR at national lines, Poverty gap 2011 PPP, Healthcare spending by government as a share of total healthcare spending
wb_data = wb.data.DataFrame(indicators_list, skipAggs=True, numericTimeKeys=True)

wb_data.to_csv(apps_dir + "/WB_Data.csv")