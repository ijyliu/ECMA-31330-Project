# Get_WBData.py 
# Retrives world bank data for requested indicators, countries, and time periods from the API

# Import objects from the setup file
import os
import sys
sys.path.append(os.path.expanduser('~/repo/ECMA-31330-Project/Source'))
from ME_Setup import *

# Packages
import wbgapi as wb

# Load in some series
# Life expectancy at birth (total), GDP PC international PPP, GDP PC Current USD, GNI PC PPP, GNI PC Atlas Method, Survey Mean Income/Consumption Per Capita, Survey mean inc/cons for the bottom 40% of the population, ILO GDP per person employed, Net Foreign Assets Per Capita, Total Reserves, Poverty HCR 2011 PPP percent pop, Poverty HCR at national lines, Poverty gap 2011 PPP, Healthcare spending by government as a share of total healthcare spending
wb_data = wb.data.DataFrame(['SP.DYN.LE00.IN', 'NY.GDP.PCAP.PP.CD', 'NY.GDP.PCAP.CD', 'NY.GNP.PCAP.PP.CD', 'NY.GNP.PCAP.CD', 'SI.SPR.PCAP', 'SI.SPR.PC40', 'SL.GDP.PCAP.EM.KD', 'NW.NFA.PC', 'FI.RES.TOTL.CD', 'SI.POV.DDAY', 'SI.POV.NAHC', 'SI.POV.GAPS', 'SH.XPD.GHED.CH.ZS'], skipAggs=True, numericTimeKeys=True)

wb_data.to_csv(input_dir + "/WB_Data.csv")