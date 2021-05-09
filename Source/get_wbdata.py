# Get_WBData.py 
# Retrives world bank data for requested indicators, countries, and time periods from the API

# Import objects from the setup file
from ME_Setup import *

# Packages
import wbgapi as wb

# Load in some series
# Life expectancy at birth (total), GDP PC international PPP, GNI PC PPP, Survey Mean Income/Consumption Per Capita, ILO GDP per person employed, Net Foreign Assets Per Capita
wb_data = wb.data.DataFrame(['SP.DYN.LE00.IN', 'NY.GDP.PCAP.PP.CD', 'NY.GNP.PCAP.PP.CD', 'SI.SPR.PCAP', 'SL.GDP.PCAP.EM.KD', 'NW.NFA.PC'], skipAggs=True, numericTimeKeys=True)

wb_data.to_csv(data_dir + "/WB_Data.csv")
