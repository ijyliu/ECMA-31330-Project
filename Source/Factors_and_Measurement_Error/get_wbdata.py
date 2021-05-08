
import pandas as pd
import wbgapi as wb

# Data location
data_dir = "~/Box/ECMA-31330-Project"

# Load in some series
# Life expectancy at birth (total), GDP PC international PPP, GNI PC PPP, Survey Mean Income/Consumption Per Capita, ILO GDP per person employed, Net Foreign Assets Per Capita
wb_data = wb.data.DataFrame(['SP.DYN.LE00.IN', 'NY.GDP.PCAP.PP.CD', 'NY.GNP.PCAP.PP.CD', 'SI.SPR.PCAP', 'SL.GDP.PCAP.EM.KD', 'NW.NFA.PC'])

wb_data.to_csv(data_dir + "/WB_Data.csv")
