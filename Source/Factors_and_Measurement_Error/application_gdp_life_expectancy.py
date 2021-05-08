
import numpy as np
import pandas as pd
import wbgapi as wb

# Load in some series
# Life expectancy at birth (total), GDP PC international PPP, GNI PC PPP, Survey Mean Income/Consumption Per Capita, ILO GDP per person employed, Net Foreign Assets Per Capita
wb_data = wb.data.DataFrame(['SP.DYN.LE00.IN', 'NY.GDP.PCAP.PP.CD', 'NY.GNP.PCAP.PP.CD', 'SI.SPR.PCAP', 'SL.GDP.PCAP.EM.KD', 'NW.NFA.PC'])

print(wb_data)
