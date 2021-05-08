
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data_dir = "~/Box/ECMA-31330-Project"

# Load in the data
wb_data = (pd.read_csv(data_dir + "/WB_Data.csv", index_col=['economy', 'series'])
             # Reshape and create a balanced panel
             .transpose()
             .stack(level = 'economy')
             .rename_axis(None, axis = 1)
             .rename({"SP.DYN.LE00.IN":"life_exp", "NY.GDP.PCAP.PP.CD":"gdp_pc", "NY.GNP.PCAP.PP.CD":"gnp_pc", "SI.SPR.PCAP":"survey_inc_con_pc", "SL.GDP.PCAP.EM.KD":"gdp_per_emp"})
             .rename_axis(['year','country'])
             # Sort by country for interpolation
             .sort_index(level=['country', 'year'])
             # Linear interpolation
             .interpolate(limit_area='inside')
             .dropna()
             )

print(wb_data)

X = wb_data.iloc[:, 0:3].to_numpy()
y = wb_data['SP.DYN.LE00.IN'].to_numpy()

# Perform the factor analysis
pca = PCA(n_components=1)
X = pca.fit_transform(X)

# Conduct instrumental variables regression
