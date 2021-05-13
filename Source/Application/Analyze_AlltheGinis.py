# Analyze_AlltheGinis.py
# Try some correlations, etc with the AlltheGinis dataset

# Import objects from the setup file
import os
import sys
sys.path.append(os.path.expanduser('~/repo/ECMA-31330-Project/Source'))
from ME_Setup import *

# Packages
import pandas as pd

all_the_ginis = (pd.read_excel(apps_dir + '/allginis_2013.xls', sheet_name = 'data'))

columns_to_keep = [variable for variable in all_the_ginis.columns if 'gini' in variable] + ['Giniall', 'year', 'contcod']

all_the_ginis = (all_the_ginis.filter(columns_to_keep)
                              .set_index(['contcod', 'year'])
                              .sort_index(level=['contcod', 'year'])
                              .interpolate(limit_area = 'inside')
                              .dropna())

std_data = pd.DataFrame(StandardScaler().fit_transform(all_the_ginis), index=all_the_ginis.index, columns=all_the_ginis.columns)

# Exploring correlations between the variables
sns.set(font_scale=0.25)
sns.heatmap(std_data.corr())
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.savefig(figures_dir + "/All_the_Ginis_Correlations.pdf")
plt.close()

print(std_data)
