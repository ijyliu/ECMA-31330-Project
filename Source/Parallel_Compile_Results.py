# Parallel_Compile_Results.py
# Collects all of individual simulations results files and combines them, then performs an analysis of the results

# Packages
import os
import pandas as pd

# Functions and objects
from ME_Setup import *

print('began compiling results')

# List target simulation results files
data_files = [data_dir + "/" + f for f in os.listdir(os.path.expanduser(data_dir)) if 'sim_results_' in f]

# File loading function from the list
def load_files(filenames):
	for filename in filenames:
		yield (pd.read_csv(filename))
		
# Compile the files
full_sim_results = pd.concat(load_files(data_files))

# Perform the analysis with of the simulations from the cluster
perform_analysis(full_sim_results, "cluster")

print('finished analysis')
