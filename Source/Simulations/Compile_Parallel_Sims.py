# Directory structure
import os
repo_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')
input_dir = repo_dir + "/Input"
output_dir = repo_dir + "/Output"
figures_dir = output_dir + "/Figures"
tables_dir = output_dir + "/Tables"
sim_results_dir = output_dir + "/Sim_Results"

# Packages
import pandas as pd
import glob

# Read all the csv files
output = pd.concat([pd.read_csv(f) for f in glob.glob(sim_results_dir + '/*_Parallel_Sim_Results.csv')], ignore_index = True)

# Save
output.to_csv(sim_results_dir + '/n_3000_results.csv', index = False)
