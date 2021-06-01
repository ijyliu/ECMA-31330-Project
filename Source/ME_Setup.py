# ME_Setup.py
# Defines the project directory structure as well as other shared functions

# Packages
import os

# Directory structure
repo_dir = os.path.join(os.path.dirname( __file__ ), '..')
input_dir = repo_dir + "/Input"
output_dir = repo_dir + "/Output"
figures_dir = output_dir + "/Figures"
tables_dir = output_dir + "/Tables"
sim_results_dir = output_dir + "/Sim_Results"
