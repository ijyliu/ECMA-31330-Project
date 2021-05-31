# ME_Setup.py
# Defines the project directory structure as well as other shared functions

# Packages
import os

# Directory structure
box_dir = "~/Box/ECMA-31330-Project"
parameters_dir = box_dir + "/Parameters"
sim_results_dir = box_dir + "/Simulation_Results"
scenario_files_dir = sim_results_dir + "/Scenario_Files"
apps_dir = box_dir + "/Applications_Data"
repo_dir = os.path.join(os.path.dirname( __file__ ), '../..')
input_dir = repo_dir + "/Input"
output_dir = repo_dir + "/Output"
figures_dir = output_dir + "/Figures"
tables_dir = output_dir + "/Tables"
