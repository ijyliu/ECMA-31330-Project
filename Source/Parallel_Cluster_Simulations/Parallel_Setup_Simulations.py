# Setup_Simulations.py
# Produces a csv file with simulation parameters which can be given to individual jobs

# Import objects from the setup file
import os
import sys
sys.path.append(os.path.expanduser('~/repo/ECMA-31330-Project/Source'))
from ME_Setup import *

# Packages
import os
import glob

# Simulations dataframe with variations of parameter values
num_sims = 1000
Ns = [100, 1000]
betas = [1, 10]
# The numpy arrays of measurement error values have to be written out as strings for storage and converted later
me_means = ['[0,0]', '[1, 0]']
me_covs = ['[[100, 0], [0, 1]]', '[[1, 0], [0, 1]]']
kappas = [1]

# Scenarios dataframe
scenarios = produce_scenarios_cartesian(Ns, betas, me_means, me_covs, kappas)
scenarios['num_sims'] = num_sims

# Remove any pre-existing parameter combinations file
files_to_remove = glob.glob(box_dir + '/*_parameter_combos.csv')
for filePath in files_to_remove:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)

# Save the csv of parameter combinations, with label of the number of combos
scenarios.to_csv(box_dir + "/" + str(len(scenarios)) + "_parameter_combos.csv")