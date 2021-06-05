# ECMA-31330-Project

Econometrics and Machine Learning Group Project

Principal Component Regression as a Solution to Measurement Error Bias

Isaac Liu, Nico Martorelli, Paul Opheim

The paper can be found [here](Release/PCR_and_Measurement_Error.pdf).

## Contents

Here is the structure of this repository, along with links to relevant folders.

### [Input](Input)

Contains all data files which are inputs to analysis for the application and simulations.

### [Output](Output)

Contains tables, figures, and simulation results.

### [Release](Release)

Contains the abstract and paper pdfs and associated LaTeX files and bibliography material.

### [Source](Source)

Contains the project source code.

#### [Application](Source/Application)

Contains code for the life expectancy and government health share application.

#### [Simulations](Source/Simulations)

Contains code to run the Monte Carlo simulations and produce relevant tables.

## Replication Instructions

1. Download or clone the repository. There should be no need to modify directory structure, but the relevant python packages must be installed.
2. The simulation files are structured such that they may be run on a computing cluster. These steps each have their own .sh scripts and [Parallel_Simulations.sh](Source/Simulations/Parallel_Simulations.sh) is set up to run them all sequentially while requesting the appropriate amount of computing resources.
   1. [Setup_Parallel_Sims.py](Source/Simulations/Setup_Parallel_Sims.py) defines a csv of parameter values and combinations.
   2. [Run_Parallel_Sim.py](Source/Simulations/Run_Parallel_Sim.py) is executed in parallel and performs 1,000 simulations for each parameter combination considered and outputs the results.
   3. [Compile_Parallel_Sims.py](Source/Simulations/Compile_Parallel_Sims.py) combines the results from the previous step into a single file.
3. To produce the statistics for tables based off of the simulation results, run [Produce_Tables_Parallel.ipynb](Source/Simulations/Produce_Tables_Parallel.ipynb).
4. To download the World Bank Data used for the application, run [Get_WB_Data.py](Source/Application/Get_WB_Data.py). This is definitely an optional step because the World Bank may update the data and the results may change slightly.
5. To run the primary empirical analysis for the application, run [Application_Gov_Health_Spending_Share_LE.py](Source/Application/Application_Gov_Health_Spending_Share_LE.py).
