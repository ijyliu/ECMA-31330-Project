# ECMA-31330-Project
Econometrics and Machine Learning Group Project
PCR and Measurement Error
Isaac Liu, Nico Martorelli, Paul Opheim

## Contents

Here is the structure of this repository, along with links to files and folders.

### [Input](Input)

Contains the World Bank Data for the application.

### [Output](Output)

Contains [tables](Output/Tables), [regressions](Output/Regressions) and [figures](Output/Figures) for the project.

### [Release](Release)

Contains the [abstract](Release/Abstract.pdf) and [paper](Release/Factors_and_Measurement_Error.pdf) pdfs and LaTeX files.

### [Source](Source)

Contains the project source code.

The most important file is [ME_Setup.py](Source/ME_Setup.py) which defines functions and objects used throughout the project (for both the local and parallelized parts).

#### [Application](Source/Application)

Contains code for the GDP and Life Expectancy application.

#### [Local Simulations](Source/Local_Simulations)

Contains code to run the project locally. [Benchmark_Estimator.py](Source/Local_Simulations/Benchmark_Estimator.py) is the most important file which sets up the scenarios and runs the analysis.

#### [Parallel_Cluster_Simulations](Source/Parallel_Cluster_Simulations)

Contains code to run many simulations and scenarios with many different parameter values on a computing cluster.
