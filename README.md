# EiM Material Proxy
This code consists of an EiM simulation that uses differential evolution (DE).
The material is replaced with a SPICE based model, produced by integrating NGSpice with Python via PySpice.
This supports the paper:

Title: Co-design of algorithms and nanomaterials for use as Evolvable Processors
Authors: Benedict Jones, John Chouard, Bianca C.C. Branco, Eléonore G.B. Vissol-Gaudin, Christopher Pearson, Dagou A. Zeze, Michael C. Petty, Christopher Groves

Note that the data used in the paper is included (~400MB) making up the bulk of the download.

## System Requirements

The code has been developed on the Windows Subsystem for Linux (WSL) and should work on Ubuntu; though these platforms are not regularily tested.
The code can also be run on windows 10 based platform using anaconda, though this has slower multiprocess speeds due to the lack of forking. Single process speeds have been otimised for fast performance on windows 10, but PySpice/NGSPice currently has a bug which leads to memory not being reclaimed by the os - a problem for larger simulations.   


For best performance:
  - Use 8GB of RAM or more (Development has taken place using 16GB).
  - A CPU with multiple cores (the number of cores to be used can be selected). More cores = more speed.


## Instillation

Having downloaded this repository, ensure that the correct packages and NGSpice are installed.
Normal installation time will be around 5 minutes (depending on upload/download speed).
Python 3.8.5 is used.

### Step 1 - Packages
To run the model/code in this repository please **install the correct modules** :

Ensure you have installed these packages (a full list of my enviroments dependences is given in dependences.txt):
    - PySpice
    - h5py
    - tqdm
    - pandas
    - tables
    - pickle
    - scipy
    - sklearn
    - numpy
    - yaml

    
    (You may need to upgrade previous versions, e.g. ```run pip install --upgrade --force-reinstall h5py```)
    


### Step 2 - NGSpice
For PySpice to work **NGSpice must also be installed**.  
(Full instructions for PySpice/NGSpice installation can be found at https://pyspice.fabrice-salvaire.fr/releases/v1.4/installation.html)  
Note: this code has been developed using NGSpice 31 (later versions such as NGSpice 33 are not currently compatible)!

#### Windows
A version of NGSpice is included in /NGSpice/ngspice.zip, the extracted contents must be placed in the appropriate location.  
For PySpice 1.4 (the current release) place Spice64 and Spice_dll in the PySpice module inside *PySpice\Spice\NgSpice*.  
    e.g. for conda enviroment eim_env: *C:\Users\user\anaconda3\envs\eim_env\Lib\site-packages\PySpice\Spice\NgSpice*
  
Note: For the older version of PySpice 1.3, place Spice64 and Spice_dll in *C:\Program Files*.

#### Linux
On linux NGSpice is installed via the command line.  
For Ubuntu use: ```sudo apt-get install -y ngspice```


## Demo

### Single Run
For a single run of a random material, run demo1.py.
Once finished the analysis class will execute and produce a convergence plot.
After this, material graphs (surface plot of the network response Y) will also be generated for:

- The best genome (i.e population member).
- The default (unconfigured) material

You can also toggle the settings such that further plots are produced for (increases execution time):

- The Effect of varying the configuration voltages (set MG_vary_Vconfig=1)
- The effect of using different input permutations/shuffle (set MG_vary_PermConfig=1)
- The effect of varying input weights (set MG_vary_InWeight=1)
- The effect of varying output weights (set MG_vary_OutWeight=1)

These results are saved. They can be found by navigating the results folder to the appropriate date and time, where results can be observed.
(Note: This pathway can be used in anly_run.py to re-produce plots from the saved data)

Often when using the simplest dataset (the 2DDS) the evolutionary algorithm will find a configuration which achieve 0% error before the evolutionary period expires.
Try changing the data set to the more challenging concentric 2d data set (con2DDS).
Or varying the algorithm features.

For 4 cores running at ~4.2Ghz the **execution time** is around 3s (windows ~15s).
Depending on the speed of your processor and the number of cores used it may be faster or slower.

### Multiple Materials
Rather than a single material, we can repeat DE algorithm on different randomly generated materials (num_circuits).
We can also just repeat the DE on the same circuit/material (num_repetitions).
For an example of this run demo2.py.
Once finished the analysis class will produce convergence plots of the mean best population fitness during the evolutionary period (the overall mean from both the new circuits and the repetitions on them).
A histogram of the final training fitnesses, and verification vs training fitness scatter graph will also be produced.

To mitigate for randomness in the DE convergence we can repeat the DE algorithm several times on the same material.
Do this by setting num_repetitions to more then one.
Note: the more repetitions the longer it will take!!

Again, results will be saved in the Results folder at the appropriate date and time.

For 4 cores running at ~4.2Ghz the **execution time** is around 20s (windows ~7mins, faster if set to a single core!).

### Loading a network
When comparing the effect of different parameters or algorithm features, we should re use the same materials. This allows for a direct comparison.
To do this we can set a target results file (which contains a saved network topology, produced when material re-use is not enabled).
An example can be seen in demo3.py, this will run the DE on a previously saved network topology.


### Running an Experiment
To compare the effect of varying different parameters on a material processor we can run an experiment, for example demo_exp.py. This saves the data differently then the single runs described above.

In this example the effect of using shuffle is examined, by testing its use (or not) on 3 different Diode Random Networks (DRN). There are 5 DE repetitions on each network.

The graph outputs and results can be found in the Experiment_List folder under the selected data set (in demo_exp.py this is currently con2DDS).

For 4 cores running at ~4.2Ghz the **execution time** is around 5mins (windows - not tested).
Depending on the speed of your processor and the number of cores used it may be faster or slower.


## Analysis

### Demo
To examine any particular single run or experiment and produce plots from the saved data we can use anly_run.py.
The target folder can be set to either:

- A results folder (e.g. *'Results/2020_06_17/__13_59_18__con2DDS__EXP_Demo_Exp_D_RN'*)
- An experiment folder (e.g. *'Experiment_List/con2DDS/2020_06_17__13_59_18___EXP_Demo_Exp_D_RN'*)

An example is given in anly_run_demo.py.

### Reading Paper Results
To reproduce the plots used in the paper, execute anly_run_paper.py (for the graphs) and anly_run_paper_mg.py (for the responce surface plots).
This will generate and save the plots to a folder named *CustomSave/Paper* by loading the data that was generated from the original experiments.

## Recreating Paper Results
To recreate the papers results, run exp_paper.py. This will use/load the same 15 RRN, NLRN and DRN networks as was used to produce the papers results.
USing the selected dataset, the DE algorithm repeats itself 5 times on each of the networks, each run has 40 iterations/generations.
This is repeated for the different algorithm feature combinations (shuffle, input and output weights).
Depending on the speed of your processor and the number of cores used, the **execution time** may be up to **a day** or more.

## Data Sets

The datasets used in the paper (the 2DDS and con2DDS) can be found as hdf5 files in EiM_MaterialProxy\mod_load\data.
Note that the dataset is normalised and scaled when it is loaded in using EiM_MaterialProxy\mod_load\LoadData.py .

