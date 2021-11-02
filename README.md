# EiM Material Proxy
This code consists of an EiM simulation that uses differential evolution (DE).
The material is replaced with a SPICE based model, produced by integrating NGSpice with Python via PySpice.
This supports the paper:

Title: Towards Intelligently Designed Evolvable Processors

Authors: Benedict Jones, John Chouard, Bianca C.C. Branco, Eléonore G.B. Vissol-Gaudin, Christopher Pearson, Dagou A. Zeze, Michael C. Petty, Christopher Groves


## System Requirements

The code has been developed on the Windows Subsystem for Linux (WSL) and should work on Ubuntu which leverages multiprocessing for increased speed.
The code can also be run on windows 10 based platform using anaconda, but is slower.

For best performance:
  - Use 8GB of RAM or more (Development has taken place using 16GB).
  - A CPU with multiple cores (the number of cores to be used can be selected). More cores = more speed when on linux.


## Instillation

Having downloaded this repository, ensure that the correct packages and NGSpice are installed.
Normal installation time will be around 5 minutes (depending on upload/download speed).
Python 3.8.5 was used.

### Step 1 - Python Packages
To run the model/code in this repository please **install the correct modules** :

Ensure you have installed these main packages. The core packages include:
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

#### Linux
On linux NGSpice is installed via the command line.  
For Ubuntu use: ```sudo apt-get install -y ngspice```

#### Windows
A version of NGSpice is included in /NGSpice/ngspice.zip, the extracted contents must be placed in the appropriate location.  
For PySpice 1.4 (the current release) place Spice64 and Spice_dll in the PySpice module inside *PySpice\Spice\NgSpice*.  
    e.g. for conda enviroment eim_env: *C:\Users\user\anaconda3\envs\eim_env\Lib\site-packages\PySpice\Spice\NgSpice*

Note: For the older version of PySpice 1.3, place Spice64 and Spice_dll in *C:\Program Files*.

## Demos

Within each of the demo.py files there is more top matter explaining how they work.

### Demo 1
For a single run of a random material, run demo1.py.
This will show how the runEiM function can be used to create and optimise an EiM classifier using the default parameters.

Using a Ryzen 9 3900X, the **execution time** on Windows 10 Anadonda was about 35 seconds, and on linux is about 5 seconds.

### Demo 2
This will show how the loaded in default parameters can be edited locally.
It shows how to load in a previously generated material.

### Demo 3
This shows how to make a basic "experiment" which is a loop where we can vary a parameter and compare the perfomance between the differently defined systems.

### Demo 4
This shows how to set up a more complex "experiment" with severla changing paramaters.

## Paper Related Info

### Recreating Paper Results
To recreate the papers results, run exp_paper.py. This will use/load the same 15 RRN, NLRN and DRN networks as was used to produce the papers results.
Using the selected dataset, the DE algorithm repeats itself 5 times on each of the networks, each run has 50 iterations/generations.
This is repeated for the different algorithm feature combinations (shuffle, input and output weights).
Depending on the speed of your processor and the number of cores used, the **execution time** may be up to **a day** or more.

### Data Sets

The datasets used in the paper, the 2d dataset (here called d2DDS) and concentric dataset (here called c2DDS), can be found as hdf5 files in mod_load\data.
