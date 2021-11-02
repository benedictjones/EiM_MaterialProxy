from runEiM import RunEiM
from mod_settings.GenParam import GenSimParam, LoadPrm

"""
*** Demo 2***

This is the same as Demo 1, execp we change some paramaters 'locally'.

If we want to loop though some different paramaters, it is useful to be able to
load in one defualt paramater file, then edit it on the fly.

We can do this by changing the underlying dictionary that we have loaded in.
e.g., here we change:
    - The number of times the DE algorithm is run on each system. This gives a
      better insight into how the material perfoms, mitigating the stocastic
      convergnce of the EA algorithm.
    - The Number of epochs
    - We set the function to re-use the materials that were generated in a
      previous run. We do this by seting "ReUse_dir" to the path of the
      previous result. When doing this, ensure that the the loaded material is
      of the smae size!


"""

# Run Evolutionary Process
if __name__ == "__main__":

    # load Template Raw (unformatted) Paramaters from param%s.yaml
    tprm = LoadPrm(param_file='')

    # Change some Prms
    tprm['num_repetitions'] = 3
    tprm['DE']['epochs'] = 30
    tprm['DE']['training_data'] = 'c2DDS'
    tprm['ReUse_dir'] = 'Results/2021_11_02/__demo1'

    # Gen final prm file
    prm = GenSimParam(param_file=tprm)  # Produce PAramater File

    # Run EiM
    RunEiM(prm)


#

#

# fin
