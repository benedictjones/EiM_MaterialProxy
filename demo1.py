from runEiM import RunEiM
from mod_settings.GenParam import GenSimParam, LoadPrm

"""
*** Demo 1 ***

The defualt paramaters can be defined in the param.yaml file.
This is loaded in, then passed to "GenSimParam", which checks the paramaters &
generates a fully defined paramater dictionary which it passes out.
This is then used to execute RunEiM and perform EiM.

Once the EiM is finished, the analysis class used within RunEiM executes to
produce a convergence plot, histogram of the test data output responces and
more.
The "material graphs" can be enabled, and are in the default file.
These are surface plot of the network response Y, showing you the decision
boundary of the classifier.

The results (hdf5 format) and graphs are saved.
They can be found by navigating the results folder to the appropriate date
and time.
We can find the generated and saved SPICE netlist saved in "MD_CircuitTop".
(Note: This pathway can be used in anly_run.py to re-produce
plots from the saved data)

Often when using the simplest dataset (the d2DDS) the evolutionary algorithm
will find a configuration which achieve 0% error before the evolutionary
period expires. Try changing the data set to the more challenging
concentric 2d data set (c2DDS).
Or varying the algorithm features (i.e. shuffle gene, input or output weights).

Estimated time to complete:
 > Windows Anadonda ~35s
 > Linux ~5s (more cores = faster)

"""

# Run Evolutionary Process
if __name__ == "__main__":

    # load Template Raw (unformatted) Paramaters from param%s.yaml
    tprm = LoadPrm(param_file='')

    # Gen final prm file
    prm = GenSimParam(param_file=tprm)  # Produce PAramater File

    # Run EiM
    RunEiM(prm)


#

#

# fin
