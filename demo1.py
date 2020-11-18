from runEiM import RunEiM


"""
*** Demo for a single run of a random material.***

The paramaters can be defined in the param.yaml file (the defualt param file),
or can be passed into the function which overwrites the defualt values in the
param file (as seen bellow).


Once the EiM is finished, the analysis class used within RunEiM executes to
produce a convergence plot, verification vs training fitness scatter
graph (if more then one rep or circuit is considered), and material graphs
(surface plot of the network response Y) for the best and default genome.

The results (hdf5 format) and graphs are saved.
They can be found by navigating the results folder to the appropriate date
and time.
(Note: This pathway can be used in anly_run.py to re-produce
plots from the saved data)

Often when using the simplest dataset (the 2DDS) the evolutionary algorithm
will find a configuration which achieve 0% error before the evolutionary
period expires. Try changing the data set to the more challenging
concentric 2d data set (con2DDS).
Or varying the algorithm features (i.e. shuffle gene, input or output weights).

Estimated time to complete:
 > Windows Anadonda ~15s
 > Linux ~3s (more cores = faster)

"""


if __name__ == "__main__":

    # Run Evolutionary Process
    output_dir = RunEiM(num_circuits=1, num_repetitions=1,
                        its=10, popsize=10,
                        #
                        model='D_RN',  # R_RN or D_RN
                        num_input=2, num_output=2, num_config=2,  # number of each node type
                        #
                        plotMG=1,  # Sets surface plots of Y to be produced.
                        plot_defualt=1,
                        MG_vary_Vconfig=0, MG_vary_PermConfig=0,
                        MG_vary_InWeight=0, MG_vary_OutWeight=0,
                        #
                        training_data='2DDS',  # Select a DataSet: 2DDS, O2DDS, con2DDS, MMDS
                        #
                        shuffle_gene=0,  # Shuffle gene
                        InWeight_gene=0,  # Input Weights
                        OutWeight_gene=0,  # Output weights
                        #
                        IntpScheme='pn_binary',
                        FitScheme='error',  # fitness scheme
                        TestVerify=1,  # can toggle whether we use verification/test data
                        #
                        num_processors=4  # Number of cores to be used, can set to 'max'
                        )


#

#

# fin
