from runEiM import RunEiM


"""
*** Demo for a multiple runs.***

Rather then a single material, we can repeat DE algorithm on different randomly
generated materials (num_circuits).
We can also just repeat the DE on the same circuit/material (num_repetitions).

Once finished the analysis class will produce convergence plots of the
mean best population fitness during the evolutionary period (the overall mean
from both the new circuits and the repetitions on them).
A histogram of the final training fitnesses, and verification vs training
fitness scatter graph will also be produced.


To mitigate for randomness in the DE convergence we can repeat the DE algorithm
several times on the same material.
Do this by setting num_repetitions to more then one.
Note: the more repetitions the longer it will take!!

Again, results will be saved in the Results folder at the appropriate date and time.

Estimated time to complete:
 > Windows Anadonda ~7min  (single core limited, large multiprocessing initiation time)
 > Linux ~20s

"""


if __name__ == "__main__":

    # Run Evolutionary Process
    output_dir = RunEiM(num_circuits=3, num_repetitions=1,
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
