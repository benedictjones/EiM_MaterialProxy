from runEiM import RunEiM


"""
*** Demo for a multiple runs.***

When comparing the effect of different parameters or algorithm features, we
should re use the same materials. This allows for a direct comparison.
To do this we can set a target results file (which contains a saved network
topology, produced when material re-use is not enabled).
An example can be seen in demo3.py, this will run the DE on a previously saved
netowrk topology.
"""


if __name__ == "__main__":

    # # load in previous files to use previous material processors
    Model = 'D_RN'  # select material model

    if Model == 'R_RN':
        # load a Resistor Random Netowrk from a target file
        # this file contains 10 circuits each with exp_num_input=2, exp_num_output=2, exp_num_config=3
        ReUse_dir = 'Results/_2020_11_17_eg/__17_26_54__2DDS__R_RN__EiM'
    elif Model == 'D_RN':
        # load a Diode Random Netowrk from a target file
        # this file contains 10 circuits each with exp_num_input=2, exp_num_output=2, exp_num_config=3
        ReUse_dir = 'Results/_2020_11_17_eg/__17_24_56__2DDS__D_RN__EiM'


    # # Run Evolutionary Process
    output_dir = RunEiM(num_circuits=3, num_repetitions=1,
                        its=10, popsize=10,
                        #
                        model=Model,  # R_RN or D_RN
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
                        exp_ReUse_dir=ReUse_dir,
                        #
                        num_processors=4  # Number of cores to be used, can set to 'max'
                        )




#

#

# fin
