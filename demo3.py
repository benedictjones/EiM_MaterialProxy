
# Imports
from DE import RunDE
from Module_Analysis.Analysis import analysis

"""
*** Demo for a multiple runs.***

When comparing the effect of different parameters or algorithm features, we
should re use the same materials. This allows for a direct comparison.
To do this we can set a target results file (which contains a saved network
topology, produced when material re-use is not enabled).
An example can be seen in demo3.py, this will run the DE on a previously saved
netowrk topology.
You can compare the produced default (unconfigured) network with the one saved
in the target folder!
"""


if __name__ == "__main__":

    # # load in previous files to use previous material processors
    Model = 'D_RN'  # select material model
    if Model == 'R_RN':
        # load a Resistor Random Netowrk from a target file
        # this file contains 10 circuits each with exp_num_input=2, exp_num_output=2, exp_num_config=3
        ReUse_dir = 'Results/2020_04_06/__11_04_56__2DDS__EXP_NewVaryScheme_R_RN'
    elif Model == 'D_RN':
        # load a Diode Random Netowrk from a target file
        # this file contains 10 circuits each with exp_num_input=2, exp_num_output=2, exp_num_config=3
        ReUse_dir = 'Results/2020_04_07/__04_31_06__2DDS__EXP_NewVaryScheme_D_RN'


    # # Run Evolutionary Process
    output_dir = RunDE(exp_num_circuits=1, exp_num_repetitions=1,
                       exp_its=5, exp_popsize=20,
                       #
                       exp_the_model=Model,  # R_RN or D_RN
                       exp_num_input=2, exp_num_output=2, exp_num_config=3,  # number of each node type
                       #
                       plotMG=1,  # Sets surface plots of Y to be produced. (always produced a plot for the best genome)
                       plot_defualt=1, MG_vary_Vconfig=0, MG_vary_PermConfig=0,
                       MG_vary_InWeight=0, MG_vary_OutWeight=0,
                       #
                       exp_training_data='con2DDS',  # Select a DataSet: 2DDS, O2DDS, con2DDS, MMDS
                       #
                       exp_shuffle_gene=0,  # Shuffle gene
                       exp_InWeight_gene=0,  # Input Weights
                       exp_OutWeight_gene=0,  # Output weights
                       #
                       exp_FitScheme='error',  # fitness scheme
                       exp_TestVerify=1,  # can toggle whether we use verification/test data
                       #
                       exp_num_processors=4,  # Number of cores to be used, can set to 'max
                       #
                       exp_ReUse_Circuits=1, exp_ReUse_dir=ReUse_dir  # makes susequent experimets use same material models
                       )



    # Use the analysis object to produce and display graphs
    obj_anly = analysis(output_dir)

    sel_dict = {'plt_mean':0,'plt_std':0,'plt_finalveri':1,'plt_popmean':0,'plt_hist':1}  # selects what graphs to plot
    obj_anly.Plt_basic(sel_dict=sel_dict, Save_NotShow=0, fill=1, ExpErrors=1, StandardError=1)
    obj_anly.Plt_mg(Save_NotShow=0, Bgeno=1, Dgeno=1, VC=1, VP=1, VoW=1, ViW=1)

#

#

# fin
