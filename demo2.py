
# Imports
from DE import RunDE
from Module_Analysis.Analysis import analysis

"""
*** Demo for a multiple runs.***

Rather then a single material, we can repeat DE algorithm on different randomly
generated materials.

Once finished the analysis class will produce convergence plots of the
mean best population fitness during the evolutionary period.
A histogram of the final training fitnesses, and verification vs training
fitness scatter graph will also be produced.
Following this Graphs from the best population member from each of the runs on
the materials will pop up, as well as the network response from the
default (unconfigured) material.

To mitigate for randomness in the DE convergence we can repeat the DE algorithm
several times on the same material.
Do this by setting exp_num_repetitions to a larger number.
Note: that the more repetitions the longer it will take!!

Again, results will be saved in the Results folder at the appropriate date and time.
"""


if __name__ == "__main__":

    # Run Evolutionary Process
    output_dir = RunDE(exp_num_circuits=3, exp_num_repetitions=1,
                       exp_its=5, exp_popsize=20,
                       #
                       exp_the_model='D_RN',  # R_RN or D_RN
                       exp_num_input=2, exp_num_output=2, exp_num_config=1,  # number of each node type
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
                       exp_num_processors=4  # Number of cores to be used, can set to 'max
                       )



    # Use the analysis object to produce and display graphs
    obj_anly = analysis(output_dir)

    sel_dict = {'plt_mean':0,'plt_std':0,'plt_finalveri':1,'plt_popmean':0,'plt_hist':1}  # selects what graphs to plot
    obj_anly.Plt_basic(sel_dict=sel_dict, Save_NotShow=0, fill=1, ExpErrors=1, StandardError=1)
    obj_anly.Plt_mg(Save_NotShow=0, Bgeno=1, Dgeno=1, VC=1, VP=1, VoW=1, ViW=1)

#

#

# fin
