
# Imports
from DE import RunDE
from Module_Analysis.Analysis import analysis

"""
*** Demo for a single run of a random material.***

Once finished the analysis class will execute and produce a convergence plot,
verification vs training fitness scatter graph, and material graphs (surface
plot of the network response Y) for the best and default genome.

These results are also saved. They can be found by navigating the results
folder to the appropriate date and time, where results can be observed.
(Note: This pathway can be used in anly_run.py to re-produce
plots from the saved data)

Often when using the simplest dataset (the 2DDS) the evolutionary algorithm
will find a configuration which achieve 0% error before the evolutionary
period expires. Try changing the data set to the more challenging
concentric 2d data set (con2DDS).
Or varying the algorithm features (i.e. shuffle gene, input or output weights).
"""


if __name__ == "__main__":

    # Run Evolutionary Process
    output_dir = RunDE(exp_num_circuits=1, exp_num_repetitions=1,
                       exp_its=10, exp_popsize=20,
                       #
                       exp_the_model='D_RN',  # R_RN or D_RN
                       exp_num_input=2, exp_num_output=2, exp_num_config=2,  # number of each node type
                       #
                       plotMG=1,  # Sets surface plots of Y to be produced.
                       plot_defualt=1,
                       MG_vary_Vconfig=0, MG_vary_PermConfig=0,
                       MG_vary_InWeight=0, MG_vary_OutWeight=0,
                       #
                       exp_training_data='2DDS',  # Select a DataSet: 2DDS, O2DDS, con2DDS, MMDS
                       #
                       exp_shuffle_gene=0,  # Shuffle gene
                       exp_InWeight_gene=0,  # Input Weights
                       exp_OutWeight_gene=0,  # Output weights
                       #
                       exp_FitScheme='error',  # fitness scheme
                       exp_TestVerify=1,  # can toggle whether we use verification/test data
                       #
                       exp_num_processors=4  # Number of cores to be used, can set to 'max'
                       )



    # Use the analysis object to produce and display graphs
    obj_anly = analysis(output_dir)

    sel_dict = {'plt_mean':1,'plt_std':0,'plt_finalveri':0,'plt_popmean':0,'plt_hist':0}  # selects what graphs to plot
    obj_anly.Plt_basic(sel_dict=sel_dict, Save_NotShow=0, fill=1, ExpErrors=1, StandardError=1)
    obj_anly.Plt_mg(Save_NotShow=0, Bgeno=1, Dgeno=1, VC=1, VP=1, VoW=1, ViW=1)

#

#

# fin
