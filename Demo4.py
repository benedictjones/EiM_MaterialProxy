from datetime import datetime
import os
import copy

from runEiM import RunEiM
from mod_analysis.Analysis import analysis
from mod_settings.GenParam import GenSimParam, LoadPrm

"""
## Demo 4 ##

You can run the Demo4.py file which allows you to loop though some
sequence of parameters, executing each on the RunEiM function.

This time we have a list of paramaters defined by a list of descripting
string labels. This is acompanied by one or more lists of the actual paramater
we would like to change.
We iterate thought the descriptive "Param_array" and use the "ex_loop" to
index the actual paramater array(s).

In this example we can vary the shuffle gene, input weight and output weight
all at once.

In this experiment we generate a new group of num_systems number of
materials. These are then reused each loop.

The experiment results will be saved in the Results_experiemts folder;
however, the actual underlying data from the individual loops are still saved
in the Results folder.

"""


# Main script
if __name__ == "__main__":

    # load Template Raw (unformatted) Paramaters from param%s.yaml
    trprm = LoadPrm(param_file='')

    # #set the model types which the experiments will be repeated for
    Model_types = ['D_RN']  # or use ['R_RN', 'D_RN', 'NL_RN']
    datasets = ['c2DDS']  # or use ['d2DDS', 'c2DDS', 'flipped_2DDS']

    #

    #

    # # Run for each dataset
    for dataset in datasets:

        # Run Experiment for each material model
        for Model in Model_types:
            print("*************************************************************")
            print("Experiment - Material Model:", Model)
            print("*************************************************************")

            # # Collect time and dat stamp of experiment
            now = datetime.now()
            real_d_string = now.strftime("%d_%m_%Y")
            d_string = now.strftime("%Y_%m_%d")
            t_string = now.strftime("%H_%M_%S")
            print("Date:", real_d_string, ", Time Stamp:", t_string)

            #

            # # Assign Details
            num_inputs = 2

            #

            # # Name the experiment, and add to folder name
            exp_name = 'Example_Vary_Shuffle_IW_OW__%s' % (Model)
            Param_Varying = 'Vary the use of the shuffle gene'
            experiment_file = '%s/%s__%s___EXP_%s' % (dataset, d_string, t_string, exp_name)

            #

            # # Set parameter which will vary and appear in fig legends
            Param_array = ["[Sh=0,Iw=0,Ow=0]",
                           "[Sh=0,Iw=1,Ow=1]",
                           "[Sh=1,Iw=1,Ow=1]"]

            # individual paramater lists to change
            Param_shuffle = [0, 0, 1]
            Param_IW =      [0, 1, 1]
            Param_OW =      [0, 1, 1]

            # # Do not load in previously generated circuits
            ReUse_dir = 'na'  # set defualt to 'na, this is assigned at the end of first loop'

            #

            num_experiments = len(Param_array)  # extracts the total num of loops
            for ex_loop, Param in enumerate(Param_array):

                print("##################################################")
                print("Experiment", ex_loop, "Out of:", num_experiments-1)
                print("##################################################")

                # # Create a raw (unformatted) paramater file from the loadted template
                rprm = copy.deepcopy(trprm)

                # Alter Prms
                rprm['ReUse_dir'] = ReUse_dir
                rprm['num_systems'] = 1
                rprm['num_repetitions'] = 3

                rprm['DE']['epochs'] = 20
                rprm['DE']['training_data'] = dataset
                rprm['DE']['save_fitvit'] = 0

                rprm['network']['model'] = Model
                rprm['network']['num_input'] = num_inputs

                rprm['genome']['Shuffle']['active'] = Param_shuffle[ex_loop]
                rprm['genome']['InWeight']['active'] = Param_IW[ex_loop]
                rprm['genome']['OutWeight']['active'] = Param_OW[ex_loop]

                # Gen final prm dict
                prm = GenSimParam(param_file=rprm,
                                  experiment=1,
                                  experiment_file=experiment_file,
                                  exp_name=exp_name)

                # # Run EiM
                RunEiM(prm, experiment_loop=ex_loop)

                # # Save the results path and toggle the reuse paramater
                if ex_loop == 0:

                    # Set the exp to use the same materials from the first for each subsequent loop
                    ReUse_dir = prm['SaveDir']

                    # save current dir to file
                    with open('%s/DataDir.csv' % (prm['experiment']['file']), 'w') as dir_result_file:
                        dir_result_file.write(prm['SaveDir'])
                        dir_result_file.write("\n")

                else:
                    # Append new dir of the results to file
                    with open('%s/DataDir.csv' % (prm['experiment']['file']), 'a') as dir_result_file:
                        dir_result_file.write(prm['SaveDir'])
                        dir_result_file.write("\n")


                # increment experiment loop
                ex_loop = ex_loop + 1

            #######################################################################
            # Save some data about the experiment
            #######################################################################
            Param_String = ''
            for Parameter in Param_array:
                Param_String = '%s%s \n' % (Param_String, str(Parameter))
            Param_String = '%s\n%s' % (Param_String, str(Param_Varying))

            path_params = "%s/Param_array.txt" % (prm['experiment']['file'])
            file1 = open(path_params, "w")
            file1.write(Param_String)
            file1.close()

            # ########################################################################
            # Run Analysis and save plots to exp dir
            # ########################################################################
            print("\nProducing Exp analysis graphs...")
            obj_anly = analysis(prm['experiment']['file'], format='png')

            obj_anly.Plt_basic(Save_NotShow=1, fill=1, ExpErrors=1, StandardError=1)
            obj_anly.Plt__ani(Save_NotShow=1)

            # # Print end time
            now = datetime.now()
            d_string_fin = now.strftime("%d_%m_%Y")
            t_string_fin = now.strftime("%H_%M_%S")
            print("\nExperiment Finished at:", t_string_fin)

    #

    #

    #

    #

    #

    #






# fin
