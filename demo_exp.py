from datetime import datetime
import os

from DE import RunDE
from Module_Analysis.Analysis import analysis


if __name__ == "__main__":

    # set the model types which the experiments will run over
    Model_types = ['D_RN']  # ['R_RN', 'D_RN']

    #

    #

    # Begin Experiment
    for Model in Model_types:
        print("*************************************************************")
        print("Experiments 1 - Material Model:", Model)
        print("*************************************************************")

        # # # # # # # # # # # #
        # Collect time and dat stamp of experiment
        now = datetime.now()
        real_d_string = now.strftime("%d_%m_%Y")
        d_string = now.strftime("%Y_%m_%d")
        print("Date:", real_d_string)
        t_string = now.strftime("%H_%M_%S")
        print("Time Stamp:", t_string)

        # ################################################################
        # Name the experiment, to pass to DE.py
        Exp_name = 'Demo_Exp_%s' % (Model)
        Param_Varying = 'Using Shuffle or not'
        # ################################################################

        # ################################################################
        training_data = 'con2DDS'
        num_inputs = 2
        # ################################################################


        # produce file dir with time stamp
        experiment_file = 'Experiment_List/%s/%s__%s___EXP_%s' % (training_data, d_string, t_string, Exp_name)
        path_Experiment_dir = '%s/DataDir.csv' % (experiment_file)
        dir_exp_list = []

        # Set parameter which will vary
        Param_array = ["Shuffle=off","Shuffle=on"]
        Param_shuffle = [0, 1]

        # Begin experiment loop
        ex_loop = 0
        ReUse_dir = 'na'  # set defualt to 'na, this is assigned at the end of first loop'

        num_experiments = len(Param_array)
        for Param in Param_array:

            print("##################################################")
            print("Experiment", ex_loop, "Out of:", num_experiments-1)
            print("##################################################")

            dir = RunDE(exp_num_circuits=3, exp_num_repetitions=3,
                        experiment=1, exp_name=Exp_name,
                        experiment_file=experiment_file, experiment_loop=ex_loop,
                        #
                        exp_its=15, exp_popsize=20,
                        exp_the_model=Model,
                        exp_num_input=num_inputs, exp_num_output=2, exp_num_config=3,
                        #
                        plotMG=1,
                        #
                        exp_shuffle_gene=Param_shuffle[ex_loop],
                        exp_InWeight_gene=0,
                        exp_OutWeight_gene=0,
                        #
                        exp_training_data=training_data,
                        exp_ReUse_Circuits=1, exp_ReUse_dir=ReUse_dir,  # makes susequent experimets use same material models
                        exp_TestVerify=1,
                        #
                        exp_num_processors=4  # Number of cores to be used
                        )


            # # # # # # # # # # # #
            # Save list of directories used for Analysis, append each exp-loop
            if ex_loop == 0:

                # if the new folder does not yet exist, create it
                if not os.path.exists(experiment_file):
                    os.makedirs(experiment_file)

                # save current dir to file
                with open(path_Experiment_dir, 'w') as dir_result_file:
                    dir_result_file.write(dir)
                    dir_result_file.write("\n")

                # assign the dir from the first loop to the re-use path
                ReUse_dir = dir  # turn this off if using a fixed ReUse Dir

            else:
                # Append new dir of the results to file
                with open(path_Experiment_dir, 'a') as dir_result_file:
                    dir_result_file.write(dir)
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

        path_params = "%s/Param_array.txt" % (experiment_file)
        file1 = open(path_params, "w")
        file1.write(Param_String)
        file1.close()

        # ########################################################################
        # Run Analysis and save plots to exp dir
        # ########################################################################
        print("\nProducing analysis graphs...")
        obj_anly = analysis(experiment_file, format='png')

        obj_anly.Plt_basic(Save_NotShow=1, fill=1, ExpErrors=1, StandardError=1)
        obj_anly.Plt_mg(Save_NotShow=1, Bgeno=1, Dgeno=1, VC=0, VP=0, VoW=0, ViW=0)

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