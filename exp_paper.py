from datetime import datetime
import os

from runEiM import RunEiM
from mod_analysis.Analysis import analysis




# Main script
if __name__ == "__main__":

    # set the model types which the experiments will run over
    Model_types = ['R_RN', 'D_RN', 'NL_RN']

    datasets = ['2DDS', 'con2DDS', 'flipped_2DDS']

    #

    #

    tog = 0

    # # Run for each dataset
    for dataset in datasets:

        # Run Experiment for each material model
        for Model in Model_types:
            print("*************************************************************")
            print("Experiment - Material Model:", Model)
            print("*************************************************************")

            # # # # # # # # # # # #
            # Collect time and dat stamp of experiment
            now = datetime.now()
            real_d_string = now.strftime("%d_%m_%Y")
            d_string = now.strftime("%Y_%m_%d")
            print("Date:", real_d_string)
            t_string = now.strftime("%H_%M_%S")
            print("Time Stamp:", t_string)

            #

            # ################################################################
            # Name the experiment, to pass to DE.py
            Exp_name = 'PaperReplication__%s' % (Model)
            Param_Varying = 'Vary Schemes: shuffle, input and output weights'
            # ################################################################

            #

            # ################################################################
            training_data = dataset
            num_inputs = 2
            # ################################################################

            #

            # produce file dir with time stamp
            experiment_file = 'Experiment_List/%s/%s__%s___EXP_%s' % (training_data, d_string, t_string, Exp_name)
            path_Experiment_dir = '%s/DataDir.csv' % (experiment_file)
            dir_exp_list = []

            # # # # # Set parameter which will vary
            Param_array = ["[Sh=0,Iw=0,Ow=0]",
                           "[Sh=0,Iw=1,Ow=0]",
                           "[Sh=0,Iw=0,Ow=1]",
                           "[Sh=0,Iw=1,Ow=1]",
                           "[Sh=1,Iw=0,Ow=0]",
                           "[Sh=1,Iw=1,Ow=0]",
                           "[Sh=1,Iw=0,Ow=1]",
                           "[Sh=1,Iw=1,Ow=1]"]

            Param_shuffle = [0, 0, 0, 0, 1, 1, 1, 1]
            Param_IW =      [0, 1, 0, 1, 0, 1, 0, 1]
            Param_OW =      [0, 0, 1, 1, 0, 0, 1, 1]


            # Begin experiment loop
            ex_loop = 0
            ReUse_dir = 'na'  # set defualt to 'na, this is assigned at the end of first loop'

            # # load in previous files to use previous material processors
            """
            if Model == 'R_RN':
                ReUse_dir = 'Results/2020_04_06/__11_04_56__2DDS__EXP_NewVaryScheme_R_RN'
            elif Model == 'D_RN':
                ReUse_dir = 'Results/2020_04_07/__04_31_06__2DDS__EXP_NewVaryScheme_D_RN'
            else:
                raise ValueError('Select available material model')
            #"""


            #

            num_experiments = len(Param_array)
            for Param in Param_array:

                if Model == 'R_RN' and ex_loop == 0 and training_data == '2DDS':
                    ReUse_dir = 'na'
                elif Model == 'R_RN':
                    ReUse_dir = RRN_ReUse_dir

                if Model == 'D_RN' and ex_loop == 0 and training_data == '2DDS':
                    ReUse_dir = 'na'
                elif Model == 'D_RN':
                    ReUse_dir = DRN_ReUse_dir

                if Model == 'NL_RN' and ex_loop == 0 and training_data == '2DDS':
                    ReUse_dir = 'na'
                elif Model == 'NL_RN':
                    ReUse_dir = NLRN_ReUse_dir



                print("##################################################")
                print("Experiment", ex_loop, "Out of:", num_experiments-1)
                print("##################################################")

                dir = RunEiM(num_circuits=15, num_repetitions=5,
                             experiment=1, exp_name=Exp_name,
                             experiment_file=experiment_file, experiment_loop=ex_loop,
                             #
                             its=40, popsize=20,
                             model=Model,
                             num_input=num_inputs, num_output=2, num_config=3,
                             #
                             plotMG=0, MG_vary_Vconfig=0, MG_vary_PermConfig=0,  # plot MG graphs?
                             #
                             save_fitvit=0,
                             #
                             shuffle_gene=Param_shuffle[ex_loop], perm_crossp_model='none',
                             InWeight_gene=Param_IW[ex_loop], InWeight_sheme='random',
                             OutWeight_gene=Param_OW[ex_loop], OutWeight_scheme='random',
                             #
                             training_data=training_data,
                             ReUse_dir=ReUse_dir,  # makes susequent experimets use same material models
                             TestVerify=1,  # USe verification of trained network
                             #
                             Add_Attribute_To_Data=0,
                             PlotExtraAttribute=0,
                             #
                             num_processors=11  # Number of cores to be used
                             )

                # Save list of directories used for Analysis, append each exp-loop
                if ex_loop == 0:

                    # if the new folder does not yet exist, create it
                    if not os.path.exists(experiment_file):
                        os.makedirs(experiment_file)

                    # save current dir to file
                    with open(path_Experiment_dir, 'w') as dir_result_file:
                        dir_result_file.write(dir)
                        dir_result_file.write("\n")

                    if Model == 'R_RN' and ex_loop == 0 and training_data == '2DDS':
                        RRN_ReUse_dir = dir

                    elif Model == 'D_RN' and ex_loop == 0 and training_data == '2DDS':
                        DRN_ReUse_dir = dir

                    elif Model == 'NL_RN' and ex_loop == 0 and training_data == '2DDS':
                        NLRN_ReUse_dir = dir


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

            # in a retry and

            # ########################################################################
            # Run Analysis and save plots to exp dir
            # ########################################################################
            print("\nProducing analysis graphs...")
            obj_anly = analysis(experiment_file, format='png')

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
