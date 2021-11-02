from datetime import datetime
import os
import copy

from runEiM import RunEiM
from mod_analysis.Analysis import analysis
from mod_settings.GenParam import GenSimParam, LoadPrm



# Main script
if __name__ == "__main__":

    # load Template Raw (undormatted) Paramaters
    trprm = LoadPrm(param_file='')

    # set the model types which the experiments will run over
    # Model_types = ['R_RN', 'D_RN', 'NL_RN']
    # datasets = ['2DDS', 'con2DDS', 'flipped_2DDS']

    Model_types = ['D_RN', 'NL_RN', 'R_RN']
    # datasets = ['c2DDS', 'MMDS', 'd2DDS', 'flipped_d2DDS']
    datasets = ['d2DDS', 'flipped_d2DDS']

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
            training_data = dataset
            num_inputs = 2

            #

            # #Name the experiment
            exp_name = 'VaryParams__%s' % (Model)
            Param_Varying = 'Vary Schemes: shuffle, input and output weights'
            experiment_file = '%s/%s__%s___EXP_%s' % (training_data, d_string, t_string, exp_name)

            # # Set parameter which will vary
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


            # # Intiialise Some variables
            ReUse_dir = 'na'  # set defualt to 'na, this is assigned at the end of first loop'

            # # load in previous files to use previous material processors
            if Model == 'R_RN':
                ReUse_dir = 'Results/15Materials/RRN_10node'
                num_nodes = 10
            elif Model == 'D_RN':
                ReUse_dir = 'Results/15Materials/DRN_10node'
                num_nodes = 10
            elif Model == 'NL_RN':
                ReUse_dir = 'Results/15Materials/NLRN_10node'
                num_nodes = 10

            if '2DDS' in dataset:
                num_inputs = 2
                num_output = 3
                num_config = num_nodes - num_inputs - num_output
            elif 'MMDS' in dataset:
                num_inputs = 4
                num_output = 5
                num_config = num_nodes - num_inputs - num_output
            #

            num_experiments = len(Param_shuffle)
            for ex_loop, Param in enumerate(Param_array):

                print("##################################################")
                print("Experiment", ex_loop, "Out of:", num_experiments-1)
                print("##################################################")

                # # Create a raw (unformatted) paramater file from the loadted template
                rprm = copy.deepcopy(trprm)

                # Alter Prms
                rprm['ReUse_dir'] = ReUse_dir
                rprm['num_systems'] = 15
                rprm['num_repetitions'] = 5

                rprm['DE']['epochs'] = 30  # 50
                rprm['DE']['training_data'] = training_data
                rprm['DE']['batch_size'] = 0
                rprm['DE']['batch_scheme'] = 'none'
                #rprm['DE']['batch_window_size'] = W_sizes[ex_loop]
                rprm['DE']['BreakOnNcomp'] = 'na'  # Used to break!

                rprm['DE']['IntpScheme'] = 'pn_binary'  # 'pn_binary'
                rprm['DE']['FitScheme'] = 'error'

                rprm['network']['model'] = Model
                rprm['network']['num_input'] = num_inputs
                rprm['network']['num_config'] = num_config
                rprm['network']['num_output'] = num_output

                rprm['genome']['Shuffle']['active'] = Param_shuffle[ex_loop]
                rprm['genome']['InWeight']['active'] = Param_IW[ex_loop]
                rprm['genome']['OutWeight']['active'] = Param_OW[ex_loop]

                rprm['DE']['save_fitvit'] = 0
                rprm['mg']['plotMG'] = 0

                # Gen final prm dict
                prm = GenSimParam(param_file=rprm,
                                  experiment=1,
                                  experiment_file=experiment_file,
                                  exp_name=exp_name)

                # # Run EiM
                RunEiM(prm, experiment_loop=ex_loop)

                # Save list of directories used for Analysis, append each exp-loop
                if ex_loop == 0:

                    # Set the exp to use the same material each paramater change
                    # ReUse_dir = prm['SaveDir']

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
