import matplotlib.pyplot as plt
import numpy as np
import h5py
from datetime import datetime

import gc
import os
import sys

######################################################
# Custom settings
######################################################

# Import custom file & functions
# import TheModel as model
# import Functions as f
from Module_Models.FunClass import f
from Module_MG.MaterialGraphs import materialgraphs
from Module_Analysis.Analysis import analysis

from Module_Settings.Set_Save import SaveSettings
from Module_Settings.Set_Change import ChangeSettings
from Module_Settings.SaveParamToText import save_param

from Module_Models.Run_Verification import FetchVerificationFitness

from Module_LoadData.NewWeightedData_CreateTemp import Load_NewWeighted_SaveTemp


def blockPrint():  # used to plock print outs
    sys.stdout = open(os.devnull, 'w')


def enablePrint():  # Restore
    sys.stdout = sys.__stdout__

######################################################
# # # OPERATION # # #
# This script can be operated directly by running DE.py
# or it can be run in an experimental set up for multiple runs with different
# parameters, using the exp.py file
######################################################


''' # NOTES
F = mut
CR = crossp, ususally between [0.8,1] for fast convergence
NP =  the population number, generally between [5*D,10*D]

A good initial guess is 10*D. Depending on the difficulty of the
problem NP can be lower than 10*D or must be higher than 10*D
to achieve convergence.
If the parameters are correlated, high values of CR work better.
The reverse is true for no correlation.


num_output_readings = the dimension of the output (e.g 1 bit, 2 bit etc.)


Network types =
        'NL_RN':  # non-linear resistor network
        'R_RN':  # random resistor network
        'custom_RN':  # custom defined resistor network
        'D_RN':  # random resistor network with diodes
        'POD_RN'  # random resistor network with parallel oposing diodes

perm_crossp_model = 'none'   - random
                    'linear' - reduces the perm crossp linearly
                    'box'    - perm crossp is 1 for the first half of the
                               total number of iterations, then 0
                    'quad'   - reduces the perm crossp quadratically
                    'unity'  - always crossp=1

'''

#

#

######################################################
# Main body
######################################################


def RunDE(exp_num_circuits=10, exp_num_repetitions=3,
          plotMG=1, plot_defualt=1, MG_vary_Vconfig=0, MG_vary_PermConfig=0,
          MG_vary_InWeight=0, MG_vary_OutWeight=0,
          MG_animation=0, MG_VaryInWeightsAni=0, MG_VaryOutWeightsAni=0,
          MG_VaryLargeOutWeightsAni=0, MG_VaryLargeInWeightsAni=0,
          experiment=0, experiment_file='bin/', experiment_loop=0, exp_name='test',
          exp_popsize=20, exp_its=5,
          exp_num_input=2, exp_num_output=2, exp_num_config=2,
          exp_the_model='R_RN',
          exp_shuffle_gene=1, exp_perm_crossp_model='none',
          exp_InWeight_gene=0, exp_InWeight_sheme='random',
          exp_OutWeight_gene=0, exp_OutWeight_scheme='random',
          exp_InterpScheme='pn_binary',
          exp_FitScheme='error',
          exp_EN_Dendrite=0, exp_NumDendrite=3, # allow Dendrite averaging of multiple network responces
          exp_training_data='2DDS',
          exp_ReUse_Circuits=0, exp_ReUse_dir='na',
          exp_TestVerify=0,
          exp_num_processors='max',
          exp_Add_Attribute_To_Data=0, exp_PlotExtraAttribute=0,
          exp_UseCustom_NewAttributeData=0, exp_UseOriginalNetowrk=0, exp_NewAttrDataDir='na'):

    # error check
    if exp_num_circuits < 1:
        print("Error (DE.py): exp_num_circuits less than 1")
        exit()
    elif exp_num_repetitions < 1:
        print("Error (DE.py): exp_num_repetitions less than 1")
        exit()


    # # # # # # # # # # # #
    # Collect time and dat stamp of experiment
    now = datetime.now()
    real_d_string = now.strftime("%d_%m_%Y")
    d_string = now.strftime("%Y_%m_%d")
    print("Date:", real_d_string)
    print("Date:", d_string)
    t_string = now.strftime("%H_%M_%S")
    print("Time Stamp:", t_string)

    # # # # # # # # # # # #
    # Set custom resister array

    # for 5 node
    R = [5, 1, 10, 3, 0.1, 1, 2, 0.7, 0.5, 1000]  # in kOhm
    # R = [6.3, 31.8, 10, 98.9, 90.4, 91, 92, 22.8, 30.9, 92]

    # for 6 node
    # R = [5, 1, 10, 3, 0.1, 1, 2, 0.7, 0.5, 6.3, 31.8, 10, 1000, 1000, 1000]  # in kOhm
    # R = [5,7.7,4.8,3.2,2,1.5,8.1,1.5,6.7,4.6,7.5,0.9,1,4.5,2.2]

    # format some data to pass in about the material graphs being plotted
    mg_dict = {'plotMG': plotMG, 'plot_defualt': plot_defualt,
               'MG_vary_Vconfig': MG_vary_Vconfig, 'MG_vary_PermConfig': MG_vary_PermConfig,
               'MG_vary_InWeight': MG_vary_InWeight, 'MG_vary_OutWeight': MG_vary_OutWeight,
               'MG_animation': MG_animation,
               'MG_VaryInWeightsAni': MG_VaryInWeightsAni, 'MG_VaryOutWeightsAni': MG_VaryOutWeightsAni,
               'MG_VaryLargeOutWeightsAni': MG_VaryLargeOutWeightsAni}

    # # # # # # # # # # # #
    # Create the setting object (which si reused)
    SaveSettings(popsize=exp_popsize, its=exp_its,
                 num_input=exp_num_input, num_output=exp_num_output, num_config=exp_num_config,
                 MG_dict=mg_dict,
                 the_model=exp_the_model, R_array_in=R,
                 shuffle_gene=exp_shuffle_gene, OutWeight_gene=exp_OutWeight_gene,
                 InWeight_gene=exp_InWeight_gene, InWeight_sheme=exp_InWeight_sheme,
                 num_output_readings=1,
                 perm_crossp_model=exp_perm_crossp_model,
                 scheme=exp_InterpScheme,
                 FitScheme=exp_FitScheme,
                 OutWeight_scheme=exp_OutWeight_scheme,
                 training_data=exp_training_data,
                 EN_Dendrite=exp_EN_Dendrite, NumDendrite=exp_NumDendrite,  # allow Dendrite averaging of multiple network responces
                 num_circuits=exp_num_circuits, num_repetitions=exp_num_repetitions,
                 ReUse_Circuits=exp_ReUse_Circuits, ReUse_dir=exp_ReUse_dir,
                 TestVerify=exp_TestVerify,
                 num_processors=exp_num_processors,
                 Add_Attribute_To_Data=exp_Add_Attribute_To_Data, PlotExtraAttribute=exp_PlotExtraAttribute,
                 UseCustom_NewAttributeData=exp_UseCustom_NewAttributeData, UseOriginalNetowrk=exp_UseOriginalNetowrk, NewAttrDataDir=exp_NewAttrDataDir)

    loop = 0
    for circuit_loop in range(exp_num_circuits):  # generate new 'material' circuits

        for repetition_loop in range(exp_num_repetitions):  # repetition on current 'material circuit'

            print("")
            print("-------------------------------------------------------------")
            print("Main Loop number:", loop, "/", exp_num_circuits*exp_num_repetitions-1)
            print("Material Circuit:", circuit_loop, "/", exp_num_circuits-1, "Repetition:", repetition_loop, "/", exp_num_repetitions-1)
            print("")
            ParamDict = ChangeSettings(AlsoLoad=1, loop=loop, circuit_loop=circuit_loop, repetition_loop=repetition_loop)

            # Make directory for data to be saved
            if loop == 0:
                # change saved name to make it more distinguishable
                if experiment == 0:
                    new_dir = "Results/%s/__%s__%s__%s" % (d_string, t_string, ParamDict['training_data'], exp_the_model)
                elif experiment == 1:
                    new_dir = "Results/%s/__%s__%s__EXP_%s" % (d_string, t_string, ParamDict['training_data'], exp_name)
                os.makedirs(new_dir)
                ParamDict = ChangeSettings(AlsoLoad=1, SaveDir=new_dir)   # used in Resnet.py to save material models

            if exp_UseCustom_NewAttributeData == 1:
                Load_NewWeighted_SaveTemp(exp_NewAttrDataDir, circuit_loop, repetition_loop, exp_TestVerify)


            # # initialise DE class using settings object
            de_obj = f()

            result = list(de_obj.de())  # list the output yield vals

            print(" ")
            #print("Final result:", result[-1])
            last_res = result[-1]
            best_genome = last_res[0]
            best_fitness = last_res[1]
            print("Best genome: ", last_res[0])
            print("Best fitness:", last_res[1])

            #

            # # # # # # # # # # # #
            # print fitness vs iteration (it zero is original pop)
            best_genome_list_zip, fr_zip, mean_fit_zip, std_fit_zip = zip(*result)

            # Produce list from unzipped/transposed data
            fr = []
            best_genome_list = []
            mean_fit = []
            std_fit = []

            for i in range(len(fr_zip)):
                fr.append(fr_zip[i])
                best_genome_list.append(best_genome_list_zip[i])
                mean_fit.append(mean_fit_zip[i])
                std_fit.append(std_fit_zip[i])



            # add in zeros and copy best genome to ensure one loop fills up
            # itteration number sized list
            if len(fr) < ParamDict['its']+1:  # +1 is to include initial pop fitness
                location = len(fr)
                while True:
                    fr.append(best_fitness)
                    best_genome_list.append(best_genome)
                    mean_fit.append(np.nan)
                    std_fit.append(np.nan)

                    location = location + 1

                    if location >= ParamDict['its']+1:
                        break


            fig_evole = plt.figure()
            axes_evole = fig_evole.add_axes([0.125, 0.125, 0.75, 0.75])
            axes_evole.plot(fr)  # NOTE: This includes the best fitness of original population
            axes_evole.set_title('Best fitness variation over the evolution period (no Verification)')
            axes_evole.set_xlabel('Iteration')
            axes_evole.set_ylabel('Best Fitness')

            #

            # # # # # # # # # # # #
            # # write data to DE hdf5 group
            location = "%s/data_%d_rep%d.hdf5" % (new_dir, circuit_loop, repetition_loop)
            with h5py.File(location, 'a') as hdf:
                G = hdf.create_group('DE_data')  # create group
                G.create_dataset('fitness', data=fr)
                G.create_dataset('best_genome', data=best_genome_list)
                G.create_dataset('pop_mean_fit', data=mean_fit)
                G.create_dataset('pop_std_fit', data=std_fit)


            ###################################################################
            # Perform Verification on trained model
            # For each circuit and each repetition
            ###################################################################
            if ParamDict['TestVerify'] == 1:
                scheme_fitness, Verification_fitness = FetchVerificationFitness(best_genome)

                # # # # # # # # # # # #
                # # write data to DE hdf5 group
                location = "%s/data_%d_rep%d.hdf5" % (new_dir, circuit_loop, repetition_loop)
                with h5py.File(location, 'a') as hdf:
                    hdf.create_dataset('/DE_data/veri_fit', data=Verification_fitness)
                    hdf.create_dataset('/DE_data/veri_scheme_fit', data=scheme_fitness)


                new_title = 'Best fitness variation over the evolution period\n Verification (error) Fit = %.4f, Scheme (%s) Fit = %.4f' % (Verification_fitness, ParamDict['FitScheme'], scheme_fitness)
                axes_evole.set_title(new_title)

            # Save figure for current loops best fitness variation
            fig0_path = "%s/%d_Rep%d_FIG_FitvIt.pdf" % (new_dir, circuit_loop, repetition_loop)
            fig_evole.savefig(fig0_path)
            plt.close(fig_evole)

            #



            #

            ###################################################################
            # Print material graphs
            ###################################################################
            MG_vary_Vconfig_NumGraph = 0  # used for the meta data
            MG_vary_PermConfig_NumGraph = 0  # used for the meta data

            if plotMG == 1:
                MG_obj = materialgraphs(de_obj.model_obj.network_obj)
                MG_obj.MG(plot_defualt, last_res)


                if MG_vary_Vconfig == 1:
                    if ParamDict['num_config'] <= 2:
                        MG_obj.MG_VaryConfig()
                        MG_vary_Vconfig_NumGraph = len(MG_obj.Vconfig_1)*len(MG_obj.Vconfig_2)
                    elif ParamDict['num_config'] == 3:
                        MG_obj.MG_VaryConfig3()
                    else:
                        print("Too many config inputs to print 2d MG map!")

                if MG_vary_PermConfig == 1:
                    selected_OW = 0  # don't randomly vary OW
                    MG_obj.MG_VaryPerm(OutWeight=selected_OW)
                    MG_vary_PermConfig_NumGraph = MG_obj.PC_num_chunks

                if MG_vary_InWeight == 1:
                    MG_obj.MG_VaryInWeights(assending=1)

                if MG_vary_OutWeight == 1:
                    MG_obj.MG_VaryOutWeights(assending=1)

                if MG_VaryInWeightsAni == 1:
                    MG_obj.MG_VaryInWeightsAni()

                if MG_VaryOutWeightsAni == 1:
                    MG_obj.MG_VaryOutWeightsAni()

                if MG_VaryLargeOutWeightsAni == 1:
                    MG_obj.MG_VaryLargeOutWeightsAni()

                if MG_VaryLargeInWeightsAni == 1:
                    MG_obj.MG_VaryLargeInWeightsAni()

            else:
                MG_obj = 'na'

            # increment loop
            loop = loop + 1

        # '''

    #########################################################################
    # Save some data about the experiment to text
    #########################################################################
    now = datetime.now()
    d_string_fin = now.strftime("%d_%m_%Y")
    t_string_fin = now.strftime("%H_%M_%S")
    Deets = save_param(ParamDict, experiment, now,
                       MG_obj, plotMG, MG_vary_Vconfig, MG_vary_PermConfig,
                       exp_num_circuits, exp_num_repetitions,
                       MG_vary_Vconfig_NumGraph, MG_vary_PermConfig_NumGraph)
    #

    # ########################################################################
    # Create a new file containing the experiment path, file labled with
    # simulation (Network) model, and number of runs
    # ########################################################################:

    if experiment == 1:

        # if the new folder does not yet exist, create it
        if not os.path.exists(experiment_file):
            os.makedirs(experiment_file)

        # Save the deets under a descriptive name
        path_Experiment_List = '%s/ExpLoop%d.txt' % (experiment_file, experiment_loop)
        file2 = open(path_Experiment_List, "w")
        file2.write(Deets)
        file2.close()

    # ########################################################################
    # Run Analysis
    # ########################################################################
    blockPrint()
    if exp_num_circuits*exp_num_repetitions > 1:
        sel_dict = {'plt_mean':1,'plt_std':1,'plt_finalveri':1,'plt_popmean':1,'plt_hist':1}
    else:
        sel_dict = {'plt_mean':0,'plt_std':0,'plt_finalveri':1,'plt_popmean':1,'plt_hist':0}

    print("\nProducing analysis graphs...")
    obj_anly = analysis(new_dir)

    obj_anly.Plt_basic(sel_dict=sel_dict, Save_NotShow=1, fill=1, ExpErrors=1, StandardError=1)
    #obj_anly.Plt__ani(Save_NotShow=1, PlotOnly='all')
    obj_anly.Plt_mg(Save_NotShow=1, VoW_ani=1, ViW_ani=1, VoiW_ani=1, VloW_ani=1)
    enablePrint()
    # ########################################################################
    # Attempt to clean up memory
    # ########################################################################
    del de_obj

    if plotMG == 1:
        del MG_obj

    gc.collect()

    # # Print end time
    print("\nfin at:", t_string_fin)

    # # Return the directories
    return new_dir

    #

    #

    #

# script


if __name__ == "__main__":

    # Run Script
    RunDE(exp_num_circuits=1, exp_num_repetitions=1,
          exp_its=2, exp_popsize=20, exp_the_model='D_RN',  # R_RN
          exp_num_input=2, exp_num_output=2, exp_num_config=3,
          plotMG=0, plot_defualt=0, MG_vary_Vconfig=1, MG_vary_PermConfig=0,
          MG_vary_InWeight=0, MG_vary_OutWeight=0,
          MG_animation=0, MG_VaryInWeightsAni=0, MG_VaryOutWeightsAni=0,
          MG_VaryLargeOutWeightsAni=0, MG_VaryLargeInWeightsAni=0,
          exp_training_data='2DDS',  # 2DDS, O2DDS, can2DDS MMDS
          exp_shuffle_gene=1, exp_perm_crossp_model='none',
          exp_InWeight_gene=1, exp_InWeight_sheme='random',
          exp_OutWeight_gene=1, exp_OutWeight_scheme='random',
          exp_InterpScheme='pn_binary',
          exp_FitScheme='error',
          exp_EN_Dendrite=0, exp_NumDendrite=4,  # sum several sets of differently weighted network responces
          exp_TestVerify=1,
          exp_Add_Attribute_To_Data=1, exp_PlotExtraAttribute=1,
          exp_num_processors=4)


# fin
