import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import h5py
from datetime import datetime
import yaml

import gc
import os
import sys

from sklearn.decomposition import PCA
from sklearn import preprocessing

import sklearn.cluster as cl
import sklearn.metrics as met

from mod_MG.MaterialGraphs import materialgraphs
from mod_analysis.Analysis import analysis

from mod_load.LoadData import Load_Data

from mod_settings.GenParam import GenSimParam
from mod_settings.Set_Change import ChangeSettings
from mod_settings.SaveParamToText import save_param

from mod_load.NewWeightedData_CreateTemp import Load_NewWeighted_SaveTemp

from mod_algorithm.DiffEvo import de

"""import warnings
warnings.filterwarnings('error')"""

#matplotlib.use('Agg')


def blockPrint():  # used to plock print outs
    sys.stdout = open(os.devnull, 'w')


def enablePrint():  # Restore
    sys.stdout = sys.__stdout__


######################################################
# # # OPERATION # # #
# This script can be operated directly by running DE.py
# or it can be run in an experimental set up for multiple runs with different
# parameters, using the exp.py file
#
# Works with PySpice Version: 1.3.2
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

'''

#

#

######################################################
# Main body
######################################################


def RunEiM(param_file='',
           experiment=0, experiment_file='bin/', experiment_loop=0, exp_name='test',
           **kwargs):

    # # Collect time and dat stamp of experiment # #
    now = datetime.now()
    d_string = now.strftime("%Y_%m_%d")
    t_string = now.strftime("%H_%M_%S")
    print("\n>> Time Stamp:", t_string, " <<")

    # # Produce PAramater File
    prm, param_file = GenSimParam(param_file, algorithm='EiM', **kwargs)  # save_param_file='DEL',


    # # initialise DE class using settings object
    de_obj = de(param_file)

    loop = 0
    for circuit_loop in range(prm['num_circuits']):  # generate new 'material' circuits

        for repetition_loop in range(prm['num_repetitions']):  # repetition on current 'material circuit'

            print("\n-------------------------------------------------------------")
            print("Main Loop number:", loop, "/", prm['num_circuits']*prm['num_repetitions']-1)
            print("Material Circuit:", circuit_loop, "/", prm['num_circuits']-1, "Repetition:", repetition_loop, "/", prm['num_repetitions']-1)

            # Make directory for data to be saved
            if loop == 0:

                # change saved name to make it more distinguishable
                if experiment == 0:
                    new_dir = "Results/%s/__%s__%s__%s__%s" % (d_string, t_string, prm['DE']['training_data'], prm['network']['model'], prm['algorithm'])
                elif experiment == 1:
                    new_dir = "Results/%s/__%s__%s__EXP_%s__%s" % (d_string, t_string, prm['DE']['training_data'], exp_name, prm['algorithm'])
                os.makedirs(new_dir)
                prm = ChangeSettings(param_file, SaveDir=new_dir)   # used in Resnet.py to save material models

                with open(r'%s/Experiment_MetaData%s.yaml' % (new_dir, param_file), 'w') as sfile:
                    yaml.dump(prm, sfile)

            if prm['DE']['UseCustom_NewAttributeData'] == 1:
                Load_NewWeighted_SaveTemp(prm['DE']['NewAttrDataDir'], circuit_loop, repetition_loop, prm['DE']['TestVerify'])


            # run the object on the new settings
            best_genome_list, mean_fit_list, std_fit_list, unzipped_best_results_list  = de_obj.perform_de(circuit_loop, repetition_loop)  # list the output yield vals
            best_fit_list, best_responceY_list, best_err_fit_list, best_Vout_list = unzipped_best_results_list




            best_genome = best_genome_list[-1]
            best_fitness = best_fit_list[-1]
            best_Vout = best_Vout_list[-1]
            print("Best fitness:", best_fitness, " Using the best genome:", best_genome)

            scaled_data = preprocessing.scale(best_Vout)
            pca = PCA() # create a PCA object
            pca.fit(scaled_data) # do the math
            eigenvalues_raw = pca.explained_variance_
            eigenvalues = eigenvalues_raw/np.max(eigenvalues_raw)
            if np.max(eigenvalues_raw) == 0:
                print("Max eigen val is zero? Eigenvalues:", eigenvalues_raw)
            max_entropy = np.log(len(eigenvalues_raw))
            entropy = 0
            for l in eigenvalues:
                entropy = entropy - (l*np.log(l))
            print("Raw Output Voltages (PCA) Entropy=%f/%f (larger is better)" % (entropy, max_entropy))

            # add in zeros and copy best genome to ensure one loop fills up
            # itteration number sized list
            if len(best_fit_list) < prm['DE']['its']+1:  # +1 is to include initial pop fitness
                location = len(best_fit_list)
                while True:
                    best_fit_list.append(best_fitness)
                    best_genome_list.append(best_genome)
                    mean_fit_list.append(np.nan)
                    std_fit_list.append(np.nan)

                    location = location + 1

                    if location >= prm['DE']['its']+1:
                        break

            fig_evole = plt.figure(tight_layout=False)
            axes_evole = fig_evole.add_axes([0.125, 0.125, 0.75, 0.75])
            axes_evole.plot(best_fit_list)  # NOTE: This includes the best fitness of original population
            axes_evole.set_title('Best fitness variation over the evolution period (no Verification)\nOutput (PCA) entropy=%.3f/%.3f' % (entropy, max_entropy))
            axes_evole.set_xlabel('Iteration')
            axes_evole.set_ylabel('Best Fitness')

            #

            #

            # format best_genome_list for hdf5
            formatted_genome_list = []
            for genome in best_genome_list:
                formatted_genome_list.append(np.concatenate(np.asarray(genome)))
            formatted_genome_array = np.asarray(formatted_genome_list).astype(np.float64)
            #print("formatted_genome", formatted_genome_array)

            # # # # # # # # # # # #
            # # write data to DE hdf5 group
            location = "%s/data.hdf5" % (new_dir)
            with h5py.File(location, 'a') as hdf:
                data_group = hdf.create_group("%d_rep%d" % (circuit_loop, repetition_loop))
                G = data_group.create_group('DE_data')  # create group
                G.create_dataset('fitness', data=best_fit_list)
                G.create_dataset('best_genome', data=formatted_genome_array)
                G.create_dataset('gen_grouping', data=prm['genome']['grouping'])
                G.create_dataset('pop_mean_fit', data=mean_fit_list)
                G.create_dataset('pop_std_fit', data=std_fit_list)
                G.create_dataset('Y_data', data=best_responceY_list[-1])
                G.create_dataset('best_Vout', data=best_Vout)
                G.create_dataset('max_entropy', data=max_entropy)
                G.create_dataset('OutputLayer_threshold', data=np.nan)

            ###################################################################
            # Perform Verification on trained model
            # For each circuit and each repetition
            ###################################################################
            if prm['DE']['TestVerify'] == 1:
                scheme_fitness, rY, Verification_fitness, Vout = de_obj.cap.solve_processor(best_genome, cir=circuit_loop, rep=repetition_loop, the_data='veri')

                # anomoly detection !
                if best_fitness <= 0.1 and scheme_fitness == 0.5:
                    raise ValueError("Anomaly Detected!")

                scaled_data = preprocessing.scale(Vout)
                pca = PCA() # create a PCA object
                pca.fit(scaled_data) # do the math
                eigenvalues_raw = pca.explained_variance_
                eigenvalues = eigenvalues_raw/np.max(eigenvalues_raw)
                entropyVeri = 0
                for landa in eigenvalues:
                    entropyVeri = entropyVeri - (landa*np.log(landa))

                # # # # # # # # # # # #
                # # write data to DE hdf5 group
                location = "%s/data.hdf5" % (new_dir)
                with h5py.File(location, 'a') as hdf:
                    hdf.create_dataset("%d_rep%d/DE_data/veri_fit" % (circuit_loop, repetition_loop), data=Verification_fitness)
                    hdf.create_dataset("%d_rep%d/DE_data/veri_scheme_fit" % (circuit_loop, repetition_loop), data=scheme_fitness)
                    hdf.create_dataset("%d_rep%d/DE_data/veri_entropy" % (circuit_loop, repetition_loop), data=entropyVeri)
                    hdf.create_dataset("%d_rep%d/DE_data/veri_Vout" % (circuit_loop, repetition_loop), data=Vout)

                new_title = 'Best fitness variation over the evolution period\n Verification (error) Fit = %.4f, Scheme (%s) Fit = %.4f\nOutput (PCA) entropy=%.3f/%.3f' % (Verification_fitness, prm['DE']['FitScheme'], scheme_fitness, entropy, max_entropy)
                axes_evole.set_title(new_title)

            # Save figure for current loops best fitness variation
            fig0_path = "%s/%d_Rep%d_FIG_FitvIt.png" % (new_dir, circuit_loop, repetition_loop)
            if prm['DE']['save_fitvit'] == 1:
                fig_evole.savefig(fig0_path, dpi=300)
            plt.close(fig_evole)

            #

            #

            ###################################################################
            # Print material graphs
            ###################################################################
            if prm['mg']['plotMG'] == 1 and prm['network']['num_input'] == 2:
                MG_obj = materialgraphs(circuit_loop, repetition_loop, param_file=param_file)
                MG_obj.MG(prm['mg']['plot_defualt'], [best_genome, best_fitness])

                if prm['mg']['MG_vary_Vconfig'] == 1:
                    if prm['network']['num_config'] <= 2:
                        MG_obj.MG_VaryConfig()
                    elif prm['network']['num_config'] == 3:
                        MG_obj.MG_VaryConfig3()
                    else:
                        print("Too many config inputs to print 2d MG map!")

                if prm['mg']['MG_vary_PermConfig'] == 1:
                    selected_OW = 0  # don't randomly vary OW
                    MG_obj.MG_VaryPerm(OutWeight=selected_OW)

                if prm['mg']['MG_vary_InWeight'] == 1:
                    MG_obj.MG_VaryInWeights(assending=1)

                if prm['mg']['MG_vary_OutWeight'] == 1:
                    MG_obj.MG_VaryOutWeights(assending=1)

                if prm['mg']['MG_VaryInWeightsAni'] == 1:
                    MG_obj.MG_VaryInWeightsAni()

                if prm['mg']['MG_VaryOutWeightsAni'] == 1:
                    MG_obj.MG_VaryOutWeightsAni()

                if prm['mg']['MG_VaryLargeOutWeightsAni'] == 1:
                    MG_obj.MG_VaryLargeOutWeightsAni()

                if prm['mg']['MG_VaryLargeInWeightsAni'] == 1:
                    MG_obj.MG_VaryLargeInWeightsAni()

            else:
                MG_obj = 'na'

            # # # ########
            # If clustering
            if 'Kmean' in prm['DE']['IntpScheme'] and 'Kmean' not in prm['DE']['FitScheme']:

                data_X, data_y = Load_Data('train', prm)

                model = cl.KMeans(prm['DE']['num_classes'])
                model.fit(data_X)
                yhat_og = model.predict(data_X) # assign a cluster to each example
                og_com = met.completeness_score(data_y, yhat_og)

                fig, ax = plt.subplots(ncols=3, nrows=1, sharey=True, sharex=True)

                ax[0].scatter(data_X[:,0], data_X[:,1], c=data_y)
                ax[0].set_title('OG Data')
                ax[0].set_xlabel('a1')
                ax[0].set_ylabel('a2')
                ax[0].set_aspect('equal', adjustable='box')

                ax[1].scatter(data_X[:,0], data_X[:,1], c=yhat_og)
                ax[1].set_title('Clusters On OG')
                ax[1].set_xlabel('a1')
                ax[1].set_aspect('equal', adjustable='box')

                ax[2].scatter(data_X[:,0], data_X[:,1], c=best_responceY_list[-1])
                ax[2].set_title('Predicted Clusters')
                ax[2].set_xlabel('a1')
                ax[2].set_aspect('equal', adjustable='box')

                fig.suptitle('IntpScheme: %s, FitScheme: %s.' % (prm['DE']['IntpScheme'], prm['DE']['FitScheme']))


                fig0_path = "%s/%d_Rep%d_FIG_Cluster.png" % (new_dir, circuit_loop, repetition_loop)
                fig.savefig(fig0_path, dpi=300)
                plt.close(fig)

            elif 'Kmean' not in prm['DE']['IntpScheme'] and 'Kmean' in prm['DE']['FitScheme']:
                rY = best_responceY_list[-1]

                if len(rY[0, :]) == 2:

                    data_X, data_y = Load_Data('train', prm)

                    fig, ax = plt.subplots()
                    ax.scatter(rY[:,0], rY[:,1], c=data_y)
                    ax.set_title('Output')
                    ax.set_xlabel('$op_1$')
                    ax.set_ylabel('$op_2$')
                    fig.suptitle('IntpScheme: %s, FitScheme: %s.' % (prm['DE']['IntpScheme'], prm['DE']['FitScheme']))
                    fig0_path = "%s/%d_Rep%d_FIG_TransformedOutput.png" % (new_dir, circuit_loop, repetition_loop)
                    fig.savefig(fig0_path, dpi=300)
                    plt.close(fig)

                elif len(rY[0, :]) == 1:

                    data_X, data_y = Load_Data('train', prm)

                    fig, ax = plt.subplots()
                    ax.scatter(np.arange(len(rY)), rY, c=data_y)
                    ax.set_title('Output')
                    ax.set_xlabel('instance')
                    ax.set_ylabel('$op_1$')
                    fig.suptitle('IntpScheme: %s, FitScheme: %s.' % (prm['DE']['IntpScheme'], prm['DE']['FitScheme']))
                    fig0_path = "%s/%d_Rep%d_FIG_TransformedOutput.png" % (new_dir, circuit_loop, repetition_loop)
                    fig.savefig(fig0_path, dpi=300)
                    plt.close(fig)


            # increment loop and clean up
            gc.collect()
            loop = loop + 1

        # '''

    #########################################################################
    # Save some data about the experiment to text
    #########################################################################
    now = datetime.now()
    d_string_fin = now.strftime("%d_%m_%Y")
    t_string_fin = now.strftime("%H_%M_%S")
    Deets = save_param(prm, experiment, now)
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
        path_Experiment_List = '%s/ExpLoop%s.txt' % (experiment_file, str(experiment_loop))
        file2 = open(path_Experiment_List, "w")
        file2.write(Deets)
        file2.close()

    # ########################################################################
    # Run Analysis
    # ########################################################################
    blockPrint()
    if prm['num_circuits']*prm['num_repetitions'] > 1:
        sel_dict = {'plt_mean':1,'plt_std':1,'plt_finalveri':1,'plt_popmean':1,'plt_hist':1}
    else:
        sel_dict = {'plt_mean':0,'plt_std':0,'plt_finalveri':0,'plt_popmean':1,'plt_hist':0}

    print("\nProducing analysis graphs...")
    obj_anly = analysis(new_dir, format='png')

    obj_anly.Plt_basic(sel_dict=sel_dict, Save_NotShow=1, fill=1, ExpErrors=0, StandardError=1)
    obj_anly.Plt__ani(Save_NotShow=1, PlotOnly='all')
    obj_anly.Plt_mg(Save_NotShow=1, Bgeno=1, Dgeno=1, VC=1, VP=1,
                    VoW=1, ViW=1, VoW_ani=1, ViW_ani=1, VoiW_ani=1, VloW_ani=1, VliW_ani=1,
                    titles='on')
    enablePrint()

    # # Clean Up # #
    print("\nfin at:", t_string_fin)
    plt.close('all')

    # # delete temp param files
    os.remove('Temp_Param%s.yaml' % (str(param_file)))

    # Return the directory
    return new_dir

#

#

#

#

#

# # script # #


if __name__ == "__main__":

    # Run Script
    #mp.set_start_method('fork')
    ReUse_dir = 'na'

    ReUse_dir = 'Results/2020_11_11/__10_41_27__con2DDS__real_D_RN__EiM'

    RunEiM(num_circuits=1, num_repetitions=1,
           model='NL_RN',  # R_RN, D_RN, NL_RN, NL_uRN, custom_RN
           #IntpScheme='raw',
           #FitScheme='KmeanDist',
           #num_readout_nodes=2, OutWeight_gene=1, IntpScheme='HOW',
           #ReUse_dir=ReUse_dir,
           num_processors=10)

#

# fin
