import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import h5py
from datetime import datetime
import yaml

import gc
import os
import sys


from sklearn import preprocessing

import sklearn.cluster as cl
import sklearn.metrics as met

from mod_MG.MaterialGraphs import materialgraphs
from mod_analysis.Analysis import analysis
from mod_analysis.EiMplotters import *
from mod_analysis.plt_rc import rc_plotter

from mod_material.eim_processor import material_processor

from mod_settings.GenParam import GenSimParam, LoadPrm
from mod_settings.Set_Change import ChangeSettings

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


def RunEiM(prm, experiment_loop=0):

    # # initialise DE class using settings object
    DEobj = de(prm)  # pass in prm file

    #

    loop = 0
    for system_loop in range(prm['num_systems']):  # generate new 'material' circuits

        for repetition_loop in range(prm['num_repetitions']):  # repetition on current 'material circuit'

            print("\n----------------------------------------------------------------------")
            print("Main Loop number:", loop, "/", prm['num_systems']*prm['num_repetitions']-1, " ( Material System:", system_loop, "/", prm['num_systems']-1, ", Repetition:", repetition_loop, "/", prm['num_repetitions']-1, ")")

            # Make directory for data to be saved
            if loop == 0:
                with open(r'%s/Experiment_MetaData.yaml' % (prm['SaveDir']), 'w') as sfile:
                    yaml.dump(prm, sfile)

            if prm['DE']['UseCustom_NewAttributeData'] == 1:
                Load_NewWeighted_SaveTemp(prm['DE']['NewAttrDataDir'], system_loop, repetition_loop, prm['DE']['TestVerify'])

            # # run DE with the new settings
            DEobj.perform_de(system_loop, repetition_loop)
            train_bFits, train_bYs, train_bErrFits, train_bVouts, train_bRLs = DEobj.train_bResult_unzipped
            vali_bFits_RAW, vali_bYs, vali_bErrFits, vali_bVouts, vali_bRLs = DEobj.vali_bResult_unzipped
            Global_bFits_RAW, Global_bYs, Global_bFits_err, Global_bVouts, Global_bRLs = DEobj.Gbest_bResult_unzipped

            # # Toggle the saved Global and valid pop fit saved, dependign on
            # # what was used (Training fit not always equal to valid fit)
            if prm['DE']['Gbest_fit'] == 'raw':
                Global_bFits = Global_bFits_RAW
                vali_bFits = vali_bFits_RAW
            elif prm['DE']['Gbest_fit'] == 'error':
                Global_bFits = Global_bFits_err
                vali_bFits = vali_bErrFits

            best_genome = DEobj.Gbest_bGenomes[-1]
            best_RidgeLayer = Global_bRLs[-1]
            train_bFit = train_bFits[-1]
            best_Vout = Global_bVouts[-1]

            if prm['DE']['batch_size'] == 0:
                print("Final Training fitness:", train_bFit, " Using the best genome:", best_genome)
                bGV_fit = 'na'
            else:
                bGV_fit = Global_bFits[-1]
                bGV_fit = round(bGV_fit, 3)
                print("Final Global Validation fitness:", bGV_fit, " Using the best genome:", best_genome)

            #

            entropy, max_entropy = PCA_entropy(best_Vout)
            print("Raw Output Voltages (PCA) Entropy=%f/%f (larger is better)" % (entropy, max_entropy))

            # add in zeros and copy best genome to ensure one loop fills up
            # itteration number sized list
            """train_bFits = list(train_bFits)
            if len(train_bFits) < prm['DE']['epochs']:  # +1 is to include initial pop fitness
                location = len(train_bFits)
                while True:
                    train_bFits.append(train_bFit)
                    DEobj.train_bGenomes.append(best_genome)
                    DEobj.train_meanFits.append(np.nan)
                    DEobj.train_stdFits.append(np.nan)

                    location = location + 1

                    if location >= prm['DE']['epochs']:
                        break"""

            #

            #

            # format DEobj.train_bGenomes for hdf5
            formatted_bTGenomes = []
            for genome in DEobj.train_bGenomes:
                formatted_bTGenomes.append(np.concatenate(np.asarray(genome)))
            formatted_bTGenomes = np.asarray(formatted_bTGenomes).astype(np.float64)

            formatted_GbGenomes = []
            for genome in DEobj.Gbest_bGenomes:
                formatted_GbGenomes.append(np.concatenate(np.asarray(genome)))
            formatted_GbGenomes = np.asarray(formatted_GbGenomes).astype(np.float64)

            # # # # # # # # # # # #
            # # write data to DE hdf5 group
            location = "%s/data.hdf5" % (prm['SaveDir'])
            with h5py.File(location, 'a') as hdf:
                data_group = hdf.create_group("%d_rep%d" % (system_loop, repetition_loop))
                G = data_group.create_group('DE_data')  # create group
                G.create_dataset('gen_grouping', data=prm['genome']['grouping'])

                G_train = G.create_group('training')  # create group
                G_train.create_dataset('best_fits', data=train_bFits)
                G_train.create_dataset('best_genomes', data=formatted_bTGenomes)
                G_train.create_dataset('pop_mean_fits', data=DEobj.train_meanFits)
                G_train.create_dataset('pop_std_fits', data=DEobj.train_stdFits)
                G_train.create_dataset('best_Y_data', data=train_bYs[-1])
                G_train.create_dataset('best_Vout', data=best_Vout)
                G_train.create_dataset('max_entropy', data=max_entropy)
                G_train.create_dataset('best_entropy', data=entropy)
                G_train.create_dataset('Ncomps', data=DEobj.batch_Ncomps)

                G_best = G.create_group('global_best')  # create group
                Gbest_genome = G_best.create_dataset('best_genomes', data=formatted_GbGenomes)
                if prm['DE']['batch_size'] == 0:
                    Gbest_genome.attrs['type'] = 'bTrain'
                else:
                    Gbest_genome.attrs['type'] = 'bVali'

                Gbest_fit = G_best.create_dataset('fits', data=Global_bFits)
                G_best.create_dataset('Ncomps', data=DEobj.epoch_Ncomps)
                G_best.create_dataset('error_fits', data=Global_bFits_err)
                if prm['DE']['Gbest_fit'] == 'raw':
                    Gbest_fit.attrs['type'] = 'raw'
                elif prm['DE']['Gbest_fit'] == 'error':
                    Gbest_fit.attrs['type'] = 'error'
                #

                if prm['DE']['batch_size'] != 0:
                    G_valid = G.create_group('validation')  # create group
                    G_valid.create_dataset('non_global_fits', data=vali_bFits)
                    G_valid.create_dataset('fits', data=Global_bFits)
                    G_valid.create_dataset('num_batches', data=DEobj.lobj.num_batches)
                    G_valid.create_dataset('batch_size_list', data=DEobj.lobj.batch_size)
                    G_valid.create_dataset('Ncomps', data=DEobj.epoch_Ncomps)
                #

                G_ridge = G.create_group('ridge')  # create group
                if prm['op_layer'] == 'ridge':
                    best_weights, best_bias, Threshold = best_RidgeLayer
                    G_ridge.create_dataset('weights', data=best_weights)
                    G_ridge.create_dataset('bias', data=best_bias)
                    G_ridge.create_dataset('threshold', data=Threshold)
                else:
                    G_ridge.create_dataset('threshold', data=np.nan)



            ###################################################################
            # Perform Test Dataset on trained model
            ###################################################################
            txdata, tydata = DEobj.lobj.fetch_data('test')
            #results, GenList_results = self.cap.run_processor(pop_denorm, syst, rep, xdata, ydata, ret_str='both', the_data='train')
            scheme_fitness, rY, TestFit, Vout = DEobj.cap.run_processor(best_genome, syst=system_loop, rep=repetition_loop,
                                                                         input_data=txdata, target_output=tydata,
                                                                         ridge_layer=best_RidgeLayer, the_data='test')
            noise = 5
            ntxdata, ntydata = DEobj.lobj.fetch_data('test', noise=noise, noise_type='per')
            noise_scheme_fitness, noise_rY, noise_Veri_fit, noise_Vout = DEobj.cap.run_processor(best_genome, syst=system_loop, rep=repetition_loop,
                                                                                                  input_data=ntxdata, target_output=ntydata,
                                                                                                  ridge_layer=best_RidgeLayer, the_data='test')

            print("Test Error Fit =", TestFit, ", Test %s Per noise Fit =" % (noise), noise_Veri_fit, ", Scheme fitness:", scheme_fitness)

            # anomoly detection !

            if prm['DE']['batch_size'] == 0:
                best_f = train_bErrFits[-1] # train_bFit
            else:
                best_f = Global_bFits_err[-1] # Global_bFits_RAW[-1]

            # # Check errors for anomolus result
            if best_f <= 0.1 and TestFit > 0.5:
                raise ValueError("Anomaly Detected!")

            # calc PCA entropy
            test_entropy, max_entropy = PCA_entropy(Vout)

            # # # # # # # # # # # #
            # # write data to DE hdf5 group
            location = "%s/data.hdf5" % (prm['SaveDir'])
            with h5py.File(location, 'a') as hdf:
                hdf.create_dataset("%d_rep%d/DE_data/test/fit" % (system_loop, repetition_loop), data=TestFit)
                hdf.create_dataset("%d_rep%d/DE_data/test/scheme_fit" % (system_loop, repetition_loop), data=scheme_fitness)
                hdf.create_dataset("%d_rep%d/DE_data/test/entropy" % (system_loop, repetition_loop), data=test_entropy)
                hdf.create_dataset("%d_rep%d/DE_data/test/Vout" % (system_loop, repetition_loop), data=Vout)

                hdf.create_dataset("%d_rep%d/DE_data/noisy_test/fit" % (system_loop, repetition_loop), data=noise_Veri_fit)
                hdf.create_dataset("%d_rep%d/DE_data/noisy_test/scheme_fit" % (system_loop, repetition_loop), data=noise_scheme_fitness)
                hdf.create_dataset("%d_rep%d/DE_data/noisy_test/Vout" % (system_loop, repetition_loop), data=noise_Vout)
                hdf.create_dataset("%d_rep%d/DE_data/noisy_test/noise_per" % (system_loop, repetition_loop), data=noise)

                noise_list = np.arange(0, 10.5, 0.5)
                hdf.create_dataset("%d_rep%d/DE_data/noisy_test/noise_per_list" % (system_loop, repetition_loop), data=noise_list)
                for n in noise_list:
                    ntxdata, ntydata = DEobj.lobj.fetch_data('test', noise=n, noise_type='per')
                    noise_scheme_fitness, noise_rY, noise_err_fit, noise_Vout = DEobj.cap.run_processor(best_genome, syst=system_loop, rep=repetition_loop,
                                                                                                          input_data=ntxdata, target_output=ntydata,
                                                                                                          ridge_layer=best_RidgeLayer, the_data='test')
                    hdf.create_dataset("%d_rep%d/DE_data/noisy_test/%s_per/fit" % (system_loop, repetition_loop, n), data=noise_err_fit)
                    hdf.create_dataset("%d_rep%d/DE_data/noisy_test/%s_per/scheme_fit" % (system_loop, repetition_loop, n), data=noise_scheme_fitness)

                    hdf.create_dataset("%d_rep%d/DE_data/noisy_test/%s_per/fit_change" % (system_loop, repetition_loop, n), data=noise_err_fit-TestFit)
                    hdf.create_dataset("%d_rep%d/DE_data/noisy_test/%s_per/scheme_fit_change" % (system_loop, repetition_loop, n), data=noise_scheme_fitness-scheme_fitness)

                # # Create uniform 2d inputs, and plot outputs
                if prm['DE']['num_classes'] == 2:
                    pClass, rY, Vout = DEobj.cap.run_processor(best_genome, syst=system_loop, rep=repetition_loop,
                                                               input_data=txdata, target_output='na',
                                                               ridge_layer=best_RidgeLayer, the_data='test')
                    hdf.create_dataset("%d_rep%d/DE_data/test_predicted/class" % (system_loop, repetition_loop), data=pClass)
                    hdf.create_dataset("%d_rep%d/DE_data/test_predicted/rY" % (system_loop, repetition_loop), data=rY)
                    hdf.create_dataset("%d_rep%d/DE_data/test_predicted/real_class" % (system_loop, repetition_loop), data=tydata)


            #

            FitEvo('Epoch', prm, system_loop, repetition_loop, train_bFits, TestFit, Global_bFits, DEobj.lobj.num_batches, vali_bFits, DEobj.batch_Ncomps, DEobj.epoch_Ncomps, DEobj.lobj.num_batches)
            
            # ####################################################
            # Check the saved params generate the same results
            # Note: Can't check if windowing with epoch start re-evaluation
            # ####################################################
            """if 'window' not in prm['DE']['batch_scheme']:
                bidx, xdata, ydata = DEobj.lobj.prev_train_batch[1]
                print("\n bidx", bidx)
                #results, GenList_results = self.cap.run_processor(pop_denorm, syst, rep, xdata, ydata, ret_str='both', the_data='train')
                ff, ry, ef, vo = DEobj.cap.run_processor(DEobj.train_bGenomes[-1], syst=system_loop, rep=repetition_loop,
                                                            input_data=xdata, target_output=ydata,
                                                            ridge_layer=train_bRLs[-1], the_data='train')
                #
                if train_bFit != ff and 'Kmean' not in prm['DE']['FitScheme']:
                    e = "ReRun best genome on training data gets wrong fitness: %f (Best Fit = %f)" % (ff, train_bFit)
                    raise ValueError(e)"""

            if prm['DE']['batch_size'] != 0:
                Vxdata, Vydata = DEobj.lobj.get_data('validation', iterate=0)
                ff, ry, ef, vo = DEobj.cap.run_processor(best_genome, syst=system_loop, rep=repetition_loop,
                                                            input_data=Vxdata, target_output=Vydata,
                                                            ridge_layer=best_RidgeLayer, the_data='validation')
                #
                if Global_bFits_RAW[-1] != ff and 'Kmean' not in prm['DE']['FitScheme']:
                    e = "ReRun best genome on training data gets wrong fitness: %f (Best Fit = %f)" % (ff, Global_bFits_RAW[-1])
                    raise ValueError(e)
            else:
                bidx, xdata, ydata = DEobj.lobj.prev_train_batch[1]
                ff, ry, ef, vo = DEobj.cap.run_processor(DEobj.train_bGenomes[-1], syst=system_loop, rep=repetition_loop,
                                                            input_data=xdata, target_output=ydata,
                                                            ridge_layer=train_bRLs[-1], the_data='train')
                #

                if np.around(train_bFit, 6) != np.around(ff, 6) and 'Kmean' not in prm['DE']['FitScheme']:
                    e = "ReRun best genome on training data gets wrong fitness: %f (Best Fit = %f)" % (ff, train_bFit)
                    print(ff, train_bFit)
                    print(np.around(train_bFit, 6), np.around(ff, 6))
                    raise ValueError(e)
            #

            #

            ###################################################################
            # Print material graphs
            ###################################################################
            if prm['mg']['plotMG'] == 1 and prm['network']['num_input'] == 2 and prm['DE']['IntpScheme'] != 'raw':
                MG_obj = materialgraphs(system_loop, repetition_loop,  DEobj.lobj, prm)
                MG_obj.MG([best_genome, train_bFit, Global_bFits[-1], TestFit], obj_RidgeLayer=best_RidgeLayer)

                MG_obj.MG_VaryConfig()
                MG_obj.MG_VaryConfig3()

                MG_obj.MG_VaryPerm(OutWeight=0)  # selected_OW=0 --> don't randomly vary OW

                MG_obj.MG_VaryInWeights(assending=1)  # 0=fixed, 1=rotating, 2=zooming
                MG_obj.MG_VaryInWeightsAni()
                MG_obj.MG_VaryLargeInWeightsAni()

                MG_obj.MG_VaryOutWeights(assending=1)  # 0=fixed, 1=rotating, 2=zooming
                MG_obj.MG_VaryOutWeightsAni()
                MG_obj.MG_VaryLargeOutWeightsAni()

                MG_obj.MG_VaryInBias()
                MG_obj.MG_VaryOutputBias()

            else:
                MG_obj = 'na'

            # # # ########
            # If clustering
            data_X, data_y = DEobj.lobj.get_data(the_data='train', iterate=0)  # load all data
            if 'Kmean' in prm['DE']['IntpScheme'] and 'Kmean' not in prm['DE']['FitScheme']:

                cluster2d(prm, system_loop, repetition_loop, DEobj.lobj, 'train', train_bYs)

            elif 'Kmean' not in prm['DE']['IntpScheme'] and 'Kmean' in prm['DE']['FitScheme']:

                plot_TransformedOutput(prm, system_loop, repetition_loop, DEobj.lobj, 'train', train_bYs)
                perfomance_comparison(prm, system_loop, repetition_loop, DEobj.lobj, train_bYs[-1], 'train')

            #plot_TransformedOutput(prm, system_loop, repetition_loop, DEobj.lobj, 'train', train_bYs)

            #plot_InToOps(prm, system_loop, repetition_loop, other_InNodes='float')
            #plot_InToOps(prm, system_loop, repetition_loop, other_InNodes=0)
            #plot_InToOps(prm, system_loop, repetition_loop, other_InNodes=1)

            #plot_TransConductance(prm, system_loop, repetition_loop, StaticVin2=1, ShuntR=0.1)
            #plot_TransConductance(prm, system_loop, repetition_loop, StaticVin2=1, ShuntR=1) # kohm
            #plot_TransConductance(prm, system_loop, repetition_loop, StaticVin2=1, ShuntR=70) # kohm

            # increment loop and clean up
            gc.collect()
            loop = loop + 1

        # '''

    #

    # ########################################################################
    # Create a new file containing the experiment path,
    # Save some data about the experiment to text
    # ########################################################################:

    if prm['experiment']['active'] == 1:

        # if the new folder does not yet exist, create it
        if not os.path.exists(prm['experiment']['file']):
            os.makedirs(prm['experiment']['file'])

        # Save the deets under a descriptive name
        with open(r'%s/ExpLoop%s.yaml' % (prm['experiment']['file'], str(experiment_loop)), 'w') as exp_file:
            yaml.dump(prm, exp_file)

    # ########################################################################
    # Run Analysis
    # ########################################################################
    blockPrint()
    if prm['num_systems']*prm['num_repetitions'] > 1:
        sel_dict = {'plt_mean':1,'plt_accuracy':1,'plt_std':1,'plt_finalveri':1,'plt_popmean':1,'plt_box':1,'plt_hist':1,'plt_genes':1, 'plt_rT':1}
    else:
        sel_dict = {'plt_mean':0,'plt_accuracy':0,'plt_std':0,'plt_finalveri':0,'plt_popmean':1,'plt_box':0,'plt_hist':0,'plt_genes':0, 'plt_rT':1}

    print("\nProducing analysis graphs...")
    obj_anly = analysis(prm['SaveDir'], format='png')

    obj_anly.Plt_basic(sel_dict=sel_dict, Save_NotShow=1, fill=1, ExpErrors=1, StandardError=1)
    obj_anly.Plt__ani(Save_NotShow=1, PlotOnly='all', format='gif')
    obj_anly.Plt_mg(Save_NotShow=1, Bgeno=1, Dgeno=1, VC=1, VP=1,
                    VoW=1, ViW=1, VoW_ani=1, ViW_ani=1, VoiW_ani=1, VloW_ani=1, VliW_ani=1,
                    titles='on')
    enablePrint()

    if prm['DE']['epochs'] == 0:
        rcp_test = rc_plotter(prm['SaveDir'], 'test')
        rcp_test.FitvH(1)
        rcp_test.FitvStdVo(1)
        rcp_test.HvStdVo(1)

    #

    # # Clean Up
    print("\nfin at:", datetime.now().strftime("%H_%M_%S"))
    plt.close('all')

    return

#

#

#

#

#

# # script # #


if __name__ == "__main__":

    # load Template Paramaters
    tprm = LoadPrm(param_file='')

    # Alter Prms
    #tprm['ReUse_dir'] = 'Results/2021_03_19/__14_56_15__con2DDS__D_RN__RidgeEiM'
    #tprm['ReUse_dir'] = 'Results/15Materials/DRN_2I_2C_3O'
    #tprm['DE']['epochs'] = 20
    #tprm['DE']['IntpScheme'] = 'Ridge'  # 'Ridge', 'pn_binary'
    #tprm['ReUse_dir'] = 'Results/2021_04_14/__16_46_57__sc2DDS__D_RN__EiM'
    #tprm['network']['num_output'] = 1

    # Gen final prm file
    prm = GenSimParam(param_file=tprm)  # Produce PAramater File

    # Run EiM
    RunEiM(prm)

    #
    #

    """
    # Gen final prm file
    #tprm['DE']['batch_size'] = 0.005
    #tprm['DE']['IntpScheme'] = 'Ridge'
    #tprm['DE']['FitScheme'] = 'BinCrossEntropy'
    tprm['DE']['mut_scheme'] = 'rand1'
    tprm['ReUse_dir'] = prm['SaveDir']

    # Run EiM
    prm = GenSimParam(param_file=tprm)  # Produce PAramater File
    RunEiM(prm)
    #"""

    #

    """
    tprm['DE']['IntpScheme'] = 'pn_binary'
    tprm['DE']['epochs'] = 100
    prm = GenSimParam(param_file=tprm)  # Produce PAramater File
    RunEiM(prm)
    #"""

    #

    #

    """
    tprm = LoadPrm(param_file='')
    tprm['ReUse_dir'] = 'Results/2021_03_16/__10_28_32__con2DDS__D_RN__EiM'
    tprm['DE']['epochs'] = 10
    tprm['DE']['mut_scheme'] = 'best1'
    prm = GenSimParam(param_file=tprm)  # Produce PAramater File
    RunEiM(prm)
    # """

    """
    RunEiM(num_systems=1, num_repetitions=1,
           model='R_RN',  # R_RN, D_RN, NL_RN, NL_uRN, custom_RN
           #mut=['DE', 10],
           batch_size=0, epochs=20,
           #IntpScheme='raw',
           #FitScheme='SpectralDist',  # 'KmeanDist',
           #num_readout_nodes=2, OutWeight_gene=1, IntpScheme='HOW',
           ReUse_dir=dir,
           #IntpScheme='HOW', num_readout_nodes='na', training_data='iris', num_input=4, num_output=3,
           num_processors=11)

    #"""

    """RunEiM(num_systems=3, num_repetitions=1,
           model='D_RN',  # R_RN, D_RN, NL_RN, NL_uRN, custom_RN
           IntpScheme='raw',
           FitScheme='KmeanDist',
           num_readout_nodes=2,
           training_data='iris', num_input=4, num_output=6, num_config=3,
           num_processors=10)"""

    """
    dir = RunEiM(num_systems=1, num_repetitions=1,
           model='D_RN',  # R_RN, D_RN, NL_RN, NL_uRN, custom_RN
           IntpScheme='raw',
           FitScheme='KmeanDist',
           num_readout_nodes=2,
           training_data='iris', num_input=4, num_output=6, num_config=3,
           num_processors=10)

    RunEiM(num_systems=1, num_repetitions=1,
           model='D_RN',  # R_RN, D_RN, NL_RN, NL_uRN, custom_RN
           IntpScheme='raw',
           FitScheme='KmeanSpace',
           num_readout_nodes=2,
           training_data='iris', num_input=4, num_output=6, num_config=3,
           ReUse_dir=dir,
           num_processors=10)
    # """
#

# fin
