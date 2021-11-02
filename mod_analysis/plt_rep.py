# # Top Matter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import pandas as pd
import scipy
import h5py

import os
from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap

from mod_analysis.Set_Load_meta import LoadMetaData


''' Description:
This class file is designed to provide various functions to analise saved data
from the looped DE.
'''
#########################################################################
# Plot Average

''' # # # # # # # # # # # # # # #
Produce a plot of the average fitness (per it) vs the iterations
with error bars
'''
def Plt__Av_FvIt(DirList, Save_NotShow=0, show=1, fill=0, ExpErrors=0,
                 StandardError=0, logplot=0, PlotVeri=1,
                 HistBins=0.02, HistMin=0, HistMax='auto', PlotOnly='all'):
    # print("Create graphs with error bar of ")

    ExpList_fi_mean = []
    ExpList_fi_std = []
    ExpList_it = []
    ExpList_num_zeros = []
    Exp_list_FinalDist = []
    ExpList_fi_matrix = []
    ExpList_MeanVeriFit = []
    ExpList_BestVeriFit =[]

    biggest_mean_fitness = 0
    smallest_mean_fitness = 1

    # # create dir list to look at
    if PlotOnly == 'all':
        new_DirList = DirList
    else:
        new_DirList = []
        for val in PlotOnly:
            new_DirList.append(DirList[val])


    # # save the data for plotting
    for curr_dir in range(len(new_DirList)):

        MetaData = LoadMetaData(DirList[curr_dir])

        cir_MEAN_fit_list = []
        cir_MEAN_pop_mean_fit_list = []
        cir_MEAN_pop_std_fit_list = []
        cir_MEAN_veri_fit = []
        cir_std_list = []
        for syst in range(MetaData['num_systems']):

            if MetaData['num_repetitions'] > 1:
                rep_fit_list = []
                rep_pop_mean_fit_list = []
                rep_pop_std_fit_list = []
                rep_veri_fit_list = []
                for rep in range(MetaData['num_repetitions']):
                    location = "%s/data_%d_rep%d.hdf5" % (DirList[curr_dir], syst, rep)
                    with h5py.File(location, 'r') as hdf:
                        rep_fit_list.append(np.array(hdf.get('/DE_data/fitness')))
                        rep_pop_mean_fit_list.append(np.array(hdf.get('/DE_data/pop_mean_fit')))
                        rep_pop_std_fit_list.append(np.array(hdf.get('/DE_data/pop_std_fit')))
                        rep_veri_fit_list.append(np.array(hdf.get('/DE_data/veri_fit')))

                rep_fit_matrix = np.concatenate((rep_fit_list), axis=1)
                rep_pop_mean_fit_matrix = np.concatenate((rep_pop_mean_fit_list), axis=1)
                rep_pop_std_fit_matrix = np.concatenate((rep_pop_std_fit_list), axis=1)
                rep_veri_fit_matrix = np.concatenate((rep_veri_fit_list), axis=1)

                cir_MEAN_fit_list.append(np.mean(rep_fit_matrix, axis=1))
                cir_MEAN_pop_mean_fit_list.append(np.mean(rep_pop_mean_fit_matrix, axis=1))
                cir_MEAN_pop_std_fit_list.append(np.mean(rep_pop_std_fit_matrix, axis=1))
                cir_MEAN_veri_fit.append(np.mean(rep_veri_fit_matrix, axis=1))

                if StandardError == 0:
                    cir_std_list.append(np.std(rep_fit_matrix, axis=1))
                elif StandardError == 1:
                    SE = np.std(rep_fit_matrix, axis=1)/((len(rep_fit_matrix[k,:]))**0.5)
                    cir_std_list.append(SE)

            else:
                # if only one rep, just let equal
                location = "%s/data_%d_rep%d.hdf5" % (DirList[curr_dir], syst, 0)
                with h5py.File(location, 'r') as hdf:
                    cir_MEAN_fit_list = np.array(hdf.get('/DE_data/fitness'))
                    cir_MEAN_pop_mean_fit_list = np.array(hdf.get('/DE_data/pop_mean_fit'))
                    cir_MEAN_pop_std_fit_list = np.array(hdf.get('/DE_data/pop_std_fit'))
                    cir_MEAN_veri_fit = np.array(hdf.get('/DE_data/veri_fit'))
                    cir_std_list = np.zeros(len(cir_MEAN_fit_list))


            if MetaData['num_systems'] > 1:

                cir_MEAN_fit_matrix = np.concatenate((cir_MEAN_fit_list), axis=1)
                cir_MEAN_pop_mean_fit_matrix = np.concatenate((cir_MEAN_pop_mean_fit_list), axis=1)
                cir_MEAN_pop_std_fit_matrix = np.concatenate((cir_MEAN_pop_std_fit_list), axis=1)
                cir_MEAN_veri_fit_matrix = np.concatenate((cir_MEAN_veri_fit), axis=1)

                exp_MEAN_fit = np.mean(cir_MEAN_fit_matrix, axis=1)
                exp_MEAN_pop_fit = np.mean(cir_MEAN_pop_mean_fit_matrix, axis=1)
                exp_MEAN_pop_std = np.mean(cir_MEAN_pop_std_fit_matrix, axis=1)
                exp_MEAN_veri_fit = np.mean(cir_MEAN_veri_fit_matrix, axis=1)

                if StandardError == 0:
                    cir_std_list.append(np.std(exp_MEAN_fit, axis=1))
                elif StandardError == 1:
                    SE = np.std(exp_MEAN_fit, axis=1)/((len(exp_MEAN_fit[0,:]))**0.5)
                    cir_std_list.append(SE)

            else:
                # if only one rep, just let equal

    # ************************************************************
    # # # # # # # # # # # # # # # # # # # # #
    # plot all on one graph

    markers = ['+', '1', '.', 'v', '*', 's', 'D', '^', '2', 'h', 'd', 'o', '3', '4', '8', 'P', 'X', ',', '<', '>']


    # Plot mean graphs for all exp's
    fig = plt.figure()
    k = 0
    for fi_mean_list in ExpList_fi_mean:


        if PlotVeri == 1 and self.TestVerify == 1:
            the_lablel = "%s %s, MeanVeri=%.3f, BestVeri=%.3f" % (self.FileRefNames[k], self.Param_array[k], ExpList_MeanVeriFit[k], ExpList_BestVeriFit[k])
            if PlotOnly != 'all':
                k_temp = lable_index[k]
                the_lablel = "%s %s, MeanVeri=%.3f, BestVeri=%.3f" % (self.FileRefNames[k_temp], self.Param_array[k_temp], ExpList_MeanVeriFit[k_temp], ExpList_BestVeriFit[k_temp])
        else:
            the_lablel = "%s %s" % (self.FileRefNames[k], self.Param_array[k])
            if PlotOnly != 'all':
                k_temp = lable_index[k]
                the_lablel = "%s %s" % (self.FileRefNames[k_temp], self.Param_array[k_temp])

        plt.plot(ExpList_it[k], fi_mean_list, label=the_lablel, marker=markers[k])
        if ExpErrors == 1:
            plt.fill_between(it, np.array(fi_mean_list)-np.array(ExpList_fi_std[k]), np.array(fi_mean_list)+np.array(ExpList_fi_std[k]), alpha=0.4)
        k = k + 1
    plt.legend()
    if ExpErrors == 1 and StandardError == 0:
        plt.title('Mean Best fitness against itteration for the different parameters (std)')
    elif ExpErrors == 1 and StandardError == 1:
        plt.title('Mean Best fitness against itteration for the different parameters (SE)')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Best Fitness ')
    if logplot == 1:
        plt.yscale('log')

    # # show & save plots
    if Save_NotShow == 1:
        fig_path = "%s/ExpALL__FIG_mean_FvIt.%s" % (self.save_dir, self.format)
        fig.savefig(fig_path, dpi=self.dpi)
        plt.close(fig)

    #

    #

    # # # # # # # # # # # #  #
    # Plot Std graphs for all exp's
    fig = plt.figure()
    k = 0
    for fi_std_list in ExpList_fi_std:
        the_lablel = "%s %s" % (self.FileRefNames[k], self.Param_array[k])
        if PlotOnly != 'all':
            k_temp = lable_index[k]
            the_lablel = "%s %s" % (self.FileRefNames[k_temp], self.Param_array[k_temp])

        plt.plot(ExpList_it[k], fi_std_list, label=the_lablel, marker=markers[k])
        k = k + 1
    plt.legend()
    if StandardError == 0:
        plt.title('Plot of std of fitness against itteration for the different parameters')
        plt.ylabel('Std of Fitness ')
    elif StandardError == 1:
        plt.title('Plot of Standard Error of fitness against itteration for the different parameters')
        plt.ylabel('SE of Fitness Measurments')
    plt.xlabel('Iteration')

    if logplot == 1:
        plt.yscale('log')

    # # show & save plots
    if Save_NotShow == 1:
        fig_path = "%s/ExpALL__FIG_std_FvIt.%s" % (self.save_dir, self.format)
        fig.savefig(fig_path, dpi=self.dpi)
        plt.close(fig)

    #

    #

    # # # # # # # # # # # #  #
    # Plot histogram of final fitnesses
    fig = plt.figure()

    # generate the lables
    the_lablel = []
    for k in range(len(ExpList_fi_mean)):
        if PlotOnly == 'all':
            the_lablel.append("%s %s, BestFit=%.4f" % (self.FileRefNames[k], self.Param_array[k], min(Exp_list_FinalDist[k])))
        else:
            k_temp = lable_index[k]
            the_lablel.append("%s %s, BestFit=%.4f" % (self.FileRefNames[k_temp], self.Param_array[k_temp], min(Exp_list_FinalDist[k_temp])))

    # determin limits of the histogrtam
    if HistMax == 'auto':
        hist_max = biggest_mean_fitness+0.02
    else:
        hist_max = HistMax

    if HistMin == 'auto':
        hist_min = smallest_mean_fitness-0.02
        if hist_min < 0:
            hist_min = 0
    else:
        hist_min = HistMin

    plt.hist(Exp_list_FinalDist, bins=np.arange(hist_min, hist_max, HistBins), label=the_lablel)

    plt.legend()
    plt.suptitle('Histergram of final fitness distribution over all the loops')
    sub_title = "Bins = %.4f" % HistBins
    plt.title(sub_title, fontsize=10)

    plt.xlabel('Final Fitness')
    plt.ylabel('Cont')

    # # show & save plots
    if Save_NotShow == 1:
        fig_path = "%s/ExpALL__FIG_FinalFit_Hist.%s" % (self.save_dir, self.format)
        fig.savefig(fig_path, dpi=self.dpi)
        plt.close(fig)

    #

    #

    #

    # Show Exp graphs
    if show == 1 and Save_NotShow == 0:
        plt.show()


# fin
