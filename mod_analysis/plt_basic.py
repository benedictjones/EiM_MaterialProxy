# # Top Matter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import pandas as pd
import scipy
import h5py

from matplotlib.ticker import FormatStrFormatter

import os
from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap

from mod_analysis.Set_Load_meta import LoadMetaData



''' # # # # # # # # # # # # # # #
Produce a plot of the basic set of graphs
'''


class plotting(object):

    def Plt_basic(self, sel_dict='na', Save_NotShow=0, show=1, fill=0, ExpErrors=0,
                     StandardError=0, logplot=0, PlotVeri=1,
                     HistBins=0.02, HistMin=0, HistMax='auto',
                     plt_mean_yaxis='auto',
                     PlotOnly='all', Specific_Cir_Rep='all',
                     titles='on', legends='on',
                     fig_letter='na'):

        self.markers = ['+', '*', '1', '2', '^', 'D', 'o', 'v', '<', 'h', 'd', 'o', '3', '4', '8', 'P', 'X', ',', 's', '>']
        self.fig_letter = fig_letter

        print("Plotting BAsic graphs...")

        # # Assign paramaters to self
        if sel_dict == 'na':
            self.sel_dict = {'plt_mean':1,'plt_std':1,'plt_finalveri':1,'plt_popmean':1,'plt_hist':1}
        else:
            self.sel_dict = sel_dict
        self.Save_NotShow = Save_NotShow
        self.show = show
        self.fill = fill
        self.ExpErrors = ExpErrors
        self.StandardError = StandardError
        self.logplot = logplot
        self.PlotVeri = PlotVeri
        self.HistBins = HistBins
        self.HistMin = HistMin
        self.HistMax = HistMax
        self.plt_mean_yaxis = plt_mean_yaxis

        matplotlib .rcParams['font.family'] = 'Arial'  # 'serif'
        matplotlib .rcParams['font.size'] = 8  # tixks and title
        matplotlib .rcParams['figure.titlesize'] = 'medium'
        matplotlib .rcParams['axes.labelsize'] = 10  # axis labels
        matplotlib .rcParams['axes.linewidth'] = 1  # box edge
        #matplotlib .rcParams['mathtext.fontset'] = 'Arial'  # 'cm'
        matplotlib.rc('pdf', fonttype=42)  # embeds the font, so can import to inkscape
        matplotlib .rcParams["legend.labelspacing"] = 0.25

        matplotlib .rcParams['lines.linewidth'] = 0.85
        matplotlib .rcParams['lines.markersize'] = 3.5
        matplotlib .rcParams['lines.markeredgewidth'] = 0.5

        if titles == 'off':
            matplotlib .rcParams["figure.figsize"] = [3.5,2.7]
        else:
            matplotlib .rcParams["figure.figsize"] = [6.4, 4.8]
        matplotlib .rcParams["figure.autolayout"] = True

        #matplotlib .rcParams['lines.linewidth'] = 1.3
        self.title = titles
        self.legends = legends
        # for hist
        self.Exp_list_FinalDist = []
        self.hist_the_lablel = []

        # save pop size
        exp_pop_size = []
        result_stats_text = []

        self.best_veri = 1000

        # # create dir list to look at
        if PlotOnly == 'all':
            self.new_DirList = self.data_dir_list
            self.new_FileRefNames = self.FileRefNames
            self.new_Param_array = self.Param_array
        else:
            self.new_DirList = []
            self.new_FileRefNames = []
            self.new_Param_array = []
            for val in PlotOnly:
                self.new_DirList.append(self.data_dir_list[val])
                self.new_FileRefNames.append(self.FileRefNames[val])
                self.new_Param_array.append(self.Param_array[val])

        # # save the data for plotting
        self.dir_loop = 0
        for self.curr_dir in self.new_DirList:

            self.MetaData = LoadMetaData(self.curr_dir)
            self.ParamDict = self.MetaData['DE']
            self.NetworkDict = self.MetaData['network']
            self.GenomeDict = self.MetaData['genome']


            exp_pop_size.append(self.ParamDict['popsize'])

            fit_list = []
            pop_mean_fit_list = []
            self.pop_std_fit_list = []
            self.veri_fit_list = []
            self.best_veri = 1000

            cir_range = range(self.MetaData['num_circuits'])
            rep_range = range(self.MetaData['num_repetitions'])

            if len(Specific_Cir_Rep) != 2:
                if Specific_Cir_Rep != 'all':
                    print("Error: Specific_Cir_Rep must be 'all' or be a specific [cir, rep]")
                    return
            else:
                if Specific_Cir_Rep[0] >= self.MetaData['num_circuits']:
                    print("Error: Invalid Circuit loop selected to plot")
                    return
                elif Specific_Cir_Rep[1] >= self.MetaData['num_repetitions']:
                    print("Error: Invalid Repetition loop selected to plot")
                    return
                else:
                    cir_range = [Specific_Cir_Rep[0]]
                    rep_range = [Specific_Cir_Rep[1]]


            for cir in cir_range:
                for rep in rep_range:

                    legacy = os.path.isfile("%s/data_%d_rep%d.hdf5" % (self.curr_dir, cir, rep))
                    if legacy is True:
                        location = "%s/data_%d_rep%d.hdf5" % (self.curr_dir, cir, rep)
                        hdf = h5py.File(location, 'r')
                        DE_data = hdf.get('/DE_data')

                    elif legacy is False:
                        location = "%s/data.hdf5" % (self.curr_dir)
                        hdf = h5py.File(location, 'r')
                        DE_data = hdf.get('/%d_rep%d/DE_data' % (cir, rep))

                    fit_list.append(np.array(DE_data.get('fitness')))
                    pop_mean_fit_list.append(np.array(DE_data.get('pop_mean_fit')))
                    self.pop_std_fit_list.append(np.array(DE_data.get('pop_std_fit')))
                    if self.ParamDict['TestVerify'] == 1:
                        self.veri_fit_list.append(np.array(DE_data.get('veri_fit')))

                    hdf.close()  # close file

            cont = self.MetaData['num_circuits'] + self.MetaData['num_repetitions']
            self.fit_matrix = np.asarray(fit_list)

            #print("self.fit_matrix:\n", self.fit_matrix)

            # cal means
            self.exp_MEAN_fit = np.mean(self.fit_matrix, axis=0)
            self.exp_MEAN_pop_fit = np.mean(np.asarray(pop_mean_fit_list), axis=0)

            # exp_MEAN_pop_std = np.mean(np.asarray(self.pop_std_fit_list), axis=0)  # must find pooled!
            if self.ParamDict['TestVerify'] == 1:
                self.exp_MEAN_veri_fit = np.mean(np.asarray(self.veri_fit_list), axis=0)
                self.exp_std_veri_fit = np.std(np.asarray(self.veri_fit_list), axis=0)

                self.best_veri = min(np.asarray(self.veri_fit_list))  # best veri fitness
                self.mean_veri = np.mean(np.asarray(self.veri_fit_list))
                self.std_veri = np.std(np.asarray(self.veri_fit_list))
            else:
                self.exp_MEAN_veri_fit = 'na'

            if self.StandardError == 0:
                self.exp_std = np.std(np.asarray(fit_list), axis=0)
            elif self.StandardError == 1:
                num_runs = len(np.asarray(fit_list))
                self.exp_std = np.std(np.asarray(fit_list), axis=0)/(num_runs**0.5)


            self.mean_training_fit = np.mean(self.fit_matrix[:,-1])
            self.std_training_fit = np.std(self.fit_matrix[:,-1])
            self.best_training_fit = min(self.fit_matrix[:,-1])

            # produce statsiscts and results of run
            if self.ParamDict['TestVerify'] == 1:
                results_stats = "Mean Train Fit = %.3f, Best Train Fit = %.3f, Std Train Fit = %.3f | Mean Veri Fit = %.3f, Best Veri Fit = %.3f, Std Veri Fit = %.3f" % (self.mean_training_fit, self.best_training_fit, self.std_training_fit, self.mean_veri, self.best_veri, self.std_veri)
            else:
                results_stats = "Mean Train Fit = %.3f, Best Train Fit = %.3f, Std Train Fit = %.3f" % (self.mean_training_fit, self.best_training_fit, self.std_training_fit)
            print(results_stats)
            result_stats_text.append(results_stats)
            #

            # Plot's'

            self.plt_mean()

            self.plt_std()

            self.plt_final_Vs_veri()

            self.plt_popmean()

            #

            # store histogram data of final fitnesses
            self.Exp_list_FinalDist.append(self.fit_matrix[:,-1])
            if titles == 'on':
                if self.ParamDict['TestVerify'] == 1:
                    self.hist_the_lablel.append("%s %s, BestFit=%.4f" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], min(self.fit_matrix[:,-1])))
                else:
                    self.hist_the_lablel.append("%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop]))
            else:
                self.hist_the_lablel.append("%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop]))
            # increment!
            self.dir_loop = self.dir_loop + 1

        # plot hist
        self.plt_hist()

        # #  show Exp graphs
        if self.show == 1 and self.Save_NotShow == 0:
            plt.show()
        else:
            plt.close('all')

        if self.Save_NotShow == 1:
            Deets = ''
            for Deets_Line in result_stats_text:
                Deets = '%s%s \n' % (Deets, Deets_Line)
            path_deets = "%s/Results&Stats.txt" % (self.save_dir)
            file1 = open(path_deets, "w")
            file1.write(Deets)
            file1.close()

        plt.close('all')
        return

    #

    #

    # ***********************************************************************
    # FUNCTIONS
    # ***********************************************************************

    # #######################################################
    # Plots mean (best pop) fitness over itterations
    # #######################################################
    def plt_mean(self):
        if 'plt_mean' in self.sel_dict:
            if self.sel_dict['plt_mean'] == 1:

                fig = plt.figure(1)
                if self.title == 'on':
                    if self.PlotVeri == 1 and self.ParamDict['TestVerify'] == 1:
                        the_lablel = "%s %s, MeanVeri=%.3f, BestVeri=%.3f" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], self.exp_MEAN_veri_fit, self.best_veri)
                    else:
                        the_lablel = "%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop])
                else:
                    the_lablel = "%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop])

                plt.plot(self.exp_MEAN_fit, label=the_lablel, marker=self.markers[self.dir_loop])

                if self.ExpErrors == 1:
                    bot = self.exp_MEAN_fit-self.exp_std
                    top = self.exp_MEAN_fit+self.exp_std
                    plt.fill_between(range(len(self.exp_MEAN_fit)), bot, top, alpha=0.4)

                if self.legends == 'on':
                    plt.legend()

                #fig.tight_layout()

                plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                if self.fig_letter != 'na':
                    fig.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

                if self.title == 'on':
                    if self.ExpErrors == 1 and self.StandardError == 0:
                        plt.title('Mean Best fitness against itteration for the different parameters (std)')
                    elif self.ExpErrors == 1 and self.StandardError == 1:
                        plt.title('Mean Best fitness against itteration for the different parameters (SE)')
                    else:
                        plt.title('Mean Best fitness against itteration')
                plt.xlabel('Iteration')
                plt.ylabel('Mean best fitness ')
                if self.plt_mean_yaxis != 'auto':
                    plt.ylim((self.plt_mean_yaxis))
                if self.logplot == 1:
                    plt.yscale('log')

                # # self.show & save plots
                if self.curr_dir == self.new_DirList[-1]:
                    if self.Save_NotShow == 1:
                        fig_path = "%s/ExpALL__FIG_mean_FvIt.%s" % (self.save_dir, self.format)
                        fig.savefig(fig_path, dpi=self.dpi)
                        plt.close(fig)

    #

    #

    #
    # #######################################################
    # Plots std of (best pop) fitness over itterations
    # #######################################################
    def plt_std(self):
        if 'plt_std' in self.sel_dict:
            if self.sel_dict['plt_std'] == 1:
                fig = plt.figure(2)

                the_lablel = "%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop])
                plt.plot(self.exp_std, label=the_lablel, marker=self.markers[self.dir_loop])

                if self.legends == 'on':
                    plt.legend()

                if self.StandardError == 0:
                    plt.title('Plot of std of fitness against \n itteration for the different parameters')
                    plt.ylabel('Std of Fitness ')
                elif self.StandardError == 1:
                    plt.title('Plot of Standard Error of fitness \n against itteration for the different parameters')
                    plt.ylabel('SE of Fitness Measurments')
                plt.xlabel('Iteration')

                if self.logplot == 1:
                    plt.yscale('log')

                # # self.show & save plots
                if self.curr_dir == self.new_DirList[-1]:
                    if self.Save_NotShow == 1:
                        fig_path = "%s/ExpALL__FIG_std_FvIt.%s" % (self.save_dir, self.format)
                        fig.savefig(fig_path, dpi=self.dpi)
                        plt.close(fig)

    #

    #

    #

    # #######################################################
    # Plot final fitness vs verification fitness
    # #######################################################
    def plt_final_Vs_veri(self):
        if self.sel_dict['plt_finalveri'] == 1:
            if self.ParamDict['TestVerify'] == 1:
                fig = plt.figure(3)

                the_lablel = "%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop])

                #S = 10 + 100/((1+self.dir_loop)**3)
                num_datasets = len(self.new_DirList)
                s_range = np.flip(np.arange(num_datasets) + 1)**1.5
                S = 10 + s_range[self.dir_loop]*10

                # s=1+(len(self.fit_matrix[:,-1])/10-self.dir_loop)**3
                #print("size scale s =", S )


                #print("self.veri_fit_list:\n", self.veri_fit_list)

                plt.scatter(self.fit_matrix[:,-1], self.veri_fit_list, label=the_lablel,
                            marker=self.markers[self.dir_loop], s=S) # , alpha=(len(self.fit_matrix[:,-1])-self.dir_loop)/len(self.fit_matrix[:,-1])

                if self.legends == 'on':
                    plt.legend()

                if self.fig_letter != 'na':
                    fig.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

                plt.title('Scatter of final training fitness against the Verification Fitness')
                plt.ylabel('Verification Fitness')
                plt.xlabel('Final training fitness')


                # # self.show & save plots
                if self.curr_dir == self.new_DirList[-1]:
                    if self.Save_NotShow == 1:
                        fig_path = "%s/ExpALL__FIG_Scatter_FinalTrainVsVeri.%s" % (self.save_dir, self.format)
                        fig.savefig(fig_path, dpi=self.dpi)
                        plt.close(fig)

    #

    #

    #

    # #######################################################
    # Plot pop mean fitness (with std/SE)
    # #######################################################
    def plt_popmean(self):
        if 'plt_popmean' in self.sel_dict:
            if self.sel_dict['plt_popmean'] == 1:
                fig = plt.figure(4)

                col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

                # # calc pooled std/SE
                pop_std_fit_matrix = np.asarray(self.pop_std_fit_list)
                pooled_pop_std = []  # create list to store pooled pop std
                pop = self.ParamDict['popsize']

                for k in range(len(pop_std_fit_matrix[0, :])):  # iterates over the "its"
                    if self.StandardError == 0:
                        numerator = 0
                        denomenator = 0
                        # calculate pooled std
                        for j in range(len(pop_std_fit_matrix[:, k])):
                            numerator = ((pop-1)*(pop_std_fit_matrix[j, k])**2) + numerator
                            denomenator = pop - 1 + denomenator
                        std_pooled = (numerator/denomenator)**0.5
                        pooled_pop_std.append(std_pooled)
                    elif self.StandardError == 1:
                        numerator = 0
                        denomenator = 0
                        total_pop = 0
                        # calculate pooled std
                        for j in range(len(pop_std_fit_matrix[:, k])):
                            numerator = ((pop-1)*(pop_std_fit_matrix[j, k])**2) + numerator
                            denomenator = pop - 1 + denomenator
                            total_pop = total_pop + pop
                        std_pooled = (numerator/denomenator)**0.5
                        SE = std_pooled/(total_pop**0.5)
                        pooled_pop_std.append(SE)

                #print("pop_std_fit_matrix\n", pop_std_fit_matrix)
                #print("pooled_pop_std\n", pooled_pop_std)

                if self.ParamDict['TestVerify'] == 1:
                    the_lablel = "%s %s, Av VeriFit=%f" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], self.exp_MEAN_veri_fit)
                else:
                    the_lablel = "%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop])

                # Select a standard error plot, or a std band self.fill plot
                if self.fill == 0:
                    plt.errorbar(self.exp_MEAN_pop_fit, yerr=pooled_pop_std, linestyle='None', marker='s', capsize=5, label=the_lablel, color=col_list[self.dir_loop])
                elif self.fill == 1:
                    plt.plot(self.exp_MEAN_pop_fit, '-x', label=the_lablel)
                    plt.fill_between(range(len(self.exp_MEAN_pop_fit)), self.exp_MEAN_pop_fit-pooled_pop_std, self.exp_MEAN_pop_fit+pooled_pop_std, alpha=0.5, color=col_list[self.dir_loop])

                if self.legends == 'on':
                    plt.legend()

                # #####
                # Plot verification fitness line here??
                # #####
                if self.ParamDict['TestVerify'] == 1:
                    plt.plot([0, len(self.exp_MEAN_pop_fit)-1],[self.exp_MEAN_veri_fit, self.exp_MEAN_veri_fit],'--', color=col_list[self.dir_loop]) # set to right colour!

                    if self.StandardError == 0:
                        plt.text((len(self.exp_MEAN_pop_fit)-2)/2, self.exp_MEAN_veri_fit, 'Mean Verification Fitness (w/ std)')
                        plt.fill_between([0, len(self.exp_MEAN_pop_fit)-1], [self.exp_MEAN_veri_fit-self.exp_std_veri_fit,self.exp_MEAN_veri_fit-self.exp_std_veri_fit],
                                         [self.exp_MEAN_veri_fit+self.exp_std_veri_fit, self.exp_MEAN_veri_fit+self.exp_std_veri_fit], alpha=0.2, color=col_list[self.dir_loop])
                    elif self.StandardError == 1:
                        SE_VeriFit = self.exp_std_veri_fit/(len(self.exp_MEAN_pop_fit)**0.5)
                        plt.text((len(self.exp_MEAN_pop_fit)-2)/2, self.exp_MEAN_veri_fit, 'Mean Verification Fitness (w/ SE)')
                        plt.fill_between([0, len(self.exp_MEAN_pop_fit)-1], [self.exp_MEAN_veri_fit-SE_VeriFit,self.exp_MEAN_veri_fit-SE_VeriFit],
                                         [self.exp_MEAN_veri_fit+SE_VeriFit, self.exp_MEAN_veri_fit+SE_VeriFit], alpha=0.2, color=col_list[self.dir_loop])



                if self.ExpErrors == 1 and self.StandardError == 0:
                    plt.title('Pop Mean fitness vs itteration for the different parameters (pooled std)')
                elif self.ExpErrors == 1 and self.StandardError == 1:
                    plt.title('Pop fitness vs itteration for the different parameters (SE)')
                plt.xlabel('Iteration')
                plt.ylabel('Mean Fitness ')

                if self.logplot == 1:
                    plt.yscale('log')

                # # self.show & save plots
                if self.curr_dir == self.new_DirList[-1]:
                    if self.Save_NotShow == 1:
                        fig_path = "%s/ExpALL__FIG_PopMean.%s" % (self.save_dir, self.format)
                        fig.savefig(fig_path, dpi=self.dpi)
                        plt.close(fig)

    #

    #

    #

    # #######################################################
    # Plot histogram of final fitnesses
    # #######################################################
    def plt_hist(self):
        if 'plt_hist' in self.sel_dict:
            if self.sel_dict['plt_hist'] == 1:

                # determin limits of the histogrtam
                if self.HistMax == 'auto':
                    hist_max = max(np.concatenate(self.Exp_list_FinalDist))+0.02
                    hist_max = hist_max
                else:
                    hist_max = self.HistMax

                if self.HistMin == 'auto':
                    hist_min = min(np.concatenate(self.Exp_list_FinalDist))-0.02
                    hist_min = hist_min
                    if hist_min < 0:
                        hist_min = 0
                else:
                    hist_min = self.HistMin

                fig = plt.figure(5)
                #print(">>hist lables:\n", self.hist_the_lablel)
                #print(">>hist limits:", hist_min, hist_max)
                plt.hist(self.Exp_list_FinalDist, bins=np.arange(hist_min, hist_max, self.HistBins), label=self.hist_the_lablel)

                if self.legends == 'on':
                    plt.legend()

                if self.title == 'on':
                    sub_title = "Bins = %.4f" % self.HistBins
                    plt.title('Histergram of final fitness distribution over all the loops\n%s' % sub_title)

                if self.fig_letter != 'na':
                    fig.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

                plt.xlabel('Final Fitness')
                plt.ylabel('Count')

                # # self.show & save plots
                if self.Save_NotShow == 1:
                    fig_path = "%s/ExpALL__FIG_FinalFit_Hist.%s" % (self.save_dir, self.format)
                    fig.savefig(fig_path, dpi=self.dpi)
                    plt.close(fig)







# fin
