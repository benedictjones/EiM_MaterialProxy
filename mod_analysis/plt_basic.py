# # Top Matter
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pickle
import pandas as pd
import scipy
import h5py

from matplotlib.ticker import FormatStrFormatter

import os
from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap

from mod_analysis.Set_Load_meta import LoadMetaData

from mod_methods.RoundDown import round_decimals_down
from mod_methods.Grouping import group_genome

#


class plotting(object):

    ''' # # # # # # # # # # # # # # #
    Produce a plot of the basic set of graphs
    '''

    def Plt_basic(self, sel_dict='na', Save_NotShow=0, show=1, fill=0, ExpErrors=0,
                     StandardError=0, logplot=0, PlotTest=1,
                     HistBins=0.02, HistMin=0, HistMax='auto',
                     plt_mean_yaxis='auto',
                     PlotOnly='all', Specific_Sys_Rep='all',
                     titles='on', legends='on',
                     fig_letter='na'):

        self.markers = ['+', '*', '1', '2', '^', 'D', 'o', 'v', '<', 'h', 'd', 'o', '3', '4', '8', 'P', 'X', ',', 's', '>']
        self.fig_letter = fig_letter

        print("Plotting Basic graphs...")

        # # Assign paramaters to self
        if sel_dict == 'na':
            self.sel_dict = {'plt_mean':1,'plt_accuracy':1,'plt_std':1,'plt_finalveri':1,'plt_popmean':1,'plt_box':1,'plt_hist':1,'plt_genes':1, 'plt_rT':1}
        else:
            self.sel_dict = sel_dict

        #self.sel_dict['plt_box'] = 0

        self.Save_NotShow = Save_NotShow
        self.show = show
        self.fill = fill
        self.ExpErrors = ExpErrors
        self.StandardError = StandardError
        self.logplot = logplot
        self.PlotTest = PlotTest
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
            matplotlib .rcParams["figure.figsize"] = [2.8,2.3]  # [3.5,2.7]
        else:
            matplotlib .rcParams["figure.figsize"] = [6.4, 4.8]
        matplotlib .rcParams["figure.autolayout"] = True

        #matplotlib .rcParams['lines.linewidth'] = 1.3
        self.title = titles
        self.legends = legends

        # # for hist
        self.Exp_list_FinalTrainFits = []
        self.Exp_list_FinalGbestFits = []
        self.Exp_list_TestFits = []
        self.Exp_list_NoisyTestFits = []
        self.Exp_list_VaryNoiseTestFits = []
        self.Exp_list_RandomTest = []

        self.hist_the_lablel = []

        self.Exp_list_MetaData = []  # store each loop prm, so can identify active genes

        # # Gene Hist
        self.Exp_list_BestGenomesList = []

        # save pop size
        exp_pop_size = []
        result_stats_Untrained = ['Param, Mean UnTrained Fit, Std UnTrained Train Fit, Best UnTrained Fit']
        result_stats_training = ['Param, Mean Final Train Fit, Std Final Train Fit, Best FinalTrain Fit']
        result_stats_first_validation = ['Param, Mean First Vali Fit, Std First Vali Fit, Best First Vali Fit']
        result_stats_validation = ['Param, Mean Final Vali Fit, Std Final Vali Fit, Best Final Vali Fit']
        result_stats_test = ['Param, Mean Test Fit, Std Test Fit, Best Test Fit']
        result_stats_noisytest = ['Param, Mean Noisy Test Fit, Std Noisy Test Fit, Best Noisy Test Fit']



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

            exp_pop_size.append(self.ParamDict['popsize'])

            train_fits = []
            self.train_ncomps = []
            self.epoch_ncomps = []
            self.train_fits_per_sys = []
            self.Gbest_Errs_per_sys = []
            pop_meanTrain_fits = []
            vali_fits = []
            Gbest_fits = []
            Gbest_Errs = []
            best_genomes = []
            randomTest = []

            VaryNoiseTestFit = []

            self.pop_std_fit_list = []
            self.test_fits = []
            self.noisytest_fits = []
            self.best_test_fit = 1000

            syst_range = range(self.MetaData['num_systems'])
            rep_range = range(self.MetaData['num_repetitions'])

            if len(Specific_Sys_Rep) != 2:
                if Specific_Sys_Rep != 'all':
                    print("Error: Specific_Sys_Rep must be 'all' or be a specific [cir, rep]")
                    return
            else:
                if Specific_Sys_Rep[0] >= self.MetaData['num_systems']:
                    print("Error: Invalid Syscuit loop selected to plot")
                    return
                elif Specific_Sys_Rep[1] >= self.MetaData['num_repetitions']:
                    print("Error: Invalid Repetition loop selected to plot")
                    return
                else:
                    syst_range = [Specific_Sys_Rep[0]]
                    rep_range = [Specific_Sys_Rep[1]]


            for cir in syst_range:
                train_fits_per_rep = []
                Gbest_Errs_per_rep = []
                for rep in rep_range:

                    # # Legacy name test
                    #legacy = os.path.isfile("%s/data_%d_rep%d.hdf5" % (self.curr_dir, cir, rep))

                    # # Load
                    location = "%s/data.hdf5" % (self.curr_dir)
                    # print(">", location)
                    hdf = h5py.File(location, 'r')
                    DE_data = hdf.get('/%d_rep%d/DE_data' % (cir, rep))
                    DE_items = list(DE_data.keys())

                    # # Fetch Group Names
                    G_train = hdf.get('/%d_rep%d/DE_data/training' % (cir, rep))
                    G_best = hdf.get('/%d_rep%d/DE_data/global_best' % (cir, rep))
                    G_test = hdf.get('/%d_rep%d/DE_data/test' % (cir, rep))
                    G_noisy_test = hdf.get('/%d_rep%d/DE_data/noisy_test' % (cir, rep))

                    # # Get Training Fits
                    train_fits.append(np.array(G_train.get('best_fits')))
                    train_fits_per_rep.append(np.array(G_train.get('best_fits')))
                    pop_meanTrain_fits.append(np.array(G_train.get('pop_mean_fits')))
                    self.pop_std_fit_list.append(np.array(G_train.get('pop_std_fits')))
                    self.train_ncomps.append(np.array(G_train.get('Ncomps')))

                    self.epoch_ncomps.append(np.array(G_best.get('Ncomps')))


                    best_genome_list = np.array(G_best.get('best_genomes'))
                    best_genome_grouping = np.array(DE_data.get('gen_grouping'))
                    if best_genome_list.any() != None:  # catch to see if there is a saved genome value
                        best_genomes.append(group_genome(best_genome_list[-1], best_genome_grouping) )
                    Gbest_fits.append(np.array(G_best.get('fits')))
                    Gbest_Errs.append(np.array(G_best.get('error_fits')))
                    Gbest_Errs_per_rep.append(np.array(G_best.get('error_fits')))

                    # # Get Validation Fits
                    if 'validation' in DE_items:
                        G_vali = hdf.get('/%d_rep%d/DE_data/validation' % (cir, rep))
                        vali_fits.append(np.array(G_vali.get('fits')))
                        num_batch_per_epoch = np.array(G_vali.get('num_batches'))
                        self.batch_weight = 1/num_batch_per_epoch
                    else:
                        vali_fits = 'na'
                        self.batch_weight = 1

                    if 'test_predicted' in DE_items:
                        G_rT = hdf.get('/%d_rep%d/DE_data/test_predicted' % (cir, rep))
                        predicted_class = np.array(G_rT.get('class'))
                        predicted_rY = np.array(G_rT.get('rY'))
                        real_rY = np.array(G_rT.get('real_class'))
                        randomTest.append(np.stack((real_rY, predicted_rY), axis=-1))


                    # # Get Test Fit
                    self.test_fits.append(np.array(G_test.get('fit')))
                    self.noisytest_fits.append(np.array(G_noisy_test.get('fit')))

                    # # Get Noise data
                    self.VaryNoiseTestFit_list = np.array(G_noisy_test.get('noise_per_list'))
                    all_nTests = []
                    for n in self.VaryNoiseTestFit_list:
                        all_nTests.append(np.array(G_noisy_test.get('%s_per/fit' % (n))))
                    VaryNoiseTestFit.append(all_nTests)

                    hdf.close()  # close file
                self.train_fits_per_sys.append(np.asarray(train_fits_per_rep))
                self.Gbest_Errs_per_sys.append(np.asarray(Gbest_Errs_per_rep))
            #

            # # # # Organise Data # # # #
            cont = self.MetaData['num_systems'] * self.MetaData['num_repetitions']
            self.train_fit_matrix = np.asarray(train_fits)
            self.Gbest_fit_matrix = np.asarray(Gbest_fits)

            self.Gbest_error_matrix = np.asarray(Gbest_Errs)

            self.Exp_list_VaryNoiseTestFits.append(VaryNoiseTestFit)

            self.Exp_list_BestGenomesList.append(best_genomes)
            self.Exp_list_MetaData.append(self.MetaData)

            # Save random input test results
            if 'test_predicted' in DE_items and cont > 1:
                randomTest = np.concatenate(randomTest)
            elif 'test_predicted' in DE_items and cont == 1:
                randomTest = randomTest[0]
            self.Exp_list_RandomTest.append(randomTest)

            # # Calc Training means
            #print(pop_meanTrain_fits)
            self.exp_MEAN_pop_fit = np.mean(np.asarray(pop_meanTrain_fits), axis=0)
            # exp_MEAN_pop_std = np.mean(np.asarray(self.pop_std_fit_list), axis=0)  # must find pooled!


            # # Calc errors
            if self.StandardError == 0:
                self.exp_std = np.std(self.train_fit_matrix, axis=0)
            elif self.StandardError == 1:
                num_runs = len(self.train_fit_matrix)
                # print("----> Std error num_runs", num_runs)
                self.exp_std = np.std(self.train_fit_matrix, axis=0)/(num_runs**0.5)

            loop_lab = "%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop])

            # # Save to Text the FIRST Training Stats
            mean_Untrained_fit = np.mean(self.train_fit_matrix[:,0])
            std_Untrained_fit = np.std(self.train_fit_matrix[:,0])
            best_Untrained_fit = min(self.train_fit_matrix[:,0])
            result_stats_Untrained.append("%s, %.3f, %.3f, %.3f" % (loop_lab, mean_Untrained_fit, std_Untrained_fit, best_Untrained_fit))

            # # Save to Text the Final Training Stats
            mean_training_fit = np.mean(self.train_fit_matrix[:,-1])
            std_training_fit = np.std(self.train_fit_matrix[:,-1])
            best_training_fit = min(self.train_fit_matrix[:,-1])
            result_stats_training.append("%s, %.3f, %.3f, %.3f" % (loop_lab, mean_training_fit, std_training_fit, best_training_fit))

            # # Save to Text the Final Validation Stats
            if vali_fits != 'na':
                self.vali_train_fit_matrix = np.asarray(vali_fits)

                mean_Fvali_fit = np.mean(self.vali_train_fit_matrix[:,0])
                std_Fvali_fit = np.std(self.vali_train_fit_matrix[:,0])
                best_Fvali_fit = min(self.vali_train_fit_matrix[:,0])
                result_stats_first_validation.append("%s, %.3f, %.3f, %.3f" % (loop_lab, mean_Fvali_fit, std_Fvali_fit, best_Fvali_fit))

                mean_vali_fit = np.mean(self.vali_train_fit_matrix[:,-1])
                std_vali_fit = np.std(self.vali_train_fit_matrix[:,-1])
                best_vali_fit = min(self.vali_train_fit_matrix[:,-1])
                result_stats_validation.append("%s, %.3f, %.3f, %.3f" % (loop_lab, mean_vali_fit, std_vali_fit, best_vali_fit))

            # # Find & Save to Text the Test Stats
            self.exp_MEAN_test_fit = np.mean(np.asarray(self.test_fits))
            self.exp_std_test_fit = np.std(np.asarray(self.test_fits))
            self.best_test_fit = np.min(np.asarray(self.test_fits))  # best veri fitness
            result_stats_test.append("%s, %.3f, %.3f, %.3f" % (loop_lab, self.exp_MEAN_test_fit, self.exp_std_test_fit, self.best_test_fit))

            # # Find & Save to Text the Noisy Test Stats
            mean_nTest = np.mean(np.asarray(self.noisytest_fits))
            std_nTest = np.std(np.asarray(self.noisytest_fits))
            best_nTest = min(np.asarray(self.noisytest_fits))
            result_stats_noisytest.append("%s, %.3f, %.3f, %.3f" % (loop_lab, mean_nTest, std_nTest, best_nTest))

            #

            # Plot's'

            self.plt_mean()

            self.plt_std()

            self.plt_final_Vs_veri()

            self.plt_popmean()

            #

            # store histogram data of final fitnesses
            self.Exp_list_FinalTrainFits.append(self.train_fit_matrix[:,-1])
            self.Exp_list_FinalGbestFits.append(self.Gbest_fit_matrix[:,-1])
            self.Exp_list_TestFits.append(self.test_fits)
            self.Exp_list_NoisyTestFits.append(self.noisytest_fits)
            if titles == 'on':
                self.hist_the_lablel.append("%s %s, $G_{best}$ Fit=%.4f" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], min(self.Gbest_fit_matrix[:,-1])))
            else:
                self.hist_the_lablel.append("%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop]))
            # increment!
            self.dir_loop = self.dir_loop + 1

        # plot hist
        self.plt_hist()
        self.plt_hist_random_test()

        self.plt_hist_gene('Config')
        self.plt_hist_gene('InWeight')
        #self.plt_hist_gene('InBias')
        self.plt_hist_gene('OutWeight')
        #self.plt_hist_gene('OutBias')
        #self.plt_hist_gene('Shuffle')

        #self.plt_box(type='Training')
        self.plt_box(type='Test')
        self.plt_box(type='TestAccr')
        self.plt_box(type='Gbest')
        self.plt_box(type='NTestDiff', force_plot=1)

        self.plt_noise_change()

        # #  show Exp graphs
        if self.show == 1 and self.Save_NotShow == 0:
            plt.show()
        else:
            plt.close('all')

        if self.Save_NotShow == 1:

            with open("%s/Results&Stats_UnTrain.txt" % (self.save_dir), "w") as res_file:
                Deets = ''
                for Deets_Line in result_stats_Untrained:
                    Deets = '%s%s \n' % (Deets, Deets_Line)
                res_file.write(Deets)

            with open("%s/Results&Stats_Train.txt" % (self.save_dir), "w") as res_file:
                Deets = ''
                for Deets_Line in result_stats_training:
                    Deets = '%s%s \n' % (Deets, Deets_Line)
                res_file.write(Deets)

            if vali_fits != 'na':

                with open("%s/Results&Stats_Vali_FirstEpoch.txt" % (self.save_dir), "w") as res_file:
                    Deets = ''
                    for Deets_Line in result_stats_first_validation:
                        Deets = '%s%s \n' % (Deets, Deets_Line)
                    res_file.write(Deets)

                with open("%s/Results&Stats_Vali_Final.txt" % (self.save_dir), "w") as res_file:
                    Deets = ''
                    for Deets_Line in result_stats_validation:
                        Deets = '%s%s \n' % (Deets, Deets_Line)
                    res_file.write(Deets)

            with open("%s/Results&Stats_Test.txt" % (self.save_dir), "w") as res_file:
                Deets = ''
                for Deets_Line in result_stats_test:
                    Deets = '%s%s \n' % (Deets, Deets_Line)
                res_file.write(Deets)

            with open("%s/Results&Stats_testNoisy.txt" % (self.save_dir), "w") as res_file:
                Deets = ''
                for Deets_Line in result_stats_noisytest:
                    Deets = '%s%s \n' % (Deets, Deets_Line)
                res_file.write(Deets)

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
    def plt_mean(self, x='Epoch'):
        if 'plt_mean' in self.sel_dict:
            if self.sel_dict['plt_mean'] == 1:

                if x == 'Epoch':
                    fig = plt.figure(10)
                    plt.xlabel('Iteration')
                elif x == 'Ncomps':
                    fig = plt.figure(11)
                    plt.xlabel(x)


                if self.title == 'on':
                    if self.PlotTest == 1:
                        the_lablel = "%s %s, MeanTest=%.3f, BestTest=%.3f" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], self.exp_MEAN_test_fit, self.best_test_fit)
                    else:
                        the_lablel = "%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop])
                else:
                    the_lablel = "%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop])


                if len(self.new_DirList) > 1:

                    exp_MEAN_fit = np.mean(self.train_fit_matrix, axis=0)

                    if x == 'Epoch':
                        new_stop = self.batch_weight*(len(exp_MEAN_fit)-1)
                        nspaces = len(exp_MEAN_fit)
                        x_vals = np.linspace(0, new_stop, nspaces)
                    elif x == 'Ncomps':
                        x_vals = self.train_ncomps[0]

                    plt.plot(x_vals, exp_MEAN_fit, label=the_lablel, marker=self.markers[self.dir_loop])

                    if self.ExpErrors == 1:
                        bot = exp_MEAN_fit-self.exp_std
                        top = exp_MEAN_fit+self.exp_std
                        plt.fill_between(x_vals, bot, top, alpha=0.4)

                else:  # for syst comparison
                    for i in range(self.MetaData['num_systems']):

                        mean_sys_fits = np.mean(self.train_fits_per_sys[i], axis=0)
                        lab = "Sys %d" % (i)

                        if x == 'Epoch':
                            new_stop = self.batch_weight*(len(mean_sys_fits)-1)
                            nspaces = len(mean_sys_fits)
                            x_vals = np.linspace(0, new_stop, nspaces)
                        elif x == 'Ncomps':
                            x_vals = self.train_ncomps[0]

                        plt.plot(x_vals, mean_sys_fits, label=lab, marker=self.markers[self.dir_loop])

                        if self.StandardError == 0:
                            exp_std = np.std(mean_sys_fits, axis=0)
                        elif self.StandardError == 1:
                            num_runs = len(self.train_fits_per_sys[i])
                            exp_std = np.std(mean_sys_fits, axis=0)/(num_runs**0.5)

                        if self.ExpErrors == 1:
                            bot = mean_sys_fits-exp_std
                            top = mean_sys_fits+exp_std
                            #plt.fill_between(range(len(mean_sys_fits)), bot, top, alpha=0.4)
                            plt.fill_between(x_vals, bot, top, alpha=0.4)

                if self.legends == 'on':
                    plt.legend()

                #fig.tight_layout()

                plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                if self.fig_letter != 'na':
                    fig.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

                if self.title == 'on':
                    if self.ExpErrors == 1 and self.StandardError == 0:
                        plt.title('Mean fitness of $p_{best}$ (std)')
                    elif self.ExpErrors == 1 and self.StandardError == 1:
                        plt.title('Mean fitness of $p_{best}$ (SE)')
                    else:
                        plt.title('Mean fitness of $p_{best}$')

                plt.ylabel('Mean Best fitness')
                if self.plt_mean_yaxis != 'auto':
                    plt.ylim((self.plt_mean_yaxis))
                if self.logplot == 1:
                    plt.yscale('log')

                # # self.show & save plots
                if self.curr_dir == self.new_DirList[-1]:
                    if self.Save_NotShow == 1:
                        fig_path = "%s/ExpALL__FIG_mean_Fv%s.%s" % (self.save_dir, x, self.format)
                        fig.savefig(fig_path, dpi=self.dpi)
                        plt.close(fig)

    #

    #

    #

    # #######################################################
    # Plots accuracy of Global best  fitness over epochs
    # #######################################################
    def plt_accuracy(self, x='Epoch'):
        if 'plt_accuracy' in self.sel_dict:
            if self.sel_dict['plt_accuracy'] == 1:

                if x == 'Epoch':
                    fig = plt.figure(6)
                elif x == 'Ncomps':
                    fig = plt.figure(7)


                if self.title == 'on':
                    if self.PlotTest == 1:
                        the_lablel = "%s %s, MeanTest=%.3f, BestTest=%.3f" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], self.exp_MEAN_test_fit, self.best_test_fit)
                    else:
                        the_lablel = "%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop])
                else:
                    the_lablel = "%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop])


                if len(self.new_DirList) > 1:
                    vali_acc = 1 - self.Gbest_error_matrix
                    exp_MEAN_fit = np.mean(vali_acc, axis=0)


                    if x == 'Epoch':
                        x_vals = np.arange(1, len(exp_MEAN_fit)+1, 1)
                    elif x == 'Ncomps':
                        x_vals = self.epoch_ncomps[0]
                        # print(self.epoch_ncomps[0])
                    plt.plot(x_vals, exp_MEAN_fit, label=the_lablel, marker=self.markers[self.dir_loop])

                    if self.ExpErrors == 1:
                        # # Calc errors
                        if self.StandardError == 0:
                            exp_std = np.std(vali_acc, axis=0)
                        elif self.StandardError == 1:
                            num_runs = len(vali_acc)
                            # print("----> Std error num_runs", num_runs)
                            exp_std = np.std(vali_acc, axis=0)/(num_runs**0.5)

                        bot = exp_MEAN_fit-exp_std
                        top = exp_MEAN_fit+exp_std
                        plt.fill_between(x_vals, bot, top, alpha=0.4)

                else:
                    for i in range(self.MetaData['num_systems']):

                        accuracy = 1 - self.Gbest_Errs_per_sys[i]
                        mean_sys_accurcy = np.mean(accuracy, axis=0)
                        lab = "Sys %d" % (i)

                        if x == 'Epoch':
                            x_vals = np.arange(1, len(mean_sys_accurcy)+1, 1)
                        elif x == 'Ncomps':
                            x_vals = self.epoch_ncomps[0]

                        plt.plot(x_vals, mean_sys_accurcy, label=lab, marker=self.markers[self.dir_loop])

                        if self.StandardError == 0:
                            exp_std = np.std(mean_sys_accurcy, axis=0)
                        elif self.StandardError == 1:
                            num_runs = len(accuracy)
                            exp_std = np.std(mean_sys_accurcy, axis=0)/(num_runs**0.5)

                        if self.ExpErrors == 1:
                            bot = mean_sys_accurcy-exp_std
                            top = mean_sys_accurcy+exp_std
                            #plt.fill_between(range(len(mean_sys_fits)), bot, top, alpha=0.4)
                            plt.fill_between(x_vals, bot, top, alpha=0.4)

                if self.legends == 'on':
                    plt.legend()

                #fig.tight_layout()

                plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                if self.fig_letter != 'na':
                    fig.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

                if self.title == 'on':
                    if self.ExpErrors == 1 and self.StandardError == 0:
                        plt.title('Mean Accuracy of $G_{best}$ (std)')
                    elif self.ExpErrors == 1 and self.StandardError == 1:
                        plt.title('Mean Accuracy of $G_{best}$  (SE)')
                    else:
                        plt.title('Mean Accuracy of $G_{best}$')
                plt.xlabel(x)
                plt.ylabel('Mean Accuracy')
                if self.plt_mean_yaxis != 'auto':
                    plt.ylim((self.plt_mean_yaxis))
                if self.logplot == 1:
                    plt.yscale('log')

                # # self.show & save plots
                if self.curr_dir == self.new_DirList[-1]:
                    if self.Save_NotShow == 1:
                        fig_path = "%s/ExpALL__FIG_mean_Av%s.%s" % (self.save_dir, x, self.format)
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

                new_stop = self.batch_weight*(len(self.exp_std)-1)
                nspaces = len(self.exp_std)
                x_vals = np.linspace(0, new_stop, nspaces)

                plt.plot(x_vals, self.exp_std, label=the_lablel, marker=self.markers[self.dir_loop])

                if self.legends == 'on':
                    plt.legend()

                if self.StandardError == 0:
                    plt.title('std Population of fitnesses')
                    plt.ylabel('Std of Pop Fitnesses')
                elif self.StandardError == 1:
                    plt.title('Standard Error Population of fitnesses')
                    plt.ylabel('SE of Pop Fitnesses')
                plt.xlabel('Epoch')

                if self.logplot == 1:
                    plt.yscale('log')

                # # self.show & save plots
                if self.curr_dir == self.new_DirList[-1]:
                    if self.Save_NotShow == 1:
                        fig_path = "%s/ExpALL__FIG_std_FvE.%s" % (self.save_dir, self.format)
                        fig.savefig(fig_path, dpi=self.dpi)
                        plt.close(fig)

    #

    #

    #

    # #######################################################
    # Plot final fitness vs Test fitness
    # #######################################################
    def plt_final_Vs_veri(self, noise=0):
        if self.sel_dict['plt_finalveri'] == 1:

            fig = plt.figure(3)

            if noise == 0:
                y_data = self.test_fits
                fig_path = "%s/ExpALL__FIG_Scatter_FinalTrainVsVeri.%s" % (self.save_dir, self.format)
                ylab = 'Test Fitness'
            elif noise == 1:
                y_data = self.noisytest_fits
                fig_path = "%s/ExpALL__FIG_Scatter_FinalTrainVsNoisyVeri.%s" % (self.save_dir, self.format)
                ylab = 'Noisy Test Fitness'

            the_lablel = "%s %s" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop])

            #S = 10 + 100/((1+self.dir_loop)**3)
            num_datasets = len(self.new_DirList)
            s_range = np.flip(np.arange(num_datasets) + 1)**1.5
            S = 10 + s_range[self.dir_loop]*10

            # s=1+(len(self.train_fit_matrix[:,-1])/10-self.dir_loop)**3
            #print("size scale s =", S )


            #print("self.test_fits:\n", self.test_fits)

            plt.scatter(self.Gbest_fit_matrix[:,-1], y_data, label=the_lablel,
                        marker=self.markers[self.dir_loop], s=S) # , alpha=(len(self.train_fit_matrix[:,-1])-self.dir_loop)/len(self.train_fit_matrix[:,-1])

            if self.legends == 'on':
                plt.legend()

            if self.fig_letter != 'na':
                fig.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

            plt.title('Final $G_{best}$ fitness against the Test Fitness')
            plt.ylabel(ylab)
            plt.xlabel('Final $G_{best}$ fit from training')


            # # self.show & save plots
            if self.curr_dir == self.new_DirList[-1]:
                if self.Save_NotShow == 1:
                    fig.savefig(fig_path, dpi=self.dpi)
                    plt.close(fig)

    #

    #

    #

    # #######################################################
    # Plot pop mean fitness (with std/SE)
    # #######################################################
    def plt_popmean(self, include_test_fit=1):
        if 'plt_popmean' in self.sel_dict:
            if self.sel_dict['plt_popmean'] == 1:
                fig = plt.figure(4)

                col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

                # # calc pooled std/SE
                pop_std_train_fit_matrix = np.asarray(self.pop_std_fit_list)
                pooled_pop_std = []  # create list to store pooled pop std
                pop = self.ParamDict['popsize']

                for k in range(len(pop_std_train_fit_matrix[0, :])):  # iterates over the "its"
                    if self.StandardError == 0:
                        numerator = 0
                        denomenator = 0
                        # calculate pooled std
                        for j in range(len(pop_std_train_fit_matrix[:, k])):
                            numerator = ((pop-1)*(pop_std_train_fit_matrix[j, k])**2) + numerator
                            denomenator = pop - 1 + denomenator
                        std_pooled = (numerator/denomenator)**0.5
                        pooled_pop_std.append(std_pooled)
                    elif self.StandardError == 1:
                        numerator = 0
                        denomenator = 0
                        total_pop = 0
                        # calculate pooled std
                        for j in range(len(pop_std_train_fit_matrix[:, k])):
                            numerator = ((pop-1)*(pop_std_train_fit_matrix[j, k])**2) + numerator
                            denomenator = pop - 1 + denomenator
                            total_pop = total_pop + pop
                        std_pooled = (numerator/denomenator)**0.5
                        SE = std_pooled/(total_pop**0.5)
                        pooled_pop_std.append(SE)

                #print("pop_std_train_fit_matrix\n", pop_std_train_fit_matrix)
                #print("pooled_pop_std\n", pooled_pop_std)

                the_lablel = "%s %s, Av TestFit=%f" % (self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], self.exp_MEAN_test_fit)

                new_stop = self.batch_weight*(len(self.exp_MEAN_pop_fit)-1)
                nspaces = len(self.exp_MEAN_pop_fit)
                x_vals = np.linspace(0, new_stop, nspaces)

                # Select a standard error plot, or a std band self.fill plot
                if self.fill == 0:
                    plt.errorbar(x_vals, self.exp_MEAN_pop_fit, yerr=pooled_pop_std, linestyle='None', marker='s', capsize=5, label=the_lablel, color=col_list[self.dir_loop])
                elif self.fill == 1:
                    plt.plot(x_vals, self.exp_MEAN_pop_fit, '-x', label=the_lablel)
                    #plt.fill_between(range(len(self.exp_MEAN_pop_fit)), self.exp_MEAN_pop_fit-pooled_pop_std, self.exp_MEAN_pop_fit+pooled_pop_std, alpha=0.5, color=col_list[self.dir_loop])
                    plt.fill_between(x_vals, self.exp_MEAN_pop_fit-pooled_pop_std, self.exp_MEAN_pop_fit+pooled_pop_std, alpha=0.5, color=col_list[self.dir_loop])

                if self.legends == 'on':
                    plt.legend()

                # #####
                # Plot Test fitness line here??
                # #####
                if include_test_fit == 1:

                    plt.plot([x_vals[0], x_vals[-1]],[self.exp_MEAN_test_fit, self.exp_MEAN_test_fit],'--', color=col_list[self.dir_loop]) # set to right colour!

                    if self.StandardError == 0:
                        plt.text((x_vals[-2])/2, self.exp_MEAN_test_fit, 'Mean Test Fitness (w/ std)')
                        plt.fill_between([x_vals[0], x_vals[-1]], [self.exp_MEAN_test_fit-self.exp_std_test_fit,self.exp_MEAN_test_fit-self.exp_std_test_fit],
                                         [self.exp_MEAN_test_fit+self.exp_std_test_fit, self.exp_MEAN_test_fit+self.exp_std_test_fit], alpha=0.2, color=col_list[self.dir_loop])
                    elif self.StandardError == 1:
                        SE_VeriFit = self.exp_std_test_fit/(len(self.exp_MEAN_pop_fit)**0.5)
                        plt.text((x_vals[-2])/2, self.exp_MEAN_test_fit, 'Mean Test Fitness (w/ SE)')
                        plt.fill_between([x_vals[0], x_vals[-1]], [self.exp_MEAN_test_fit-SE_VeriFit,self.exp_MEAN_test_fit-SE_VeriFit],
                                         [self.exp_MEAN_test_fit+SE_VeriFit, self.exp_MEAN_test_fit+SE_VeriFit], alpha=0.2, color=col_list[self.dir_loop])



                if self.ExpErrors == 1 and self.StandardError == 0:
                    plt.title('Evolution of Mean Population fitness (pooled std)')
                elif self.ExpErrors == 1 and self.StandardError == 1:
                    plt.title('Evolution of Mean Population fitness (SE)')
                plt.xlabel('Epoch')
                plt.ylabel('Mean Pop Fitness ')

                if self.logplot == 1:
                    plt.yscale('log')

                # # self.show & save plots
                if self.curr_dir == self.new_DirList[-1]:
                    if self.Save_NotShow == 1:
                        fig_path = "%s/ExpALL__FIG_mean_PopFit.%s" % (self.save_dir, self.format)
                        fig.savefig(fig_path, dpi=self.dpi)
                        plt.close(fig)

    #

    #

    #

    # #######################################################
    # Plot final fitness as a box plot
    # #######################################################
    def plt_box(self, type='Gbest', force_plot=0):
        if self.sel_dict['plt_box'] == 1 or force_plot == 1:

            fig = plt.figure()
            y1, y2 = -0.005, 0.45  # y lim
            if type == 'Training':
                dat = self.Exp_list_FinalTrainFits
                ylab = '%s Fitness' % (type)
                fig_path = "%s/ExpALL__FIG_BoxPlot_FinalTrain.%s" % (self.save_dir, self.format)

            elif type == 'Gbest':
                dat = self.Exp_list_FinalGbestFits
                ylab = '%s Fitness' % (type)
                fig_path = "%s/ExpALL__FIG_BoxPlot_FinalGbest.%s" % (self.save_dir, self.format)

            elif type == 'Test':
                dat = self.Exp_list_TestFits
                ylab = '%s Fitness' % (type)
                fig_path = "%s/ExpALL__FIG_BoxPlot_Test.%s" % (self.save_dir, self.format)

            elif type == 'TestAccr':
                dat_fit = self.Exp_list_TestFits
                dat = []
                for d in dat_fit:
                    dat.append(1-np.asarray(d))
                ylab = 'Test Accuracy'
                fig_path = "%s/ExpALL__FIG_BoxPlot_TestAccr.%s" % (self.save_dir, self.format)
                y1, y2 = 0.45, 1

            elif type == 'NTest':
                dat = self.Exp_list_NoisyTestFits
                ylab = '%s Fitness' % (type)
                fig_path = "%s/ExpALL__FIG_BoxPlot_NoisyTest.%s" % (self.save_dir, self.format)

            elif type == 'NTestDiff':
                if len(self.new_DirList) > 1:
                    dat = []
                    for d in range(len(self.new_DirList)):
                        dat.append(abs(np.asarray(self.Exp_list_TestFits[d])-np.asarray(self.Exp_list_NoisyTestFits[d])))
                else:
                    dat = abs(np.asarray(self.Exp_list_TestFits)-np.asarray(self.Exp_list_NoisyTestFits))
                ylab = '$abs(\Phi^{test}-\Phi^{test}_{noisy})$'
                fig_path = "%s/ExpALL__FIG_BoxPlot_NoisyTestDIFF.%s" % (self.save_dir, self.format)
            else:
                raise ValueError("Box plot type invalid")

            if len(self.new_DirList) > 1:
                # plot noraml box plot
                plt.boxplot(x=dat, labels=self.new_Param_array)
                plt.xlabel('Paramater')
            else:
                # group the reps together, agaist circuit
                shaped_dat = np.reshape(dat, (self.MetaData['num_systems'], self.MetaData['num_repetitions'])).T
                #print("shaped_dat:\n", shaped_dat)
                plt.boxplot(x=shaped_dat, labels=np.arange(self.MetaData['num_systems']))
                plt.xlabel('Sys')


            if self.fig_letter != 'na':
                fig.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

            plt.ylabel(ylab)

            if np.max(np.asarray(dat)) > y2:
                y2 = np.max(np.asarray(dat))

            if np.min(np.asarray(dat)) < y1:
                y1 = np.min(np.asarray(dat))

            plt.ylim(y1, y2)

            # # self.show & save plots
            if self.curr_dir == self.new_DirList[-1]:
                if self.Save_NotShow == 1:
                    fig.savefig(fig_path, dpi=self.dpi)
                    plt.close(fig)

                    bad_hit = 0
                    float_prm = []
                    print(">>>", self.new_Param_array)
                    for x in self.new_Param_array:
                        try:
                            float_prm.append(float(x))
                        except ValueError:
                            bad_hit = 1

                    # # Also plot error plot if params are all numeric
                    if type == 'Test' and bad_hit == 0:
                        fig = plt.figure()
                        for exp, test_fits in enumerate(self.Exp_list_TestFits):
                            plt.errorbar(float_prm[exp], np.mean(test_fits), yerr=np.std(test_fits), marker='x', fmt='')
                        fig.savefig("%s/ExpALL__FIG_ErrorPlt_Test.%s" % (self.save_dir, self.format), dpi=self.dpi)
                        plt.close(fig)
    #

    #

    #

    #

    # #######################################################
    # Plot noisy test fitness (change vs OG test no noise fit)
    # #######################################################
    def plt_noise_change(self, force_plot=1):
        if force_plot == 1:

            fig = plt.figure()
            plt.style.use('seaborn-dark-palette')  # 'seaborn-dark-palette'

            for d in range(len(self.new_DirList)):
                VaryNoiseTestFits = self.Exp_list_VaryNoiseTestFits[d]

                data = np.zeros((len(VaryNoiseTestFits), len(self.VaryNoiseTestFit_list)))
                for row, row_array in enumerate(VaryNoiseTestFits):
                    for col, n in enumerate(self.VaryNoiseTestFit_list):
                        data[row, col] = row_array[col]

                the_lablel = "%s %s" % (self.new_FileRefNames[d], self.new_Param_array[d])
                data_mean = np.mean(data, axis=0)
                data_std = np.std(data, axis=0)
                plt.plot(self.VaryNoiseTestFit_list, data_mean, label=the_lablel, marker=self.markers[self.dir_loop])

                if self.ExpErrors == 1:
                    bot = data_mean-data_std
                    top = data_mean+data_std
                    plt.fill_between(self.VaryNoiseTestFit_list, bot, top, alpha=0.4)


            if self.fig_letter != 'na':
                fig.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

            #ylab = 'mean $abs(\Phi^{test}-\Phi^{test}_{noisy})$'
            ylab = '$mean(\Phi^{test}_{noisy})$'
            plt.ylabel(ylab)
            plt.xlabel('% Noise Added to Test Data')
            plt.legend()

            # # self.show & save plots
            if self.curr_dir == self.new_DirList[-1]:
                if self.Save_NotShow == 1:
                    fig_path = "%s/ExpALL__FIG_VaryNoisyTest.%s" % (self.save_dir, self.format)
                    fig.savefig(fig_path, dpi=self.dpi)
                    plt.close(fig)
        plt.style.use('default')
        return
    #

    #

    # #######################################################
    # Plot histogram of final fitnesses
    # #######################################################
    def plt_hist(self, type='Gbest'):
        if 'plt_hist' in self.sel_dict:
            if self.sel_dict['plt_hist'] == 1:

                if type == 'Train':
                    data = self.Exp_list_FinalTrainFits
                    xlab = 'Final Training Fitness'
                    fig_path = "%s/ExpALL__FIG_FinalTrainFit_Hist.%s" % (self.save_dir, self.format)
                elif type == 'Gbest':
                    data = self.Exp_list_FinalGbestFits
                    xlab = 'Final $G_{best}$ Fit from training period'
                    fig_path = "%s/ExpALL__FIG_FinalGbestFit_Hist.%s" % (self.save_dir, self.format)
                elif type == 'Test':
                    data = self.Exp_list_TestFits
                    xlab = 'Final Test Fitness'
                    fig_path = "%s/ExpALL__FIG_FinalTestFit_Hist.%s" % (self.save_dir, self.format)

                # determin limits of the histogrtam
                if self.HistMax == 'auto':
                    hist_max = max(np.concatenate(data))+0.02
                    hist_max = hist_max
                else:
                    hist_max = self.HistMax

                if self.HistMin == 'auto':
                    hist_min = min(np.concatenate(data))-0.02
                    hist_min = hist_min
                    if hist_min < 0:
                        hist_min = 0
                else:
                    hist_min = self.HistMin

                fig = plt.figure(5)
                #print(">>hist lables:\n", self.hist_the_lablel)
                #print(">>hist limits:", hist_min, hist_max)
                plt.hist(data, bins=np.arange(hist_min, hist_max, self.HistBins), label=self.hist_the_lablel)

                if self.legends == 'on':
                    plt.legend()

                if self.title == 'on':
                    sub_title = "Bins = %.4f" % self.HistBins
                    plt.title('Histogram of final fitness distribution over all the loops\n%s' % sub_title)

                if self.fig_letter != 'na':
                    fig.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

                plt.xlabel(xlab)
                plt.ylabel('Count')

                # # self.show & save plots
                if self.Save_NotShow == 1:
                    fig.savefig(fig_path, dpi=self.dpi)
                    plt.close(fig)
    #

    #

    #

    # #######################################################
    # Plot histogram of Gene Values
    # #######################################################
    def plt_hist_gene(self, gene, HistBins=0.1):
        if 'plt_genes' in self.sel_dict:
            if self.sel_dict['plt_genes'] == 1:

                """# # Colled valid gene keys
                active_genes = []
                for key in self.MetaData['genome']:
                    if 'loc' in key:
                        active_genes.append(key)

                # # test for gene
                loc_key = 'na'
                for key in active_genes:
                    if gene in key:
                        loc_key = key"""

                if gene not in self.MetaData['genome']:
                    print("Passed in gene '%s' is an invalide decision paramater." % (gene))
                    return

                """if 'loc' not in self.MetaData['genome'][gene].keys():
                    print("Cannot plot histogram of gene values as '%s' is not a valid/saved gene." % (gene))
                    return

                if self.MetaData['genome'][gene]['active'] != 1:
                    print("Cannot plot histogram of gene selected (%s) is not active" % (gene))
                    return"""

                print("\nHist for Genes:", gene)

                # Select gene and format
                Exp_list_BestGeneList = []
                bad_hit = 0
                MaxMinList = []
                hit_label = []

                for exp, BestGenomesList in enumerate(self.Exp_list_BestGenomesList):

                    prm = self.Exp_list_MetaData[exp]

                    if prm['genome'][gene]['active'] == 1:

                        num_genes = len(self.Exp_list_BestGenomesList[exp][0][prm['genome'][gene]['loc']])  # in selected group
                        num_genomes = len(self.Exp_list_BestGenomesList[exp])
                        Exp_list_GeneList = np.zeros((num_genomes, num_genes))

                        for row, Genome  in enumerate(BestGenomesList):
                            genGroup = Genome[prm['genome'][gene]['loc']]
                            for idx, g in enumerate(genGroup):
                                Exp_list_GeneList[row, idx] = g

                        Exp_list_BestGeneList.append(Exp_list_GeneList)
                        MaxMinList.append(Exp_list_GeneList)
                        hit_label.append(self.hist_the_lablel[exp])
                    else:
                        Exp_list_BestGeneList.append('bad')
                        bad_hit += 1

                if bad_hit >= len(self.Exp_list_MetaData):
                    print("None of the results foldered contained results using that gene (%s) i.e. active" % (gene))
                    # print(" > ",  bad_hit, len(self.Exp_list_MetaData))
                    return

                # determin limits of the histogrtam
                hist_max = np.nanmax(MaxMinList)+0.5
                hist_min = np.nanmin(MaxMinList)-0.5

                #print("self.Exp_list_BestGeneList:\n", self.Exp_list_BestGeneList)
                fig_gh, ax = plt.subplots(ncols=1, nrows=num_genes, sharey=True, sharex=True , figsize=(10, 6), squeeze=False)

                colours = ['b','r','g','c','m', 'y', 'k']

                #print(">>>", len(self.Exp_list_BestGenomesList))

                if len(self.Exp_list_BestGenomesList) > 1:
                    """for exp in range(len(Exp_list_BestGeneList)):
                        BestGeneArray = Exp_list_BestGeneList[exp]

                        for g in range(num_genes):
                            ax[g].hist(BestGeneArray[:, g], bins=np.arange(hist_min, hist_max, HistBins), alpha=0.5, label=self.hist_the_lablel[exp])
                            ax[g].set_ylabel('G%d' % (g+1))"""

                    for g in range(num_genes):

                        vals = []

                        for exp in range(len(Exp_list_BestGeneList)):
                            BestGeneArray = Exp_list_BestGeneList[exp]
                            if BestGeneArray == 'bad':
                                empty_hit = 1
                            else:
                                vals.append(BestGeneArray[:, g])
                                empty_hit = 0

                        if empty_hit == 1:
                            continue

                        ax[g,0].hist(vals, bins=np.arange(hist_min, hist_max, HistBins), label=hit_label)
                        ax[g,0].set_ylabel('G%d' % (g+1))

                else:

                    lab = []
                    for i in range(self.MetaData['num_systems']):
                        lab.append("Sys %d" % (i))

                    BestGeneArray = Exp_list_BestGeneList[0]
                    for g in range(num_genes):

                        gene_vals = np.asarray(BestGeneArray[:, g])
                        #print(g, self.MetaData['num_systems'], self.MetaData['num_repetitions'], "gene_vals\n", gene_vals)
                        gene_vals = gene_vals.reshape(self.MetaData['num_systems'], self.MetaData['num_repetitions']).T
                        # print(g, "reshape gene_vals\n", gene_vals)

                        ax[g,0].hist(gene_vals, bins=np.arange(hist_min, hist_max, HistBins), label=lab)
                        #ax[g].hist(gene_vals[g], bins=np.arange(hist_min, hist_max, HistBins), label=lab[g], color=colours[g])
                        ax[g,0].set_ylabel('G%d' % (g+1))

                if self.legends == 'on':
                    plt.legend()

                if self.title == 'on':
                    sub_title = "Bins = %.2f" % HistBins
                    plt.suptitle('Histogram of %s gene distribution over all the loops\n%s' % (gene, sub_title))

                if self.fig_letter != 'na':
                    fig_gh.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

                plt.xlabel('Gene Value')

                # # self.show & save plots
                if self.Save_NotShow == 1:
                    fig_path = "%s/ExpALL__FIG_GeneHist_%s.%s" % (self.save_dir, gene, self.format)
                    fig_gh.savefig(fig_path, dpi=self.dpi)
                    plt.close(fig_gh)

                # print("Genome hist done")
        return

    #

    #

    #

    # #######################################################
    # Plot histogram of Gene Values
    # #######################################################
    def plt_hist_random_test(self, HistBins=0.1):
        """
        Plots the processed (best solution) outputs, and plots them.
        Could use random data, or test data.
        Allows us to see how the porocessor seperates them.
        """

        if 'plt_rT' in self.sel_dict:
            if self.sel_dict['plt_rT'] == 1:

                print("\nHist for Random Test inputs",)

                # Select gene and format
                hist_max = np.max(np.concatenate(np.asarray(self.Exp_list_RandomTest))[:,1])+0.5
                hist_min = np.min(np.concatenate(np.asarray(self.Exp_list_RandomTest))[:,1])-0.5

                #print("hist lims!:", hist_max, hist_min)
                for exp, RandomTest in enumerate(self.Exp_list_RandomTest):

                    pClass = RandomTest[:,0]
                    prY = RandomTest[:,1]

                    cl1_idx = np.where(pClass==1)[0]
                    cl2_idx = np.where(pClass==2)[0]

                    data = [prY[cl1_idx], prY[cl2_idx]]

                    fig = plt.figure()
                    colours = ['#009cffff','#ff8800ff','#9cff00ff','c','m', 'y', 'k']
                    #plt.hist(data, bins=np.arange(hist_min, hist_max, HistBins), label=['class 1', 'class 2'])
                    plt.hist(data[0], bins=np.arange(hist_min, hist_max, HistBins), label='class 1', color=colours[0], alpha=0.6)
                    plt.hist(data[1], bins=np.arange(hist_min, hist_max, HistBins), label='class 2', color=colours[1], alpha=0.6)

                    if self.legends == 'on':
                        plt.legend()

                    if self.title == 'on':
                        sub_title = "Bins = %.2f" % HistBins
                        if len(self.Exp_list_RandomTest) > 1:
                            loop_lab = " \n%s %s" % (self.new_FileRefNames[exp], self.new_Param_array[exp])
                            plt.title('Histogram of random value on the Final solution (%s) %s' % (sub_title, loop_lab))
                        else:
                            plt.title('Histogram of random value on the Final solution \n%s' % (sub_title))

                    if self.fig_letter != 'na':
                        fig.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

                    plt.xlabel('Response Y')
                    plt.ylabel('Count')

                    # # self.show & save plots
                    if self.Save_NotShow == 1:
                        if len(self.Exp_list_RandomTest) > 1:
                            fig_path = "%s/ExpALL__FIG_Hist_RandomTest_ExpLoop%s.%s" % (self.save_dir, exp, self.format)
                        else:
                            fig_path = "%s/ExpALL__FIG_Hist_RandomTest.%s" % (self.save_dir, self.format)
                        fig.savefig(fig_path, dpi=self.dpi)
                        plt.close(fig)

                # #############
                # Plot all together
                if len(self.Exp_list_RandomTest) > 1:
                    fig = plt.figure()
                    #c = ['#3776ab', '#ab373b', 'aba537']
                    plt.style.use('seaborn-dark-palette')  # 'seaborn-dark-palette'
                    for exp, RandomTest in enumerate(self.Exp_list_RandomTest):

                        pClass = RandomTest[:,0]
                        prY = RandomTest[:,1]

                        plt.hist(prY, bins=np.arange(hist_min, hist_max, HistBins), label=self.new_Param_array[exp], alpha=0.5)

                        if self.legends == 'on':
                            plt.legend()

                        if self.title == 'on':
                            sub_title = "Bins = %.4f" % HistBins
                            plt.title('Histogram of random value on the Final solution \n%s' % (sub_title))

                        if self.fig_letter != 'na':
                            fig.text(0.02, 0.94, self.fig_letter, fontsize=14, fontweight='bold')

                        plt.xlabel('Response Y')
                        plt.ylabel('Count')

                    # # self.show & save plots
                    if self.Save_NotShow == 1:
                        fig_path = "%s/ExpALL__FIG_Hist_RandomTest_All%s.%s" % (self.save_dir, exp, self.format)
                        fig.savefig(fig_path, dpi=self.dpi)
                        plt.close(fig)

                plt.style.use('default')
        return



# fin
