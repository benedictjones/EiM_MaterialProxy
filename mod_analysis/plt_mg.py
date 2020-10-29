# # Top Matter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import h5py
import os

import matplotlib.animation as animation

from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap

from mod_analysis.Set_Load_meta import LoadMetaData

from mod_load.LoadData import Load_Data

from sklearn.decomposition import PCA
from sklearn import preprocessing

''' Description:
This class file is designed to provide various functions to analise saved data
from the looped DE.
'''
#########################################################################
# functions


def fetch_genome_string(genome, gen_grouping):
    # # form genome text
    rounded_genome = np.around(genome, decimals=3)

    print("rounded_genome", rounded_genome)

    best_genome_text = '['

    i = 0
    group = 0
    for size in gen_grouping:
        for c in range(size):

            best_genome_text = best_genome_text + str(rounded_genome[i])
            i = i + 1

            if c == size-1 and group != len(gen_grouping)-1:
                best_genome_text = best_genome_text + ' | '
            elif c != size-1:
                best_genome_text = best_genome_text + ', '
            else:
                best_genome_text = best_genome_text + ']'
        group = group + 1

    return best_genome_text

#########################################################################


''' # # # # # # # # # # # # # # #
Produce a plot of the average fitness (per it) vs the iterations
with error bars
'''


class material_graphs(object):

    def Plt_mg(self, Save_NotShow=0, show=1, Bgeno=0, Dgeno=0, VC=0, VP=0,
               VoW=0, ViW=0, VoW_ani=0, ViW_ani=0, VoiW_ani=1, VloW_ani=1, VliW_ani=1,
               Specific_Cir_Rep='all', PlotOnly='all', format='gif',
               titles='on'):

        if format != 'gif' and format != 'mp4':
            print("Error (animation): only mp4 and gif formats allowed")
        else:
            self.ani_format = format


        matplotlib .rcParams['font.family'] = 'Arial' # 'serif'
        matplotlib .rcParams['font.size'] = 8  # tixks and title
        matplotlib .rcParams['figure.titlesize'] = 'medium'
        matplotlib .rcParams['axes.labelsize'] = 10  # axis labels
        matplotlib .rcParams['axes.linewidth'] = 1
        #matplotlib .rcParams['mathtext.fontset'] = 'cm'
        matplotlib .rcParams["legend.labelspacing"] = 0.25
        #matplotlib .rcParams["figure.autolayout"] = True
        if titles == 'off':
            matplotlib .rcParams["figure.figsize"] = [3.5,2.7]
        else:
            matplotlib .rcParams["figure.figsize"] = [6.4, 4.8]
        matplotlib .rcParams["figure.autolayout"] = False

        matplotlib.rc('pdf', fonttype=42)  # embeds the font, so can import to inkscape
        #matplotlib.rcParams['text.usetex'] = True
        self.title = titles

        self.Save_NotShow = Save_NotShow
        #self.interp = 'gaussian'
        self.interp = 'none'
        #basic_cols = ['#009cff', '#ffffff','#ff8800']  # pastal orange/white/blue
        basic_cols = ['#009cff', '#6d55ff', '#ffffff', '#ff6d55','#ff8800']  # pastal orange/red/white/purle/blue

        self.fps = 10

        # # create dir list to look at
        if PlotOnly == 'all':
            new_DirList = self.data_dir_list
            new_FileRefNames = self.FileRefNames
            new_Param_array = self.Param_array
        else:
            new_DirList = []
            new_FileRefNames = []
            new_Param_array = []
            for val in PlotOnly:
                new_DirList.append(self.data_dir_list[val])
                new_FileRefNames.append(self.FileRefNames[val])
                new_Param_array.append(self.Param_array[val])

        # # save the data for plotting
        self.dir_loop = 0
        for curr_dir in new_DirList:

            self.MetaData = LoadMetaData(curr_dir)
            self.ParamDict = self.MetaData['DE']
            self.NetworkDict = self.MetaData['network']
            self.GenomeDict = self.MetaData['genome']

            cir_range = range(self.MetaData['num_circuits'])
            rep_range = range(self.MetaData['num_repetitions'])

            if 'HOW' in self.ParamDict['IntpScheme'] or self.ParamDict['IntpScheme'] == 'band':
                self.class_cp = 1  # allows colour bars to change if there it is only plotting the class
                self.max_colour_val = self.GenomeDict['max_class_value']
                self.min_colour_val = self.GenomeDict['min_class_value']
                self.my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)
            else:
                val = 0.2  # 0.2
                self.max_colour_val = val
                self.min_colour_val = -val
                # basic_cols=['#0000ff', '#000000', '#ff0000']
                self.my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)
                self.my_cmap.set_bad(color='#ffff00')
                self.class_cp = 0

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

                    legacy = os.path.isfile("%s/data_%d_rep%d.hdf5" % (curr_dir, cir, rep))
                    if legacy == True:
                        location = "%s/data_%d_rep%d.hdf5" % (curr_dir, cir, rep)
                        hdf =  h5py.File(location, 'r')
                        MG = hdf.get('/MG')

                    elif legacy == False:
                        location = "%s/data.hdf5" % (curr_dir)
                        hdf =  h5py.File(location, 'r')
                        MG = hdf.get('/%d_rep%d/MG' % (cir, rep))

                    try:
                        MG_items = list(MG.keys())
                        print("The keys: \n", MG_items)
                    except:
                        print(">>Could not fetch keys (or no MG keys found to plot)<<")
                        break

                    # Check if Ridged regression was used, and adjust colour bar
                    threshold = np.array(hdf.get('/%d_rep%d/DE_data/OutputLayer_threshold' % (cir, rep)))
                    #print(">>>> plt_mg, threshold", threshold)
                    if np.isnan(threshold) == False:
                        self.max_colour_val = threshold + 1
                        self.min_colour_val = threshold - 1
                        self.class_cp == 0

                    self.curr_dir = curr_dir

                    self.plt_best(hdf, MG, MG_items, Bgeno, cir, rep)
                    self.plt_bestGeneOp(hdf, MG, MG_items, Bgeno, cir, rep)

                    self.plt_defualt(hdf, MG, MG_items, Dgeno, cir, rep)

                    self.plt_VaryConfig(hdf, MG, MG_items, VC, cir, rep)
                    self.plt_VaryConfigOp(hdf, MG, MG_items, VC, cir, rep)

                    self.plt_VaryPerm(hdf, MG, MG_items, VP, cir, rep)

                    self.plt_VaryOutWeight(hdf, MG, MG_items, VoW, cir, rep)
                    the_OW_animation = self.plt_VaryOutWeight_ani(hdf, MG, MG_items, VoW_ani, cir, rep)

                    self.plt_VaryInWeight(hdf, MG, MG_items, ViW, cir, rep)
                    the_IW_animation = self.plt_VaryInWeight_ani(hdf, MG, MG_items, ViW_ani, cir, rep)

                    the_IOW_animation = self.plt_VaryOutInWeight_ani(hdf, MG, MG_items, VoiW_ani, cir, rep)

                    the_lOW_animation = self.plt_VaryLargeOutWeight_ani(hdf, MG, MG_items, VloW_ani, cir, rep, Both=1)

                    the_lIW_animation = self.plt_VaryLargeInWeight_ani(hdf, MG, MG_items, VliW_ani, cir, rep, Both=1)

                    hdf.close()  # exit file
            #

            # #######################################################
            # increment!
            self.dir_loop = self.dir_loop + 1
            # #######################################################

        # # Show all graphs
        if show == 1 and Save_NotShow == 0:
            plt.show()


    # # Function which loads and plots best genome material graph
    def plt_best(self, hdf, MG, MG_items, Bgeno, cir, rep):

        if Bgeno == 0 or 'BestGenome' not in MG_items:
            return

        if self.title == 'off':
            matplotlib .rcParams["figure.figsize"] = [3.5,2.7]
        else:
            matplotlib .rcParams["figure.figsize"] = [6.4, 4.8]

        # fetch data
        responceY = np.array(MG.get('BestGenome/responceY'))
        the_extent = np.array(MG.get('BestGenome/extent'))
        best_genome = np.array(MG.get('BestGenome/the_best_genome'))
        gen_grouping = np.array(MG.get('BestGenome/gen_grouping'))
        best_fitness_val = np.array(MG.get('BestGenome/best_fitness_val'))

        # plot responce data
        fig = plt.figure()
        plt.imshow(responceY.T, origin="lower",
                   extent=the_extent,
                   interpolation=self.interp,
                   vmin=self.min_colour_val, vmax=self.max_colour_val,
                   cmap=self.my_cmap)
        cbar = plt.colorbar(format='% 2.2f')
        cbar.set_label("Network Responce Y", fontsize=14)
        cbar.ax.tick_params(labelsize=11)  # cbar ticks size

        plt.subplots_adjust(left=0.125, bottom=0.15, right=0.87, top=0.85)

        # # form genome text
        best_genome_text = fetch_genome_string(best_genome, gen_grouping)

        # add title and axis
        if self.title == 'on':
            if self.MetaData['SaveDir'] == self.save_dir:
                title_pt1 = 'Class outputs for the best genome (cir %d, rep %d)' % (cir, rep)
            else:
                title_pt1 = 'Class outputs for the best genome (file %d, cir %d, rep %d)' % (self.dir_loop, cir, rep)
            #Best_Genome_str = 'Best Genome %s\nfitness %.3f' % (str(np.around(best_genome, decimals=3)), best_fitness_val)
            Best_Genome_str = 'Best Genome %s\nfitness %.3f' % (str(best_genome_text), best_fitness_val)

            plt.title("%s\n%s" % (title_pt1, Best_Genome_str), fontsize=8)
        plt.xlabel('$a_1$')
        plt.ylabel('$a_2$')


        # plot training data too
        data_set = self.ParamDict['training_data']
        if data_set == 'HL' or data_set == 'HL_nom':
            data_set = 'con2DDS'
        train_data_X, train_data_Y = Load_Data('train', self.MetaData)
        c = 0
        cl1_hit = 0
        cl2_hit = 0
        for row in train_data_X:

            if train_data_Y[c] == 1:
                if cl1_hit == 0:
                    plt.plot(row[0], row[1],  'o', alpha=0.8, markersize=3.5, label="Class 1",
                             markerfacecolor='#009cffff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
                    cl1_hit = 1
                else:
                    plt.plot(row[0], row[1],  'o', alpha=0.8, markersize=3.5,
                             markerfacecolor='#009cffff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
            else:
                if cl2_hit == 0:
                    plt.plot(row[0], row[1],  '*', alpha=0.8, markersize=5.5, label="Class 2",
                             markerfacecolor='#ff8800ff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
                    cl2_hit = 1
                else:
                    plt.plot(row[0], row[1],  '*', alpha=0.8, markersize=5.5,
                             markerfacecolor='#ff8800ff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
            c = c + 1
        plt.legend(markerscale=1.5, fancybox=True)  # , facecolor='k'


        # # show & save plots
        if self.Save_NotShow == 1:
            if self.MetaData['SaveDir'] == self.save_dir:
                fig_path = "%s/%d_rep%d_BestGenome.%s" % (self.save_dir, cir, rep, self.format)
            else:
                fig_path = "%s/File%d__cir%d_rep%d_BestGenome.%s" % (self.save_dir, self.dir_loop, cir, rep, self.format)
            fig.savefig(fig_path, dpi=self.dpi)
            plt.close(fig)

    #

    #

    #

    # # Function which loads and plots defualt genome material graph
    def plt_defualt(self, hdf, MG, MG_items, Dgeno, cir, rep):

        if Dgeno == 0 or 'DefaultGenome' not in MG_items:
            if cir == 0 and rep == 0 and Dgeno !=0:
                print("Defualt material graphs not saved")
            return

        responceY = np.array(MG.get('DefaultGenome/responceY'))
        the_extent = np.array(MG.get('DefaultGenome/extent'))
        defualt_genome = np.array(MG.get('DefaultGenome/default_genome'))
        gen_grouping = np.array(MG.get('DefaultGenome/gen_grouping'))
        defualt_genome_text = fetch_genome_string(defualt_genome, gen_grouping)

        fig = plt.figure(2)
        plt.imshow(responceY.T, origin="lower",
                   extent=the_extent,
                   interpolation=self.interp, vmin=self.min_colour_val, vmax=self.max_colour_val, cmap=self.my_cmap)

        cbar = plt.colorbar(format='% 2.2f')
        cbar.set_label("Network Responce Y", fontsize=14)
        cbar.ax.tick_params(labelsize=11)  # cbar ticks size

        # add title and axis
        if self.title == 'on':
            if self.MetaData['SaveDir'] == self.save_dir:
                title_pt1 = 'Class outputs for the unconficured material (cir %d, rep %d) \n' % (cir, rep)
            else:
                title_pt1 = 'Class outputs for the unconficured material (file %d, cir %d, rep %d) \n' % (self.dir_loop, cir, rep)
            Defualt_Genome_str = 'Default Genome %s\n' % (defualt_genome_text)
            plt.title("%s\n%s" % (title_pt1, Defualt_Genome_str), fontsize=8)
        plt.xlabel('$a_1$')
        plt.ylabel('$a_2$')

        # # show & save plots
        if self.Save_NotShow == 1:
            if self.MetaData['SaveDir'] == self.save_dir:
                fig_path = "%s/%d_rep%d_DefualtGenome.%s" % (self.save_dir, cir, rep, self.format)
            else:
                fig_path = "%s/File%d__cir%d_rep%d_DefualtGenome.%s" % (self.save_dir, self.dir_loop, cir, rep, self.format)
            fig.savefig(fig_path, dpi=self.dpi)
            plt.close(fig)

    #

    #

    #

    # # Function which loads and plots defualt genome material graph
    def plt_bestGeneOp(self, hdf, MG, MG_items, Bgeno, cir, rep):

        if Bgeno == 0 or 'BestGenome' not in MG_items:
            return

        # fetch data
        responceY = np.array(MG.get('BestGenome/responceY'))
        the_extent = np.array(MG.get('BestGenome/extent'))
        best_genome = np.array(MG.get('BestGenome/the_best_genome'))
        gen_grouping = np.array(MG.get('BestGenome/gen_grouping'))
        best_fitness_val = np.array(MG.get('BestGenome/best_fitness_val'))
        op_list = np.array(MG.get('BestGenome/op_list'))

        if self.ParamDict['FitScheme'] == 'PCAH':
            vMAX = 2
            vMIN = -2
        else:
            vMAX = self.max_colour_val
            vMIN = self.min_colour_val

        if self.class_cp == 1:
            plt_title = 'Class'
            op_graph_vMAX = 0.5
            op_graph_vMIN = -0.5
        elif self.class_cp == 0:
            plt_title = 'Overall Responce (Y)'
            op_graph_vMAX = vMAX
            op_graph_vMIN = vMIN

        location = "%s/data.hdf5" % (self.curr_dir)
        with h5py.File(location, 'r') as de_hdf:
            loc = "/%d_rep%d/DE_data/best_Vout" % (cir, rep)
            best_Vout = np.array(de_hdf.get(loc))

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


        # # set up figure
        fig, ax = plt.subplots(nrows=1, ncols=int(len(op_list)+1),  sharey='row')
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        # # rin loop to plot
        k = 0
        for col, op_raw in enumerate(op_list):

            op = np.asarray(op_raw)

            im = ax[col].imshow(op.T, origin="lower", extent=the_extent,
                                   interpolation=self.interp, vmin=op_graph_vMIN,
                                   vmax=op_graph_vMAX, cmap=self.my_cmap)


            subplot_title = '$V_{o%d}$' % (col+1)
            ax[col].set_title(subplot_title, fontsize=9, verticalalignment='top')

            ax[col].xaxis.set_tick_params(labelsize=8)
            #ax[row, col].xaxis.set_ticks([-5,0,5])
            ax[col].yaxis.set_tick_params(labelsize=8)
            #ax[row, col].yaxis.set_ticks([-5,0,5])

            ax[col].set_xlabel('$a_1$', fontsize=10)

            if col == 0:
                ax[col].set_ylabel('$a_2$', fontsize=10)

            # iterate to next matrix bound in the list
            k = k + 1

        # plot colour and title
        #cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
        #fig.colorbar(im, cax=cb_ax)

        fig.suptitle('Individual Outputs (weighted)\nEntropy (from the selected data)=%.3f/%.3f' % (entropy, max_entropy), fontsize=12)

        imY = ax[len(op_list)].imshow(responceY.T, origin="lower", extent=the_extent,
                                 interpolation=self.interp, vmin=vMIN,
                                 vmax=vMAX, cmap=self.my_cmap)
        ax[len(op_list)].set_title(plt_title, fontsize=9, verticalalignment='top')
        ax[len(op_list)].set_xlabel('$a_1$', fontsize=10)
        ax[len(op_list)].xaxis.set_tick_params(labelsize=8)
        ax[len(op_list)].yaxis.set_tick_params(labelsize=8)



        if self.class_cp == 1:
            cax = fig.add_axes([0.25, 0.2, 0.2, 0.03])
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            cbar.set_label("Vout Value", fontsize=10)
            cbar.ax.tick_params(labelsize=8)  # cbar ticks size

            caxY = fig.add_axes([0.6, 0.2, 0.2, 0.03])
            cbar2 = fig.colorbar(imY, cax=caxY, orientation='horizontal')
            cbar2.set_label(plt_title, fontsize=10)
            cbar2.ax.tick_params(labelsize=8)  # cbar ticks size
        elif self.class_cp == 0:
            cax = fig.add_axes([0.25, 0.15, 0.5, 0.05])
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            #cbar = plt.colorbar(format='% 2.2f', cax=ax[len(op_list)])
            cbar.set_label(plt_title, fontsize=10)
            cbar.ax.tick_params(labelsize=8)  # cbar ticks size


        # # show & save plots
        if self.Save_NotShow == 1:
            if self.MetaData['SaveDir'] == self.save_dir:
                fig_path = "%s/%d_rep%d_Op.%s" % (self.save_dir, cir, rep, self.format)
            else:
                fig_path = "%s/File%d__cir%d_rep%d_Op.%s" % (self.save_dir, self.dir_loop, cir, rep, self.format)
            fig.savefig(fig_path, dpi=self.dpi)
            plt.close(fig)

    #

    #

    #

    # # Function which loads and plots defualt genome material graph
    def plt_VaryConfig(self, hdf, MG, MG_items, VC, cir, rep):

        if VC == 0 or 'VaryConfig' not in MG_items:
            if cir == 0 and rep == 0 and VC !=0:
                print("VaryConfig material graphs not saved")
            return

        responceY_list = np.array(MG.get('VaryConfig/responceY_list'))
        the_extent = np.array(MG.get('VaryConfig/extent'))
        Vconfig_1 = np.array(MG.get('VaryConfig/Vconfig_1'))
        Vconfig_2 = np.array(MG.get('VaryConfig/Vconfig_2'))
        Perm_list = np.array(MG.get('VaryConfig/Perm_list'))
        x1_VC = np.array(MG.get('VaryConfig/x1_data'))
        x2_VC = np.array(MG.get('VaryConfig/x2_data'))
        pop_array = np.array(MG.get('VaryConfig/pop_array'))

        # # set up figure
        #fig = plt.figure(3)
        if self.NetworkDict['num_config'] == 1:
            cols, rows = len(Vconfig_1), len(Vconfig_2)
            fig, ax = plt.subplots(ncols=cols, sharey='row')
        elif self.NetworkDict['num_config'] == 2:
            rows, cols = len(Vconfig_1), len(Vconfig_2)
            fig, ax = plt.subplots(rows, cols,  #  figsize=(6,5)
                                   sharex='col',
                                   sharey='row')
            #fig.subplots_adjust(hspace=0.2, wspace=0.05)
            fig.subplots_adjust(left=0.125, bottom=0.125, right=0.85, top=0.85, wspace=0.05, hspace=0.2)
        else:
            print("")
            print("ERROR (MG): MG_VaryConfig() can only compute up to 2 inputs")

        # # run loop to plot
        k = 0
        for col in range(cols):
            for row in range(rows):
                responce_Y = responceY_list[k]



                if self.NetworkDict['num_config'] == 1:  # for 1 Vconfig input

                    im = ax[col].imshow(responce_Y.T, origin="lower", extent=the_extent,
                                             interpolation=self.interp, vmin=self.min_colour_val,
                                             vmax=self.max_colour_val, cmap=self.my_cmap)


                    if row == 0:
                        # ax[row].text( (x1_VC.max()-abs(x1_VC.min()))/2.1, (x2_VC.max() + (x2_VC.max()-abs(x2_VC.min()))/6 ) , str(Vconfig_2[col]), size=12)
                        ax[col].set_title( str(Vconfig_1[col]), size=12)

                    ax[col].set_xlabel('$a_1$', fontsize=9, labelpad=1)

                    if col == 0:
                        ax[col].set_ylabel('$a_2$', fontsize=9, labelpad=-5)

                elif self.NetworkDict['num_config'] == 2:  # for 2 Vconfig input
                    im = ax[row, col].imshow(responce_Y.T, origin="lower", extent=the_extent,
                                             interpolation=self.interp, vmin=self.min_colour_val,
                                             vmax=self.max_colour_val, cmap=self.my_cmap)

                    # remove tick_params
                    plt.setp(ax[row, col].get_xticklabels(), fontsize=8)
                    plt.setp(ax[row, col].get_yticklabels(), fontsize=8)

                    #plt.setp(ax[row, col].get_xticklabels(), visible=False)
                    #plt.setp(ax[row, col].get_yticklabels(), visible=False)
                    #ax[row, col].tick_params(axis='both', which='both', length=0)

                    if row == 4:
                        # ax[row, col].text( (x1_VC.max()-abs(x1_VC.min()))/2.1, (x2_VC.max() + (x2_VC.max()-abs(x2_VC.min()))/6 ) , str(Vconfig_2[col]), size=12)
                        #ax[row, col].set_title( str(Vconfig_2[col])+'V', size=12)
                        #ax[row, col].set_xlabel(str(Vconfig_1[col])+'V', fontsize=11)
                        ax[row, col].set_xlabel('$a_1$', fontsize=9, labelpad=1)

                    if row == 0:
                        if Vconfig_2[col] < 0:
                            ax[row, col].text( (x1_VC.max()-abs(x1_VC.min()))-3.5 , x1_VC.max()+1.5 ,'$\\bf{%s %s}$' %( str(Vconfig_1[col]), 'V'), size=9)
                        else:
                            ax[row, col].text( (x1_VC.max()-abs(x1_VC.min()))-1.5 , x1_VC.max()+1.5 ,'$\\bf{%s %s}$' %( str(Vconfig_1[col]), 'V'), size=9)
                    if col == 4:
                        ax[row, col].text( x2_VC.max()+1.25 , (x2_VC.max()-abs(x2_VC.min()))-1.5 , '$\\bf{%s %s}$' % (str(Vconfig_2[row]), 'V'), size=9)

                    #ax[row, col].text( 0.25 , 0.25 , '%s' % pop_array[k] , size=9)

                    # if row == rows-1:
                        # ax[row, col].set_xlabel('$a_1$')

                    if col == 0:
                        #ax[row, col].text(-((x1_VC.max()+abs(x1_VC.min()))+0.2), (x2_VC.max()-abs(x2_VC.min()))/2.25, str(Vconfig_1[row])+'V', size=12)
                        #ax[row, col].set_ylabel(str(Vconfig_1[4-row])+'V', fontsize=11)
                        ax[row, col].set_ylabel('$a_2$', fontsize=9, labelpad=-5)


                    # if col == 0:
                        # ax[row, col].set_ylabel('$a_2$')

                # iterate to next matrix bound in the list
                k = k + 1

        # plot colour and title
        #cb_ax = fig.add_axes([0.89, 0.1, 0.02, 0.8])
        #fig.colorbar(im, cax=cb_ax)
        if self.title == 'on':
            fig.suptitle('Varying Vconfig on unconfigured material')

        #fig.tight_layout()

        #ax[4,0].set_ylabel('This is a long label shared among more axes', fontsize=14)
        # # bottom
        #fig.text(0.45, 0.03, '$V_{config_1}$', fontsize=18)
        #fig.text(0.04, 0.45, '$V_{config_2}$', fontsize=18, rotation=90)

        # # top
        if self.NetworkDict['num_config'] == 1:
            fig.text(0.49, 0.7, '$\\bf{V_{c1}}$', fontsize=12)
        else:
            fig.text(0.49, 0.93, '$\\bf{V_{c1}}$', fontsize=12)
            fig.text(0.915, 0.47, '$\\bf{V_{c2}}$', fontsize=12, rotation=0)

        # # show & save plots
        if self.Save_NotShow == 1:
            if self.MetaData['SaveDir'] == self.save_dir:
                fig_path = "%s/%d_rep%d_VaryConfig.%s" % (self.save_dir, cir, rep, self.format)
            else:
                fig_path = "%s/File%d__cir%d_rep%d_VaryConfig.%s" % (self.save_dir, self.dir_loop, cir, rep, self.format)
            fig.savefig(fig_path, dpi=self.dpi)
            plt.close(fig)

    #

    #

    #

    # # Function which loads and plots defualt genome material graph
    def plt_VaryConfigOp(self, hdf, MG, MG_items, VC, cir, rep):

        if VC == 0 or 'VaryConfig' not in MG_items:
            if cir == 0 and rep == 0 and VC !=0:
                print("VaryConfig material graphs not saved")
            return

        responceY_list = np.array(MG.get('VaryConfig/responceY_list'))
        the_extent = np.array(MG.get('VaryConfig/extent'))
        Vconfig_1 = np.array(MG.get('VaryConfig/Vconfig_1'))
        Vconfig_2 = np.array(MG.get('VaryConfig/Vconfig_2'))
        Perm_list = np.array(MG.get('VaryConfig/Perm_list'))
        x1_VC = np.array(MG.get('VaryConfig/x1_data'))
        x2_VC = np.array(MG.get('VaryConfig/x2_data'))
        op_list_Glist = np.array(MG.get('VaryConfig/op_list_Glist'))
        pop_array = np.array(MG.get('VaryConfig/pop_array'))

        if self.ParamDict['FitScheme'] == 'PCAH':
            vMAX = 2
            vMIN = -2
        else:
            vMAX = self.max_colour_val
            vMIN = self.min_colour_val

        if self.ParamDict['IntpScheme'] == 'HOW' or self.ParamDict['IntpScheme'] == 'band':
            vMAX = 2
            vMIN = -2

        init_op_list = op_list_Glist[0]
        init_responceY = responceY_list[0]

        # # set up figure
        fig, ax = plt.subplots(nrows=1, ncols=int(len(init_op_list)+1),  sharey='row')
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        # # rin loop to plot
        image_list = []
        k = 0
        for col, op_raw in enumerate(init_op_list):

            op = np.asarray(op_raw)

            im = ax[col].imshow(op.T, origin="lower", extent=the_extent,
                                   interpolation=self.interp, vmin=vMIN,
                                   vmax=vMAX, cmap=self.my_cmap)
            image_list.append(im)

            subplot_title = '$V_{o%d}$' % (col+1)
            ax[col].set_title(subplot_title, fontsize=9, verticalalignment='top')

            ax[col].xaxis.set_tick_params(labelsize=8)
            #ax[row, col].xaxis.set_ticks([-5,0,5])
            ax[col].yaxis.set_tick_params(labelsize=8)
            #ax[row, col].yaxis.set_ticks([-5,0,5])

            ax[col].set_xlabel('$a_1$', fontsize=10)

            if col == 0:
                ax[col].set_ylabel('$a_2$', fontsize=10)

            # iterate to next matrix bound in the list
            k = k + 1

        # Add overall responce
        title_text = fig.suptitle('Individual Outputs (weighted) for varying Vconfig', fontsize=12)

        imY = ax[len(init_op_list)].imshow(init_responceY.T, origin="lower", extent=the_extent,
                                 interpolation=self.interp, vmin=self.min_colour_val,
                                 vmax=self.max_colour_val, cmap=self.my_cmap)
        image_list.append(imY)
        ax[len(init_op_list)].set_title('Overall Responce (Y)', fontsize=9, verticalalignment='top')
        ax[len(init_op_list)].set_xlabel('$a_1$', fontsize=10)
        ax[len(init_op_list)].xaxis.set_tick_params(labelsize=8)
        ax[len(init_op_list)].yaxis.set_tick_params(labelsize=8)

        cax = fig.add_axes([0.25, 0.15, 0.5, 0.05])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        #cbar = plt.colorbar(format='% 2.2f', cax=ax[len(op_list)])
        cbar.set_label("Vout Value", fontsize=10)
        cbar.ax.tick_params(labelsize=8)  # cbar ticks size

        # # the function which updates the animation
        def op_animate(frame, ax):
            op_list = op_list_Glist[frame]
            responceY = responceY_list[frame]
            title_text.set_text('Individual Outputs for varying configuration Voltages\n Vconfig = %s' % str(pop_array[frame]))
            for col, op in enumerate(op_list):
                #ax[col].set_data(op.T)
                image_list[col].set_array(op.T)
            image_list[-1].set_array(responceY.T)
            return

        # # produce animation
        the_animation = animation.FuncAnimation(
            fig,
            op_animate,
            np.arange(len(pop_array)),
            fargs=[ax],
            interval=20
            )

        # # show & save plots
        if self.Save_NotShow == 1:
            if self.MetaData['SaveDir'] == self.save_dir:
                fig_path = "%s/%d_rep%d_VaryConfigOp.gif" % (self.save_dir, cir, rep)
            else:
                fig_path = "%s/File%d__cir%d_rep%d_VaryConfigOp.gif" % (self.save_dir, self.dir_loop, cir, rep)
            the_animation.save(fig_path, writer='pillow', fps=2, dpi=self.dpi)
            plt.close(fig)
    #

    #

    #

    #

    #

    #

    # # Function which loads and plots defualt genome material graph
    def plt_VaryPerm(self, hdf, MG, MG_items, VP, cir, rep):

        if VP == 0 or 'VaryShuffle' not in MG_items:
            if cir == 0 and rep == 0 and VP !=0:
                print("VaryShuffle material graphs not saved")
            return

        responceY_list = np.array(MG.get('VaryShuffle/responceY_list'))
        the_extent = np.array(MG.get('VaryShuffle/extent'))
        rows = np.array(MG.get('VaryShuffle/rows'))
        cols = np.array(MG.get('VaryShuffle/cols'))
        Perm_list = np.array(MG.get('VaryShuffle/Perm_list'))
        OutWeight_toggle = np.array(MG.get('VaryShuffle/OutWeight'))
        Weight_list = np.array(MG.get('VaryShuffle/Weight_list'))

        # # set up figure
        fig, ax = plt.subplots(int(rows), int(cols), sharex='col', sharey='row', figsize=(6.25,6))
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        # # rin loop to plot
        k = 0
        for row in range(rows):
            for col in range(cols):
                responce_Y = responceY_list[k]

                im = ax[row, col].imshow(responce_Y.T, origin="lower", extent=the_extent,
                                         interpolation=self.interp, vmin=self.min_colour_val,
                                         vmax=self.max_colour_val, cmap=self.my_cmap)

                if self.title == 'on':
                    if OutWeight_toggle == 0:
                        subplot_title = 'Perm %s' % (str(Perm_list[k]))
                    elif OutWeight_toggle == 1:
                        subplot_title = 'Perm %s\nWeight: %s' % (str(Perm_list[k]), str(Weight_list[k]))
                    ax[row, col].set_title(subplot_title, fontsize=5, verticalalignment='top')

                ax[row, col].xaxis.set_tick_params(labelsize=9)
                #ax[row, col].xaxis.set_ticks([-5,0,5])
                ax[row, col].yaxis.set_tick_params(labelsize=9)
                #ax[row, col].yaxis.set_ticks([-5,0,5])

                if row == rows-1:
                    ax[row, col].set_xlabel('$a_1$', fontsize=13)

                if col == 0:
                    ax[row, col].set_ylabel('$a_2$', fontsize=13)

                # iterate to next matrix bound in the list
                k = k + 1

        # plot colour and title
        #cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
        #fig.colorbar(im, cax=cb_ax)
        if self.title == 'on':
            if OutWeight_toggle == 0:
                fig.suptitle('Varying Permutation of inputs of an unconfigured material')
            elif OutWeight_toggle == 1:
                fig.suptitle('Varying Permutation of inputs of an unconfigured material,\nwith random output weights')


        # # show & save plots
        if self.Save_NotShow == 1:
            if self.MetaData['SaveDir'] == self.save_dir:
                fig_path = "%s/%d_rep%d_VaryShuffle.%s" % (self.save_dir, cir, rep, self.format)
            else:
                fig_path = "%s/File%d__cir%d_rep%d_VaryShuffle.%s" % (self.save_dir, self.dir_loop, cir, rep, self.format)
            fig.savefig(fig_path, dpi=self.dpi)
            plt.close(fig)

    #

    #

    #

    # # Function which loads and plots defualt genome material graph with different output weights
    def plt_VaryOutWeight(self, hdf, MG, MG_items, VoW, cir, rep):

        if VoW == 0 or 'VaryOutWeight' not in MG_items:
            if cir == 0 and rep == 0 and VoW !=0:
                #print('MG_items:', MG_items)
                print("VaryOutWeight material graphs not saved")
            return

        responceY_list = np.array(MG.get('VaryOutWeight/responceY_list'))
        the_extent = np.array(MG.get('VaryOutWeight/extent'))
        rows = np.array(MG.get('VaryOutWeight/rows'))
        cols = np.array(MG.get('VaryOutWeight/cols'))
        Weight_list = np.array(MG.get('VaryOutWeight/Weight_list'))

        # # set up figure
        fig, ax = plt.subplots(int(rows), int(cols), sharex='col', sharey='row')


        cb_ax = fig.add_axes([0.86, 0.125, 0.02, 0.8])
        fig.subplots_adjust(left=0.125, bottom=0.15, right=0.87, top=0.9, wspace=0.001, hspace=0.4)


        # # rin loop to plot
        k = 0
        for row in range(rows):
            for col in range(cols):
                responce_Y = responceY_list[k]

                im = ax[row, col].imshow(responce_Y.T, origin="lower", extent=the_extent,
                                         interpolation=self.interp, vmin=self.min_colour_val,
                                         vmax=self.max_colour_val, cmap=self.my_cmap)

                subplot_title = '$W^{out}$ %s' % (str(Weight_list[k]))
                ax[row, col].set_title(subplot_title, fontsize=5, verticalalignment='top')

                ax[row, col].xaxis.set_tick_params(labelsize=7)
                ax[row, col].yaxis.set_tick_params(labelsize=7)

                if row == rows-1:
                    ax[row, col].set_xlabel('$a_1$', fontsize=8)

                if col == 0:
                    ax[row, col].set_ylabel('$a_2$', fontsize=8)

                # iterate to next matrix bound in the list
                k = k + 1

        # plot colour and title
        fig.colorbar(im, cax=cb_ax)
        if self.title == 'on':
            fig.suptitle('Varying output weights for defualt material and no shuffle inputs')


        # # show & save plots
        if self.Save_NotShow == 1:
            if self.MetaData['SaveDir'] == self.save_dir:
                fig_path = "%s/%d_rep%d_VaryOutWeight.%s" % (self.save_dir, cir, rep, self.format)
            else:
                fig_path = "%s/File%d__cir%d_rep%d_VaryOutWeight.%s" % (self.save_dir, self.dir_loop, cir, rep, self.format)
            fig.savefig(fig_path, dpi=self.dpi)
            plt.close(fig)

    #

    #

    #

    # # Function which loads and plots defualt genome material graph with different output weights
    def plt_VaryInWeight(self, hdf, MG, MG_items, ViW, cir, rep):

        if ViW == 0 or 'VaryInWeight' not in MG_items:
            if cir == 0 and rep == 0 and ViW != 0:
                #print('MG_items:', MG_items)
                print("VaryInWeight material graphs not saved")
            return

        responceY_list = np.array(MG.get('VaryInWeight/responceY_list'))
        the_extent = np.array(MG.get('VaryInWeight/extent'))
        rows = np.array(MG.get('VaryInWeight/rows'))
        cols = np.array(MG.get('VaryInWeight/cols'))
        Weight_list = np.array(MG.get('VaryInWeight/Weight_list'))

        # # set up figure
        fig, ax = plt.subplots(int(rows), int(cols), sharex='col', sharey='row')


        cb_ax = fig.add_axes([0.86, 0.125, 0.02, 0.8])
        fig.subplots_adjust(left=0.125, bottom=0.15, right=0.87, top=0.9, wspace=0.001, hspace=0.4)
        #fig.tight_layout()

        # # rin loop to plot
        k = 0
        for row in range(rows):
            for col in range(cols):
                responce_Y = responceY_list[k]

                im = ax[row, col].imshow(responce_Y.T, origin="lower", extent=the_extent,
                                         interpolation=self.interp, vmin=self.min_colour_val,
                                         vmax=self.max_colour_val, cmap=self.my_cmap)

                subplot_title = '$W^{in}$ %s' % (str(Weight_list[k]))
                ax[row, col].set_title(subplot_title, fontsize=5, verticalalignment='top')

                ax[row, col].xaxis.set_tick_params(labelsize=7)
                ax[row, col].yaxis.set_tick_params(labelsize=7)


                if row == rows-1:
                    ax[row, col].set_xlabel('$a_1$', fontsize=8)

                if col == 0:
                    ax[row, col].set_ylabel('$a_2$', fontsize=8)

                # iterate to next matrix bound in the list
                k = k + 1

        # plot colour and title
        fig.colorbar(im, cax=cb_ax)
        if self.title == 'on':
            fig.suptitle('Varying input weights for defualt material and no shuffle inputs')
            fig.subplots_adjust(left=0.12, bottom=0.12, right=0.8, top=0.8, wspace=0.05, hspace=0.2)

        # # show & save plots
        if self.Save_NotShow == 1:
            if self.MetaData['SaveDir'] == self.save_dir:
                fig_path = "%s/%d_rep%d_VaryInWeight.%s" % (self.save_dir, cir, rep, self.format)
            else:
                fig_path = "%s/File%d__cir%d_rep%d_VaryInWeight.%s" % (self.save_dir, self.dir_loop, cir, rep, self.format)
            fig.savefig(fig_path, dpi=self.dpi)
            plt.close(fig)

    #

    #

    #

    # # Function which loads and plots defualt genome material graph with different output weights
    def plt_VaryInWeight_ani(self, hdf, MG, MG_items, ViW_ani, cir, rep):
        if ViW_ani == 0 or 'VaryInWeightAni' not in MG_items:
            if cir == 0 and rep == 0 and ViW_ani != 0 and self.dir_loop == 0:
                #print('MG_items:', MG_items)
                print("VaryInWeight material graphs not saved")
            return

        responceY_list = np.array(MG.get('VaryInWeightAni/responceY_list'))
        the_extent = np.array(MG.get('VaryInWeightAni/extent'))
        Weight_list = np.array(MG.get('VaryInWeightAni/Weight_list'))

        # Now we can do the plotting!
        fig, ax = plt.subplots(1)
        total_number_of_frames = len(responceY_list)

        # Remove a bunch of stuff to make sure we only 'see' the actual imshow
        # Stretch to fit the whole plane
        #fig.subplots_adjust(0, 0, 1, 1)

        # Remove bounding line
        #ax.axis("off")


        # Initialise our plot. Make sure you set vmin and vmax!
        image = ax.imshow(responceY_list[0].T, origin="lower",
                          extent=the_extent,
                          interpolation=self.interp,
                          vmin=self.min_colour_val, vmax=self.max_colour_val,
                          cmap=self.my_cmap)
        plt.colorbar(image)

        #ax.plot([0,1,2],[2,4,6])
        #ax.set_title("Animation (File %d, cir=%d, rep=%d) - Iteration %d" % (self.dir_loop, cir, rep, 0))
        ax.set_title("Animation - File %d, cir=%d, rep=%d \n Input Weight: %s" % (self.dir_loop, cir, rep, str(Weight_list[0])))

        ax.set_ylabel("$a_2$")
        ax.set_xlabel("$a_1$")

        # # the function which updates the animation
        def animate(frame):
            #Animation function. Takes the current frame number (to select the potion of
            #data to plot) and a line selfect to update.

            # Not strictly neccessary, just so we know we are stealing these from
            # the global scope
            #global responceY_list, image, ax

            # We want up-to and _including_ the frame'th element
            image.set_array(responceY_list[frame].T)  # transposing the data so it is plotted correctly
            ax.set_title("Animation - File %d, cir=%d, rep=%d \n Input Weight: %s" % (self.dir_loop, cir, rep, str(Weight_list[frame])))

            return image

        # # produce animation
        the_animation = animation.FuncAnimation(
            # Your Matplotlib Figure selfect
            fig,
            # The function that does the updating of the Figure
            animate,
            # Frame information (here just frame number)
            np.arange(total_number_of_frames),
            # Extra arguments to the animate function
            fargs=[],
            # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
            interval=1000/self.fps
            )

        # # show & save plots
        if self.Save_NotShow == 1:
            if self.ani_format == 'mp4':
                fig_path = "%s/File%d_VaryInWeightAni_Cir%d_Rep%d.mp4" % (self.save_dir, self.dir_loop, cir, rep)
                the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
            elif self.ani_format == 'gif':
                fig_path = "%s/File%d_VaryInWeightAni_Cir%d_Rep%d.gif" % (self.save_dir, self.dir_loop, cir, rep)
                the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            plt.close(fig)

        return the_animation

    #

    #

    #

    # # Function which loads and plots defualt genome material graph with different output weights
    def plt_VaryOutWeight_ani(self, hdf, MG, MG_items, VoW_ani, cir, rep):
        if VoW_ani == 0 or 'VaryOutWeightAni' not in MG_items:
            if cir == 0 and rep == 0 and VoW_ani != 0 and self.dir_loop == 0:
                #print('MG_items:', MG_items)
                print("VaryOutWeight animation material graphs not saved")
            return

        responceY_list = np.array(MG.get('VaryOutWeightAni/responceY_list'))
        the_extent = np.array(MG.get('VaryOutWeightAni/extent'))
        Weight_list = np.array(MG.get('VaryOutWeightAni/Weight_list'))

        # Now we can do the plotting!
        fig, ax = plt.subplots(1)
        total_number_of_frames = len(responceY_list)
        # Remove a bunch of stuff to make sure we only 'see' the actual imshow
        # Stretch to fit the whole plane
        #fig.subplots_adjust(0, 0, 1, 1)

        # Remove bounding line
        #ax.axis("off")


        # Initialise our plot. Make sure you set vmin and vmax!
        image = ax.imshow(responceY_list[0].T, origin="lower",
                          extent=the_extent,
                          interpolation=self.interp,
                          vmin=self.min_colour_val, vmax=self.max_colour_val,
                          cmap=self.my_cmap)
        plt.colorbar(image)

        #ax.plot([0,1,2],[2,4,6])
        #ax.set_title("Animation (File %d, cir=%d, rep=%d) - Iteration %d" % (self.dir_loop, cir, rep, 0))
        ax.set_title("Animation - File %d, cir=%d, rep=%d \n Output Weight: %s" % (self.dir_loop, cir, rep, str(Weight_list[0])))

        ax.set_ylabel("$a_2$")
        ax.set_xlabel("$a_1$")

        # # the function which updates the animation
        def animate(frame):
            #Animation function. Takes the current frame number (to select the potion of
            #data to plot) and a line selfect to update.

            # Not strictly neccessary, just so we know we are stealing these from
            # the global scope
            #global responceY_list, image, ax

            # We want up-to and _including_ the frame'th element
            image.set_array(responceY_list[frame].T)  # transposing the data so it is plotted correctly
            ax.set_title("Animation - File %d, cir=%d, rep=%d \n Output Weight: %s" % (self.dir_loop, cir, rep, str(Weight_list[frame])))

            return image

        # # produce animation
        the_animation = animation.FuncAnimation(
            # Your Matplotlib Figure selfect
            fig,
            # The function that does the updating of the Figure
            animate,
            # Frame information (here just frame number)
            np.arange(total_number_of_frames),
            # Extra arguments to the animate function
            fargs=[],
            # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
            interval=1000/self.fps
            )

        # # show & save plots
        if self.Save_NotShow == 1:
            if self.ani_format == 'mp4':
                fig_path = "%s/File%d_VaryOutWeightAni_Cir%d_Rep%d.mp4" % (self.save_dir, self.dir_loop, cir, rep)
                the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
            elif self.ani_format == 'gif':
                fig_path = "%s/File%d_VaryOutWeightAni_Cir%d_Rep%d.gif" % (self.save_dir, self.dir_loop, cir, rep)
                the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            plt.close(fig)

        return the_animation



    # # Function which loads and plots defualt genome material graph with different output weights
    def plt_VaryOutInWeight_ani(self, hdf, MG, MG_items, VoiW_ani, cir, rep):
        if VoiW_ani == 0 or 'VaryOutWeightAni' not in MG_items or 'VaryInWeightAni' not in MG_items:
            if cir == 0 and rep == 0 and VoiW_ani != 0 and self.dir_loop == 0:
                #print('MG_items:', MG_items)
                print("Vary OutWeight or InWeight animation material graphs not saved")
            return

        OW_responceY_list = np.array(MG.get('VaryOutWeightAni/responceY_list'))
        OW_the_extent = np.array(MG.get('VaryOutWeightAni/extent'))
        OW_Weight_list = np.array(MG.get('VaryOutWeightAni/Weight_list'))

        IW_responceY_list = np.array(MG.get('VaryInWeightAni/responceY_list'))
        IW_the_extent = np.array(MG.get('VaryInWeightAni/extent'))
        IW_Weight_list = np.array(MG.get('VaryInWeightAni/Weight_list'))

        # # plot fig
        fig, (ax_iw, ax_ow) = plt.subplots(1, 2, figsize=(14, 6))
        if len(OW_responceY_list) != len(IW_responceY_list):
            print("Error: OW Y list and IW Y length not equal")
            return
        total_number_of_frames = len(OW_responceY_list)

        fig.suptitle("Animation - File %d, cir=%d, rep=%d" % (self.dir_loop, cir, rep))

        # Initialise our plot. Make sure you set vmin and vmax!
        image_iw = ax_iw.imshow(IW_responceY_list[0].T, origin="lower",
                             extent=IW_the_extent,
                             interpolation=self.interp,
                             vmin=self.min_colour_val, vmax=self.max_colour_val,
                             cmap=self.my_cmap)
        ax_iw.set_title("Input Weight: %s" % (str(IW_Weight_list[0])))
        ax_iw.set_ylabel("$a_2$")
        ax_iw.set_xlabel("$a_1$")

        image_ow = ax_ow.imshow(OW_responceY_list[0].T, origin="lower",
                             extent=OW_the_extent,
                             interpolation=self.interp,
                             vmin=self.min_colour_val, vmax=self.max_colour_val,
                             cmap=self.my_cmap)
        ax_ow.set_title("Output Weight: %s" % (str(OW_Weight_list[0])))
        ax_ow.set_ylabel("$a_2$")
        ax_ow.set_xlabel("$a_1$")

        # # the function which updates the animation
        def animate(frame):
            #Animation function. Takes the current frame number (to select the potion of
            #data to plot) and a line selfect to update.

            # We want up-to and _including_ the frame'th element
            image_iw.set_array(IW_responceY_list[frame].T)  # transposing the data so it is plotted correctly
            ax_iw.set_title("Input Weight: %s" % (str(IW_Weight_list[frame])))

            image_ow.set_array(OW_responceY_list[frame].T)  # transposing the data so it is plotted correctly
            ax_ow.set_title("Output Weight: %s" % (str(OW_Weight_list[frame])))

            return image_iw, image_ow

        # # produce animation
        the_animation = animation.FuncAnimation(
            # Your Matplotlib Figure selfect
            fig,
            # The function that does the updating of the Figure
            animate,
            # Frame information (here just frame number)
            np.arange(total_number_of_frames),
            # Extra arguments to the animate function
            fargs=[],
            # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
            interval=1000/self.fps
            )

        # # show & save plots
        if self.Save_NotShow == 1:
            if self.ani_format == 'mp4':
                fig_path = "%s/File%d_Vary_OutIn_WeightAni_Cir%d_Rep%d.mp4" % (self.save_dir, self.dir_loop, cir, rep)
                the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
            elif self.ani_format == 'gif':
                fig_path = "%s/File%d_Vary_OutIn_WeightAni_Cir%d_Rep%d.gif" % (self.save_dir, self.dir_loop, cir, rep)
                the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            plt.close(fig)

        return the_animation

    #

    #

    #

    # # Function which loads and plots defualt genome material graph with different output weights
    def plt_VaryLargeOutWeight_ani(self, hdf, MG, MG_items, VloW_ani, cir, rep, Both=1):

        # check to see other data exists before plotting both
        plot_both = 0  # if 1 plot both large OW and normal OW
        test = MG.get('VaryOutWeightAni/responceY_list')
        if Both == 1 and test is not None:
            plot_both = 1

        if VloW_ani == 0 or 'VaryLargeOutWeightAni' not in MG_items:
            if cir == 0 and rep == 0 and VloW_ani !=0:
                #print('MG_items:', MG_items)
                print("VaryLargeOutWeightAni material graphs not saved")
            return

        # collect large varyation
        lOW_responceY_list = np.array(MG.get('VaryLargeOutWeightAni/responceY_list'))
        lOW_the_extent = np.array(MG.get('VaryLargeOutWeightAni/extent'))
        lOW_Weight_list = np.array(MG.get('VaryLargeOutWeightAni/Weight_list'))



        if plot_both == 0 or 'VaryOutWeight' not in MG_items:
            plot_both = 0
        else:
            plot_both = 1
            # collect normal varyation
            nOW_responceY_list = np.array(MG.get('VaryOutWeightAni/responceY_list'))
            nOW_the_extent = np.array(MG.get('VaryOutWeightAni/extent'))
            nOW_Weight_list = np.array(MG.get('VaryOutWeightAni/Weight_list'))

        # # produce a single plot, or side by side sub plot
        if plot_both == 0:
            fig, ax = plt.subplots(1)
            total_number_of_frames = len(lOW_responceY_list)

            # Initialise our plot. Make sure you set vmin and vmax!
            image = ax.imshow(lOW_responceY_list[0].T, origin="lower",
                              extent=lOW_the_extent,
                              interpolation=self.interp,
                              vmin=self.min_colour_val, vmax=self.max_colour_val,
                              cmap=self.my_cmap)
            plt.colorbar(image)


            ax.set_title("Animation - File %d, cir=%d, rep=%d \n Output Weight: %s" % (self.dir_loop, cir, rep, str(lOW_Weight_list[0])))
            ax.set_ylabel("$a_2$")
            ax.set_xlabel("$a_1$")

            # # the function which updates the animation
            def animate(frame):
                image.set_array(lOW_responceY_list[frame].T)  # transposing the data so it is plotted correctly
                ax.set_title("Animation - File %d, cir=%d, rep=%d \n Output Weight: %s" % (self.dir_loop, cir, rep, str(lOW_Weight_list[frame])))
                return image

            # # produce animation
            the_animation = animation.FuncAnimation(
                fig,
                animate,
                np.arange(total_number_of_frames),
                fargs=[],
                interval=1000/self.fps
                )

        elif plot_both == 1:

            fig, (ax_now, ax_low) = plt.subplots(1, 2, figsize=(14, 6))
            if len(lOW_responceY_list) != len(nOW_responceY_list):
                print("Error: OW Y list and LARGE OW Y length not equal")
                print("Large OW length:", len(lOW_responceY_list))
                print("OW length:", len(nOW_responceY_list))
                return
            total_number_of_frames = len(lOW_responceY_list)

            fig.suptitle("Animation - File %d, cir=%d, rep=%d" % (self.dir_loop, cir, rep))

            # Initialise our plot. Make sure you set vmin and vmax!
            image_iw = ax_now.imshow(nOW_responceY_list[0].T, origin="lower",
                                    extent=nOW_the_extent,
                                    interpolation=self.interp,
                                    vmin=self.min_colour_val, vmax=self.max_colour_val,
                                    cmap=self.my_cmap)
            ax_now.set_title("Input Weight: %s" % (str(nOW_Weight_list[0])))
            ax_now.set_ylabel("$a_2$")
            ax_now.set_xlabel("$a_1$")

            image_ow = ax_low.imshow(lOW_responceY_list[0].T, origin="lower",
                                    extent=lOW_the_extent,
                                    interpolation=self.interp,
                                    vmin=self.min_colour_val, vmax=self.max_colour_val,
                                    cmap=self.my_cmap)
            ax_low.set_title("Output Weight: %s" % (str(lOW_Weight_list[0])))
            ax_low.set_ylabel("$a_2$")
            ax_low.set_xlabel("$a_1$")

            # # the function which updates the animation
            def animate(frame):
                image_iw.set_array(nOW_responceY_list[frame].T)  # transposing the data so it is plotted correctly
                ax_now.set_title("Output Weight: %s" % (str(nOW_Weight_list[frame])))

                image_ow.set_array(lOW_responceY_list[frame].T)  # transposing the data so it is plotted correctly
                ax_low.set_title("Larger Output Weight: %s" % (str(lOW_Weight_list[frame])))

                return image_iw, image_ow

            # # produce animation
            the_animation = animation.FuncAnimation(
                fig,
                animate,
                np.arange(total_number_of_frames),
                fargs=[],
                interval=1000/self.fps
                )


        # # show & save plots
        if self.Save_NotShow == 1:
            if self.ani_format == 'mp4':
                fig_path = "%s/File%d_VaryLargeOutWeightAni_Cir%d_Rep%d.mp4" % (self.save_dir, self.dir_loop, cir, rep)
                the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
            elif self.ani_format == 'gif':
                fig_path = "%s/File%d_VaryLargeOutWeightAni_Cir%d_Rep%d.gif" % (self.save_dir, self.dir_loop, cir, rep)
                the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            plt.close(fig)

        return the_animation

    #

    #

    #

    # # Function which loads and plots defualt genome material graph with different output weights
    def plt_VaryLargeInWeight_ani(self, hdf, MG, MG_items, VliW_ani, cir, rep, Both=1):

        # check to see other data exists before plotting both
        plot_both = 0  # if 1 plot both large OW and normal OW
        test =  MG.get('VaryInWeightAni/responceY_list')
        if Both == 1 and test is not None:
            plot_both = 1


        if VliW_ani == 0 or 'VaryLargeInWeightAni' not in MG_items:
            if cir == 0 and rep == 0 and VliW_ani != 0:
                #print('MG_items:', MG_items)
                print("VaryLargeInWeightAni material graphs not saved")
            return

        liW_responceY_list = np.array(MG.get('VaryLargeInWeightAni/responceY_list'))
        lIW_the_extent = np.array(MG.get('VaryLargeInWeightAni/extent'))
        lIW_Weight_list = np.array(MG.get('VaryLargeInWeightAni/Weight_list'))

        if plot_both == 0 or 'VaryInWeightAni' not in MG_items:
            plot_both = 0
            print('MG_items:', MG_items)
        else:
            plot_both = 1
            niW_responceY_list = np.array(MG.get('VaryInWeightAni/responceY_list'))
            nIW_the_extent = np.array(MG.get('VaryInWeightAni/extent'))
            nIW_Weight_list = np.array(MG.get('VaryInWeightAni/Weight_list'))

        # # produce a single plot, or side by side sub plot
        if plot_both == 0:
            fig, ax = plt.subplots(1)
            total_number_of_frames = len(liW_responceY_list)

            # Initialise our plot. Make sure you set vmin and vmax!
            image = ax.imshow(liW_responceY_list[0].T, origin="lower",
                              extent=lIW_the_extent,
                              interpolation=self.interp,
                              vmin=self.min_colour_val, vmax=self.max_colour_val,
                              cmap=self.my_cmap)
            plt.colorbar(image)


            ax.set_title("Animation - File %d, cir=%d, rep=%d \n Input Weight: %s" % (self.dir_loop, cir, rep, str(lIW_Weight_list[0])))
            ax.set_ylabel("$a_2$")
            ax.set_xlabel("$a_1$")

            # # the function which updates the animation
            def animate(frame):
                image.set_array(liW_responceY_list[frame].T)  # transposing the data so it is plotted correctly
                ax.set_title("Animation - File %d, cir=%d, rep=%d \n Input Weight: %s" % (self.dir_loop, cir, rep, str(lIW_Weight_list[frame])))
                return image

            # # produce animation
            the_animation = animation.FuncAnimation(
                fig,
                animate,
                np.arange(total_number_of_frames),
                fargs=[],
                interval=1000/self.fps
                )

        elif plot_both == 1:
            fig, (ax_niw, ax_liw) = plt.subplots(1, 2, figsize=(14, 6))
            if len(liW_responceY_list) != len(niW_responceY_list):
                print("Error: IW Y list and LARGE IW Y length not equal")
                print("Large IW length:", len(liW_responceY_list))
                print("IW length:", len(niW_responceY_list))
                return
            total_number_of_frames = len(liW_responceY_list)

            fig.suptitle("Animation - File %d, cir=%d, rep=%d" % (self.dir_loop, cir, rep))

            # Initialise our plot. Make sure you set vmin and vmax!
            image_niw = ax_niw.imshow(niW_responceY_list[0].T, origin="lower",
                                    extent=nIW_the_extent,
                                    interpolation=self.interp,
                                    vmin=self.min_colour_val, vmax=self.max_colour_val,
                                    cmap=self.my_cmap)
            ax_niw.set_title("Input Weight: %s" % (str(nIW_Weight_list[0])))
            ax_niw.set_ylabel("$a_2$")
            ax_niw.set_xlabel("$a_1$")

            image_liw = ax_liw.imshow(liW_responceY_list[0].T, origin="lower",
                                    extent=lIW_the_extent,
                                    interpolation=self.interp,
                                    vmin=self.min_colour_val, vmax=self.max_colour_val,
                                    cmap=self.my_cmap)
            ax_liw.set_title("Input Weight: %s" % (str(lIW_Weight_list[0])))
            ax_liw.set_ylabel("$a_2$")
            ax_liw.set_xlabel("$a_1$")

            # # the function which updates the animation
            def animate(frame):
                image_niw.set_array(niW_responceY_list[frame].T)  # transposing the data so it is plotted correctly
                ax_niw.set_title("Input Weight: %s" % (str(nIW_Weight_list[frame])))

                image_liw.set_array(liW_responceY_list[frame].T)  # transposing the data so it is plotted correctly
                ax_liw.set_title("Larger Input Weights: %s" % (str(lIW_Weight_list[frame])))

                return image_niw, image_liw

            # # produce animation
            the_animation = animation.FuncAnimation(
                fig,
                animate,
                np.arange(total_number_of_frames),
                fargs=[],
                interval=1000/self.fps
                )


        # # show & save plots
        if self.Save_NotShow == 1:
            if self.ani_format == 'mp4':
                fig_path = "%s/File%d_VaryLargeInWeightAni_Cir%d_Rep%d.mp4" % (self.save_dir, self.dir_loop, cir, rep)
                the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
            elif self.ani_format == 'gif':
                fig_path = "%s/File%d_VaryLargeInWeightAni_Cir%d_Rep%d.gif" % (self.save_dir, self.dir_loop, cir, rep)
                the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            plt.close(fig)

        return the_animation



#

#

#

#

# fin
