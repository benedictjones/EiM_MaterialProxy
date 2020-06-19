# # Top Matter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import h5py

import matplotlib.animation as animation

from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap

from Module_Analysis.Set_Load_meta import LoadMetaData

from Module_LoadData.LoadData import Load_Data


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
class material_graphs(object):

    def Plt_mg(self, Save_NotShow=0, show=1, Bgeno=0, Dgeno=0, VC=0, VP=0,
               VoW=0, ViW=0, VoW_ani=0, ViW_ani=0, VoiW_ani=1, VloW_ani=1, VliW_ani=1,
               Specific_Cir_Rep='all', PlotOnly='all', format='gif',
               titles='on'):

        if format != 'gif' and format != 'mp4':
            print("Error (animation): only mp4 and gif formats allowed")
        else:
            self.ani_format = format


        matplotlib .rcParams['font.family'] = 'Arial'
        matplotlib .rcParams['font.size'] = 12  # tixks and title
        matplotlib .rcParams['figure.titlesize'] = 'medium'
        matplotlib .rcParams['axes.labelsize'] = 15  # axis labels
        matplotlib .rcParams['axes.linewidth'] = 1
        #matplotlib .rcParams['mathtext.fontset'] = 'cm'
        matplotlib .rcParams["legend.labelspacing"] = 0.25
        matplotlib.rc('pdf', fonttype=42)  # embeds the font, so can import to inkscape

        self.title = titles

        self.Save_NotShow = Save_NotShow
        #self.interp = 'gaussian'
        self.interp = 'none'
        #basic_cols = ['#009cff', '#ffffff','#ff8800']  # pastal orange/white/blue
        basic_cols = ['#009cff', '#6d55ff', '#ffffff', '#ff6d55','#ff8800']  # pastal orange/red/white/purle/blue
        val = 0.2  # 0.2
        self.max_colour_val = val
        self.min_colour_val = -val
        # basic_cols=['#0000ff', '#000000', '#ff0000']
        self.my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)
        self.my_cmap.set_bad(color='#ffff00')

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
            cir_range = range(self.MetaData['exp_num_circuits'])
            rep_range = range(self.MetaData['exp_num_repetitions'])

            if len(Specific_Cir_Rep) != 2:
                if Specific_Cir_Rep != 'all':
                    print("Error: Specific_Cir_Rep must be 'all' or be a specific [cir, rep]")
                    return
            else:
                if Specific_Cir_Rep[0] >= self.MetaData['exp_num_circuits']:
                    print("Error: Invalid Circuit loop selected to plot")
                    return
                elif Specific_Cir_Rep[1] >= self.MetaData['exp_num_repetitions']:
                    print("Error: Invalid Repetition loop selected to plot")
                    return
                else:
                    cir_range = [Specific_Cir_Rep[0]]
                    rep_range = [Specific_Cir_Rep[1]]

            for cir in cir_range:

                for rep in rep_range:

                    location = "%s/data_%d_rep%d.hdf5" % (curr_dir, cir, rep)
                    with h5py.File(location, 'r') as hdf:

                        MG = hdf.get('/MG')

                        try:
                            MG_items = list(MG.keys())
                        except:
                            break

                        self.plt_best(hdf, MG, MG_items, Bgeno, cir, rep)

                        self.plt_defualt(hdf, MG, MG_items, Dgeno, cir, rep)

                        self.plt_VaryConfig(hdf, MG, MG_items, VC, cir, rep)

                        self.plt_VaryPerm(hdf, MG, MG_items, VP, cir, rep)

                        self.plt_VaryOutWeight(hdf, MG, MG_items, VoW, cir, rep)
                        the_OW_animation = self.plt_VaryOutWeight_ani(hdf, MG, MG_items, VoW_ani, cir, rep)

                        self.plt_VaryInWeight(hdf, MG, MG_items, ViW, cir, rep)
                        the_IW_animation = self.plt_VaryInWeight_ani(hdf, MG, MG_items, ViW_ani, cir, rep)

                        the_IOW_animation = self.plt_VaryOutInWeight_ani(hdf, MG, MG_items, VoiW_ani, cir, rep)

                        the_lOW_animation = self.plt_VaryLargeOutWeight_ani(hdf, MG, MG_items, VloW_ani, cir, rep, Both=1)

                        the_lIW_animation = self.plt_VaryLargeInWeight_ani(hdf, MG, MG_items, VliW_ani, cir, rep, Both=1)
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

        # fetch data
        responceY = np.array(hdf.get('/MG/BestGenome/responceY'))
        the_extent = np.array(hdf.get('/MG/BestGenome/extent'))
        best_genome = np.array(hdf.get('/MG/BestGenome/the_best_genome'))
        best_fitness_val = np.array(hdf.get('/MG/BestGenome/best_fitness_val'))

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


        # add title and axis
        if self.title == 'on':
            title_pt1 = 'Class outputs for the best genome (file %d, cir %d, rep %d) \n' % (self.dir_loop, cir, rep)
            Best_Genome_str = 'Best Genome %s\nfitness %.3f' % (str(np.around(best_genome, decimals=3)), best_fitness_val)
            plt.suptitle(title_pt1, fontsize=10, fontweight='bold')
            plt.title(Best_Genome_str, fontsize=8)
        plt.xlabel('$a_1$')
        plt.ylabel('$a_2$')


        # plot training data too
        train_data_X, train_data_Y = Load_Data('train', self.MetaData['num_input'], self.MetaData['num_output_readings'],
                                               self.MetaData['training_data'], 0, self.MetaData['UseCustom_NewAttributeData'])
        c = 0
        cl1_hit = 0
        cl2_hit = 0
        for row in train_data_X:

            if train_data_Y[c] == 1:
                if cl1_hit == 0:
                    plt.plot(row[0], row[1],  'o', alpha=0.8, markersize=4.5, label="Class 1",
                             markerfacecolor='#009cffff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
                    cl1_hit = 1
                else:
                    plt.plot(row[0], row[1],  'o', alpha=0.8, markersize=4.5,
                             markerfacecolor='#009cffff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
            else:
                if cl2_hit == 0:
                    plt.plot(row[0], row[1],  '*', alpha=0.8, markersize=7, label="Class 2",
                             markerfacecolor='#ff8800ff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
                    cl2_hit = 1
                else:
                    plt.plot(row[0], row[1],  '*', alpha=0.8, markersize=7,
                             markerfacecolor='#ff8800ff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
            c = c + 1
        plt.legend(markerscale=1.5, fancybox=True)  # , facecolor='k'


        # # show & save plots
        if self.Save_NotShow == 1:
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

        responceY = np.array(hdf.get('/MG/DefaultGenome/responceY'))
        the_extent = np.array(hdf.get('/MG/DefaultGenome/extent'))
        defualt_genome = np.array(hdf.get('/MG/DefaultGenome/default_genome'))

        fig = plt.figure()
        plt.imshow(responceY.T, origin="lower",
                   extent=the_extent,
                   interpolation=self.interp, vmin=self.min_colour_val, vmax=self.max_colour_val, cmap=self.my_cmap)

        cbar = plt.colorbar(format='% 2.2f')
        cbar.set_label("Network Responce Y", fontsize=14)
        cbar.ax.tick_params(labelsize=11)  # cbar ticks size

        # add title and axis
        if self.title == 'on':
            title_pt1 = 'Class outputs for the unconficured material (file %d, cir %d, rep %d) \n' % (self.dir_loop, cir, rep)
            Defualt_Genome_str = 'Default Genome %s\n' % (str(np.around(defualt_genome, decimals=3)))
            plt.suptitle(title_pt1, fontsize=10, fontweight='bold')
            plt.title(Defualt_Genome_str, fontsize=8)
        plt.xlabel('$a_1$')
        plt.ylabel('$a_2$')

        # # show & save plots
        if self.Save_NotShow == 1:
            fig_path = "%s/File%d__cir%d_rep%d_DefualtGenome.%s" % (self.save_dir, self.dir_loop, cir, rep, self.format)
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

        responceY_list = np.array(hdf.get('/MG/VaryConfig/responceY_list'))
        the_extent = np.array(hdf.get('/MG/VaryConfig/extent'))
        Vconfig_1 = np.array(hdf.get('/MG/VaryConfig/Vconfig_1'))
        Vconfig_2 = np.array(hdf.get('/MG/VaryConfig/Vconfig_2'))
        Perm_list = np.array(hdf.get('/MG/VaryConfig/Perm_list'))
        x_data = np.array(hdf.get('/MG/VaryConfig/x_data'))
        x1_VC = x_data[0]
        x2_VC = x_data[1]


        # # set up figure
        #fig = plt.figure(3)
        if self.MetaData['num_config'] == 1:
            rows, cols = len(Vconfig_1), len(Vconfig_2)
            fig, ax = plt.subplots(rows, sharex='col')
        elif self.MetaData['num_config'] == 2:
            rows, cols = len(Vconfig_1), len(Vconfig_2)
            fig, ax = plt.subplots(rows, cols, figsize=(6,5),
                                   sharex='col',
                                   sharey='row')

            fig.subplots_adjust(hspace=0.2, wspace=0.05)
        else:
            print("")
            print("ERROR (MG): MG_VaryConfig() can only compute up to 2 inputs")

        v_list = [-6,-3,0,3,6]

        # # rin loop to plot
        k = 0
        for row in range(rows):
            for col in range(cols):
                responce_Y = responceY_list[k]



                if self.MetaData['num_config'] == 1:  # for 1 Vconfig input

                    im = ax[row].imshow(responce_Y.T, origin="lower", extent=the_extent,
                                             interpolation=self.interp, vmin=self.min_colour_val,
                                             vmax=self.max_colour_val, cmap=self.my_cmap)
                    if col == 0:
                        ax[row].text(-(x1_VC.max()+abs(x1_VC.min())+0.2), (x2_VC.max()-abs(x2_VC.min()))/2.25, str(Vconfig_1[row]), size=12)

                    if row == 0:
                        # ax[row].text( (x1_VC.max()-abs(x1_VC.min()))/2.1, (x2_VC.max() + (x2_VC.max()-abs(x2_VC.min()))/6 ) , str(Vconfig_2[col]), size=12)
                        ax[row, col].set_title( str(Vconfig_2[col]), size=12)

                elif self.MetaData['num_config'] == 2:  # for 2 Vconfig input
                    im = ax[row, col].imshow(responce_Y.T, origin="lower", extent=the_extent,
                                             interpolation=self.interp, vmin=self.min_colour_val,
                                             vmax=self.max_colour_val, cmap=self.my_cmap)

                    # remove tick_params
                    plt.setp(ax[row, col].get_xticklabels(), visible=False)
                    plt.setp(ax[row, col].get_yticklabels(), visible=False)
                    ax[row, col].tick_params(axis='both', which='both', length=0)

                    if row == 4:
                        # ax[row, col].text( (x1_VC.max()-abs(x1_VC.min()))/2.1, (x2_VC.max() + (x2_VC.max()-abs(x2_VC.min()))/6 ) , str(Vconfig_2[col]), size=12)
                        #ax[row, col].set_title( str(Vconfig_2[col])+'V', size=12)
                        ax[row, col].set_xlabel(str(Vconfig_1[col])+'V', fontsize=11)

                    # if row == rows-1:
                        # ax[row, col].set_xlabel('$a_1$')

                    if col == 0:
                        #ax[row, col].text(-((x1_VC.max()+abs(x1_VC.min()))+0.2), (x2_VC.max()-abs(x2_VC.min()))/2.25, str(Vconfig_1[row])+'V', size=12)
                        ax[row, col].set_ylabel(str(Vconfig_1[4-row])+'V', fontsize=11)



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
        fig.text(0.45, 0.03, '$V_{config_1}$', fontsize=18)
        fig.text(0.04, 0.45, '$V_{config_2}$', fontsize=18, rotation=90)


        # # show & save plots
        if self.Save_NotShow == 1:
            fig_path = "%s/File%d__cir%d_rep%d_VaryConfig.%s" % (self.save_dir, self.dir_loop, cir, rep, self.format)
            fig.savefig(fig_path, dpi=self.dpi)
            plt.close(fig)

    #

    #

    #

    # # Function which loads and plots defualt genome material graph
    def plt_VaryPerm(self, hdf, MG, MG_items, VP, cir, rep):

        if VP == 0 or 'VaryShuffle' not in MG_items:
            if cir == 0 and rep == 0 and VP !=0:
                print("VaryShuffle material graphs not saved")
            return

        responceY_list = np.array(hdf.get('/MG/VaryShuffle/responceY_list'))
        the_extent = np.array(hdf.get('/MG/VaryShuffle/extent'))
        rows = np.array(hdf.get('/MG/VaryShuffle/rows'))
        cols = np.array(hdf.get('/MG/VaryShuffle/cols'))
        Perm_list = np.array(hdf.get('/MG/VaryShuffle/Perm_list'))
        OutWeight_toggle = np.array(hdf.get('/MG/VaryShuffle/OutWeight'))
        Weight_list = np.array(hdf.get('/MG/VaryShuffle/Weight_list'))

        # # set up figure
        fig, ax = plt.subplots(int(rows), int(cols), sharex='col', sharey='row', figsize=(6.25,6))



        fig.subplots_adjust(hspace=0.2, wspace=0.2)

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
                ax[row, col].xaxis.set_ticks([-5,0,5])
                ax[row, col].yaxis.set_tick_params(labelsize=9)
                ax[row, col].yaxis.set_ticks([-5,0,5])

                if row == rows-1:
                    ax[row, col].set_xlabel('$a_1$', fontsize=16)



                if col == 0:
                    ax[row, col].set_ylabel('$a_2$', fontsize=16)

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

        responceY_list = np.array(hdf.get('/MG/VaryOutWeight/responceY_list'))
        the_extent = np.array(hdf.get('/MG/VaryOutWeight/extent'))
        rows = np.array(hdf.get('/MG/VaryOutWeight/rows'))
        cols = np.array(hdf.get('/MG/VaryOutWeight/cols'))
        Weight_list = np.array(hdf.get('/MG/VaryOutWeight/Weight_list'))

        # # set up figure
        fig, ax = plt.subplots(int(rows), int(cols), sharex='col', sharey='row')


        cb_ax = fig.add_axes([0.89, 0.1, 0.02, 0.8])
        fig.subplots_adjust(hspace=0.2, wspace=0.05)

        # # rin loop to plot
        k = 0
        for row in range(rows):
            for col in range(cols):
                responce_Y = responceY_list[k]

                im = ax[row, col].imshow(responce_Y.T, origin="lower", extent=the_extent,
                                         interpolation=self.interp, vmin=self.min_colour_val,
                                         vmax=self.max_colour_val, cmap=self.my_cmap)

                subplot_title = 'Weight: %s' % (str(Weight_list[k]))
                ax[row, col].set_title(subplot_title, fontsize=5, verticalalignment='top')

                if row == rows-1:
                    ax[row, col].set_xlabel('$a_1$')

                if col == 0:
                    ax[row, col].set_ylabel('$a_2$')

                # iterate to next matrix bound in the list
                k = k + 1

        # plot colour and title
        fig.colorbar(im, cax=cb_ax)
        fig.suptitle('Varying output weights for defualt material and no shuffle inputs')

        # # show & save plots
        if self.Save_NotShow == 1:
            fig_path = "%s/File%d__cir%d_rep%d_VaryOutWeight.%s" % (self.save_dir, self.dir_loop, cir, rep, self.format)
            fig.savefig(fig_path, dpi=self.dpi)
            plt.close(fig)

    #

    #

    #

    # # Function which loads and plots defualt genome material graph with different output weights
    def plt_VaryInWeight(self, hdf, MG, MG_items, ViW, cir, rep):

        if ViW == 0 or 'VaryInWeight' not in MG_items:
            if cir == 0 and rep == 0 and ViW !=0:
                #print('MG_items:', MG_items)
                print("VaryInWeight material graphs not saved")
            return

        responceY_list = np.array(hdf.get('/MG/VaryInWeight/responceY_list'))
        the_extent = np.array(hdf.get('/MG/VaryInWeight/extent'))
        rows = np.array(hdf.get('/MG/VaryInWeight/rows'))
        cols = np.array(hdf.get('/MG/VaryInWeight/cols'))
        Weight_list = np.array(hdf.get('/MG/VaryInWeight/Weight_list'))

        # # set up figure
        fig, ax = plt.subplots(int(rows), int(cols), sharex='col', sharey='row')


        cb_ax = fig.add_axes([0.89, 0.1, 0.02, 0.8])
        fig.subplots_adjust(hspace=0.2, wspace=0.05)

        # # rin loop to plot
        k = 0
        for row in range(rows):
            for col in range(cols):
                responce_Y = responceY_list[k]

                im = ax[row, col].imshow(responce_Y.T, origin="lower", extent=the_extent,
                                         interpolation=self.interp, vmin=self.min_colour_val,
                                         vmax=self.max_colour_val, cmap=self.my_cmap)

                subplot_title = 'Weight: %s' % (str(Weight_list[k]))
                ax[row, col].set_title(subplot_title, fontsize=5, verticalalignment='top')

                if row == rows-1:
                    ax[row, col].set_xlabel('$a_1$')

                if col == 0:
                    ax[row, col].set_ylabel('$a_2$')

                # iterate to next matrix bound in the list
                k = k + 1

        # plot colour and title
        fig.colorbar(im, cax=cb_ax)
        fig.suptitle('Varying input weights for defualt material and no shuffle inputs')

        # # show & save plots
        if self.Save_NotShow == 1:
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
                print("VaryOutWeight material graphs not saved")
            return

        responceY_list = np.array(hdf.get('/MG/VaryInWeightAni/responceY_list'))
        the_extent = np.array(hdf.get('/MG/VaryInWeightAni/extent'))
        Weight_list = np.array(hdf.get('/MG/VaryInWeightAni/Weight_list'))

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

        responceY_list = np.array(hdf.get('/MG/VaryOutWeightAni/responceY_list'))
        the_extent = np.array(hdf.get('/MG/VaryOutWeightAni/extent'))
        Weight_list = np.array(hdf.get('/MG/VaryOutWeightAni/Weight_list'))

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

        OW_responceY_list = np.array(hdf.get('/MG/VaryOutWeightAni/responceY_list'))
        OW_the_extent = np.array(hdf.get('/MG/VaryOutWeightAni/extent'))
        OW_Weight_list = np.array(hdf.get('/MG/VaryOutWeightAni/Weight_list'))

        IW_responceY_list = np.array(hdf.get('/MG/VaryInWeightAni/responceY_list'))
        IW_the_extent = np.array(hdf.get('/MG/VaryInWeightAni/extent'))
        IW_Weight_list = np.array(hdf.get('/MG/VaryInWeightAni/Weight_list'))

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

        plot_both = Both  # if 1 plot both large OW and normal OW

        if VloW_ani == 0 or 'VaryLargeOutWeightAni' not in MG_items:
            if cir == 0 and rep == 0 and VloW_ani !=0:
                #print('MG_items:', MG_items)
                print("VaryLargeOutWeightAni material graphs not saved")
            return

        lOW_responceY_list = np.array(hdf.get('/MG/VaryLargeOutWeightAni/responceY_list'))
        lOW_the_extent = np.array(hdf.get('/MG/VaryLargeOutWeightAni/extent'))
        lOW_Weight_list = np.array(hdf.get('/MG/VaryLargeOutWeightAni/Weight_list'))

        if plot_both == 0 or 'VaryOutWeight' not in MG_items:
            plot_both = 0
        else:
            plot_both = 1
            nOW_responceY_list = np.array(hdf.get('/MG/VaryOutWeightAni/responceY_list'))
            nOW_the_extent = np.array(hdf.get('/MG/VaryOutWeightAni/extent'))
            nOW_Weight_list = np.array(hdf.get('/MG/VaryOutWeightAni/Weight_list'))

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

        plot_both = Both  # if 1 plot both large OW and normal OW

        if VliW_ani == 0 or 'VaryLargeInWeightAni' not in MG_items:
            if cir == 0 and rep == 0 and VliW_ani !=0:
                #print('MG_items:', MG_items)
                print("VaryLargeInWeightAni material graphs not saved")
            return

        liW_responceY_list = np.array(hdf.get('/MG/VaryLargeInWeightAni/responceY_list'))
        lIW_the_extent = np.array(hdf.get('/MG/VaryLargeInWeightAni/extent'))
        lIW_Weight_list = np.array(hdf.get('/MG/VaryLargeInWeightAni/Weight_list'))

        if plot_both == 0 or 'VaryInWeightAni' not in MG_items:
            plot_both = 0
            print('MG_items:', MG_items)
        else:
            plot_both = 1
            niW_responceY_list = np.array(hdf.get('/MG/VaryInWeightAni/responceY_list'))
            nIW_the_extent = np.array(hdf.get('/MG/VaryInWeightAni/extent'))
            nIW_Weight_list = np.array(hdf.get('/MG/VaryInWeightAni/Weight_list'))

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
