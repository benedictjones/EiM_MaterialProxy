# # Top Matter
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import numpy as np
import pickle
import pandas as pd
import scipy
import h5py

import os
from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap

from mod_methods.RoundDown import round_decimals_down
from mod_methods.Grouping import fetch_genome_string

from mod_analysis.Set_Load_meta import LoadMetaData

from mod_load.FetchDataObj import FetchDataObj



''' # # # # # # # # # # # # # # #
Produce an animation
> Need ImageMagick installed on the PC
> Saves as an mp4 or a gif
> Ensure Ffmpeg is selected to install alongside
'''


class Animation(object):

    def __init__(self):
        self.fps = 3
        Writer = animation.writers['ffmpeg']
        self.the_writer = Writer(fps=self.fps, metadata=dict(artist='Me'), bitrate=1800)

    def Plt__ani(self, Save_NotShow=0, PlotOnly='all', Specific_Sys_Rep='all', format='gif'):

        if format != 'gif' and format != 'mp4':
            print("Error (animation): only mp4 and gif formats allowed")
        else:
            self.ani_format = format

        print("Producing Animation...!")

        # # create dir list to look at
        if PlotOnly == 'all':
            new_DirList = self.data_dir_list
            self.new_FileRefNames = self.FileRefNames
            self.new_Param_array = self.Param_array
        else:
            new_DirList = []
            self.new_FileRefNames = []
            self.new_Param_array = []
            for val in PlotOnly:
                new_DirList.append(self.data_dir_list[val])
                self.new_FileRefNames.append(self.FileRefNames[val])
                self.new_Param_array.append(self.Param_array[val])

        # # save the data for plotting
        self.dir_loop = 0
        for curr_dir in new_DirList:

            MetaData = LoadMetaData(curr_dir)
            self.lobj = FetchDataObj(MetaData)
            syst_range = range(MetaData['num_systems'])
            rep_range = range(MetaData['num_repetitions'])
            self.ParamDict = MetaData['DE']
            NetworkDict = MetaData['network']
            GenomeDict = MetaData['genome']
            self.genome_grouping = np.asarray(GenomeDict['grouping'])

            if 'HOW' in self.ParamDict['IntpScheme'] or self.ParamDict['IntpScheme'] == 'band':
                self.max_colour_val = GenomeDict['BandClass']['max']
                self.min_colour_val = GenomeDict['BandClass']['min']

            elif self.ParamDict['IntpScheme'] == 'thresh_binary':
                self.max_colour_val = self.ParamDict['threshold']+0.2
                self.min_colour_val = self.ParamDict['threshold']-0.2

            elif 'Ridge' in self.ParamDict['IntpScheme']:
                hdf = h5py.File("%s/data.hdf5" % (curr_dir), 'r')
                threshold = np.array(hdf.get('/%d_rep%d/DE_data/ridge/threshold' % (0, 0)))
                hdf.close()
                if np.isnan(threshold) == False:
                    self.max_colour_val = threshold + 0.1
                    self.min_colour_val = threshold - 0.1
            else:
                val = 0.05  # 0.2
                self.max_colour_val = val
                self.min_colour_val = -val


            if MetaData['mg']['MG_animation'] == 0:
                #print("Directory", self.dir_loop, "did not save animations data")
                self.dir_loop = self.dir_loop + 1
                continue

            if len(Specific_Sys_Rep) != 2:
                if Specific_Sys_Rep != 'all':
                    print("Error: Specific_Sys_Rep must be 'all' or be a specific [syst, rep]")
                    return
            else:
                if Specific_Sys_Rep[0] >= MetaData['num_systems']:
                    print("Error: Invalid Syscuit loop selected to plot")
                    return
                elif Specific_Sys_Rep[1] >= MetaData['num_repetitions']:
                    print("Error: Invalid Repetition loop selected to plot")
                    return
                else:
                    syst_range = [Specific_Sys_Rep[0]]
                    rep_range = [Specific_Sys_Rep[1]]


            # # loop over selected syst and rep
            for syst in syst_range:

                for rep in rep_range:

                    # # Extract data
                    current_file = "%s/data_ani.hdf5" % (curr_dir)
                    with h5py.File(current_file, 'r') as hdf:

                        Ani_Train_data = hdf.get('/%d_rep%d/Train' % (syst, rep))
                        Ani_Vali_data = hdf.get('/%d_rep%d/Vali' % (syst, rep))

                        ls = list(Ani_Train_data.keys())
                        #print("list of top level keys in this file:/n", ls)

                        #print("num its = ", len(ls)-1)
                        num_frames = len(ls)  # extract number of its

                        if syst == 0 and rep == 0:
                            the_extent = np.array(hdf.get('extent'))
                            self.num_batches = np.array(hdf.get('num_batches'))
                            if self.num_batches == 0:
                                self.batch_weight = 1
                            else:
                                self.batch_weight = 1/self.num_batches

                        self.all_data_train = []
                        self.all_data_train_bfits = []
                        self.all_data_train_bGenome = []

                        self.all_data_vali = []
                        self.all_data_vali_bfits = []
                        self.all_data_vali_bGenome = []


                        for it in range(num_frames):
                            dataset_name = "MG_dat_it_%d" % (it)
                            dataset = Ani_Train_data.get(dataset_name)
                            self.all_data_train.append(np.array(dataset))
                            self.all_data_train_bfits.append(dataset.attrs['best_fit'])
                            self.all_data_train_bGenome.append(dataset.attrs['best_genome'])

                        if self.ParamDict['batch_size'] != 0:
                            for ep in range(len(list(Ani_Vali_data.keys()))):
                                Vdataset_name = "MG_dat_it_%d" % (ep+1)
                                Vdataset = Ani_Vali_data.get(Vdataset_name)
                                self.all_data_vali.append(np.array(Vdataset))
                                self.all_data_vali_bfits.append(Vdataset.attrs['best_fit'])
                                self.all_data_vali_bGenome.append(Vdataset.attrs['best_genome'])


                    # # do a pixel colour plot
                    #basic_cols = ['#009cff','#ffffff','#ff8800']   # pastal orange/blue
                    #basic_cols = ['#0000ff', '#5800ff', '#000000', '#ff5800', '#ff0000']  # red/blue
                    basic_cols = ['#009cff', '#6d55ff', '#ffffff', '#ff6d55','#ff8800']  # pastal orange/red/white/purle/blue
                    # basic_cols=['#0000ff', '#000000', '#ff0000']
                    my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

                    # # load training data to plot on graph
                    #data_X, class_Y = Load_Data('train', MetaData)
                    #data_X, class_Y = self.lobj.fetch_data('train')
                    data_X, class_Y = self.lobj.get_data(the_data='train', iterate=0)  # the whole training dataset

                    # Some global variables to define the whole run
                    self.total_number_of_frames = num_frames

                    #the_animation = self.basic_ani(syst, rep, my_cmap, the_extent, data_X, class_Y, Save_NotShow)
                    #the_animation = self.sub_ani(syst, rep, my_cmap, the_extent, data_X, class_Y, Save_NotShow)
                    the_animation = self.sub_ani2(syst, rep, my_cmap, the_extent, data_X, class_Y, Save_NotShow)

            # increment directory loop number
            self.dir_loop = self.dir_loop + 1

        if Save_NotShow == 0:
            plt.show()

    #

    #

    #

    #

    def basic_ani(self, syst, rep, my_cmap, the_extent, data_X, class_Y, Save_NotShow=0):
        """ Produces a basic animation of the responce graph
        """


        # Now we can do the plotting!
        fig, ax = plt.subplots(1)

        # Remove a bunch of stuff to make sure we only 'see' the actual imshow
        # Stretch to fit the whole plane
        #fig.subplots_adjust(0, 0, 1, 1)

        # Remove bounding line
        #ax.axis("off")


        # Initialise our plot. Make sure you set vmin and vmax!
        image_basic = ax.imshow(self.all_data_train[0].T, origin="lower",
                                  extent=the_extent,
                                  interpolation='none',
                                  vmin=self.min_colour_val, vmax=self.max_colour_val,
                                  cmap=my_cmap)
        cb = plt.colorbar(image_basic)
        cb.set_label("Y", fontsize=10, rotation=90)


        #ax.plot([0,1,2],[2,4,6])
        #ax.set_title("Animation (File %d, syst=%d, rep=%d) - Iteration %d" % (self.dir_loop, syst, rep, 0))
        genome_text = self.all_data_train_bGenome[0].replace('array()', '')
        genome_text = genome_text.replace(')', '')
        ax.set_title("Animation (File %d, syst=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, syst, rep, 0, self.all_data_train_bfits[0], genome_text))

        ax.set_ylabel("a2")
        ax.set_xlabel("a1")

        # # plot training data too
        data_X, data_Y = self.lobj.get_data(the_data='train', iterate=0)
        class_values = np.unique(data_Y)
        class_values.sort()
        markers = ["o", "*", "x", "v", "+", "." , "^" , "<", ">"]
        colours = ['#009cffff','#ff8800ff','#9cff00ff','c','m', 'y', 'k']
        sizes = [3.5, 5.5, 3.5, 3.5, 3.5, 3.5, 3.5]

        # Group Classes
        class_data = []
        for cla in class_values:
            class_group = []
            for instance, yclass in enumerate(data_Y):
                if yclass == cla:
                    class_group.append([data_X[instance,0], data_X[instance,1]])
            class_group = np.asarray(class_group)
            class_data.append(class_group)

        # Plot the grouped classes
        for idx, cla in enumerate(class_values):
            data = class_data[idx]
            ax.plot(data[:,0], data[:,1], color=colours[idx],
                     marker=markers[idx], markersize=sizes[idx], ls='',
                     label="Class %d" % (idx+1),
                     alpha=0.8, markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))

        ax.legend(markerscale=1.5, fancybox=True)  # , facecolor='k'

        # # the function which updates the animation
        def animate(frame):
            #Animation function. Takes the current frame number (to select the potion of
            #data to plot) and a line selfect to update.

            # Not strictly neccessary, just so we know we are stealing these from
            # the global scope
            #global self.all_data_train, image_basic, ax

            # We want up-to and _including_ the frame'th element
            image_basic.set_array(self.all_data_train[frame].T)  # transposing the data so it is plotted correctly
            genome_text = self.all_data_train_bGenome[frame].replace('array(', '')
            genome_text = genome_text.replace(')', '')
            ax.set_title("Animation (File %d, syst=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, syst, rep, frame, self.all_data_train_bfits[frame], genome_text))

            return image_basic

        # # produce animation
        the_animation = animation.FuncAnimation(
            # Your Matplotlib Figure selfect
            fig,
            # The function that does the updating of the Figure
            animate,
            # Frame information (here just frame number)
            np.arange(self.total_number_of_frames),
            # Extra arguments to the animate function
            fargs=[],
            # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
            interval=1000/self.fps
            )

        # # show & save plots
        plt.subplots_adjust(top=0.75)
        if Save_NotShow == 1:
            if self.exp == 1:
                if self.ani_format == 'mp4':
                    fig_path = "%s/%s_%s_AniMG_Sys%d_Rep%d.mp4" % (self.save_dir, self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], syst, rep)
                    the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
                elif self.ani_format == 'gif':
                    fig_path = "%s/%s_%s_AniMG_Sys%d_Rep%d.gif" % (self.save_dir, self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], syst, rep)
                    the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            else:
                if self.ani_format == 'mp4':
                    fig_path = "%s/AniMG_Sys%d_Rep%d.mp4" % (self.save_dir, syst, rep)
                    the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
                elif self.ani_format == 'gif':
                    fig_path = "%s/AniMG_Sys%d_Rep%d.gif" % (self.save_dir, syst, rep)
                    the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            plt.close(fig)

        return the_animation

    #

    #

    #

    #

    #

    #

    def sub_ani(self, syst, rep, my_cmap, the_extent, data_X, class_Y, Save_NotShow=0):
        """ Produces a basic animation of the responce graph, with a graph
        following it's best fitness vs itteration subplotted
        """

        # Now we can do the plotting!
        fig, (ax_mg, ax_bf) = plt.subplots(1, 2, figsize=(6, 3), gridspec_kw={'width_ratios': [1, 1.25]})

        #fig.suptitle("Animation (File %d, syst=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, syst, rep, 0, self.all_data_train_bfits[0], self.all_data_train_bGenome[0]), fontsize=12)
        fig.suptitle("Animation (File %d, syst=%d, rep=%d), Best Batch Fit = %.4f\n" % (self.dir_loop, syst, rep, self.all_data_train_bfits[0]), fontsize=12)

        # Remove a bunch of stuff to make sure we only 'see' the actual imshow
        # Stretch to fit the whole plane
        #plt.tight_layout()
        #fig.subplots_adjust(0, 0, 1, 1)
        #fig.subplots_adjust(top=0.5)
        #plt.subplots_adjust(right=0.7)
        #fig.subplots_adjust(left=0.125, bottom=0.125, right=0.85, top=0.75, wspace=0.1, hspace=0.2)

        # Remove bounding line
        #ax.axis("off")

        batchs = []
        b = 0
        for i in range(self.total_number_of_frames):
            batchs.append(b)
            b = b + 1
            if b >= (1/self.batch_weight):
                b = 0

        # Initialise our plot. Make sure you set vmin and vmax!
        image = ax_mg.imshow(self.all_data_train[0].T, origin="lower",
                          extent=the_extent,
                          interpolation='none',
                          vmin=self.min_colour_val, vmax=self.max_colour_val,
                          cmap=my_cmap, aspect='equal')

        #plt.colorbar(image, ax=ax_mg)  # ad to ax plot
        cbar = plt.colorbar(image, ax=ax_mg, format='% 2.2f')
        cbar.set_label("Y", fontsize=10, rotation=90)
        cbar.ax.tick_params(labelsize=8)  # cbar ticks size
        #cbar.solids.set_edgecolor("face")

        #cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.5])
        #plt.colorbar(image, cax=cb_ax)  # place manually

        #ax.plot([0,1,2],[2,4,6])
        #ax.set_title("Animation (File %d, syst=%d, rep=%d) - Iteration %d" % (self.dir_loop, syst, rep, 0))
        #ax_mg.set_title("Animation (File %d, syst=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, syst, rep, 0, self.all_data_train_bfits[0], str(np.around(self.all_data_train_bGenome[0],decimals=2))))
        ax_mg.set_title("Responce", fontsize=10)
        ax_mg.set_ylabel("a2", fontsize=9)
        ax_mg.set_xlabel("a1", fontsize=9)
        ax_mg.xaxis.set_tick_params(labelsize=8)
        ax_mg.yaxis.set_tick_params(labelsize=8)

        # # plot training data too
        data_X, data_Y = self.lobj.get_data(the_data='train', iterate=0)
        class_values = np.unique(data_Y)
        class_values.sort()
        markers = ["o", "*", "x", "v", "+", "." , "^" , "<", ">"]
        colours = ['#009cffff','#ff8800ff','#9cff00ff','c','m', 'y', 'k']
        sizes = [3.5, 5.5, 3.5, 3.5, 3.5, 3.5, 3.5]

        # Group Classes
        class_data = []
        for cla in class_values:
            class_group = []
            for instance, yclass in enumerate(data_Y):
                if yclass == cla:
                    class_group.append([data_X[instance,0], data_X[instance,1]])
            class_group = np.asarray(class_group)
            class_data.append(class_group)

        # Plot the grouped classes
        data_scat = []
        for idx, cla in enumerate(class_values):
            data = class_data[idx]
            graph, = ax_mg.plot(data[:,0], data[:,1], color=colours[idx],
                                     marker=markers[idx], markersize=sizes[idx], ls='',
                                     label="Class %d" % (idx+1),
                                     alpha=0.8, markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
            data_scat.append(graph)

        #ax_mg.legend(markerscale=1.5, fancybox=True)  # , facecolor='k'

        #asp = np.diff(ax_bf.get_xlim())[0] / np.diff(ax_bf.get_ylim())[0]
        #asp /= np.abs(np.diff(ax_mg.get_xlim())[0] / np.diff(ax_mg.get_ylim())[0])
        #ax_bf.set_aspect(asp)

        new_stop = self.batch_weight*(len(self.all_data_train_bfits)-1)
        nspaces = len(self.all_data_train_bfits)
        self.x_vals = np.linspace(0, new_stop, nspaces)

        ax_bf.plot(self.x_vals, self.all_data_train_bfits, color='#0000ff', label='training')   # plot line

        it_point, = ax_bf.plot(self.x_vals[0], self.all_data_train_bfits[0], 'o', markersize=5, markerfacecolor='#00008c00', markeredgecolor='#00008c')
        ax_bf.set_title("Best Fitness Vs Iteration", fontsize=10)
        ax_bf.set_ylabel("Best Population Fitness", fontsize=9)
        ax_bf.set_xlabel("Epoch", fontsize=9)
        ax_bf.xaxis.set_tick_params(labelsize=8)
        ax_bf.yaxis.set_tick_params(labelsize=8)

        if self.ParamDict['batch_size'] != 0:
            self.x_vals_vali = np.arange(1, len(self.all_data_vali_bfits)+1)
            #ax_bf.plot(np.arange(1, len(self.all_data_vali_bfits)),self.all_data_vali_bfits[1:], 'r', label='validation')  # plot line
            ax_bf.plot(self.x_vals_vali, self.all_data_vali_bfits, 'r', label='validation')  # plot line
            it_point_vali, = ax_bf.plot(-1, 0, 'o', markerfacecolor='#8c000000', markeredgecolor='#8c0000', markersize=5)
            ax_bf.set_xlim(left=0)
            self.ani_epoch = 0

        ax_bf.legend(fancybox=True)  # , facecolor='k'


        #plt.tight_layout()
        #fig.subplots_adjust(top=0.5)
        #fig.subplots_adjust(left=0.05, bottom=0.05, top=0.75)

        #plt.show()
        #exit()

        # # the function which updates the animation
        def animate(frame):
            # Animation function. Takes the current frame number (to select the potion of
            # data to plot) and a line selfect to update.

            # Not strictly neccessary, just so we know we are stealing these from
            # the global scope
            # global self.all_data_train, image, ax

            # We want up-to and _including_ the frame'th element
            image.set_array(self.all_data_train[frame].T)  # transposing the data so it is plotted correctly
            #ax_bf.clear()
            #ax_bf.plot(frame, self.all_data_train_bfits[frame], 'o', markersize=5, markerfacecolor='#ff880000', markeredgewidth=2, markeredgecolor='#ff0000ff')

            # # Set Training marker location
            it_point.set_ydata(self.all_data_train_bfits[frame])
            it_point.set_xdata(self.x_vals[frame])

            # # Set Validation marker location
            if self.ParamDict['batch_size'] != 0 and self.ani_epoch+1 <= (self.batch_weight*frame):
                it_point_vali.set_ydata(self.all_data_vali_bfits[self.ani_epoch])
                it_point_vali.set_xdata(self.x_vals_vali[self.ani_epoch])
                self.ani_epoch += 1

            # # Plot each batch (but can't if it was shuffled as data not saved)
            if self.ParamDict['batch_size'] != 0 and self.ParamDict['batch_scheme'] != 'shuffle':
                # Group Classes
                class_data = []
                data_X = self.lobj.batch_list[batchs[frame]][0]
                data_Y = self.lobj.batch_list[batchs[frame]][1]
                for cla in class_values:
                    class_group = []
                    for instance, yclass in enumerate(data_Y):
                        if yclass == cla:
                            class_group.append([data_X[instance,0], data_X[instance,1]])
                    class_group = np.asarray(class_group)
                    class_data.append(class_group)

                # Plot the grouped classes
                for idx, cla in enumerate(class_values):
                    data = class_data[idx]
                    data_scat[idx].set_xdata(data[:,0])
                    data_scat[idx].set_ydata(data[:,1])


            #fig.suptitle("Animation (File %d, syst=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, syst, rep, frame, self.all_data_train_bfits[frame], self.all_data_train_bGenome[frame]))
            fig.suptitle("Animation (File %d, syst=%d, rep=%d), Best Batch Fit = %.4f" % (self.dir_loop, syst, rep, self.all_data_train_bfits[frame]))

            return image

        # # produce animation
        the_animation = animation.FuncAnimation(
            # Your Matplotlib Figure selfect
            fig,
            # The function that does the updating of the Figure
            animate,
            # Frame information (here just frame number)
            np.arange(self.total_number_of_frames),
            # Extra arguments to the animate function
            fargs=[],
            # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
            interval=1000/self.fps
            )

        # # show & save plots
        #fig.subplots_adjust(top=0.75)
        #fig.subplots_adjust(left=0.125, bottom=0.125, right=0.85, top=0.75, wspace=0.1, hspace=0.2)
        #fig.subplots_adjust(left=0.05, bottom=0.05, top=0.75)
        if Save_NotShow == 1:
            if self.exp == 1:
                if self.ani_format == 'mp4':
                    fig_path = "%s/%s_%s_Animation_Sys%d_Rep%d.mp4" % (self.save_dir, self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], syst, rep)
                    the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
                elif self.ani_format == 'gif':
                    fig_path = "%s/%s_%s_Animation_Sys%d_Rep%d.gif" % (self.save_dir, self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], syst, rep)
                    the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            else:
                if self.ani_format == 'mp4':
                    fig_path = "%s/Animation_Sys%d_Rep%d.mp4" % (self.save_dir, syst, rep)
                    the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
                elif self.ani_format == 'gif':
                    fig_path = "%s/Animation_Sys%d_Rep%d.gif" % (self.save_dir, syst, rep)
                    the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            plt.close(fig)

        return the_animation

    #

    #

    #

    #

    #

    #

    def sub_ani2(self, syst, rep, my_cmap, the_extent, data_X, class_Y, Save_NotShow=0):
        """ Produces a basic animation of the responce graph, with a graph
        following it's best fitness vs itteration subplotted
        """

        # Now we can do the plotting!
        #fig, (ax_mg, ax_bf) = plt.subplots(1, 2, figsize=(6, 3), gridspec_kw={'width_ratios': [1, 1.25]})


        """gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1])
        ax_Pbest = fig.add_subplot(gs[1, 0])
        ax_Gbest = fig.add_subplot(gs[1, 1])
        ax_bf = fig.add_subplot(gs[0, :])"""
        if self.ParamDict['batch_size'] != 0:
            fig = plt.figure(figsize=(6, 5))
            gs = fig.add_gridspec(4,4)
            ax_Pbest = fig.add_subplot(gs[2:, :2])
            ax_Gbest = fig.add_subplot(gs[2:, 2:])
            ax_bf = fig.add_subplot(gs[:2, 1:3]) # [:2, 1:3]
        else:
            fig = plt.figure(figsize=(6, 3))
            gs = fig.add_gridspec(1,2)
            ax_Pbest = fig.add_subplot(gs[0])
            ax_bf = fig.add_subplot(gs[1])

        #fig.suptitle("Animation (File %d, syst=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, syst, rep, 0, self.all_data_train_bfits[0], self.all_data_train_bGenome[0]), fontsize=12)
        fig.suptitle("Animation (File %d, syst=%d, rep=%d), Best Batch Fit = %.4f\n" % (self.dir_loop, syst, rep, self.all_data_train_bfits[0]), fontsize=12)
        plt.tight_layout()
        #fig.subplots_adjust(left=0.3, right=0.8)


        batchs = []
        b = 0
        for i in range(self.total_number_of_frames):
            batchs.append(b)
            b = b + 1
            if b >= (1/self.batch_weight):
                b = 0

        # Initialise our plot. Make sure you set vmin and vmax!
        image = ax_Pbest.imshow(self.all_data_train[0].T, origin="lower",
                                  extent=the_extent,
                                  interpolation='none',
                                  vmin=self.min_colour_val, vmax=self.max_colour_val,
                                  cmap=my_cmap, aspect='equal')


        #cbar = plt.colorbar(image, ax=ax_Pbest, format='% 2.2f')
        #cbar.set_label("Y", fontsize=10, rotation=90)
        #cbar.ax.tick_params(labelsize=8)  # cbar ticks size
        ax_Pbest.set_title("$p_{best}$ Responce", fontsize=10)
        ax_Pbest.set_ylabel("a2", fontsize=9)
        ax_Pbest.set_xlabel("a1", fontsize=9)
        ax_Pbest.xaxis.set_tick_params(labelsize=8)
        ax_Pbest.yaxis.set_tick_params(labelsize=8)

        # # plot training data too
        data_X, data_Y = self.lobj.get_data(the_data='train', iterate=0)
        class_values = np.unique(data_Y)
        class_values.sort()
        markers = ["o", "*", "x", "v", "+", "." , "^" , "<", ">"]
        colours = ['#009cffff','#ff8800ff','#9cff00ff','c','m', 'y', 'k']
        sizes = [3.5, 5.5, 3.5, 3.5, 3.5, 3.5, 3.5]

        # Group Classes Batches
        class_data = []
        for cla in class_values:
            class_group = []
            for instance, yclass in enumerate(data_Y):
                if yclass == cla:
                    class_group.append([data_X[instance,0], data_X[instance,1]])
            class_group = np.asarray(class_group)
            class_data.append(class_group)

        # Plot the grouped classes
        data_scat = []
        for idx, cla in enumerate(class_values):
            data = class_data[idx]
            graph, = ax_Pbest.plot(data[:,0], data[:,1], color=colours[idx],
                                     marker=markers[idx], markersize=sizes[idx], ls='',
                                     label="Class %d" % (idx+1),
                                     alpha=0.8, markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
            data_scat.append(graph)

        #

        #

        # # Plot Validation Initial data
        if self.ParamDict['batch_size'] != 0:
            Vimage = ax_Gbest.imshow(np.zeros(np.shape(self.all_data_vali[0].T)), origin="lower",
                                      extent=the_extent,
                                      interpolation='none',
                                      vmin=self.min_colour_val, vmax=self.max_colour_val,
                                      cmap=my_cmap, aspect='equal')

            Vcbar = plt.colorbar(Vimage, ax=ax_Gbest, format='% 2.2f')
            Vcbar.set_label("Y", fontsize=10, rotation=90)
            Vcbar.ax.tick_params(labelsize=8)  # cbar ticks size
            ax_Gbest.set_title("$G_{best}$ Responce", fontsize=10)
            ax_Gbest.set_ylabel("a2", fontsize=9)
            ax_Gbest.set_xlabel("a1", fontsize=9)
            ax_Gbest.xaxis.set_tick_params(labelsize=8)
            ax_Gbest.yaxis.set_tick_params(labelsize=8)

            # # Plot Validation Data
            data_X, data_Y = self.lobj.get_data(the_data='validation', iterate=0)
            class_data = []
            for cla in class_values:
                class_group = []
                for instance, yclass in enumerate(data_Y):
                    if yclass == cla:
                        class_group.append([data_X[instance,0], data_X[instance,1]])
                class_group = np.asarray(class_group)
                class_data.append(class_group)

            # Plot the grouped classes
            for idx, cla in enumerate(class_values):
                data = class_data[idx]
                ax_Gbest.plot(data[:,0], data[:,1], color=colours[idx],
                              marker=markers[idx], markersize=sizes[idx], ls='',
                              label="Class %d" % (idx+1),
                              alpha=0.8, markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))


        #ax_Pbest.legend(markerscale=1.5, fancybox=True)  # , facecolor='k'

        #asp = np.diff(ax_bf.get_xlim())[0] / np.diff(ax_bf.get_ylim())[0]
        #asp /= np.abs(np.diff(ax_Pbest.get_xlim())[0] / np.diff(ax_Pbest.get_ylim())[0])
        #ax_bf.set_aspect(asp)

        new_stop = self.batch_weight*(len(self.all_data_train_bfits)-1)
        nspaces = len(self.all_data_train_bfits)
        self.x_vals = np.linspace(0, new_stop, nspaces)

        ax_bf.plot(self.x_vals, self.all_data_train_bfits,  color='#0000ff', label='$p_{best}$ training')   # plot line

        it_point, = ax_bf.plot(self.x_vals[0], self.all_data_train_bfits[0], 'o', markersize=5, markerfacecolor='#00008c00', markeredgecolor='#00008c')
        ax_bf.set_title("Best Fitness Vs Iteration", fontsize=10)
        ax_bf.set_ylabel("Best Population Fitness", fontsize=9)
        ax_bf.set_xlabel("Epoch", fontsize=9)
        ax_bf.xaxis.set_tick_params(labelsize=8)
        ax_bf.yaxis.set_tick_params(labelsize=8)

        if self.ParamDict['batch_size'] != 0:
            self.x_vals_vali = np.arange(1, len(self.all_data_vali_bfits)+1)
            #ax_bf.plot(np.arange(1, len(self.all_data_vali_bfits)),self.all_data_vali_bfits[1:], 'r', label='validation')  # plot line
            ax_bf.plot(self.x_vals_vali, self.all_data_vali_bfits, '-*', color='r', label='$G_{best}$ validation')  # plot line
            it_point_vali, = ax_bf.plot(-1, 0, 'o', markerfacecolor='#8c000000', markeredgecolor='#8c0000', markersize=5)
            ax_bf.set_xlim(left=0)
            self.ani_epoch = 0

        ax_bf.legend(fancybox=True)  # , facecolor='k'


        #plt.tight_layout()
        #fig.subplots_adjust(top=0.5)
        #fig.subplots_adjust(left=0.05, bottom=0.05, top=0.75)

        #plt.show()
        #exit()

        # # the function which updates the animation
        def animate(frame):
            # Animation function. Takes the current frame number (to select the potion of
            # data to plot) and a line selfect to update.

            # Not strictly neccessary, just so we know we are stealing these from
            # the global scope
            # global self.all_data_train, image, ax

            # We want up-to and _including_ the frame'th element
            image.set_array(self.all_data_train[frame].T)  # transposing the data so it is plotted correctly
            #ax_bf.clear()
            #ax_bf.plot(frame, self.all_data_train_bfits[frame], 'o', markersize=5, markerfacecolor='#ff880000', markeredgewidth=2, markeredgecolor='#ff0000ff')

            # # Set Training marker location
            it_point.set_ydata(self.all_data_train_bfits[frame])
            it_point.set_xdata(self.x_vals[frame])

            # # Set Validation marker location
            if self.ParamDict['batch_size'] != 0 and self.ani_epoch+1 <= (self.batch_weight*frame):
                it_point_vali.set_ydata(self.all_data_vali_bfits[self.ani_epoch])
                it_point_vali.set_xdata(self.x_vals_vali[self.ani_epoch])
                Vimage.set_array(self.all_data_vali[self.ani_epoch].T)
                self.ani_epoch += 1

            # # Plot each batch (but can't if it was shuffled as data not saved)
            if self.ParamDict['batch_size'] != 0 and self.ParamDict['batch_scheme'] != 'shuffle':
                
                # Group Classes
                class_data = []
                data_X = self.lobj.batch_list[batchs[frame]][0]
                data_Y = self.lobj.batch_list[batchs[frame]][1]
                for cla in class_values:
                    class_group = []
                    for instance, yclass in enumerate(data_Y):
                        if yclass == cla:
                            class_group.append([data_X[instance,0], data_X[instance,1]])
                    class_group = np.asarray(class_group)
                    class_data.append(class_group)

                # Plot the grouped classes
                for idx, cla in enumerate(class_values):
                    data = class_data[idx]
                    data_scat[idx].set_xdata(data[:,0])
                    data_scat[idx].set_ydata(data[:,1])


            #fig.suptitle("Animation (File %d, syst=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, syst, rep, frame, self.all_data_train_bfits[frame], self.all_data_train_bGenome[frame]))
            fig.suptitle("Animation (File %d, syst=%d, rep=%d), Best Batch Fit = %.4f" % (self.dir_loop, syst, rep, self.all_data_train_bfits[frame]))

            return

        # # produce animation
        the_animation = animation.FuncAnimation(
            # Your Matplotlib Figure selfect
            fig,
            # The function that does the updating of the Figure
            animate,
            # Frame information (here just frame number)
            np.arange(self.total_number_of_frames),
            # Extra arguments to the animate function
            fargs=[],
            # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
            interval=1000/self.fps
            )

        # # show & save plots
        #fig.subplots_adjust(left=0.125, bottom=0.125, right=0.85, top=0.75, wspace=0.1, hspace=0.2)
        #fig.subplots_adjust(left=0.05, bottom=0.05, top=0.75)
        if Save_NotShow == 1:
            if self.exp == 1:
                if self.ani_format == 'mp4':
                    fig_path = "%s/%s_%s_Animation_Sys%d_Rep%d.mp4" % (self.save_dir, self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], syst, rep)
                    the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
                elif self.ani_format == 'gif':
                    fig_path = "%s/%s_%s_Animation_Sys%d_Rep%d.gif" % (self.save_dir, self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], syst, rep)
                    the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            else:
                if self.ani_format == 'mp4':
                    fig_path = "%s/Animation_Sys%d_Rep%d.mp4" % (self.save_dir, syst, rep)
                    the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
                elif self.ani_format == 'gif':
                    fig_path = "%s/Animation_Sys%d_Rep%d.gif" % (self.save_dir, syst, rep)
                    the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            plt.close(fig)

        return the_animation

    #

    #

    #

    #

    #

    #


#

#

#


#

#

# fin
