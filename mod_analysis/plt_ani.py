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

from mod_analysis.Set_Load_meta import LoadMetaData

from mod_load.LoadData import Load_Data




''' # # # # # # # # # # # # # # #
Produce an animation
> Need ImageMagick installed on the PC
> Saves as an mp4 or a gif
> Ensure Ffmpeg is selected to install alongside
'''


class Animation(object):

    def __init__(self):
        print("Producing Animation...!")
        self.fps = 1.5
        Writer = animation.writers['ffmpeg']
        self.the_writer = Writer(fps=self.fps, metadata=dict(artist='Me'), bitrate=1800)

    def Plt__ani(self, Save_NotShow=0, PlotOnly='all', Specific_Cir_Rep='all', format='gif'):

        if format != 'gif' and format != 'mp4':
            print("Error (animation): only mp4 and gif formats allowed")
        else:
            self.ani_format = format

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
            cir_range = range(MetaData['num_circuits'])
            rep_range = range(MetaData['num_repetitions'])
            ParamDict = MetaData['DE']
            NetworkDict = MetaData['network']
            GenomeDict = MetaData['genome']


            if MetaData['mg']['MG_animation'] == 0:
                #print("Directory", self.dir_loop, "did not save animations data")
                self.dir_loop = self.dir_loop + 1
                continue

            if len(Specific_Cir_Rep) != 2:
                if Specific_Cir_Rep != 'all':
                    print("Error: Specific_Cir_Rep must be 'all' or be a specific [cir, rep]")
                    return
            else:
                if Specific_Cir_Rep[0] >= MetaData['num_circuits']:
                    print("Error: Invalid Circuit loop selected to plot")
                    return
                elif Specific_Cir_Rep[1] >= MetaData['num_repetitions']:
                    print("Error: Invalid Repetition loop selected to plot")
                    return
                else:
                    cir_range = [Specific_Cir_Rep[0]]
                    rep_range = [Specific_Cir_Rep[1]]


            # # loop over selected cir and rep
            for cir in cir_range:

                for rep in rep_range:

                    # # Extract data
                    current_file = "%s/MG_AniData/%d_Rep%d.hdf5" % (curr_dir, cir, rep)
                    with h5py.File(current_file, 'r') as hdf:

                        ls = list(hdf.keys())
                        #print("list of top level keys in this file:/n", ls)

                        #print("num its = ", len(ls)-1)
                        num_its = len(ls)-1  # extract number of its

                        dataset_extent = hdf.get('extent')
                        the_extent = np.array(dataset_extent)

                        self.all_data = []
                        self.all_data__best_fit = []
                        self.all_data__best_genome = []
                        for it in range(num_its):
                            dataset_name = 'MG_dat_it_%d' % (it)

                            dataset = hdf.get(dataset_name)
                            self.all_data.append(np.array(dataset))
                            self.all_data__best_fit.append(dataset.attrs['best_fit'])
                            self.all_data__best_genome.append(dataset.attrs['best_genome'])

                    # # do a pixel colour plot
                    #basic_cols = ['#009cff','#ffffff','#ff8800']   # pastal orange/blue
                    #basic_cols = ['#0000ff', '#5800ff', '#000000', '#ff5800', '#ff0000']  # red/blue
                    basic_cols = ['#009cff', '#6d55ff', '#ffffff', '#ff6d55','#ff8800']  # pastal orange/red/white/purle/blue
                    # basic_cols=['#0000ff', '#000000', '#ff0000']
                    my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

                    # # load training data to plot on graph
                    data_X, class_Y = Load_Data('train', MetaData)

                    # Some global variables to define the whole run
                    self.total_number_of_frames = len(self.all_data)  # num_its

                    the_animation = self.sub_ani(cir, rep, my_cmap, the_extent, data_X, class_Y, Save_NotShow)

            # increment directory loop number
            self.dir_loop = self.dir_loop + 1

        if Save_NotShow == 0:
            plt.show()

    #

    #

    #

    #

    def basic_ani(self, cir, rep, my_cmap, the_extent, data_X, class_Y, Save_NotShow=0):
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
        image = ax.imshow(self.all_data[0].T, origin="lower",
                          extent=the_extent,
                          interpolation='none', vmin=-0.2, vmax=0.2, cmap=my_cmap)
        plt.colorbar(image)

        #ax.plot([0,1,2],[2,4,6])
        #ax.set_title("Animation (File %d, cir=%d, rep=%d) - Iteration %d" % (self.dir_loop, cir, rep, 0))
        ax.set_title("Animation (File %d, cir=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, cir, rep, 0, self.all_data__best_fit[0], self.all_data__best_genome[0]))

        ax.set_ylabel("x2")
        ax.set_xlabel("x1")

        # # plot training data too
        c = 0
        cl1_hit = 0
        cl2_hit = 0
        for row in data_X:

            if class_Y[c] == 1:
                if cl1_hit == 0:
                    plt.plot(row[0], row[1],  'o', alpha=0.8, markersize=4, label="Class 1",
                             markerfacecolor='#009cffff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
                    cl1_hit = 1
                else:
                    plt.plot(row[0], row[1],  'o', alpha=0.8, markersize=4,
                             markerfacecolor='#009cffff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
            else:
                if cl2_hit == 0:
                    plt.plot(row[0], row[1],  '*', alpha=0.8, markersize=6, label="Class 2",
                             markerfacecolor='#ff8800ff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
                    cl2_hit = 1
                else:
                    plt.plot(row[0], row[1],  '*', alpha=0.8, markersize=6,
                             markerfacecolor='#ff8800ff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
            c = c + 1


        plt.legend(markerscale=2, fancybox=True)  # , facecolor='k'

        # # the function which updates the animation
        def animate(frame):
            #Animation function. Takes the current frame number (to select the potion of
            #data to plot) and a line selfect to update.

            # Not strictly neccessary, just so we know we are stealing these from
            # the global scope
            #global self.all_data, image, ax

            # We want up-to and _including_ the frame'th element
            image.set_array(self.all_data[frame].T)  # transposing the data so it is plotted correctly
            ax.set_title("Animation (File %d, cir=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, cir, rep, frame, self.all_data__best_fit[frame], self.all_data__best_genome[frame]))

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
        if Save_NotShow == 1:
            if self.exp == 1:
                if self.ani_format == 'mp4':
                    fig_path = "%s/%s_%s_Animation_Cir%d_Rep%d.mp4" % (self.save_dir, self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], cir, rep)
                    the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
                elif self.ani_format == 'gif':
                    fig_path = "%s/%s_%s_Animation_Cir%d_Rep%d.gif" % (self.save_dir, self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], cir, rep)
                    the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            else:
                if self.ani_format == 'mp4':
                    fig_path = "%s/Animation_Cir%d_Rep%d.mp4" % (self.save_dir, cir, rep)
                    the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
                elif self.ani_format == 'gif':
                    fig_path = "%s/Animation_Cir%d_Rep%d.gif" % (self.save_dir, cir, rep)
                    the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            plt.close(fig)

        return the_animation

    #

    #

    def sub_ani(self, cir, rep, my_cmap, the_extent, data_X, class_Y, Save_NotShow=0):
        """ Produces a basic animation of the responce graph, with a graph
        following it's best fitness vs itteration subplotted
        """

        # Now we can do the plotting!
        fig, (ax_mg, ax_bf) = plt.subplots(1, 2, figsize=(14, 6))

        fig.suptitle("Animation (File %d, cir=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, cir, rep, 0, self.all_data__best_fit[0], self.all_data__best_genome[0]))

        # Remove a bunch of stuff to make sure we only 'see' the actual imshow
        # Stretch to fit the whole plane
        #fig.subplots_adjust(0, 0, 1, 1)

        # Remove bounding line
        #ax.axis("off")


        # Initialise our plot. Make sure you set vmin and vmax!
        image = ax_mg.imshow(self.all_data[0].T, origin="lower",
                          extent=the_extent,
                          interpolation='none', vmin=-0.2, vmax=0.2, cmap=my_cmap)

        #plt.colorbar(image, ax=ax_mg)  # ad to ax plot

        #cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.5])
        #plt.colorbar(image, cax=cb_ax)  # place manually

        #ax.plot([0,1,2],[2,4,6])
        #ax.set_title("Animation (File %d, cir=%d, rep=%d) - Iteration %d" % (self.dir_loop, cir, rep, 0))
        #ax_mg.set_title("Animation (File %d, cir=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, cir, rep, 0, self.all_data__best_fit[0], str(np.around(self.all_data__best_genome[0],decimals=2))))
        ax_mg.set_title("Responce (Y)")
        ax_mg.set_ylabel("x2")
        ax_mg.set_xlabel("x1")

        # # plot training data too
        c = 0
        cl1_hit = 0
        cl2_hit = 0
        for row in data_X:

            if class_Y[c] == 1:
                if cl1_hit == 0:
                    ax_mg.plot(row[0], row[1],  'o', alpha=0.8, markersize=4, label="Class 1",
                             markerfacecolor='#009cffff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
                    cl1_hit = 1
                else:
                    ax_mg.plot(row[0], row[1],  'o', alpha=0.8, markersize=4,
                             markerfacecolor='#009cffff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
            else:
                if cl2_hit == 0:
                    ax_mg.plot(row[0], row[1],  '*', alpha=0.8, markersize=6, label="Class 2",
                             markerfacecolor='#ff8800ff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
                    cl2_hit = 1
                else:
                    ax_mg.plot(row[0], row[1],  '*', alpha=0.8, markersize=6,
                             markerfacecolor='#ff8800ff', markeredgewidth=0.5, markeredgecolor=(1, 1, 1, 1))
            c = c + 1


        ax_mg.legend(markerscale=2, fancybox=True)  # , facecolor='k'

        #asp = np.diff(ax_bf.get_xlim())[0] / np.diff(ax_bf.get_ylim())[0]
        #asp /= np.abs(np.diff(ax_mg.get_xlim())[0] / np.diff(ax_mg.get_ylim())[0])
        #ax_bf.set_aspect(asp)

        ax_bf.plot(self.all_data__best_fit, color='#0000ff')
        it_point, = ax_bf.plot(0, self.all_data__best_fit[0], 'o', markersize=5, markerfacecolor='#ff880000', markeredgewidth=2, markeredgecolor='#ff0000ff')
        ax_bf.set_title("Best Fitness Vs Itteration")
        ax_bf.set_ylabel("Best Population Fitness")
        ax_bf.set_xlabel("Itteration")

        plt.tight_layout()
        plt.subplots_adjust(left=0.4, top=0.8)

        # # the function which updates the animation
        def animate(frame):
            #Animation function. Takes the current frame number (to select the potion of
            #data to plot) and a line selfect to update.

            # Not strictly neccessary, just so we know we are stealing these from
            # the global scope
            #global self.all_data, image, ax

            # We want up-to and _including_ the frame'th element
            image.set_array(self.all_data[frame].T)  # transposing the data so it is plotted correctly
            #ax_bf.clear()
            #ax_bf.plot(frame, self.all_data__best_fit[frame], 'o', markersize=5, markerfacecolor='#ff880000', markeredgewidth=2, markeredgecolor='#ff0000ff')
            it_point.set_ydata(self.all_data__best_fit[frame])
            it_point.set_xdata(frame)
            fig.suptitle("Animation (File %d, cir=%d, rep=%d) - Iteration %d, Best Fit = %.4f\nBest Genome: %s" % (self.dir_loop, cir, rep, frame, self.all_data__best_fit[frame], self.all_data__best_genome[frame]))

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
        if Save_NotShow == 1:
            if self.exp == 1:
                if self.ani_format == 'mp4':
                    fig_path = "%s/%s_%s_Animation_Cir%d_Rep%d.mp4" % (self.save_dir, self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], cir, rep)
                    the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
                elif self.ani_format == 'gif':
                    fig_path = "%s/%s_%s_Animation_Cir%d_Rep%d.gif" % (self.save_dir, self.new_FileRefNames[self.dir_loop], self.new_Param_array[self.dir_loop], cir, rep)
                    the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            else:
                if self.ani_format == 'mp4':
                    fig_path = "%s/Animation_Cir%d_Rep%d.mp4" % (self.save_dir, cir, rep)
                    the_animation.save(fig_path, writer=self.the_writer, dpi=self.dpi)
                elif self.ani_format == 'gif':
                    fig_path = "%s/Animation_Cir%d_Rep%d.gif" % (self.save_dir, cir, rep)
                    the_animation.save(fig_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            plt.close(fig)

        return the_animation
#

#

#

#


#

#

# fin
