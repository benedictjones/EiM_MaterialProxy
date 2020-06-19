# Import
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap
import numpy as np
import pandas as pd
import h5py
import pickle
import os

import multiprocessing

import time

# My imports
from Module_Settings.Set_Load import LoadSettings
from Module_LoadData.LoadData import Load_Data
from Module_Functions.FetchPerm import IndexPerm, NPerms

from Module_MG.RunCal_1Genome import run_calc_class1
from Module_MG.RunCal_ManyGenome import run_calc_class


''' NOTES
#   Matrix Bounds are saved with x values down the col, and y vals along the
    rows therefore it must be transposed befor an imshow plot!!

'''

# from TheSettings import settings


class materialgraphs(object):

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Initialisation of class
    '''
    def __init__(self, res_obj, GenMG_animation=0):

        # # Assign setting to self from setting dict file
        ParamDict = LoadSettings()
        self.ParamDict = ParamDict

        self.num_config = ParamDict['num_config']
        self.num_input = ParamDict['num_input']
        self.num_output = ParamDict['num_output']
        self.num_output_readings = ParamDict['num_output_readings']
        self.shuffle_gene = ParamDict['shuffle_gene']
        self.OutWeight_gene = ParamDict['OutWeight_gene']
        self.num_processors = ParamDict['num_processors']
        self.IntpScheme = ParamDict['IntpScheme']
        self.OutWeight_scheme = ParamDict['OutWeight_scheme']
        self.MaxOutputWeight = ParamDict['MaxOutputWeight']
        self.TestVerify = ParamDict['TestVerify']
        self.FitScheme = ParamDict['FitScheme']
        self.max_OutResponce = ParamDict['max_OutResponce']

        self.shuffle_on = 0  # used to force shuffle when plotting perm graphs
        self.OW_on = 0  # used to force shuffle when plotting perm graphs
        self.IN_on = 0  # used to force shuffle when plotting perm graphs

        self.in_weight_gene_loc = ParamDict['in_weight_gene_loc']
        self.config_gene_loc = ParamDict['config_gene_loc']
        self.out_weight_gene_loc = ParamDict['out_weight_gene_loc']
        self.SGI = ParamDict['SGI']

        self.EN_Dendrite = ParamDict['EN_Dendrite']  # Demdrite enabled
        self.NumDendrite = ParamDict['NumDendrite']  # Dendrite number (i.e network responce number)

        self.repetition_loop = ParamDict['repetition_loop']
        self.circuit_loop = ParamDict['circuit_loop']
        self.ReUse_dir = ParamDict['ReUse_dir']
        self.data_set = ParamDict['training_data']
        self.save_dir = ParamDict['SaveDir']
        self.num_nodes = self.num_input + self.num_config + self.num_output

        self.UseCustom_NewAttributeData = ParamDict['UseCustom_NewAttributeData']

        # # for a fine graph use in BestGenome Plot
        #the_max = 3.1
        #the_min = 0.5
        #interval = 0.05  # 0.05



        if self.data_set == '2DDS':
            the_max = 4.4
            the_min = -1
            interval = 0.1  # 0.05
        elif self.data_set == 'O2DDS':
            the_max = 4.2
            the_min = -3
            interval = 0.1  # 0.05
        elif self.data_set == 'con2DDS':
            the_max = 4
            the_min = -4
            interval = 0.1  # 0.05
        else:
            the_max = 3.5
            the_min = -3.5
            interval = 0.1  # 0.05

        # change if making an animation
        if GenMG_animation == 1:
            interval = 0.2  # set a larger interval is gening a gif

        max = the_max+interval
        self.x1 = np.arange(the_min, max, interval)
        self.x2 = np.arange(the_min, max, interval)

        # # for a fast graphs use in DefualtGenome Plot:
        the_max = 5
        the_min = -5
        interval = 0.2
        max = the_max+interval
        self.x1_fast = np.arange(the_min, max, interval)
        self.x2_fast = np.arange(the_min, max, interval)

        # # Assign passed in objects to self
        self.res_obj = res_obj




        # Load the training data
        train_data_X, train_data_Y = Load_Data('train', self.num_input, self.num_output_readings, self.data_set, 0, self.UseCustom_NewAttributeData)  # self.TestVerify st to 0, so all points plotted on MG
        self.training_data = train_data_X
        self.training_data_classes = train_data_Y

        # # Assign setting s to self
        #self.num_processors = self.num_processors
        self.num_processors = multiprocessing.cpu_count()  # set to the max for MG graphs

        # Define the cmap
        #basic_cols = ['#0000ff', '#5800ff', '#000000', '#ff5800', '#ff0000']  # red/blue
        #basic_cols = ['#009cff', '#ffffff', '#ff8800']  # pastal orange/blue
        basic_cols = ['#009cff', '#6d55ff', '#ffffff', '#ff6d55','#ff8800']  # pastal orange/red/white/purle/blue
        val = 0.2  # 0.2
        self.max_colour_val = val
        self.min_colour_val = -val
        # basic_cols=['#0000ff', '#000000', '#ff0000']
        self.my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)
        self.my_cmap.set_bad(color='#ffff00')

        self.dpi = 300

        self.bit_error = 0.00001  # the error to detect a class, i.e resolution of ADC
        self.interp = 'none'  # 'none' , 'gaussian'

        if self.IntpScheme == 'HOwins':
            self.interp = 'none'
            #basic_cols = ['#0000ff', '#ff0000']
            self.max_colour_val = 1
            self.min_colour_val = -1

#

    #

    #

    #

    #

    #

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Run the multiprocessing fobj function
    '''
    def Find_Class(self, pop_denorm_in, x1_data, x2_data, bound_dict):

        #print("")
        #print("pop_denorm_in")
        #print(pop_denorm_in)

        # # Divide the population in n chunks, where n = num_processors
        if pop_denorm_in.ndim == 1:
            pop_size = 1
            the_genome = pop_denorm_in
            alist = []
            alist.append(pop_denorm_in)
            pop_denorm_in = alist  # need to force into list  for formating
        elif pop_denorm_in.ndim == 2:
            pop_size = len(pop_denorm_in[:, 0])
        else:
            print("")
            print("ERROR (MG): Input numpy pop array is not 1d or 2d")
            exit()

        # detemin the number of number of chunks
        if pop_size < self.num_processors:
            num_chunks = pop_size
        else:
            num_chunks = self.num_processors

        # split up the population into chunks
        genome_chunk_list = np.array_split(pop_denorm_in, num_chunks)


        '''
        print("")
        print("genome_chunk_list")
        print("len len(gene_chunk_list)", len(genome_chunk_list))
        for j in range(len(genome_chunk_list)):
            print(genome_chunk_list[j])
            print("--------------------")

        '''

        # Produce the set of input data chunks to be evaluated
        x_in_list = []
        for i in range(len(x1_data)):
            x_in = np.zeros((len(x2_data), self.num_input))
            for j in range(len(x2_data)):
                x_in[j, 0] = x1_data[i]
                x_in[j, 1] = x2_data[j]
            # print("i", i, " for x_in", x_in)
            x_in_list.append(x_in)

        """# form custom param dict
        bound_dict = {}
        bound_dict['shuffle_on'] = self.shuffle_on
        bound_dict['OW_on'] = self.OW_on
        bound_dict['IN_on'] = self.IN_on"""
        bound_dict['len_x1'] = len(x1_data)
        bound_dict['len_x2'] = len(x2_data)
        bound_dict['num_processors'] = self.num_processors

        save_file = "Temp_MPdata/MG_param.dat"
        with open(save_file, 'wb') as outfile:
            pickle.dump(bound_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        # Evaluate the chunk of data inputs (using multiprcoessing)
        if pop_size > 1:
            ClassOut_all, responceY_all = run_calc_class(genome_chunk_list, pop_size, x_in_list)
        else:  # use special code which splits data into chunks for faster cal
            ClassOut_all, responceY_all = run_calc_class1(the_genome, x_in_list)

        #print("responceY_all:\n", responceY_all)


        # Extract the returned class values and plot in the input space
        k = 0
        bounds_matrix_list = []
        for p in range(pop_size):
            x_dim_bounds = 0
            bounds_matrix = np.zeros((len(x1_data), len(x2_data)))
            current_genome_responceY = responceY_all[k]

            for i in range(len(x1_data)):
                responceY = current_genome_responceY[i]
                #print("responceY:\n", responceY)

                y_dim_bounds = 0
                for j in range(len(x2_data)):
                    #print("class_res[j]", class_res[j])
                    #print("responceY[j][0]", responceY[j][0])
                    bounds_matrix[x_dim_bounds, y_dim_bounds] = responceY[j][0]

                    # increment
                    y_dim_bounds = y_dim_bounds + 1
                x_dim_bounds = x_dim_bounds + 1
            k = k + 1

            # save this genomes bounds data
            bounds_matrix_list.append(bounds_matrix)

        '''NOTE: Since MatrixBounds are saved this way, for imshow we must
        transpose the matrix before plotting, this is becuase imshow expects
        [y in cols, x in rows] formatted data '''

        return bounds_matrix_list
    #

    #

    #

    #

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Produce a masked Bounds Matrix, so 0 weighting is plotted as white
    '''
    def Mask_BoundMatrix(self, bounds_matrix):

        converted_bounds_matrix = np.zeros((len(bounds_matrix[:,0]), len(bounds_matrix[0,:])))
        ii = 0
        for row in bounds_matrix:
            jj = 0
            for col in row:
                if -self.bit_error <= col <= self.bit_error:
                    converted_bounds_matrix[ii,jj] = np.nan
                else:
                    converted_bounds_matrix[ii,jj] = bounds_matrix[ii,jj]
                jj = jj + 1
            ii = ii + 1

        masked_bounds_matrix = np.ma.masked_where(converted_bounds_matrix == np.nan, converted_bounds_matrix)

        '''
        print("bounds_matrix")
        print(bounds_matrix)

        print("converted_bounds_matrix")
        print(converted_bounds_matrix)

        print("masked_bounds_matrix")
        print(masked_bounds_matrix)
        '''

        return masked_bounds_matrix

    #

    #

    #

    # #######################################################################

    # #######################################################################

    # #######################################################################

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Material Graphs main function
    '''
    def MG(self, plot_defualt, last_res, it='na', GenMG_animation=0):

        if GenMG_animation == 1:
            print("\n Getting material graph data for animation...")
        else:
            print("\nProducing material graphs...")

        tic = time.time()

        # retrieve the best results
        best_genome = last_res[0]
        best_fitness_val = last_res[1]

        # find the bounds matrix representing the class and proximity to the class boundary
        bound_dict = self.ParamDict
        bounds_matrix_list = self.Find_Class(best_genome, self.x1, self.x2, bound_dict)
        bounds_matrix = bounds_matrix_list[0]

        # produce colour plot
        fig = plt.figure()

        # # Save data and return if only saving animation data
        if GenMG_animation == 1:
            ani_file = "%s/MG_AniData" % (self.save_dir)
            ani_file_dir = "%s/%d_Rep%d.hdf5" % (ani_file, self.circuit_loop, self.repetition_loop)
            if it == 0:
                # if the new folder does not yet exist, create it
                if not os.path.exists(ani_file):
                    os.makedirs(ani_file)

                with h5py.File(ani_file_dir, 'a') as hdf:
                    dataset_name = 'MG_dat_it_%d' % (it)
                    Y_dat = hdf.create_dataset(dataset_name, data=bounds_matrix)
                    Y_dat.attrs['best_fit'] = best_fitness_val
                    Y_dat.attrs['best_genome'] = best_genome


                    dataset_extent = [self.x1.min(),self.x1.max(),self.x2.min(), self.x2.max()]
                    hdf.create_dataset('extent', data=dataset_extent)

            else:
                with h5py.File(ani_file_dir, 'a') as hdf:
                    dataset_name = 'MG_dat_it_%d' % (it)
                    Y_dat = hdf.create_dataset(dataset_name, data=bounds_matrix)
                    Y_dat.attrs['best_fit'] = best_fitness_val
                    Y_dat.attrs['best_genome'] = best_genome

            return

        # # do a pixel colour plot
        plt.imshow(bounds_matrix.T, origin="lower",
                   extent=[self.x1.min(),self.x1.max(),self.x2.min(), self.x2.max()],
                   interpolation=self.interp, vmin=self.min_colour_val, vmax=self.max_colour_val, cmap=self.my_cmap)

        cbar = plt.colorbar()
        #cbar.set_alpha(1)
        #cbar.draw_all()

        # # form genome text
        rounded_best_genome = np.around(best_genome, decimals=3)
        print("best genome is:", rounded_best_genome)

        best_genome_text = '['
        for i in range(self.config_gene_loc[0], self.config_gene_loc[1]):
            best_genome_text = best_genome_text + str(rounded_best_genome[i])
            if i != self.config_gene_loc[1]-1:
                best_genome_text = best_genome_text + ', '
            else:
                best_genome_text = best_genome_text + ' '
        if self.SGI != 'na':
            best_genome_text = best_genome_text + '| '
            best_genome_text = best_genome_text + str(rounded_best_genome[self.SGI])
            best_genome_text = best_genome_text + ' '
        if self.in_weight_gene_loc != 'na':
            best_genome_text = best_genome_text + '| '
            for i in range(self.in_weight_gene_loc[0], self.in_weight_gene_loc[1]):
                best_genome_text = best_genome_text + str(rounded_best_genome[i])
                if i != self.in_weight_gene_loc[1]-1:
                    best_genome_text = best_genome_text + ', '
                else:
                    best_genome_text = best_genome_text + ' '
        if self.out_weight_gene_loc != 'na':
            best_genome_text = best_genome_text + '| '
            for i in range(self.out_weight_gene_loc[0], self.out_weight_gene_loc[1]):
                best_genome_text = best_genome_text + str(rounded_best_genome[i])
                if i != self.out_weight_gene_loc[1]-1:
                    best_genome_text = best_genome_text + ', '
                else:
                    best_genome_text = best_genome_text + ' '
        best_genome_text = best_genome_text + ']'

        #print("\nBest genome text:\n", best_genome_text)

        # Print plot lables and plot the class 'box'
        title_pt1 = 'Class outputs for the configured material (i.e best genome)\n'
        Best_Genome_str = 'Best Genome %s\nfitness %f' % (str(best_genome_text), best_fitness_val)
        #Best_Genome_str = 'Best Genome %s\nfitness %f' % (str(np.around(best_genome, decimals=3)), best_fitness_val)
        plt.suptitle(title_pt1, fontsize=7, fontweight='bold')
        plt.title(Best_Genome_str, fontsize=6)
        plt.xlabel('x1')
        plt.ylabel('x2')
        #plt.plot([1, 1.5, 1.5, 1, 1], [1.5, 1.5, 1, 1, 1.5], color='#6666ff')
        #plt.plot([1.6, 3, 3, 1.6, 1.6], [1.4, 1.4, 2.6, 2.6, 1.4], color='#ff6666')

        # plot training data too
        c = 0
        cl1_hit = 0
        cl2_hit = 0
        for row in self.training_data:

            if self.training_data_classes[c] == 1:
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

        # # Save figure and data (bounds_matrix)
        fig1_path = "%s/%d_Rep%d_FIG_best_genome.pdf" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        fig.savefig(fig1_path, dpi=self.dpi)

        # Close Fig
        plt.close(fig)

        # # write data to MG group, best gen sub group
        location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        with h5py.File(location, 'a') as hdf:
            G = hdf.create_group('MG')
            G_sub = G.create_group('BestGenome')

            G_sub.create_dataset('responceY', data=bounds_matrix)  # write BW

            x_data = np.array([self.x1, self.x2])
            G_sub.create_dataset('x_data', data=x_data)  # write xy data

            extent = [self.x1.min(), self.x1.max(), self.x2.min(), self.x2.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('the_best_genome', data=best_genome)  # write extent
            G_sub.create_dataset('best_fitness_val', data=best_fitness_val)  # write extent




        # # #

        # # #

        # # #

        # # #

        # # #

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Print the unconfigured material and it's material effect
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        if plot_defualt == 1:

            fig = plt.figure()  # produce figure

            # # # # # # # # # # # #
            # Create the "Defualt Geneome"
            defualt_genome = []

            bound_dict = self.ParamDict
            config_gene_loc = []
            # SGI - index for the shuffle gene (Shuffle Gene Index)
            in_weight_gene_loc = []
            out_weight_gene_loc = []
            j = 0

            config_gene_loc.append(j)
            for i in range(self.num_config):
                defualt_genome.append(0)
                j = j + 1
            config_gene_loc.append(j)
            bound_dict['config_gene_loc'] = config_gene_loc

            defualt_genome = np.array(defualt_genome)

            # # set other bits appropriatly
            bound_dict['shuffle_gene'] = 0
            bound_dict['InWeight_gene'] = 0
            bound_dict['OutWeight_gene'] = 0

            # # # # # # # # # # # # # # #
            # find the bounds matrix representing the class and proximity to the class boundary
            bounds_matrix_list = self.Find_Class(defualt_genome, self.x1_fast, self.x2_fast, bound_dict)
            bounds_matrix_defa = bounds_matrix_list[0]

            # produce colour plot
            fig = plt.figure()

            plt.imshow(bounds_matrix_defa.T, origin="lower",
                       extent=[self.x1_fast.min(),self.x1_fast.max(), self.x2_fast.min(), self.x2_fast.max()],
                       interpolation=self.interp,
                       vmin=self.min_colour_val, vmax=self.max_colour_val, cmap=self.my_cmap)
            plt.colorbar()

            # Print plot lables and plot the class 'box'
            title_pt1_def = 'Class outputs for the unconficured material\n(red = class 2, blue = class 1)\n'
            Defualt_Genome_str = 'Default Genome %s\n' % (str(np.around(defualt_genome, decimals=3)))
            plt.suptitle(title_pt1_def, fontsize=7, fontweight='bold')
            plt.title(Defualt_Genome_str, fontsize=6)
            plt.xlabel('x1')
            plt.ylabel('x2')

            # # Save figure
            fig2_path = "%s/%d_Rep%d_FIG_default_genome.pdf" % (self.save_dir, self.circuit_loop, self.repetition_loop)
            fig.savefig(fig2_path, dpi=self.dpi)

            # Close Fig
            plt.close(fig)

            # # write data to MG group, defualt gen sub group
            location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)
            with h5py.File(location, 'a') as hdf:
                G_sub = hdf.create_group('/MG/DefaultGenome')
                G_sub.create_dataset('responceY', data=bounds_matrix_defa)  # write BW

                x_data = np.array([self.x1_fast, self.x2_fast])
                G_sub.create_dataset('x_data', data=x_data)  # write xy data

                extent = [self.x1_fast.min(),self.x1_fast.max(), self.x2_fast.min(), self.x2_fast.max()]
                G_sub.create_dataset('extent', data=extent)  # write extent

                G_sub.create_dataset('default_genome', data=defualt_genome)  # write extent

        # # Print execution time
        toc = time.time()
        print("MG Graphs Finished, execution time:", toc - tic)
        plt.close('all')

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

    # #######################################################################

    # #######################################################################

    # #######################################################################

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Create a defualt genome, then produce graphs for varying Vconfig voltages
    '''
    def MG_VaryConfig(self):

        print("")
        print("Vary Vconfig for a defualt_genome ...")
        tic = time.time()

        # Set Paraeters
        num_config = self.num_config
        num_input = self.num_input
        num_output = self.num_output

        the_max = 5
        the_min = -5
        interval = 0.25
        max = the_max+interval
        x1_VC = np.arange(the_min, max, interval)
        x2_VC = np.arange(the_min, max, interval)



        # Produce a sub plot in a grid format
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels

        if num_config == 1:
            Vconfig_1 = np.array([-6, -3, 0, 3, 6])
            Vconfig_2 = np.array([0])
            rows, cols = len(Vconfig_1), len(Vconfig_2)
            fig, ax = plt.subplots(rows, sharex='col')
            cb_ax = fig.add_axes([0.8, 0.1, 0.02, 0.8])
        elif num_config == 2:
            Vconfig_1 = np.array([-6, -3, 0, 3, 6])
            Vconfig_2 = np.array([-6, -3, 0, 3, 6])
            rows, cols = len(Vconfig_1), len(Vconfig_2)
            fig, ax = plt.subplots(rows, cols,
                                   sharex='col',
                                   sharey='row')
            cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
            fig.subplots_adjust(hspace=0.2, wspace=0.05)
        else:
            print("")
            print("ERROR (MG): MG_VaryConfig() can only compute up to 2 inputs")


        # Produce the genome population to be examined
        bound_dict = self.ParamDict  # select parametrs for find class
        pop_list = []
        for row in range(rows):
            for col in range(cols):

                # # # # # # # # # # # #
                # Create the "Defualt Geneome"
                defualt_genome = []

                config_gene_loc = []
                # SGI - index for the shuffle gene (Shuffle Gene Index)
                in_weight_gene_loc = []
                out_weight_gene_loc = []
                j = 0

                config_gene_loc.append(j)
                for i in range(self.num_config):
                    if i == 0:
                        defualt_genome.append(Vconfig_1[row])
                    if i == 1:
                        defualt_genome.append(Vconfig_2[col])
                    j = j + 1
                config_gene_loc.append(j)
                bound_dict['config_gene_loc'] = config_gene_loc

                defualt_genome = np.array(defualt_genome)

                # # set other bits appropriatly
                bound_dict['shuffle_gene'] = 0
                bound_dict['InWeight_gene'] = 0
                bound_dict['OutWeight_gene'] = 0

                pop_list.append(defualt_genome)


        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.array(pop_list)
        bounds_matrix_list = self.Find_Class(pop_array, x1_VC, x2_VC, bound_dict)


        # # # # # # # # # # # #
        # Produce the sub plot graph
        k = 0
        for row in range(rows):
            for col in range(cols):

                bounds_matrix = bounds_matrix_list[k]

                if num_config == 1:  # for 1 Vconfig input
                    im = ax[row].imshow(bounds_matrix.T, origin="lower",
                                        extent=[x1_VC.min(),x1_VC.max(),x2_VC.min(), x2_VC.max()],
                                        interpolation=self.interp,
                                        vmin=self.min_colour_val, vmax=self.max_colour_val,
                                        cmap=self.my_cmap)
                    if col == 0:
                        ax[row].text(-(x1_VC.max()+abs(x1_VC.min())+0.2), (x2_VC.max()-abs(x2_VC.min()))/2.25, str(Vconfig_1[row]), size=12)

                    if row == 0:
                        # ax[row].text( (x1_VC.max()-abs(x1_VC.min()))/2.1, (x2_VC.max() + (x2_VC.max()-abs(x2_VC.min()))/6 ) , str(Vconfig_2[col]), size=12)
                        ax[row, col].set_title( str(Vconfig_2[col]), size=12)

                elif num_config == 2:  # for 2 Vconfig input
                    im = ax[row, col].imshow(bounds_matrix.T, origin="lower",
                                             extent=[x1_VC.min(),x1_VC.max(),x2_VC.min(), x2_VC.max()],
                                             interpolation=self.interp,
                                             vmin=self.min_colour_val, vmax=self.max_colour_val,
                                             cmap=self.my_cmap)

                    if row == 0:
                        # ax[row, col].text( (x1_VC.max()-abs(x1_VC.min()))/2.1, (x2_VC.max() + (x2_VC.max()-abs(x2_VC.min()))/6 ) , str(Vconfig_2[col]), size=12)
                        ax[row, col].set_title( str(Vconfig_2[col]), size=12)

                    # if row == rows-1:
                        # ax[row, col].set_xlabel('x1')

                    if col == 0:
                        ax[row, col].text(-((x1_VC.max()+abs(x1_VC.min()))+0.2), (x2_VC.max()-abs(x2_VC.min()))/2.25, str(Vconfig_1[row]), size=12)

                    # if col == 0:
                        # ax[row, col].set_ylabel('x2')

                #ax[row, col].set_title(str(pop_array[k]), fontsize=3, verticalalignment='top')

                # iterate to next matrix bound in the list
                k = k + 1




        # plot colour and title
        fig.colorbar(im, cax=cb_ax)
        fig.suptitle('Varying Vconfig on unconfigured material')


        # # Save figure
        fig3_path = "%s/%d_Rep%d_FIG_defualt_genome_data_Biased.pdf" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        fig.savefig(fig3_path, dpi=self.dpi)

        # Close Fig
        plt.close(fig)

        # Assign properties to be printed in Deets files
        self.len_x1_VC = len(x1_VC)
        self.len_x2_VC = len(x2_VC)
        self.Vconfig_1 = Vconfig_1
        self.Vconfig_2 = Vconfig_2

        # # write data to MG group, vary config sub group
        # # (FIG_defualt_genome_data_Biased)
        location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group('/MG/VaryConfig')

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            x_data = np.array([x1_VC, x2_VC])
            G_sub.create_dataset('x_data', data=x_data)  # write xy data

            extent = [x1_VC.min(), x1_VC.max(), x2_VC.min(), x2_VC.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('Vconfig_1', data=Vconfig_1)
            G_sub.create_dataset('Vconfig_2', data=Vconfig_2)


        # # Print execution time
        toc = time.time()
        print("Vary Vconfig Finished, execution time:", toc - tic)
        plt.close('all')
    #

    #

    #

    #

    #

    # #######################################################################

    # #######################################################################

    # #######################################################################

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Create a defualt genome, then produce graphs for different shuffle
    permutations, with fixed Vconfig voltages
    '''
    def MG_VaryPerm(self, OutWeight=0):

        print("")
        print("Vary Input perm's (shuffle) for a defualt_genome ...")
        tic = time.time()

        # Set Paraeters
        num_config = self.num_config
        num_input = self.num_input
        num_output = self.num_output
        num_output_readings = self.num_output_readings
        self.shuffle_on = 1

        the_max = 5
        the_min = -5
        interval = 0.25
        max = the_max+interval
        x1_PC = np.arange(the_min, max, interval)
        x2_PC = np.arange(the_min, max, interval)



        # Produce a sub plot in a grid format
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels

        # create agrid of sub plots
        rows, cols = 4, 4
        fig, ax = plt.subplots(rows, cols,
                               sharex='col',
                               sharey='row')
        cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
        fig.subplots_adjust(hspace=0.2, wspace=0.05)



        # Produce the genome population to be examined
        Num_Perms = NPerms(num_input+num_config)
        perm_options = np.arange(0, Num_Perms)
        perm_array = np.random.choice(perm_options, rows*cols, replace=False)  # select a random array of values from input array
        perm_list = []
        pop_list = []
        OW_list_list = []
        actual_PermArray_list = []

        bound_dict = self.ParamDict  # assign param for find class

        k = 0
        for row in range(rows):
            for col in range(cols):

                if k == 0:  # select defualt as first graph
                    perm = 0
                else:
                    # Randomly select some perms to plot
                    perm = perm_array[k]  # select a value from the random array

                # save perm to a list
                perm_list.append(perm)  # save the perms used

                # # # # # # # # # # # #
                # Create the "Defualt Geneome" with shuffle
                defualt_genome = []
                bound_dict['InWeight_gene'] = 0
                config_gene_loc = []
                # SGI - index for the shuffle gene (Shuffle Gene Index)
                out_weight_gene_loc = []
                i = 0

                config_gene_loc.append(i)
                for n in range(self.num_config):
                    defualt_genome.append(0)
                    i = i + 1
                config_gene_loc.append(i)
                bound_dict['config_gene_loc'] = config_gene_loc

                # Assign perm (i.e the shuffle of inputs)
                defualt_genome.append(perm)
                bound_dict['SGI'] = i
                bound_dict['shuffle_gene'] = 1
                i = i + 1

                # Assign no output weightings, or random output weightings
                num_output_weights = num_output*num_output_readings
                bound_dict['out_weight_gene_loc'] = [i, i+num_output_weights]
                OW_list = []  # make list to print weights on graph
                for j in range(num_output_weights):
                    # no output weightings
                    if OutWeight == 0:
                        bound_dict['OutWeight_gene'] = 0

                    elif OutWeight == 1:
                        bound_dict['OutWeight_gene'] = 1
                        nom_OW = np.random.uniform(0, 1)
                        # scale the random number, to randomly produce weighting
                        diff = np.fabs(self.MaxOutputWeight - (-self.MaxOutputWeight))  # absolute val of each el
                        OW = -self.MaxOutputWeight + nom_OW * diff  # pop with their real values

                        if k == 0:  # select defualt as first graph
                            defualt_genome.append(1)
                            OW_list.append(1)
                        else:  # random weights for the rest
                            if self.OutWeight_scheme == 'random':
                                OW_list.append(np.around(OW, decimals=2))
                                defualt_genome.append(OW)
                            elif self.OutWeight_scheme == 'AddSub':
                                if OW >= 0:
                                    OW_list.append(1)
                                    defualt_genome.append(1)
                                else:
                                    defualt_genome.append(-1)
                                    OW_list.append(-1)
                        i = i + 1


                OW_list_list.append(OW_list)
                pop_list.append(np.array(defualt_genome))

                # iterate k to select the next random perm
                k = k + 1


        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.array(pop_list)
        #print('pop_array\n', pop_array)
        bounds_matrix_list = self.Find_Class(pop_array, x1_PC, x2_PC, bound_dict)

        # # # # # # # # # # # #
        # Produce the sub plot graph
        hdf5_perm_list = []
        k = 0
        for row in range(rows):
            for col in range(cols):
                bounds_matrix = bounds_matrix_list[k]

                # set zero values to yellow
                bounds_matrix_MASKED = bounds_matrix
                #bounds_matrix_MASKED = self.Mask_BoundMatrix(bounds_matrix)

                # # # # # # # # # # # #
                # Produce the sub plot graph
                im = ax[row, col].imshow(bounds_matrix_MASKED.T, origin="lower", extent=[x1_PC.min(),x1_PC.max(),x2_PC.min(), x2_PC.max()],
                                         interpolation=self.interp, vmin=self.min_colour_val, vmax=self.max_colour_val, cmap=self.my_cmap)

                Current_perm = IndexPerm(num_input+num_config, perm_list[k])

                if OutWeight == 0:
                    subplot_title = 'Perm %s' % (str(Current_perm))
                elif OutWeight == 1:
                    subplot_title = 'Perm %s, OW: %s' % (str(Current_perm), str(np.array(OW_list_list[k])))
                    #subplot_title = str(pop_list[k])
                ax[row, col].set_title(subplot_title, fontsize=3, verticalalignment='top')
                actual_PermArray_list.append(subplot_title)

                hdf5_perm_list.append(Current_perm)

                if row == rows-1:
                    ax[row, col].set_xlabel('x1', fontsize=4)

                if col == 0:
                    ax[row, col].set_ylabel('x2', fontsize=4)

                # iterate to next matrix bound in the list
                k = k + 1

        # plot colour and title
        fig.colorbar(im, cax=cb_ax)
        if OutWeight == 0:
            fig.suptitle('Varying Permutation of inputs of an unconfigured material')
        elif OutWeight == 1:
            fig.suptitle('Varying Permutation of inputs of an unconfigured material,\nwith random output weights')


        # # Save figure
        fig4_path = "%s/%d_Rep%d_FIG_defualt_genome_data_PERMs.pdf" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        fig.savefig(fig4_path, dpi=self.dpi)

        # Close Fig
        plt.close(fig)

        # Assign properties to be printed in Deets files
        self.len_x1_PC = len(x1_PC)
        self.len_x2_PC = len(x2_PC)
        self.PC_num_chunks = rows*cols
        self.shuffle_on = 0

        # # write data to MG group, vary shuffle sub group
        # # (FIG_defualt_genome_data_Biased)
        location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group('/MG/VaryShuffle')

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            x_data = np.array([x1_PC, x2_PC])
            G_sub.create_dataset('x_data', data=x_data)  # write xy data
            G_sub.create_dataset('OutWeight', data=OutWeight)  # write xy data

            extent = [x1_PC.min(),x1_PC.max(),x2_PC.min(), x2_PC.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('Perm_list', data=hdf5_perm_list)
            G_sub.create_dataset('Weight_list', data=OW_list_list)
            r = G_sub.create_dataset('rows', data=rows)
            G_sub.create_dataset('cols', data=cols)
            r.attrs['deets'] = 'plot all cols then rows'



        # # Print execution time
        toc = time.time()
        print("Vary Perms Finished, execution time:", toc - tic)
        plt.close('all')

    #

    #

    #

    #

    #

    # #######################################################################

    # #######################################################################

    # #######################################################################

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Create a defualt genome, then produce graphs for different
    output weights
        > Only works with 2 output and one dimension
        > 4 plots produced of '1' or '-1' combinations (assending=0)
        > series of plots produced of weight values incrementing between
          '1' and '-1' (assending=1)
    '''
    def MG_VaryOutWeights(self, assending=0):

        print("")
        print("Vary Output Weights for a defualt_genome ...")
        tic = time.time()

        # Set Paraeters
        num_config = self.num_config
        num_input = self.num_input
        num_output = self.num_output
        num_output_readings = self.num_output_readings
        self.shuffle_on = 1
        num_output_weights = num_output*num_output_readings

        if num_output_weights != 2:
            print("MG_VaryWeights currently only works for 2 weighting outputs")
            print("Aborting")
            return

        the_max = 5
        the_min = -5
        interval = 0.25
        max = the_max+interval
        x1_PC = np.arange(the_min, max, interval)
        x2_PC = np.arange(the_min, max, interval)



        # Produce a sub plot in a grid format
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels

        # create agrid of sub plots
        if assending == 0:
            rows, cols = num_output_weights, num_output_weights
            fig, ax = plt.subplots(rows, cols,
                                   sharex='col',
                                   sharey='row')
        elif assending == 1:

            weight_list = [[-1, 1], [-0.5, 1], [0, 1], [0.5, 1],
                           [1, 1], [1, 0.5], [1, 0], [1, -0.5],
                           [1, -1], [0.5, -1], [0, -1], [-0.5, -1],
                           [-1, -1], [-1, -0.5], [-1, 0], [-1, 0.5]]


            rows, cols = 4, 4
            fig, ax = plt.subplots(rows, cols,
                                   sharex='col',
                                   sharey='row')


        cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
        fig.subplots_adjust(hspace=0.2, wspace=0.05)


        # Produce the genome population to be examined
        Num_Perms = NPerms(num_input+num_config)
        perm_options = np.arange(0, Num_Perms)
        perm_list = []
        pop_list = []
        OW_list_list = []
        actual_PermArray_list = []
        bound_dict = self.ParamDict  # assign param for find class
        k = 0
        for row in range(rows):
            for col in range(cols):

                # defualt perm/input order only
                perm = 0

                # save perm to a list
                perm_list.append(perm)  # save the perms used

                # # # # # # # # # # # #
                # Create the "Defualt Geneome" with shuffle
                defualt_genome = []
                bound_dict['InWeight_gene'] = 0
                config_gene_loc = []
                # SGI - index for the shuffle gene (Shuffle Gene Index)
                i = 0

                config_gene_loc.append(i)
                for n in range(self.num_config):
                    defualt_genome.append(0)
                    i = i + 1
                config_gene_loc.append(i)
                bound_dict['config_gene_loc'] = config_gene_loc

                # Assign NO perm (i.e the shuffle of inputs)
                bound_dict['SGI'] = 'na'
                bound_dict['shuffle_gene'] = 0

                # Assign no output weightings, or random output weightings
                bound_dict['OutWeight_gene'] = 1
                bound_dict['out_weight_gene_loc'] = [i, i+2]
                if assending == 0:
                    # no output weightings
                    if k == 0:
                        defualt_genome.append(1)
                        defualt_genome.append(1)
                        OW_list_list.append([1,1])
                    elif k == 1:
                        defualt_genome.append(1)
                        defualt_genome.append(-1)
                        OW_list_list.append([1, -1])
                    elif k == 2:
                        defualt_genome.append(-1)
                        defualt_genome.append(1)
                        OW_list_list.append([-1, 1])
                    elif k == 3:
                        defualt_genome.append(-1)
                        defualt_genome.append(-1)
                        OW_list_list.append([-1, -1])
                elif assending == 1:
                    weights = weight_list[k]
                    defualt_genome.append(weights[0])
                    defualt_genome.append(weights[1])
                    OW_list_list.append(weights)

                pop_list.append(np.array(defualt_genome))


                # iterate k to select the next random perm
                k = k + 1


        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.array(pop_list)
        #print(pop_array)
        bounds_matrix_list = self.Find_Class(pop_array, x1_PC, x2_PC, bound_dict)

        # # # # # # # # # # # #
        # Produce the sub plot graph
        k = 0
        for row in range(rows):
            for col in range(cols):
                bounds_matrix = bounds_matrix_list[k]

                # set zero values to yellow
                bounds_matrix_MASKED = bounds_matrix
                #bounds_matrix_MASKED = self.Mask_BoundMatrix(bounds_matrix)

                # # # # # # # # # # # #
                # Produce the sub plot graph
                im = ax[row, col].imshow(bounds_matrix_MASKED.T, origin="lower",
                                         extent=[x1_PC.min(),x1_PC.max(),x2_PC.min(), x2_PC.max()],
                                         interpolation=self.interp,
                                         vmin=self.min_colour_val, vmax=self.max_colour_val,
                                         cmap=self.my_cmap)


                #subplot_title = 'Weight: %s' % (str(OW_list_list[k]))
                subplot_title = str(pop_list[k])
                ax[row, col].set_title(subplot_title, fontsize=5, verticalalignment='top')

                Current_perm = IndexPerm(num_input+num_config, perm_list[k])
                actual_PermArray_list.append(Current_perm)

                if row == rows-1:
                    ax[row, col].set_xlabel('x1', fontsize=4)

                if col == 0:
                    ax[row, col].set_ylabel('x2', fontsize=4)

                # iterate to next matrix bound in the list
                k = k + 1

        # plot colour and title
        fig.colorbar(im, cax=cb_ax)
        fig.suptitle('Varying output weights for defualt material and no shuffle inputs')


        # # Save figure
        fig5_path = "%s/%d_Rep%d_FIG_defualt_genome_OutWeights.pdf" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        fig.savefig(fig5_path, dpi=self.dpi)

        # Close Fig
        plt.close(fig)

        # Assign properties to be printed in Deets files
        self.len_x1_PC = len(x1_PC)
        self.len_x2_PC = len(x2_PC)
        self.PC_num_chunks = rows*cols
        self.shuffle_on = 0

        # # write data to MG group, vary shuffle sub group
        # # (FIG_defualt_genome_data_Biased)
        location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        with h5py.File(location, 'a') as hdf:

            G_sub = hdf.create_group('/MG/VaryOutWeight')

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            x_data = np.array([x1_PC, x2_PC])
            G_sub.create_dataset('x_data', data=x_data)  # write xy data

            extent = [x1_PC.min(),x1_PC.max(),x2_PC.min(), x2_PC.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('perm_list', data=actual_PermArray_list)
            G_sub.create_dataset('Weight_list', data=OW_list_list)
            G_sub.create_dataset('rows', data=rows)
            G_sub.create_dataset('cols', data=cols)

        # # Print execution time
        toc = time.time()
        print("Vary Output Weights Finished, execution time:", toc - tic)
        plt.close('all')

    #

    #

    # #############################################################
    # do a finer sweep to produce an animation
    # #############################################################
    def MG_VaryOutWeightsAni(self):

        print("")
        print("Vary Output Weights for a defualt_genome animation ...")
        tic = time.time()

        # Set Paraeters
        num_output = self.num_output
        num_output_readings = self.num_output_readings
        self.shuffle_on = 1
        num_output_weights = num_output*num_output_readings

        if num_output_weights != 2:
            print("MG_VaryWeights currently only works for 2 weighting outputs")
            print("Aborting")
            return

        the_max = 5
        the_min = -5
        interval = 0.25
        max = the_max+interval
        x1 = np.arange(the_min, max, interval)
        x2 = np.arange(the_min, max, interval)



        # Produce a sub plot in a grid format
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels

        # create weights list
        r0 = np.arange(0,1.2,0.2)
        r = np.arange(0.2,1.2,0.2)
        r2 = np.arange(0,1,0.2)

        p1 = np.asarray([np.ones(len(r0)), np.flip(r0)])
        p2 = np.asarray([np.ones(len(r)), -r])

        p3 = np.asarray([np.flip(r2), -np.ones(len(r2))])
        p4 = np.asarray([-r, -np.ones(len(r))])

        p5 = np.asarray([-np.ones(len(r2)), -np.flip(r2)])
        p6 = np.asarray([-np.ones(len(r)), r])

        p7 = np.asarray([-np.flip(r2), np.ones(len(r2))])
        p8 = np.asarray([r, np.ones(len(r))])

        # produce weights list to analysis for
        weight_list = np.around(np.concatenate((p1, p2, p3, p4, p5 ,p6, p7, p8), axis=1), 2)

        # Produce the genome population to be examined
        pop_list = []
        OW_list_list = []
        bound_dict = self.ParamDict  # assign param for find class
        k = 0
        for num in range(len(weight_list[0,:])):

            # # # # # # # # # # # #
            # Create the "Defualt Geneome" with shuffle
            defualt_genome = []
            bound_dict['InWeight_gene'] = 0
            config_gene_loc = []
            # SGI - index for the shuffle gene (Shuffle Gene Index)
            i = 0

            config_gene_loc.append(i)
            for n in range(self.num_config):
                defualt_genome.append(0)
                i = i + 1
            config_gene_loc.append(i)
            bound_dict['config_gene_loc'] = config_gene_loc

            # Assign NO perm (i.e the shuffle of inputs)
            bound_dict['SGI'] = 'na'
            bound_dict['shuffle_gene'] = 0

            # Assign no output weightings, or random output weightings
            bound_dict['OutWeight_gene'] = 1
            bound_dict['out_weight_gene_loc'] = [i, i+2]

            weights = weight_list[:, k]
            defualt_genome.append(weights[0])
            defualt_genome.append(weights[1])
            OW_list_list.append(list(weights))

            pop_list.append(np.array(defualt_genome))

            # iterate k to select the next random perm
            k = k + 1

        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.array(pop_list)
        #print(pop_array)
        bounds_matrix_list = self.Find_Class(pop_array, x1, x2, bound_dict)

        # # write data to MG group
        location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        with h5py.File(location, 'a') as hdf:

            G_sub = hdf.create_group('/MG/VaryOutWeightAni')

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            x_data = np.array([x1, x2])
            G_sub.create_dataset('x_data', data=x_data)  # write xy data

            extent = [x1.min(),x1.max(),x2.min(), x2.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('Weight_list', data=OW_list_list)


        # # Print execution time
        toc = time.time()
        print("Vary Output Weights ANIMATION Finished, execution time:", toc - tic)
        plt.close('all')

    #

    #

    # #############################################################
    # do a finer sweep to produce an animation
    # #############################################################
    def MG_VaryLargeOutWeightsAni(self):

        print("")
        print("Vary LARGE Output Weights for a defualt_genome animation ...")
        tic = time.time()

        # Set Paraeters
        num_output = self.num_output
        num_output_readings = self.num_output_readings
        self.shuffle_on = 1
        num_output_weights = num_output*num_output_readings

        if num_output_weights != 2:
            print("MG_VaryWeights currently only works for 2 weighting outputs")
            print("Aborting")
            return

        the_max = 5
        the_min = -5
        interval = 0.25
        max = the_max+interval
        x1 = np.arange(the_min, max, interval)
        x2 = np.arange(the_min, max, interval)



        # Produce a sub plot in a grid format
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels

        # create weights list
        r0 = np.arange(0,1.2,0.2)
        r = np.arange(0.2,1.2,0.2)
        r2 = np.arange(0,1,0.2)

        p1 = np.asarray([np.ones(len(r0)), np.flip(r0)])
        p2 = np.asarray([np.ones(len(r)), -r])

        p3 = np.asarray([np.flip(r2), -np.ones(len(r2))])
        p4 = np.asarray([-r, -np.ones(len(r))])

        p5 = np.asarray([-np.ones(len(r2)), -np.flip(r2)])
        p6 = np.asarray([-np.ones(len(r)), r])

        p7 = np.asarray([-np.flip(r2), np.ones(len(r2))])
        p8 = np.asarray([r, np.ones(len(r))])

        # produce weights list to analysis for
        weight_list = np.around(np.concatenate((p1, p2, p3, p4, p5 ,p6, p7, p8), axis=1), 2)*10

        # Produce the genome population to be examined
        pop_list = []
        OW_list_list = []
        bound_dict = self.ParamDict  # assign param for find class
        k = 0
        for num in range(len(weight_list[0,:])):

            # # # # # # # # # # # #
            # Create the "Defualt Geneome" with shuffle
            defualt_genome = []
            bound_dict['InWeight_gene'] = 0
            config_gene_loc = []
            # SGI - index for the shuffle gene (Shuffle Gene Index)
            i = 0

            config_gene_loc.append(i)
            for n in range(self.num_config):
                defualt_genome.append(0)
                i = i + 1
            config_gene_loc.append(i)
            bound_dict['config_gene_loc'] = config_gene_loc

            # Assign NO perm (i.e the shuffle of inputs)
            bound_dict['SGI'] = 'na'
            bound_dict['shuffle_gene'] = 0

            # Assign no output weightings, or random output weightings
            bound_dict['OutWeight_gene'] = 1
            bound_dict['out_weight_gene_loc'] = [i, i+2]

            weights = weight_list[:, k]
            defualt_genome.append(weights[0])
            defualt_genome.append(weights[1])
            OW_list_list.append(list(weights))

            pop_list.append(np.array(defualt_genome))

            # iterate k to select the next random perm
            k = k + 1

        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.array(pop_list)
        #print(pop_array)
        bounds_matrix_list = self.Find_Class(pop_array, x1, x2, bound_dict)

        # # write data to MG group
        location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        with h5py.File(location, 'a') as hdf:

            G_sub = hdf.create_group('/MG/VaryLargeOutWeightAni')

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            x_data = np.array([x1, x2])
            G_sub.create_dataset('x_data', data=x_data)  # write xy data

            extent = [x1.min(),x1.max(),x2.min(), x2.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('Weight_list', data=OW_list_list)


        # # Print execution time
        toc = time.time()
        print("Vary LARGE Output Weights ANIMATION Finished, execution time:", toc - tic)
        plt.close('all')

    #

    #

    #

    #

    #

    # #######################################################################

    # #######################################################################

    # #######################################################################

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Create a defualt genome, then produce graphs for different
    input weights
        > Only works with 2 output and one dimension
        > 4 plots produced of '1' or '-1' combinations (assending=0)
        > series of plots produced of weight values incrementing between
          '1' and '-1' (assending=1)
    '''
    def MG_VaryInWeights(self, assending=0):

        print("")
        print("Vary Input Weights for a defualt_genome ...")
        tic = time.time()

        # Set Paraeters
        num_config = self.num_config
        num_input = self.num_input
        num_output = self.num_output
        num_output_readings = self.num_output_readings
        self.shuffle_on = 1
        num_output_weights = num_output*num_output_readings

        if num_output_weights != 2:
            print("MG_VaryWeights currently only works for 2 weighting outputs")
            print("Aborting")
            return

        the_max = 5
        the_min = -5
        interval = 0.25
        max = the_max+interval
        x1_PC = np.arange(the_min, max, interval)
        x2_PC = np.arange(the_min, max, interval)



        # Produce a sub plot in a grid format
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels

        # create agrid of sub plots
        if assending == 0:
            rows, cols = num_output_weights, num_output_weights
            fig, ax = plt.subplots(rows, cols,
                                   sharex='col',
                                   sharey='row')
        elif assending == 1:

            weight_list = [[-1, 1], [-0.5, 1], [0, 1], [0.5, 1],
                           [1, 1], [1, 0.5], [1, 0], [1, -0.5],
                           [1, -1], [0.5, -1], [0, -1], [-0.5, -1],
                           [-1, -1], [-1, -0.5], [-1, 0], [-1, 0.5]]


            rows, cols = 4, 4
            fig, ax = plt.subplots(rows, cols,
                                   sharex='col',
                                   sharey='row')


        cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
        fig.subplots_adjust(hspace=0.2, wspace=0.05)


        # Produce the genome population to be examined
        Num_Perms = NPerms(num_input+num_config)
        perm_options = np.arange(0, Num_Perms)
        perm_list = []
        pop_list = []
        OW_list_list = []
        actual_PermArray_list = []
        bound_dict = self.ParamDict  # assign param for find class
        k = 0
        for row in range(rows):
            for col in range(cols):

                # defualt perm/input order only
                perm = 0

                # save perm to a list
                perm_list.append(perm)  # save the perms used

                # # # # # # # # # # # #
                # Create the "Defualt Geneome" with shuffle
                defualt_genome = []
                bound_dict['OutWeight_gene'] = 0
                config_gene_loc = []
                # SGI - index for the shuffle gene (Shuffle Gene Index)
                i = 0

                config_gene_loc.append(i)
                for n in range(self.num_config):
                    defualt_genome.append(0)
                    i = i + 1
                config_gene_loc.append(i)
                bound_dict['config_gene_loc'] = config_gene_loc

                # Assign NO perm (i.e the shuffle of inputs)
                bound_dict['SGI'] = 'na'
                bound_dict['shuffle_gene'] = 0

                # Assign no output weightings, or random output weightings
                bound_dict['InWeight_gene'] = 1

                bound_dict['in_weight_gene_loc'] = [i, i+2]
                if assending == 0:
                    # no output weightings
                    if k == 0:
                        defualt_genome.append(1)
                        defualt_genome.append(1)
                        OW_list_list.append([1,1])
                    elif k == 1:
                        defualt_genome.append(1)
                        defualt_genome.append(-1)
                        OW_list_list.append([1, -1])
                    elif k == 2:
                        defualt_genome.append(-1)
                        defualt_genome.append(1)
                        OW_list_list.append([-1, 1])
                    elif k == 3:
                        defualt_genome.append(-1)
                        defualt_genome.append(-1)
                        OW_list_list.append([-1, -1])
                elif assending == 1:
                    weights = weight_list[k]
                    defualt_genome.append(weights[0])
                    defualt_genome.append(weights[1])
                    OW_list_list.append(weights)

                pop_list.append(np.array(defualt_genome))


                # iterate k to select the next random perm
                k = k + 1


        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.array(pop_list)
        #print(pop_array)
        bounds_matrix_list = self.Find_Class(pop_array, x1_PC, x2_PC, bound_dict)

        # # # # # # # # # # # #
        # Produce the sub plot graph
        k = 0
        for row in range(rows):
            for col in range(cols):
                bounds_matrix = bounds_matrix_list[k]

                # set zero values to yellow
                bounds_matrix_MASKED = bounds_matrix
                #bounds_matrix_MASKED = self.Mask_BoundMatrix(bounds_matrix)

                # # # # # # # # # # # #
                # Produce the sub plot graph
                im = ax[row, col].imshow(bounds_matrix_MASKED.T, origin="lower",
                                         extent=[x1_PC.min(),x1_PC.max(),x2_PC.min(), x2_PC.max()],
                                         interpolation=self.interp,
                                         vmin=self.min_colour_val, vmax=self.max_colour_val,
                                         cmap=self.my_cmap)


                #subplot_title = 'Weight: %s' % (str(OW_list_list[k]))
                subplot_title = str(pop_list[k])
                ax[row, col].set_title(subplot_title, fontsize=5, verticalalignment='top')

                Current_perm = IndexPerm(num_input+num_config, perm_list[k])
                actual_PermArray_list.append(Current_perm)

                if row == rows-1:
                    ax[row, col].set_xlabel('x1', fontsize=4)

                if col == 0:
                    ax[row, col].set_ylabel('x2', fontsize=4)

                # iterate to next matrix bound in the list
                k = k + 1

        # plot colour and title
        fig.colorbar(im, cax=cb_ax)
        fig.suptitle('Varying input weights for defualt material and no shuffle inputs')


        # # Save figure
        fig6_path = "%s/%d_Rep%d_FIG_defualt_genome_InWeights.pdf" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        fig.savefig(fig6_path, dpi=self.dpi)

        # Close Fig
        plt.close(fig)

        # Assign properties to be printed in Deets files
        self.len_x1_PC = len(x1_PC)
        self.len_x2_PC = len(x2_PC)
        self.PC_num_chunks = rows*cols
        self.shuffle_on = 0

        # # write data to MG group, vary shuffle sub group
        # # (FIG_defualt_genome_data_Biased)
        location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        with h5py.File(location, 'a') as hdf:

            G_sub = hdf.create_group('/MG/VaryInWeight')

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            x_data = np.array([x1_PC, x2_PC])
            G_sub.create_dataset('x_data', data=x_data)  # write xy data

            extent = [x1_PC.min(),x1_PC.max(),x2_PC.min(), x2_PC.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('perm_list', data=actual_PermArray_list)
            G_sub.create_dataset('Weight_list', data=OW_list_list)
            G_sub.create_dataset('rows', data=rows)
            G_sub.create_dataset('cols', data=cols)

        # # Print execution time
        toc = time.time()
        print("Vary Output Weights Finished, execution time:", toc - tic)
        plt.close('all')

    #

    #

    # #############################################################
    # do a finer sweep to produce an animation
    # #############################################################
    def MG_VaryInWeightsAni(self):

        print("")
        print("Vary Input Weights for a defualt_genome animation ...")
        tic = time.time()

        # Set Paraeters
        num_output = self.num_output
        num_output_readings = self.num_output_readings
        self.shuffle_on = 1
        num_output_weights = num_output*num_output_readings

        if num_output_weights != 2:
            print("MG_VaryWeights currently only works for 2 weighting outputs")
            print("Aborting")
            return

        the_max = 5
        the_min = -5
        interval = 0.25
        max = the_max+interval
        x1 = np.arange(the_min, max, interval)
        x2 = np.arange(the_min, max, interval)



        # Produce a sub plot in a grid format
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels

        # create weights list
        r0 = np.arange(0,1.2,0.2)
        r = np.arange(0.2,1.2,0.2)
        r2 = np.arange(0,1,0.2)

        p1 = np.asarray([np.ones(len(r0)), np.flip(r0)])
        p2 = np.asarray([np.ones(len(r)), -r])

        p3 = np.asarray([np.flip(r2), -np.ones(len(r2))])
        p4 = np.asarray([-r, -np.ones(len(r))])

        p5 = np.asarray([-np.ones(len(r2)), -np.flip(r2)])
        p6 = np.asarray([-np.ones(len(r)), r])

        p7 = np.asarray([-np.flip(r2), np.ones(len(r2))])
        p8 = np.asarray([r, np.ones(len(r))])

        weight_list = np.around(np.concatenate((p1, p2, p3, p4, p5 ,p6, p7, p8), axis=1), 2)


        # Produce the genome population to be examined
        pop_list = []
        OW_list_list = []
        bound_dict = self.ParamDict  # assign param for find class
        k = 0
        for num in range(len(weight_list[0,:])):

            # # # # # # # # # # # #
            # Create the "Defualt Geneome" with shuffle
            defualt_genome = []
            bound_dict['OutWeight_gene'] = 0
            config_gene_loc = []
            # SGI - index for the shuffle gene (Shuffle Gene Index)
            i = 0

            config_gene_loc.append(i)
            for n in range(self.num_config):
                defualt_genome.append(0)
                i = i + 1
            config_gene_loc.append(i)
            bound_dict['config_gene_loc'] = config_gene_loc

            # Assign NO perm (i.e the shuffle of inputs)
            bound_dict['SGI'] = 'na'
            bound_dict['shuffle_gene'] = 0

            # Assign no output weightings, or random output weightings
            bound_dict['InWeight_gene'] = 1
            bound_dict['in_weight_gene_loc'] = [i, i+2]

            weights = weight_list[:, k]
            defualt_genome.append(weights[0])
            defualt_genome.append(weights[1])
            OW_list_list.append(list(weights))

            pop_list.append(np.array(defualt_genome))

            # iterate k to select the next random perm
            k = k + 1

        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.array(pop_list)
        #print(pop_array)
        bounds_matrix_list = self.Find_Class(pop_array, x1, x2, bound_dict)

        # # write data to MG group
        location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        with h5py.File(location, 'a') as hdf:

            G_sub = hdf.create_group('/MG/VaryInWeightAni')

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            x_data = np.array([x1, x2])
            G_sub.create_dataset('x_data', data=x_data)  # write xy data

            extent = [x1.min(),x1.max(),x2.min(), x2.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('Weight_list', data=OW_list_list)


        # # Print execution time
        toc = time.time()
        print("Vary Output Weights ANIMATION Finished, execution time:", toc - tic)
        plt.close('all')

    #

    #

    # #############################################################
    # do a finer sweep to produce an animation
    # #############################################################
    def MG_VaryLargeInWeightsAni(self):

        print("")
        print("Vary Large Input Weights for a defualt_genome animation ...")
        tic = time.time()

        # Set Paraeters
        num_output = self.num_output
        num_output_readings = self.num_output_readings
        self.shuffle_on = 1
        num_output_weights = num_output*num_output_readings

        if num_output_weights != 2:
            print("MG_VaryWeights currently only works for 2 weighting outputs")
            print("Aborting")
            return

        the_max = 5
        the_min = -5
        interval = 0.25
        max = the_max+interval
        x1 = np.arange(the_min, max, interval)
        x2 = np.arange(the_min, max, interval)



        # Produce a sub plot in a grid format
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels

        # create weights list
        r0 = np.arange(0,1.2,0.2)
        r = np.arange(0.2,1.2,0.2)
        r2 = np.arange(0,1,0.2)

        p1 = np.asarray([np.ones(len(r0)), np.flip(r0)])
        p2 = np.asarray([np.ones(len(r)), -r])

        p3 = np.asarray([np.flip(r2), -np.ones(len(r2))])
        p4 = np.asarray([-r, -np.ones(len(r))])

        p5 = np.asarray([-np.ones(len(r2)), -np.flip(r2)])
        p6 = np.asarray([-np.ones(len(r)), r])

        p7 = np.asarray([-np.flip(r2), np.ones(len(r2))])
        p8 = np.asarray([r, np.ones(len(r))])

        weight_list = np.around(np.concatenate((p1, p2, p3, p4, p5 ,p6, p7, p8), axis=1), 2)*10


        # Produce the genome population to be examined
        pop_list = []
        OW_list_list = []
        bound_dict = self.ParamDict  # assign param for find class
        k = 0
        for num in range(len(weight_list[0,:])):

            # # # # # # # # # # # #
            # Create the "Defualt Geneome" with shuffle
            defualt_genome = []
            bound_dict['OutWeight_gene'] = 0
            config_gene_loc = []
            # SGI - index for the shuffle gene (Shuffle Gene Index)
            i = 0

            config_gene_loc.append(i)
            for n in range(self.num_config):
                defualt_genome.append(0)
                i = i + 1
            config_gene_loc.append(i)
            bound_dict['config_gene_loc'] = config_gene_loc

            # Assign NO perm (i.e the shuffle of inputs)
            bound_dict['SGI'] = 'na'
            bound_dict['shuffle_gene'] = 0

            # Assign no output weightings, or random output weightings
            bound_dict['InWeight_gene'] = 1
            bound_dict['in_weight_gene_loc'] = [i, i+2]

            weights = weight_list[:, k]
            defualt_genome.append(weights[0])
            defualt_genome.append(weights[1])
            OW_list_list.append(list(weights))

            pop_list.append(np.array(defualt_genome))

            # iterate k to select the next random perm
            k = k + 1

        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.array(pop_list)
        #print(pop_array)
        bounds_matrix_list = self.Find_Class(pop_array, x1, x2, bound_dict)

        # # write data to MG group
        location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        with h5py.File(location, 'a') as hdf:

            G_sub = hdf.create_group('/MG/VaryLargeInWeightAni')

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            x_data = np.array([x1, x2])
            G_sub.create_dataset('x_data', data=x_data)  # write xy data

            extent = [x1.min(),x1.max(),x2.min(), x2.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('Weight_list', data=OW_list_list)


        # # Print execution time
        toc = time.time()
        print("Vary Large Input Weights ANIMATION Finished, execution time:", toc - tic)
        plt.close('all')

    #

    #

    #

    #

    #

    #

    #

    #

    # #######################################################################

    # #######################################################################


    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Create a defualt genome, then produce graphs for varying Vconfig voltages
    and include a varyinf 3rd config voltage
    '''
    def MG_VaryConfig3(self):

        print("")
        print("Vary Vconfig 3 for a defualt_genome ...")
        tic = time.time()

        # Set Paraeters
        num_config = self.num_config
        num_input = self.num_input
        num_output = self.num_output

        the_max = 5
        the_min = -5
        interval = 0.25
        max = the_max+interval
        x1_VC = np.arange(the_min, max, interval)
        x2_VC = np.arange(the_min, max, interval)



        # Produce a sub plot in a grid format
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels

        # # produce multiple graphs for different Vconfig3
        for Vconfig_3 in [-6, -3, 0, 3, 6]:

            if num_config == 3:
                Vconfig_1 = np.array([-6, -3, 0, 3, 6])
                Vconfig_2 = np.array([-6, -3, 0, 3, 6])
                rows, cols = len(Vconfig_1), len(Vconfig_2)
                fig, ax = plt.subplots(rows, cols,
                                       sharex='col',
                                       sharey='row')
                cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
                fig.subplots_adjust(hspace=0.2, wspace=0.05)
            else:
                print("")
                print("ERROR (MG): MG_VaryConfig() can only compute up to 2 inputs")


            # Produce the genome population to be examined
            bound_dict = self.ParamDict  # select parametrs for find class
            pop_list = []
            for row in range(rows):
                for col in range(cols):

                    # # # # # # # # # # # #
                    # Create the "Defualt Geneome"
                    defualt_genome = []

                    config_gene_loc = []
                    # SGI - index for the shuffle gene (Shuffle Gene Index)
                    in_weight_gene_loc = []
                    out_weight_gene_loc = []
                    j = 0

                    config_gene_loc.append(j)
                    for i in range(self.num_config):
                        if i == 0:
                            defualt_genome.append(Vconfig_1[row])
                        if i == 1:
                            defualt_genome.append(Vconfig_2[col])
                        if i == 2:
                            defualt_genome.append(Vconfig_3)
                        j = j + 1
                    config_gene_loc.append(j)
                    bound_dict['config_gene_loc'] = config_gene_loc

                    defualt_genome = np.array(defualt_genome)

                    # # set other bits appropriatly
                    bound_dict['shuffle_gene'] = 0
                    bound_dict['InWeight_gene'] = 0
                    bound_dict['OutWeight_gene'] = 0

                    pop_list.append(defualt_genome)

            #print("pop_list:\n", pop_list)
            # find the bounds matrix representing the class and proximity to the class boundary
            pop_array = np.array(pop_list)
            bounds_matrix_list = self.Find_Class(pop_array, x1_VC, x2_VC, bound_dict)


            # # # # # # # # # # # #
            # Produce the sub plot graph
            k = 0
            for row in range(rows):
                for col in range(cols):

                    bounds_matrix = bounds_matrix_list[k]


                    im = ax[row, col].imshow(bounds_matrix.T, origin="lower",
                                             extent=[x1_VC.min(),x1_VC.max(),x2_VC.min(), x2_VC.max()],
                                             interpolation=self.interp,
                                             vmin=self.min_colour_val, vmax=self.max_colour_val,
                                             cmap=self.my_cmap)

                    if row == 0:
                        # ax[row, col].text( (x1_VC.max()-abs(x1_VC.min()))/2.1, (x2_VC.max() + (x2_VC.max()-abs(x2_VC.min()))/6 ) , str(Vconfig_2[col]), size=12)
                        ax[row, col].set_title( str(Vconfig_2[col]), size=12)

                    if col == 0:
                        ax[row, col].text(-((x1_VC.max()+abs(x1_VC.min()))+0.2), (x2_VC.max()-abs(x2_VC.min()))/2.25, str(Vconfig_1[row]), size=12)



                    # iterate to next matrix bound in the list
                    k = k + 1




            # plot colour and title
            fig.colorbar(im, cax=cb_ax)
            fig.suptitle('Varying Vconfig on unconfigured material, Vconfig3=%d' % (Vconfig_3))


            # # Save figure
            fig3_path = "%s/%d_Rep%d_FIG_DefualtGenomeBiased___Vconfig3_%d.png" % (self.save_dir, self.circuit_loop, self.repetition_loop, Vconfig_3)
            fig.savefig(fig3_path, dpi=self.dpi)

            # Close Fig
            plt.close(fig)

        """
        # Assign properties to be printed in Deets files
        self.len_x1_VC = len(x1_VC)
        self.len_x2_VC = len(x2_VC)
        self.Vconfig_1 = Vconfig_1
        self.Vconfig_2 = Vconfig_2

        # # write data to MG group, vary config sub group
        # # (FIG_defualt_genome_data_Biased)
        location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group('/MG/VaryConfig')

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            x_data = np.array([x1_VC, x2_VC])
            G_sub.create_dataset('x_data', data=x_data)  # write xy data

            extent = [x1_VC.min(), x1_VC.max(), x2_VC.min(), x2_VC.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('Vconfig_1', data=Vconfig_1)
            G_sub.create_dataset('Vconfig_2', data=Vconfig_2)

        """
        # # Print execution time
        toc = time.time()
        print("Vary Vconfig 3 Finished, execution time:", toc - tic)
        plt.close('all')
    #

#

# fin
