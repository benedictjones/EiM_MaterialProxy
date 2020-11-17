# Import
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap

import numpy as np
import pandas as pd
import h5py
import os

import multiprocessing

import time

# My imports
from mod_settings.Set_Load import LoadSettings
from mod_load.LoadData import Load_Data
from mod_methods.FetchPerm import IndexPerm, NPerms

from mod_material.eim_processor import material_processor



''' NOTES
#   Matrix Bounds are saved with x values down the col, and y vals along the
    rows therefore it must be transposed befor an imshow plot!!

'''

# from TheSettings import settings


class materialgraphs(object):

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Initialisation of class
    '''
    def __init__(self, cir, rep, GenMG_animation=0, param_file=''):

        # # Assign setting to self from setting dict file
        self.CompiledDict = LoadSettings(param_file)
        self.ParamDict = self.CompiledDict['DE']
        self.NetworkDict = self.CompiledDict['network']
        self.GenomeDict = self.CompiledDict['genome']

        self.num_processors = self.CompiledDict['num_processors']
        self.save_dir = self.CompiledDict['SaveDir']
        self.num_nodes = self.NetworkDict['num_input'] + self.NetworkDict['num_config'] + self.NetworkDict['num_output']
        self.UseCustom_NewAttributeData = self.ParamDict['UseCustom_NewAttributeData']

        self.cir = cir
        self.rep = rep
        #print("MG plots on: cir", cir, "rep", rep)

        # change if making an animation
        if GenMG_animation == 1:
            self.GenMG_animation = 1
            interval = 0.2  # set a larger interval is gening a gif
        else:
            self.GenMG_animation = 0
            interval = 0.1

        the_max = self.CompiledDict['spice']['Vmax']+1
        the_min = self.CompiledDict['spice']['Vmin']-1
        max = the_max+interval
        self.x1 = np.arange(the_min, max, interval)
        self.x2 = np.arange(the_min, max, interval)

        # # for a fast graphs use in DefualtGenome Plot:
        interval = 0.2
        max = the_max+interval
        self.x1_fast = np.arange(the_min, max, interval)
        self.x2_fast = np.arange(the_min, max, interval)

        interval = 0.2
        max = the_max+interval
        self.x1_sweep = np.arange(the_min, max, interval)
        self.x2_sweep = np.arange(the_min, max, interval)

        # Load the training data
        train_data_X, train_data_Y = Load_Data('train', self.CompiledDict)  # self.ParamDict['TestVerify'] st to 0, so all points plotted on MG
        self.training_data = train_data_X
        self.training_data_classes = train_data_Y

        # # Assign setting s to self
        #self.num_processors = self.num_processors
        self.num_processors = multiprocessing.cpu_count()  # set to the max for MG graphs
        if self.num_processors >= 10:
            self.num_processors = 10

        # Define the cmap
        #basic_cols = ['#0000ff', '#5800ff', '#000000', '#ff5800', '#ff0000']  # red/blue
        #basic_cols = ['#009cff', '#ffffff', '#ff8800']  # pastal orange/blue
        basic_cols = ['#009cff', '#6d55ff', '#ffffff', '#ff6d55','#ff8800']  # pastal orange/red/white/purle/blue
        val = 0.2  # 0.2
        self.max_colour_val = val
        self.min_colour_val = -val
        # basic_cols=['#0000ff', '#000000', '#ff0000']
        self.my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

        self.dpi = 300

        self.interp = 'none'  # 'none' , 'gaussian'

        if self.ParamDict['IntpScheme'] == 'HOwins':
            self.interp = 'none'
            #basic_cols = ['#0000ff', '#ff0000']
            self.max_colour_val = 1
            self.min_colour_val = -1

        # compute number of output weights that would be needed
        if self.CompiledDict['DE']['num_readout_nodes'] != 'na':
            self.num_output_weights = self.CompiledDict['network']['num_output']*self.CompiledDict['DE']['num_readout_nodes']
        else:
            self.num_output_weights = self.CompiledDict['network']['num_output']

        return

#

    #

    #

    def clean_genome(self, Dict):

        Dict['genome']['shuffle_gene'] = 0
        Dict['genome']['InWeight_gene'] = 0
        Dict['genome']['OutWeight_gene'] = 0

        Dict['genome']['PulseWidth_gene'] = 0

        Dict['genome']['BandNumber_gene'] = 0
        Dict['genome']['BandEdge_gene'] = 0
        Dict['genome']['BandWidth_gene'] = 0
        Dict['genome']['BandClass_gene'] = 0

        Dict['genome']['HOWperm_gene'] = 0

        Dict['genome']['NN_weight_loc'] = 'na'

        return Dict

    #

    #

    #

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Run the multiprocessing fobj function
    '''
    def Find_Class(self, pop_denorm_in, x1_data, x2_data, bound_dict):

        # # Divide the population in n chunks, where n = num_processors
        genome_list = np.asarray(pop_denorm_in)

        # Produce the set of input data chunks to be evaluated
        x_in_list = []
        for i in range(len(x1_data)):
            x_in = np.zeros((len(x2_data), self.NetworkDict['num_input']))
            for j in range(len(x2_data)):
                x_in[j, 0] = x1_data[i]
                x_in[j, 1] = x2_data[j]
            x_in_list.append(x_in)

        x_in_list = np.asarray(x_in_list)
        x_in = np.concatenate(x_in_list)


        # form custom param dict
        bound_dict['len_x1'] = len(x1_data)
        bound_dict['len_x2'] = len(x2_data)
        bound_dict['num_processors'] = self.num_processors

        cap = material_processor(bound_dict)
        cap.gen_material(self.cir, load_only=1)  # load material
        results = cap.run_processor(genome_list, self.cir, self.rep, x_in, the_data='mg', ret_str='unzip')
        responceY_Glist_unshaped = results[1]
        Vop_Glist_unshaped = results[2]

        responceY_Glist = []
        Vop_Glist = []
        for gen in range(len(genome_list)):
            rY_unshaped = responceY_Glist_unshaped[gen]
            Vop_unshaped = Vop_Glist_unshaped[gen]
            responceY_Glist.append(np.reshape(rY_unshaped, (len(x1_data), len(x2_data))))
            ops = []
            for op_node in range(len(Vop_unshaped[0,:])):
                ops.append(np.reshape(Vop_unshaped[:, op_node], (len(x1_data), len(x2_data))))
            Vop_Glist.append(ops)

            # print("op", gen, "shape:", op_Glist[gen].shape)
            # print("ResponceY", gen, "shape:", responceY_Glist[gen].shape)

        '''NOTE: Since MatrixBounds are saved this way, for imshow we must
        transpose the matrix before plotting, this is becuase imshow expects
        [y in cols, x in rows] formatted data '''

        return responceY_Glist, Vop_Glist
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
    Material Graphs main function
    '''
    def MG(self, plot_defualt, last_res, it='na'):

        if self.GenMG_animation == 1:
            print("\nGetting material graph data for animation...")
        else:
            print("\nProducing material graphs...")
            print("MG best genome")

        tic = time.time()

        # retrieve the best results
        best_genome = last_res[0]
        best_fitness_val = last_res[1]
        self.best_genome = best_genome


        # find the bounds matrix representing the class and proximity to the class boundary
        custom_Dict = self.CompiledDict

        x1_data = self.x1
        x2_data = self.x2
        bounds_matrix_list, op_list_Glist = self.Find_Class([best_genome], x1_data, x2_data, custom_Dict)
        bounds_matrix = bounds_matrix_list[0]

        """op_list = op_list_Glist[0]
        op1 = op_list[0]
        print(op1)
        print(op1.shape)
        exit()"""

        # # Save data and return if only saving animation data
        if self.GenMG_animation == 1:

            ani_file = "%s/MG_AniData" % (self.save_dir)
            ani_file_dir = "%s/%d_Rep%d.hdf5" % (ani_file, self.cir, self.rep)
            if it == 0:
                # if the new folder does not yet exist, create it
                if not os.path.exists(ani_file):
                    os.makedirs(ani_file)
                with h5py.File(ani_file_dir, 'a') as hdf:
                    dataset_name = 'MG_dat_it_%d' % (it)
                    Y_dat = hdf.create_dataset(dataset_name, data=bounds_matrix)
                    Y_dat.attrs['best_fit'] = best_fitness_val
                    Y_dat.attrs['best_genome'] = str(best_genome)
                    dataset_extent = [self.x1.min(),self.x1.max(),self.x2.min(), self.x2.max()]
                    hdf.create_dataset('extent', data=dataset_extent)
            else:
                with h5py.File(ani_file_dir, 'a') as hdf:
                    dataset_name = 'MG_dat_it_%d' % (it)
                    Y_dat = hdf.create_dataset(dataset_name, data=bounds_matrix)
                    Y_dat.attrs['best_fit'] = best_fitness_val
                    Y_dat.attrs['best_genome'] = str(best_genome)
            return

        # # write data to MG group, best gen sub group
        location = "%s/data.hdf5" % (self.save_dir)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group("%d_rep%d/MG/BestGenome" % (self.cir, self.rep))


            G_sub.create_dataset('responceY', data=bounds_matrix)  # write BW

            G_sub.create_dataset('x1_data', data=self.x1)  # write xy data
            G_sub.create_dataset('x2_data', data=self.x2)  # write xy data

            extent = [self.x1.min(), self.x1.max(), self.x2.min(), self.x2.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('the_best_genome', data=np.float64(np.concatenate(best_genome)))  # write extent
            G_sub.create_dataset('gen_grouping', data=self.GenomeDict['grouping'])

            G_sub.create_dataset('best_fitness_val', data=best_fitness_val)
            G_sub.create_dataset('op_list', data=np.asarray(op_list_Glist[0]))

        # # #

        # # #

        # # #

        # # #

        # # #

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Print the unconfigured material and it's material effect
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        if plot_defualt == 1:
            print("\nMG Default genome")

            # # # # # # # # # # # #
            # Create the "Defualt Geneome"
            defualt_genome = []

            custom_Dict = self.CompiledDict
            custom_Dict = self.clean_genome(custom_Dict)

            temp = []
            for i in range(self.NetworkDict['num_config']):
                temp.append(0)
            defualt_genome.append(np.asarray(temp))
            custom_Dict['genome']['config_gene_loc'] = 0

            # # # # # # # # # # # # # # #
            # find the bounds matrix representing the class and proximity to the class boundary

            x1_data = self.x1_fast
            x2_data = self.x2_fast
            bounds_matrix_list, op_list_Glist = self.Find_Class(np.asarray([defualt_genome]), x1_data, x2_data, custom_Dict)
            bounds_matrix_defa = bounds_matrix_list[0]

            # # write data to MG group, defualt gen sub group
            location = "%s/data.hdf5" % (self.save_dir)
            with h5py.File(location, 'a') as hdf:
                G_sub = hdf.create_group("%d_rep%d/MG/DefaultGenome" % (self.cir, self.rep))

                G_sub.create_dataset('responceY', data=bounds_matrix_defa)  # write BW

                G_sub.create_dataset('x1_data', data=self.x1_fast)  # write xy data
                G_sub.create_dataset('x2_data', data=self.x2_fast)  # write xy data


                extent = [self.x1_fast.min(),self.x1_fast.max(), self.x2_fast.min(), self.x2_fast.max()]
                G_sub.create_dataset('extent', data=extent)  # write extent
                G_sub.create_dataset('default_genome', data=np.concatenate(defualt_genome))  # write extent
                G_sub.create_dataset('gen_grouping', data=[self.NetworkDict['num_config']])



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

        x1_VC = self.x1_sweep
        x2_VC = self.x2_sweep



        # Produce a sub plot in a grid format
        plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=7)    # fontsize of the tick labels

        if self.NetworkDict['num_config'] == 1:
            Vconfig_1 = np.asarray([-6, -3, 0, 3, 6])
            Vconfig_1 = np.asarray([0, 2, 4, 6])
            Vconfig_2 = np.asarray([0])
            rows, cols = len(Vconfig_1), len(Vconfig_2)
            fig, ax = plt.subplots(rows, sharex='col')
            cb_ax = fig.add_axes([0.8, 0.1, 0.02, 0.8])
        elif self.NetworkDict['num_config'] == 2:
            Vconfig_1 = np.asarray([-6, -3, 0, 3, 6])
            Vconfig_2 = np.asarray([6, 3, 0, -3, -6])
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

        custom_Dict = self.CompiledDict
        custom_Dict = self.clean_genome(custom_Dict)
        custom_Dict['genome']['config_gene_loc'] = 0

        pop_list = []
        for row in range(rows):
            for col in range(cols):

                # # # # # # # # # # # #
                # Create the "Defualt Geneome"
                defualt_genome = []

                j = 0
                temp = []
                for i in range(self.NetworkDict['num_config']):
                    if i == 0:
                        temp.append(Vconfig_1[row])
                    if i == 1:
                        temp.append(Vconfig_2[col])
                    j = j + 1

                defualt_genome.append(np.asarray(temp))
                pop_list.append(defualt_genome)


        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.asarray(pop_list)

        responceY_list, op_list_Glist = self.Find_Class(pop_array, x1_VC, x2_VC, custom_Dict)

        # # write data to MG group, vary config sub group
        # # (FIG_defualt_genome_data_Biased)
        location = "%s/data.hdf5" % (self.save_dir)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group("%d_rep%d/MG/VaryConfig" % (self.cir, self.rep))

            rY_list = G_sub.create_dataset('responceY_list', data=responceY_list)  # write BW
            G_sub.create_dataset('op_list_Glist', data=op_list_Glist)  # write BW
            rY_list.attrs['num_plots'] = len(responceY_list)

            G_sub.create_dataset('x1_data', data=x1_VC)  # write xy data
            G_sub.create_dataset('x2_data', data=x2_VC)  # write xy data

            extent = [x1_VC.min(), x1_VC.max(), x2_VC.min(), x2_VC.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('Vconfig_1', data=Vconfig_1)
            G_sub.create_dataset('Vconfig_2', data=Vconfig_2)
            G_sub.create_dataset('pop_array', data=pop_array)


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

        x1_PC = self.x1_sweep
        x2_PC = self.x2_sweep

        # select a size for grid of sub plots
        rows, cols = 4, 4




        # Produce the genome population to be examined
        Num_Perms = NPerms(self.NetworkDict['num_input']+self.NetworkDict['num_config'])
        perm_options = np.arange(0, Num_Perms)
        if rows*cols >= Num_Perms:
            rows = 2
            cols = 3
        perm_array = np.random.choice(perm_options, rows*cols, replace=False)  # select a random asarray of values from input asarray
        perm_list = []
        pop_list = []
        OW_list_list = []

        custom_Dict = self.CompiledDict
        custom_Dict = self.clean_genome(custom_Dict)
        custom_Dict['genome']['config_gene_loc'] = 0
        custom_Dict['genome']['shuffle_gene'] = 1
        custom_Dict['genome']['SGI'] = 1

        k = 0
        for row in range(rows):
            for col in range(cols):

                if k == 0:  # select defualt as first graph
                    perm = 0
                else:
                    # Randomly select some perms to plot
                    perm = perm_array[k]  # select a value from the random asarray

                # save perm to a list
                perm_list.append(perm)  # save the perms used

                # # # # # # # # # # # #
                # Create the "Defualt Geneome" with shuffle
                defualt_genome = []

                temp = []
                for i in range(self.NetworkDict['num_config']):
                    temp.append(0)
                defualt_genome.append(np.asarray(temp, dtype=object))

                # Assign perm (i.e the shuffle of inputs)
                defualt_genome.append(np.asarray([perm]))


                # Assign no output weightings, or random output weightings
                # num_output_weights = self.NetworkDict['num_output']*self.ParamDict['num_output_readings']

                # no output weightings
                if OutWeight == 0:
                    custom_Dict['genome']['OutWeight_gene'] = 0

                elif OutWeight == 1:
                    custom_Dict['genome']['OutWeight_gene'] = 1
                    custom_Dict['genome']['out_weight_gene_loc'] = 2
                    OW_list = []  # make list to print weights on graph
                    temp = []
                    for j in range(self.num_output_weights):

                        nom_OW = np.random.uniform(0, 1)
                        # scale the random number, to randomly produce weighting
                        diff = np.fabs(self.GenomeDict['MaxOutputWeight'] - (-self.GenomeDict['MaxOutputWeight']))  # absolute val of each el
                        OW = -self.GenomeDict['MaxOutputWeight'] + nom_OW * diff  # pop with their real values

                        if k == 0:  # select defualt as first graph
                            temp.append(1)
                            OW_list.append(1)
                        else:  # random weights for the rest
                            if self.GenomeDict['OutWeight_scheme'] == 'random':
                                OW_list.append(np.around(OW, decimals=2))
                                temp.append(OW)
                            elif self.GenomeDict['OutWeight_scheme'] == 'AddSub':
                                if OW >= 0:
                                    OW_list.append(1)
                                    temp.append(1)
                                else:
                                    temp.append(-1)
                                    OW_list.append(-1)


                    defualt_genome.append(np.asarray(temp, dtype=object))
                    OW_list_list.append(OW_list)
                pop_list.append(np.asarray(defualt_genome, dtype=object))

                # iterate k to select the next random perm
                k = k + 1


        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.asarray(pop_list)
        #print('pop_array\n', pop_array)
        bounds_matrix_list, op_list_Glist = self.Find_Class(pop_array, x1_PC, x2_PC, custom_Dict)

        # # # # # # # # # # # #
        # Produce the sub plot graph
        hdf5_perm_list = []
        k = 0
        for row in range(rows):
            for col in range(cols):
                Current_perm = IndexPerm(self.NetworkDict['num_input']+self.NetworkDict['num_config'], perm_list[k])
                hdf5_perm_list.append(Current_perm)
                k = k + 1

        # # write data to MG group, vary shuffle sub group
        # # (FIG_defualt_genome_data_Biased)
        location = "%s/data.hdf5" % (self.save_dir)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group("%d_rep%d/MG/VaryShuffle" % (self.cir, self.rep))

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            G_sub.create_dataset('x1_data', data=x1_PC)  # write xy data
            G_sub.create_dataset('x2_data', data=x2_PC)  # write xy data
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
        if self.num_output_weights != 2:
            print("MG_VaryOutWeights currently only works for 2 weighting outputs")
            print("Aborting")
            return

        x1_data = self.x1_sweep
        x2_data = self.x2_sweep

        # create agrid of sub plots
        if assending == 0:
            rows, cols = self.num_output_weights, self.num_output_weights
        elif assending == 1:
            weight_list = [[-1, 1], [-0.5, 1], [0, 1], [0.5, 1],
                           [1, 1], [1, 0.5], [1, 0], [1, -0.5],
                           [1, -1], [0.5, -1], [0, -1], [-0.5, -1],
                           [-1, -1], [-1, -0.5], [-1, 0], [-1, 0.5]]
            rows, cols = 4, 4

        # Produce the genome population to be examined
        Num_Perms = NPerms(self.NetworkDict['num_input']+self.NetworkDict['num_config'])
        perm_options = np.arange(0, Num_Perms)
        perm_list = []
        pop_list = []
        OW_list_list = []
        actual_PermArray_list = []
        custom_Dict = self.CompiledDict
        custom_Dict = self.clean_genome(custom_Dict)
        custom_Dict['genome']['config_gene_loc'] = 0
        custom_Dict['genome']['OutWeight_gene'] = 1
        custom_Dict['genome']['out_weight_gene_loc'] = 1

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

                temp = []
                for i in range(self.NetworkDict['num_config']):
                    temp.append(0)

                defualt_genome.append(np.asarray(temp, dtype=object))

                # Assign no output weightings, or random output weightings
                temp = []
                if assending == 0:
                    # no output weightings
                    if k == 0:
                        temp.append(1)
                        temp.append(1)
                        OW_list_list.append([1,1])
                    elif k == 1:
                        temp.append(1)
                        temp.append(-1)
                        OW_list_list.append([1, -1])
                    elif k == 2:
                        temp.append(-1)
                        temp.append(1)
                        OW_list_list.append([-1, 1])
                    elif k == 3:
                        temp.append(-1)
                        temp.append(-1)
                        OW_list_list.append([-1, -1])
                elif assending == 1:
                    weights = weight_list[k]
                    temp.append(weights[0])
                    temp.append(weights[1])
                    OW_list_list.append(weights)
                defualt_genome.append(np.asarray(temp, dtype=object))

                pop_list.append(np.asarray(defualt_genome, dtype=object))


                # iterate k to select the next random perm
                k = k + 1


        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.asarray(pop_list)
        #print(pop_array)
        bounds_matrix_list, op_list_Glist = self.Find_Class(pop_array, x1_data, x2_data, custom_Dict)

        # # write data to MG group, vary shuffle sub group
        # # (FIG_defualt_genome_data_Biased)
        location = "%s/data.hdf5" % (self.save_dir)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group("%d_rep%d/MG/VaryOutWeight" % (self.cir, self.rep))

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            G_sub.create_dataset('x1_data', data=x1_data)  # write xy data
            G_sub.create_dataset('x2_data', data=x2_data)  # write xy data

            extent = [x1_data.min(),x1_data.max(),x2_data.min(), x2_data.max()]
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
        if self.num_output_weights != 2:
            print("MG_VaryWeights currently only works for 2 weighting outputs")
            print("Aborting")
            return

        x1 = self.x1_sweep
        x2 = self.x2_sweep

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
        custom_Dict = self.CompiledDict
        custom_Dict = self.clean_genome(custom_Dict)
        custom_Dict['genome']['config_gene_loc'] = 0
        custom_Dict['genome']['OutWeight_gene'] = 1
        custom_Dict['genome']['out_weight_gene_loc'] = 1

        k = 0
        for num in range(len(weight_list[0,:])):

            # # # # # # # # # # # #
            # Create the "Defualt Geneome" with shuffle
            defualt_genome = []

            temp = []
            for i in range(self.NetworkDict['num_config']):
                temp.append(0)
            #temp[-1] = 2  # set last config to non-zero
            defualt_genome.append(np.asarray(temp, dtype=object))

            temp = []
            weights = weight_list[:, k]
            temp.append(weights[0])
            temp.append(weights[1])
            OW_list_list.append(list(weights))
            defualt_genome.append(np.asarray(temp, dtype=object))

            pop_list.append(np.asarray(defualt_genome, dtype=object))

            # iterate k to select the next random perm
            k = k + 1

        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.asarray(pop_list)
        #print(pop_array)
        bounds_matrix_list, op_list_Glist = self.Find_Class(pop_array, x1, x2, custom_Dict)

        # # write data to MG group
        location = "%s/data.hdf5" % (self.save_dir)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group("%d_rep%d/MG/VaryOutWeightAni" % (self.cir, self.rep))

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            G_sub.create_dataset('x1_data', data=x1)  # write xy data
            G_sub.create_dataset('x2_data', data=x2)  # write xy data

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
        if self.num_output_weights != 2:
            print("MG_VaryWeights currently only works for 2 weighting outputs")
            print("Aborting")
            return

        x1 = self.x1_sweep
        x2 = self.x2_sweep

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
        custom_Dict = self.CompiledDict
        custom_Dict = self.clean_genome(custom_Dict)
        custom_Dict['genome']['config_gene_loc'] = 0
        custom_Dict['genome']['OutWeight_gene'] = 1
        custom_Dict['genome']['out_weight_gene_loc'] = 1

        k = 0
        for num in range(len(weight_list[0,:])):

            # # # # # # # # # # # #
            # Create the "Defualt Geneome" with shuffle
            defualt_genome = []

            temp = []
            for i in range(self.NetworkDict['num_config']):
                temp.append(0)
            defualt_genome.append(np.asarray(temp, dtype=object))

            temp = []
            weights = weight_list[:, k]
            temp.append(weights[0])
            temp.append(weights[1])
            OW_list_list.append(list(weights))
            defualt_genome.append(np.asarray(temp, dtype=object))

            pop_list.append(np.asarray(defualt_genome, dtype=object))

            # iterate k to select the next random perm
            k = k + 1

        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.asarray(pop_list)
        #print(pop_array)
        bounds_matrix_list, op_list_Glist = self.Find_Class(pop_array, x1, x2, custom_Dict)

        # # write data to MG group
        location = "%s/data.hdf5" % (self.save_dir)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group("%d_rep%d/MG/VaryLargeOutWeightAni" % (self.cir, self.rep))

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            G_sub.create_dataset('x1_data', data=x1)  # write xy data
            G_sub.create_dataset('x2_data', data=x2)  # write xy data

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
        if self.NetworkDict['num_input'] != 2:
            print("MG_VaryWeights currently only works for 2 weighting inputs")
            print("Aborting")
            return

        x1 = self.x1_sweep
        x2 = self.x2_sweep

        # create agrid of sub plots
        if assending == 0:
            rows, cols = self.NetworkDict['num_input'], self.NetworkDict['num_input']
        elif assending == 1:
            weight_list = [[-1, 1], [-0.5, 1], [0, 1], [0.5, 1],
                           [1, 1], [1, 0.5], [1, 0], [1, -0.5],
                           [1, -1], [0.5, -1], [0, -1], [-0.5, -1],
                           [-1, -1], [-1, -0.5], [-1, 0], [-1, 0.5]]
            rows, cols = 4, 4

        # Produce the genome population to be examined
        Num_Perms = NPerms(self.NetworkDict['num_input']+self.NetworkDict['num_config'])
        perm_options = np.arange(0, Num_Perms)
        perm_list = []
        pop_list = []
        OW_list_list = []
        actual_PermArray_list = []
        custom_Dict = self.CompiledDict
        custom_Dict = self.clean_genome(custom_Dict)
        custom_Dict['genome']['config_gene_loc'] = 0
        custom_Dict['genome']['InWeight_gene'] = 1
        custom_Dict['genome']['in_weight_gene_loc'] = 1

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

                temp = []
                for i in range(self.NetworkDict['num_config']):
                    temp.append(0)
                defualt_genome.append(np.asarray(temp, dtype=object))

                # Assign no output weightings, or random output weightings
                temp = []
                if assending == 0:
                    # no output weightings
                    if k == 0:
                        temp.append(1)
                        temp.append(1)
                        OW_list_list.append([1,1])
                    elif k == 1:
                        temp.append(1)
                        temp.append(-1)
                        OW_list_list.append([1, -1])
                    elif k == 2:
                        temp.append(-1)
                        temp.append(1)
                        OW_list_list.append([-1, 1])
                    elif k == 3:
                        temp.append(-1)
                        temp.append(-1)
                        OW_list_list.append([-1, -1])
                elif assending == 1:
                    weights = weight_list[k]
                    temp.append(weights[0])
                    temp.append(weights[1])
                    OW_list_list.append(weights)
                defualt_genome.append(np.asarray(temp, dtype=object))

                pop_list.append(np.asarray(defualt_genome, dtype=object))

                # iterate k to select the next random perm
                k = k + 1

        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.asarray(pop_list)
        #print(pop_array)
        bounds_matrix_list, op_list_Glist = self.Find_Class(pop_array, x1, x2, custom_Dict)

        # # # # # # # # # # # #
        # Produce the sub plot graph
        k = 0
        for row in range(rows):
            for col in range(cols):
                Current_perm = IndexPerm(self.NetworkDict['num_input']+self.NetworkDict['num_config'], perm_list[k])
                actual_PermArray_list.append(Current_perm)
                k = k + 1

        # # write data to MG group, vary shuffle sub group
        # # (FIG_defualt_genome_data_Biased)
        location = "%s/data.hdf5" % (self.save_dir)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group("%d_rep%d/MG/VaryInWeight" % (self.cir, self.rep))

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            G_sub.create_dataset('x1_data', data=x1)  # write xy data
            G_sub.create_dataset('x2_data', data=x2)  # write xy data

            extent = [x1.min(),x1.max(),x2.min(), x2.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('perm_list', data=actual_PermArray_list)
            G_sub.create_dataset('Weight_list', data=OW_list_list)
            G_sub.create_dataset('rows', data=rows)
            G_sub.create_dataset('cols', data=cols)

        # # Print execution time
        toc = time.time()
        print("Vary Input Weights Finished, execution time:", toc - tic)

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
        if  self.NetworkDict['num_input'] != 2:
            print("MG_VaryInWeightsAni currently only works for 2 weighting inputs")
            print("Aborting")
            return

        x1 = self.x1_sweep
        x2 = self.x2_sweep

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
        custom_Dict = self.CompiledDict
        custom_Dict = self.clean_genome(custom_Dict)
        custom_Dict['genome']['config_gene_loc'] = 0
        custom_Dict['genome']['InWeight_gene'] = 1
        custom_Dict['genome']['in_weight_gene_loc'] = 1

        k = 0
        for num in range(len(weight_list[0,:])):

            # # # # # # # # # # # #
            # Create the "Defualt Geneome" with shuffle
            defualt_genome = []

            temp = []
            for i in range(self.NetworkDict['num_config']):
                temp.append(0)
            defualt_genome.append(np.asarray(temp, dtype=object))

            temp = []
            weights = weight_list[:, k]
            temp.append(weights[0])
            temp.append(weights[1])
            OW_list_list.append(list(weights))
            defualt_genome.append(np.asarray(temp, dtype=object))

            pop_list.append(np.asarray(defualt_genome, dtype=object))

            # iterate k to select the next random perm
            k = k + 1

        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.asarray(pop_list)
        #print(pop_array)
        bounds_matrix_list, op_list_Glist = self.Find_Class(pop_array, x1, x2, custom_Dict)

        # # write data to MG group
        location = "%s/data.hdf5" % (self.save_dir)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group("%d_rep%d/MG/VaryInWeightAni" % (self.cir, self.rep))

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            G_sub.create_dataset('x1_data', data=x1)  # write xy data
            G_sub.create_dataset('x2_data', data=x2)  # write xy data

            extent = [x1.min(),x1.max(),x2.min(), x2.max()]
            G_sub.create_dataset('extent', data=extent)  # write extent

            G_sub.create_dataset('Weight_list', data=OW_list_list)


        # # Print execution time
        toc = time.time()
        print("Vary Input Weights ANIMATION Finished, execution time:", toc - tic)

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
        if self.num_output_weights != 2:
            print("MG_VaryWeights currently only works for 2 weighting outputs")
            print("Aborting")
            return

        x1 = self.x1_sweep
        x2 = self.x2_sweep

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
        custom_Dict = self.CompiledDict
        custom_Dict = self.clean_genome(custom_Dict)
        custom_Dict['genome']['config_gene_loc'] = 0
        custom_Dict['genome']['InWeight_gene'] = 1
        custom_Dict['genome']['in_weight_gene_loc'] = 1

        k = 0
        for num in range(len(weight_list[0,:])):

            # # # # # # # # # # # #
            # Create the "Defualt Geneome" with shuffle
            defualt_genome = []

            temp = []
            for i in range(self.NetworkDict['num_config']):
                temp.append(0)
            defualt_genome.append(np.asarray(temp, dtype=object))

            temp = []
            weights = weight_list[:, k]
            temp.append(weights[0])
            temp.append(weights[1])
            OW_list_list.append(list(weights))
            defualt_genome.append(np.asarray(temp, dtype=object))

            pop_list.append(np.asarray(defualt_genome, dtype=object))

            # iterate k to select the next random perm
            k = k + 1

        # find the bounds matrix representing the class and proximity to the class boundary
        pop_array = np.asarray(pop_list)
        #print(pop_array)
        bounds_matrix_list, op_list_Glist = self.Find_Class(pop_array, x1, x2, custom_Dict)

        # # write data to MG group
        location = "%s/data.hdf5" % (self.save_dir)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group("%d_rep%d/MG/VaryLargeInWeightAni" % (self.cir, self.rep))

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            G_sub.create_dataset('x1_data', data=x1)  # write xy data
            G_sub.create_dataset('x2_data', data=x2)  # write xy data

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

        x1_VC = self.x1_sweep
        x2_VC = self.x2_sweep

        # # produce multiple graphs for different Vconfig3
        for Vconfig_3 in [-6, -3, 0, 3, 6]:

            if self.NetworkDict['num_config'] == 3:
                Vconfig_1 = np.asarray([-6, -3, 0, 3, 6])
                Vconfig_2 = np.asarray([-6, -3, 0, 3, 6])
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
            custom_Dict = self.CompiledDict
            custom_Dict = self.clean_genome(custom_Dict)
            custom_Dict['genome']['config_gene_loc'] = 0

            pop_list = []
            for row in range(rows):
                for col in range(cols):

                    # # # # # # # # # # # #
                    # Create the "Defualt Geneome"
                    defualt_genome = []

                    j = 0
                    temp = []
                    for i in range(self.NetworkDict['num_config']):
                        if i == 0:
                            temp.append(Vconfig_1[row])
                        if i == 1:
                            temp.append(Vconfig_2[col])
                        if i == 2:
                            temp.append(Vconfig_3)
                        j = j + 1
                    defualt_genome.append(np.asarray(temp, dtype=object))
                    pop_list.append(defualt_genome, dtype=object)

            #print("pop_list:\n", pop_list)
            # find the bounds matrix representing the class and proximity to the class boundary
            pop_array = np.asarray(pop_list)
            bounds_matrix_list, op_list_Glist = self.Find_Class(pop_array, x1_VC, x2_VC, custom_Dict)


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
            fig3_path = "%s/%d_Rep%d_FIG_DefualtGenomeBiased___Vconfig3_%d.png" % (self.save_dir, self.cir, self.rep, Vconfig_3)
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
        location = "%s/data_%d_rep%d.hdf5" % (self.save_dir, self.cir, self.rep)
        with h5py.File(location, 'a') as hdf:
            G_sub = hdf.create_group('/MG/VaryConfig')

            rY_list = G_sub.create_dataset('responceY_list', data=bounds_matrix_list)  # write BW
            rY_list.attrs['num_plots'] = len(bounds_matrix_list)

            x_data = np.asarray([x1_VC, x2_VC])
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
