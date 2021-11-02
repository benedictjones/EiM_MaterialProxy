# # Top Matter
import numpy as np
import time
import os
import h5py
import sys

from random import random
from multiprocessing import Pool
from functools import partial

from mod_settings.Set_Load import LoadSettings

from mod_material.spice.GenerateRN import generate_random_netork
from mod_material.spice.Network_LoadRun import LoadRun_model
from mod_material.spice.GenerateNN import generate_neuromorphic_netork

from mod_analysis.Set_Load_meta import LoadMetaData

from mod_methods.FetchPerm import IndexPerm

from mod_interp.InterpretOutputs import InterpretVoltages


class material_processor(object):

    def __init__(self, param_file=''):
        """
        Initialise object
        """

        # Check to see if dict is being passed in, or needs to be loaded
        if isinstance(param_file, dict):
            self.CompiledDict = param_file
        else:
            self.CompiledDict = LoadSettings(param_file)

        self.ParamDict = self.CompiledDict['DE']
        self.NetworkDict = self.CompiledDict['network']
        self.SpiceDict = self.CompiledDict['spice']
        self.GenomeDict = self.CompiledDict['genome']

        # # Make interpret output object
        self.interp = InterpretVoltages(self.CompiledDict)

        return

    #

    def gen_material(self, syst, rep=0, l='na', m='na'):
        """
        Generate Material if appropriate
        """
        # print("Gen material: syst", syst, "rep", rep, ", load_only:", load_only)



        # Check to see if we are re using a model
        if self.CompiledDict['ReUse_dir'] != 'na':
            print(">> ReUsing a material model <<")
            self.ModelDir = self.CompiledDict['ReUse_dir']
        else:
            self.ModelDir = self.CompiledDict['SaveDir']

        # # Add layer refference
        if l == 'na':
            lref = ''
        else:
            lref = '_l%d' % (l)

        # # Add the material (in a select layer) refference
        if m == 'na':
            mref = ''
        else:
            mref = '_m%d' % (m)

        # If the material exists, don't re-create
        dir_path = "%s/MD_CircuitTop/Network_Topology_%s_Syst%d%s%s.txt" % (self.ModelDir, str(self.CompiledDict['param_file']), syst, lref, mref)
        if os.path.exists(dir_path) is True:
            #print(">> Material Model Path found <<")
            meta = LoadMetaData(self.ModelDir, param_file=str(self.CompiledDict['param_file']))
            load_num_nodes = meta['network']['num_input'] + meta['network']['num_config'] + meta['network']['num_output']
            num_nodes = self.NetworkDict['num_input'] + self.NetworkDict['num_config']  + self.NetworkDict['num_output']
            #print("\n>>>>>", load_num_nodes, num_nodes)
            if num_nodes < load_num_nodes:
                print("Warning: Some nodes unused!! Using %d nodes, but loaded material has %d!" % (num_nodes, load_num_nodes))
            elif num_nodes > load_num_nodes:
                raise ValueError("Selected Material had %d in nodes, current settings ask for %d in nodes!" % (load_num_nodes, num_nodes))

            #print(">> Loaded Material <<")
            #print(dir_path)
            return

        elif os.path.exists(dir_path) is False and self.CompiledDict['ReUse_dir'] != 'na':
            raise ValueError("Selected ReUse Dir does not exist:\n > %s" % (dir_path))

        # if the load only toggle is on, then return only
        # (i.e never gen a new material)


        # Don't generate a new material if it is a repetition
        if rep > 0:
            return

        # Genertate a material if it is not Neuromorphic
        if self.NetworkDict['model'][-2:] != 'NN':
            print(">> Generating Material <<")
            generate_random_netork(self.CompiledDict, syst)
        return

    #

    def run_processors(self, genome_list, syst, rep, input_data, target_output='na',
                       ridge_layer_list=0, ret_str='defualt', the_data='train', force_RL_return=0):
        """
        Uses MultiProcessing to calculate the outputs, responces, assigned
        classes etc.

        The Return Structure (ret_str) can be set to:
            'defualt' - list of the results from each genome
            'unzip' - list of each property, themselves in a list according to the genome
            'both' - returns unzipped & defualt
        """
        tic = time.time()
        # # Create the iterable for the Multiprocessing
        iteralbe_idx = np.arange(len(genome_list))
        #print("genome_list:\n", genome_list)
        use_MP = 1

        # # if we are using a DC/op simulation, just do all genomes in one
        # instance of NGSpice - this could be faster but mainly prevents
        # memory errors
        """
        if self.SpiceDict['sim_type'] == 'sim_dc' and genome_list.shape[0] > 1 and sys.platform == "win32":
            # Hope was this would stop the memory leak - doesn't seem to

            # # Windows Anaconda based Shared NGSpice usages seems to have a
            # string limit or some sort on the input of a pwl.
            # >> Is the data size is bigger then aprox 20000 then the spice will
            # just exit. Not yet sure why.
            approx_num_inst = len(input_data[:,0])*genome_list.shape[0]
            if approx_num_inst < 20000:

                # '''
                results_GenomeList = self._solve_all_processors_(idx=0, genome_list=genome_list, syst=syst, rep=rep,
                                                                   input_data=input_data, target_output=target_output,
                                                                   the_data=the_data,
                                                                   ridge_layer_list=ridge_layer_list, force_RL_ret=force_RL_return)
                # '''

                ''' # prevents memory error! (but slow!)
                with Pool(processes=1) as pool:
                    func = partial(self._solve_all_processors_, genome_list=genome_list, syst=syst, rep=rep,
                                   input_data=input_data, target_output=target_output,
                                   the_data=the_data, ridge_layer_list=ridge_layer_list, force_RL_ret=force_RL_return)
                    results_GenomeList_list = pool.map(func, [0])
                results_GenomeList = results_GenomeList_list[0]

                '''
                use_MP = 0
        #"""

        # # calc outputs for each genome using multiprocessing
        if use_MP == 1:
            if self.CompiledDict['num_processors'] == 1:
                results_GenomeList = []
                for idx in iteralbe_idx:
                    results = self._solve_processor_(idx, genome_list, syst, rep, input_data, target_output,
                                                     the_data, ridge_layer_list, force_RL_return)
                    results_GenomeList.append(results)
            else:
                if self.CompiledDict['num_processors'] == 'auto':
                    with Pool() as pool:
                        func = partial(self._solve_processor_, genome_list=genome_list, syst=syst, rep=rep,
                                       input_data=input_data, target_output=target_output,
                                       the_data=the_data, ridge_layer_list=ridge_layer_list, force_RL_ret=force_RL_return)
                        results_GenomeList = pool.map(func, iteralbe_idx)
                else:
                    with Pool(processes=self.CompiledDict['num_processors']) as pool:
                        func = partial(self._solve_processor_, genome_list=genome_list, syst=syst, rep=rep,
                                       input_data=input_data, target_output=target_output,
                                       the_data=the_data, ridge_layer_list=ridge_layer_list, force_RL_ret=force_RL_return)
                        results_GenomeList = pool.map(func, iteralbe_idx)

        #
        # print(" ~ Time to run all processors:", time.time()-tic)

        # # Change the structure of the outputs  # #

        # list of each property, themselves in a list according to the genome
        if ret_str == 'unzip':
            structured_res = []
            res_list = list(map(list, zip(*results_GenomeList)))
            for property in res_list:
                #structured_res.append(np.asarray(property, dtype=object))
                #structured_res.append(np.asarray(property))
                structured_res.append(property)
            return structured_res

        # list of the results from each genome
        elif ret_str == 'defualt':
            return results_GenomeList

        # returns both
        elif ret_str == 'both':
            structured_res = []
            #res_list = list(zip(*results_GenomeList))
            res_list = list(map(list, zip(*results_GenomeList)))
            for property in res_list:
                #structured_res.append(np.asarray(property, dtype=object))
                #structured_res.append(np.asarray(property))
                structured_res.append(property)
                #print(property)
                #exit()
            return structured_res, results_GenomeList

    #

    def run_processor(self, genome, syst, rep, input_data, target_output='na',
                      ridge_layer=0, the_data='train',
                      force_RL_return=0, force_predY_return=0):
        """
        This runs the processor once, for a set of single inputs:
            - genome, Ridge layer, etc.

        This functions is a more user intuative way of running a single set of
        System Defining parameters.
            e.g. _solve_processor_ could be called directly but the user would
                 need to pass in a idx='na'.
        While not ideal, _solve_processor_ needed it's arguments formated in
        this way so pool could pass in the indexing iterable!
        """

        # # Calc results
        results = self._solve_processor_(idx='na', genome_list=genome, syst=syst, rep=rep, input_data=input_data,
                                         target_output=target_output, the_data=the_data,
                                         ridge_layer_list=ridge_layer, force_RL_ret=force_RL_return,
                                         force_predY_ret=force_predY_return)

        return results

    #

    #

    #

    def _solve_processor_(self, idx, genome_list, syst, rep,
                          input_data, target_output, the_data,
                          ridge_layer_list=0, force_RL_ret=0,
                          force_predY_ret=0):


        """
        This Solves the material for the given input System Defining Paramaters
        (e.g. genome, ridge output layer), then interprets the outputs to
        generate & return results.

        The System Definign Paramaters (genome_list & ridge_layer_list) are a
        list which is indexed by the "idx". This format is used to facilitate
        pool MP.
        Instead of passing the whole list of System Definign Paramaters, they
        could have been zipped together and passed in as the itterable. However
        this is a less flexible approach if future params might be needed.
        That method was tried and also dound to be slower (maybe due to the
        overhead of zipping and unziping lists?).

        The Voltage outputs are then interpreted using the interp scheme object
        which is set to EiM/RC on initialisation.


        >> Results format <<

        --> For Passed in Data and a Target output:
                When ridge_layer = 0 (OR when force_RL_ret = 1):
                    EiM: [scheme_fitness, responceY, err_fit, Vout, -1]
                Otherwise:
                    EiM: [scheme_fitness, responceY, err_fit, Vout]

        --> For input data with no Target data (i.e. data_Y='na' and IntpScheme != raw).
            We can't determine a supervised fitness, but can compute a reponce.
            (Note: if IntpScheme == raw, then it is assumed a unsupervised
            fit scheme is being used.)
                EiM: [class_out, responceY, Vout]

        Where: trained_ridge_layer = [weights, bias, Threshold]

        """
        ti = time.time()
        # # Extract information from the passed in System Definign Paramaters
        if idx == 'na':
            genome = genome_list
            ridge_layer = ridge_layer_list

        else:
            genome = genome_list[idx]
            if isinstance(ridge_layer_list, list) is False and ridge_layer_list == 0:
                ridge_layer = 0
            elif isinstance(ridge_layer_list, list) is False and ridge_layer_list == -1:
                ridge_layer = -1
            else:
                ridge_layer = ridge_layer_list[idx]

        #print("}}-", ridge_layer)

        # # Assign Data
        data_X = input_data
        data_Y = target_output

        # # Check data dimension against physical number of inputs
        # print(">>", self.CompiledDict['network']['num_input'], data_X.shape)
        if len(data_X[0, :]) != self.CompiledDict['network']['num_input']:
            raise ValueError("\nParam File: %s\nUsing %d input data nodes --> for %d input attributes!" % (self.CompiledDict['param_file'], self.CompiledDict['network']['num_input'], len(data_X[0,:])))

        # Calc material output voltages
        Vout = self._solve_material_(genome, data_X, syst=syst, the_data=the_data)

        #print("\n\n>>>", ridge_layer)

        # Using the interp schem object, calc output class, responce, etc.
        results = self.interp.run(Vout, genome, data_Y, syst, rep, ridge_layer, the_data,
                                  force_RL_ret, force_predY_ret)

        # print(" ~~ Time to solve process:", time.time()-ti)

        return results

    #

    def _solve_material_(self, genome, data_X, syst, the_data, multi_solve=0):
        """
        This Formats the input data and genome characterisitcs such that the
        correct input voltages are applied to the selected nodes.
        """

        # # Iterate over all training data to retrieve re-ordered &/or modified voltage inputs
        seq_len = self.NetworkDict['num_config'] + self.NetworkDict['num_input']

        # print("~~ ", genome)

        if multi_solve == 1:
            # concatenate the genome Vin so only one instance of NGSpice is called
            # here genome = passed in genome array
            Vin_matrix = self._get_multi_geneome_voltages_(data_X, genome, self.GenomeDict, seq_len)
        else:
            # for a single genome
            Vin_matrix = self._get_voltages_(data_X, genome, self.GenomeDict, seq_len)

        # print(Vin_matrix)
        # print("genome:", genome)

        # # set pulse width
        if self.GenomeDict['PulseWidth']['active'] == 1:
            genome_tp = genome[self.GenomeDict['PulseWidth']['loc']][0]
        else:
            genome_tp = self.SpiceDict['pulse_Tp']

        # # Calculate the voltage output of input training data
        # # Load circuit
        for attempt in [1,2,3]:
            try:
                if self.NetworkDict['model'][-2:] == 'NN':
                    if the_data == 'test' or the_data == 'validation':
                        SaveToText = syst
                    else:
                        SaveToText = 0
                    Vout = generate_neuromorphic_netork(self.CompiledDict, genome, Vin_matrix, genome_tp, SaveToText)
                else:
                    Vout = LoadRun_model(Vin_matrix, self.ModelDir, self.CompiledDict, genome_tp, syst) # l,m = layer-index, material-in-layer-index
                break
            except Exception as e:
                time.sleep(0.02 + random())
                print("Attempt", attempt, "failed... \n Re trying to calculate output voltages.")
                if attempt == 3:
                    print("Error (TheModel, LoadRun_model): Failed to calculate output voltages\n\n")
                    raise ValueError(e)
                else:
                    pass

        return Vout

    #

    def _get_voltages_(self, X, genome, GenomeDict, seq_len):
        """
        Applies weights to the input voltages, appends the config voltages, and
        reorders the columns acording to the shuffle index.
        """

        #print("genome:", genome, " shape", genome.shape)

        # if X is a 1d array, make it into a 2d array
        if len(np.shape(X)) == 1:
            X = np.reshape(X, (len(X), 1))


        # # Initialise lists
        num_inst = len(X[:,0])
        num_attr = len(X[0,:])
        Vin = np.zeros((num_inst,seq_len))

        # # apply input weights to attribute voltages
        if GenomeDict['InWeight']['active'] == 0:
            for attr in range(num_attr):
                Vin[:, attr] = X[:, attr]
        elif GenomeDict['InWeight']['active'] == 1:
            in_weight_list = genome[GenomeDict['InWeight']['loc']]
            for attr in range(num_attr):
                Vin[:, attr] = X[:, attr]*in_weight_list[attr]

        # Apply Input Bias to each input
        if GenomeDict['InBias']['active'] == 1:
            in_bias_list = genome[GenomeDict['InBias']['loc']]
            for attr in range(num_attr):
                Vin[:, attr] = Vin[:, attr] + in_bias_list[attr]

        # # add config voltages
        if GenomeDict['Config']['active'] == 1:
            for idx, v_config in enumerate(genome[GenomeDict['Config']['loc']]):
                Vin[:, num_attr+idx] = np.full(num_inst, v_config)

        # # shuffle
        if GenomeDict['Shuffle'] == 1:
            perm_index = genome[GenomeDict['Shuffle']['loc']][0]
            order = IndexPerm(seq_len, perm_index)
            Vin_ordered = self._reorder_cols_(Vin, order)
        else:
            Vin_ordered = Vin

        # # Round
        new_Vin = np.around(Vin_ordered, decimals=5)

        return new_Vin

    #

    def _reorder_cols_(self, matrix, index):
        """ Function which re-orders a 2d array's columns acording to the
        input permutation (i.e passed in index 1d array)
        e.g
            for matrix with 3 columns: col_A, col_B, col_C
            and passed in index array: [1,0,2]
            new maxtix column order: col_B, col_A, col_C
        """
        new_matrix = np.zeros(matrix.shape)
        for new_idx, old_idx in enumerate(index):
            # print(new_idx, old_idx)
            new_matrix[:, new_idx] = matrix[:, old_idx]
        return new_matrix

    #

    #

    #

    def _solve_all_processors_(self, idx, genome_list, syst, rep,
                                input_data, target_output='na',
                                ridge_layer_list=0, the_data='train', force_RL_ret=0):
        """
        Very similar to solve_processor() function, execpt it generates a
        concatinated array of the input voltages, and performs the SPICE
        computation on all the inpouts at once.
        This can only work for inputs that are not dependent on one anouther
        i.e. DC sweep or operational point (Not transient!!).

        """

        if idx != 0:
            raise ValueError("For _solve_all_processors_ no iterable index is needed. (idx=%d)" % (idx))

        # print(">>", input_data, output_data)
        # print("genome list shape:", genome_list.shape)

        data_X = input_data
        data_Y = target_output

        # # Check data dimension against physical number of inputs
        # print(">>", self.CompiledDict['network']['num_input'], len(data_X[0, :]))
        if len(data_X[0, :]) != self.CompiledDict['network']['num_input']:
            raise ValueError("\nParam File: %s\nUsing %d input data nodes --> for %d input attributes!" % (self.self.CompiledDict['param_file'], self.CompiledDict['network']['num_input'], len(data_X[0,:])))

        # Calc material output voltages
        Vout_all = self._solve_material_(genome_list, data_X, syst=syst, the_data=the_data, multi_solve=1)

        # split up the Voutput into the correct chunks
        shaped_Vout_all = np.reshape(Vout_all, (genome_list.shape[0], data_X.shape[0], self.NetworkDict['num_output']))

        # error check - generate the output for the first genome
        Vout = self._solve_material_(genome_list[0], data_X, syst=syst, the_data=the_data)

        # # compare first Vout array, to check it is correct
        if np.array_equal(np.around(shaped_Vout_all[0], 2), np.around(Vout, 2)) != True:
            raise ValueError("Voltages Incorrectly calculated.\nSim_Dc speed up failed!")

        # Using the interp schem object, calc output class, responce, etc.
        # for each genome
        results_GenomeList = []
        for gidx, genome in enumerate(genome_list):

            # Extract ridge layer
            if isinstance(ridge_layer_list, list) is False and ridge_layer_list == 0:
                ridge_layer = 0
            elif isinstance(ridge_layer_list, list) is False and ridge_layer_list == -1:
                ridge_layer = -1
            else:
                ridge_layer = ridge_layer_list[gidx]

            # fetch results
            results = self.interp.run(shaped_Vout_all[gidx], genome, data_Y, syst, rep, ridge_layer, the_data, force_RL_ret)
            results_GenomeList.append(results)

        return results_GenomeList

    #

    #

    def _get_multi_geneome_voltages_(self, data_X, genome_array, GenomeDict, seq_len):
        """
        Taked in the genome list/array and created the appropriate Voltage
        inputs. These are concatenated together into one large array.
        """

        Vin_list = []

        # # fetch the input voltages for each genome
        for genome in genome_array:

            # # Iterate over all training data to retrieve re-ordered &/or modified voltage inputs
            Vin = self._get_voltages_(data_X, genome, self.GenomeDict, seq_len)
            Vin_list.append(Vin)

        Vin_matrix = np.concatenate(Vin_list)

        return Vin_matrix



# fin
