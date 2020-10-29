# # Top Matter
import numpy as np
import time
import os
import h5py

from random import random
from multiprocessing import Pool
from functools import partial

from mod_settings.Set_Load import LoadSettings

from mod_material.spice.GenerateRN import generate_random_netork
from mod_material.spice.Network_LoadRun import LoadRun_model
from mod_material.spice.GenerateNN import generate_neuromorphic_netork

from mod_load.LoadData import Load_Data

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


        self.interp = InterpretVoltages(self.CompiledDict)

        return

    #

    def gen_material(self, cir, rep=0, load_only=0):
        """
        Generate Material if appropriate
        """
        # print("Gen material: cir", cir, "rep", rep, ", load_only:", load_only)



        # Check to see if we are re using a model
        if self.CompiledDict['ReUse_dir'] != 'na':
            print(">> ReUsing a material model <<")
            self.ModelDir = self.CompiledDict['ReUse_dir']
        else:
            self.ModelDir = self.CompiledDict['SaveDir']

        # If the material exists, don't re-create
        dir_path = "%s/MD_CircuitTop/Network_Topology__Cir_%d.txt" % (self.ModelDir, cir)
        if os.path.exists(dir_path) is True:
            #print(">> Material Model Path found <<")
            return
        elif os.path.exists(dir_path) is False and self.CompiledDict['ReUse_dir'] != 'na':
            raise ValueError("Selected ReUse Dir does not exist")

        # if the load only toggle is on, then return only
        # (i.e never gen a new material)
        if load_only == 1:
            return

        # Don't generate a new material if it is a repetition
        if rep > 0:
            return

        # Genertate a material if it is not Neuromorphic
        if self.NetworkDict['model'][-2:] != 'NN':
            print(">> Generating Material <<")
            generate_random_netork(self.CompiledDict, cir)
        return

    #

    def run_processor(self, genome_list, cir, rep, input_data='na', output_data='na', the_data='train', ret_str='defualt'):
        """
        Uses MultiProcessing to calculate the outputs, responces, assigned
        classes etc.

        The Return Structure (ret_str) can be set to:
            'defualt' - list of the results from each genome
            'unzip' - list of each property, themselves in a list according to the genome
            'both' - returns unzipped & defualt
        """

        # calc outputs for each genome using multiprocessing
        if self.CompiledDict['num_processors'] == 1:
            results_GenomeList = []
            for genome in genome_list:
                results = self.solve_processor(genome, cir, rep, the_data, input_data, output_data)
                results_GenomeList.append(results)
        else:
            if self.CompiledDict['num_processors'] == 'auto':
                with Pool() as pool:
                    func = partial(self.solve_processor, cir=cir, rep=rep, the_data=the_data, input_data=input_data, output_data=output_data)
                    results_GenomeList = pool.map(func, genome_list)
            else:
                with Pool(processes=self.CompiledDict['num_processors']) as pool:
                    func = partial(self.solve_processor, cir=cir, rep=rep, the_data=the_data, input_data=input_data, output_data=output_data)
                    results_GenomeList = pool.map(func, genome_list)

        # Change the structure of the outputs
        if ret_str == 'unzip':
            structured_res = []
            res_list = list(zip(*results_GenomeList))
            for property in res_list:
                structured_res.append(np.asarray(property))
            return structured_res

        elif ret_str == 'defualt':
            return results_GenomeList

        elif ret_str == 'both':
            structured_res = []
            res_list = list(zip(*results_GenomeList))
            for property in res_list:
                structured_res.append(np.asarray(property))
            return structured_res, results_GenomeList

    #

    def solve_processor(self, genome, cir='', rep='', the_data='train', input_data='na', output_data='na'):
        """
        This Formats the input data and genome characterisitcs such that the
        correct input voltages are applied to the selected nodes.

        The Voltage outputs are then interpreted using the interp scheme object
        which is set to EiM/RC on initialisation.

        > Results format <
        For the Standard Loaded Data:
            EiM: [scheme_fitness, responceY, err_fit, Vout]
            RC: [scheme_fitness, responceY, err_fit, Vout, weights, bias, Threshold]
        For the Custom input data (can't determine a fitness):
            EiM: [class_out, responceY, Vout]
            RC: [class_out, responceY, Vout, weights, bias, Threshold]

        For Ridged Fit on target output:
            []

        """

        # print(">>", input_data, output_data)

        # #  Load the training data
        if str(input_data) == 'na':
            data_X, data_Y = Load_Data(the_data, self.CompiledDict)
        else:
            data_X = input_data
            data_Y = output_data

        # # load target array
        if the_data == 'train' or the_data == 'veri':

            if str(output_data) == 'input' or self.CompiledDict['DE']['IntpScheme'] == 'Ridged_fit':
                data_Y, nn = Load_Data(the_data, self.CompiledDict)
            elif str(output_data) == 'na':
                nn, data_Y = Load_Data(the_data, self.CompiledDict)
            else:
                data_Y = output_data

        # Calc material output voltages
        Vout = self._solve_material_(genome, data_X, cir=cir, the_data=the_data)

        # Using the interp schem object, calc output class, responce, etc.
        results = self.interp.run(Vout, genome, data_Y, the_data, cir, rep)

        return results

    #

    #

    #

    def _solve_material_(self, genome, data_X, cir, the_data='train'):
        """
        This Formats the input data and genome characterisitcs such that the
        correct input voltages are applied to the selected nodes.
        """

        # # Iterate over all training data to retrieve re-ordered &/or modified voltage inputs
        seq_len = self.NetworkDict['num_config'] + self.NetworkDict['num_input']
        Vin_matrix = self._get_voltages_(data_X, genome, self.GenomeDict, seq_len)

        # print(Vin_matrix)
        # print("genome:", genome)

        # # set pulse width
        if self.GenomeDict['PulseWidth_gene'] == 1:
            genome_tp = genome[self.GenomeDict['PulseWidth_gene_loc']][0]
        else:
            genome_tp = self.SpiceDict['pulse_Tp']

        # # Calculate the voltage output of input training data
        # # Load circuit
        for attempt in [1,2,3]:
            try:
                if self.NetworkDict['model'][-2:] == 'NN':
                    if the_data == 'veri':
                        SaveToText = cir
                    else:
                        SaveToText = 0
                    Vout = generate_neuromorphic_netork(self.CompiledDict, genome, Vin_matrix, genome_tp, SaveToText)
                else:
                    Vout = LoadRun_model(Vin_matrix, self.ModelDir, self.CompiledDict, genome_tp, cir)
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

        # if X is a 1d array, make it into a 2d array
        if len(np.shape(X)) == 1:
            X = np.reshape(X, (len(X), 1))


        # # Initialise lists
        num_inst = len(X[:,0])
        num_attr = len(X[0,:])
        Vin = np.zeros((num_inst,seq_len))

        # # apply input weights to attribute voltages
        if GenomeDict['InWeight_gene'] == 0:
            for attr in range(num_attr):
                Vin[:, attr] = X[:, attr]
        elif GenomeDict['InWeight_gene'] == 1:
            in_weight_list = genome[GenomeDict['in_weight_gene_loc']]
            for attr in range(num_attr):
                Vin[:, attr] = X[:, attr]*in_weight_list[attr]

        # # add config voltages
        for idx, v_config in enumerate(genome[GenomeDict['config_gene_loc']]):
            Vin[:, num_attr+idx] = np.full(num_inst, v_config)

        # # shuffle
        if GenomeDict['shuffle_gene'] == 1:
            perm_index = genome[GenomeDict['SGI']][0]
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






# fin
