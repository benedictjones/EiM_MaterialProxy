import numpy as np
import multiprocessing
import yaml
import os
from datetime import datetime
import sys

from mod_load.LoadData_method import Load_Data
from mod_load.FetchDataObj import FetchDataObj
from mod_methods.FetchPerm import NPerms
from mod_settings.Set_Change import ChangeSettings
from mod_analysis.Set_Load_meta import LoadMetaData
#


def LoadPrm(param_file=''):
    """
    Load in a template paramater yaml file.
    """
    with open(r'mod_settings/param%s.yaml' % (str(param_file))) as file:
        prm = yaml.full_load(file)
    return prm

#


def GenSimParam(param_file='', algorithm='na',
                experiment=0, experiment_file='bin/', exp_name='test',
                pt='na', NoGenes=0, change_dict='na',
                make_save=1, label='', **kwargs):

    """
    Function which loads a param_file (using the above argument to select
    the end of the filename param%s). Depending on the parmaters, other
    properties are added such as number of processors, or the bounds for the
    different angorithm decision variables (genes).

    """



    # # Load Params from file (if not passed in)
    if isinstance(param_file, dict):
        prm = param_file
        prm['param_file'] = label
    else:
        with open(r'mod_settings/param%s.yaml' % (str(param_file))) as file:
            prm = yaml.full_load(file)
        prm['param_file'] = param_file

    # # Apply Custom kargs
    prm = ChangeSettings(prm, pt=pt, change_dict=change_dict, **kwargs)


    """# # Edit and add to Params # #"""

    #"""
    # #  assign the algorithm
    if algorithm != 'na':
        prm['algorithm'] = algorithm
        prm['ptype'] = 'na'

    #"""

    #

    #

    #  #  #  # extract the processor type from algorithm selected #  #  #  #

    # Evolution in-Materio  (Trained config paramaters)
    if 'EiM' in prm['algorithm']:
        if 'Ridge' not in prm['DE']['IntpScheme']:

            if prm['DE']['epochs'] > 0:
                prm['ptype'] = 'EiM'  # processor type
            else:
                prm['ptype'] = 'UTiM'  # UnTrained processor type
            prm['op_layer'] = 'evolved'  # processor type

        else:
            if prm['DE']['epochs'] > 0:
                prm['ptype'] = 'RidgeEiM'  # processor type
            else:
                prm['ptype'] = 'RidgeUTiM'  # UnTrained processor type
            prm['op_layer'] = 'ridge'  # processor type

    elif 'RiM_EiMpt' in prm['algorithm']:
        prm['ptype'] = 'RiM_EiMpt'  # processor type
        prm['op_layer'] = 'ridge'  # output layer type

    #elif 'RC' in prm['algorithm']:
    #    prm['ptype'] = 'RC'  # processor type
    elif 'AE' in prm['algorithm']:
        prm['ptype'] = 'AE'  # processor type



    else:
        raise ValueError("Error (Settings): The processor type (ptype) cannot be determined from the algorithm: %s" % (prm['algorithm']))

    #

    #

    #

    # # Multiprocessor Settings (for DE simulation process only, MG currently seperate)
    # print("number of cores:", multiprocessing.cpu_count())
    if prm['num_processors'] == 'auto':
        # fetch the local PC's number of cores

        if sys.platform == "win32":
            prm['num_processors'] = 1
            print("Windows Platform - Set to single core usage!")
        else:
            prm['num_processors'] = 'auto'
            print("Multi core usage enabled!")

    elif prm['num_processors'] > multiprocessing.cpu_count():
        print("Maximum number of cores is:", multiprocessing.cpu_count())
        print("Number of corse being set the the maximum possible.")
        prm['num_processors'] = int(multiprocessing.cpu_count())
    else:
        prm['num_processors'] = prm['num_processors']
    print("Number Processors Selected =", prm['num_processors'])

    # # add model to spice section
    prm['spice']['model'] = prm['network']['model']

    # # If only one layer, remove per layer values
    if prm['spice']['num_layers'] == 1:
        prm['spice']['ConfigPerLayer'] = 'na'
        prm['spice']['NodesPerLayer'] = 'na'

    #

    # # Fetch The number of nodes in materal/loaded material
    if prm['ReUse_dir'] != 'na':
        meta = LoadMetaData(prm['ReUse_dir'], param_file=str(prm['param_file']))
        prm['network']['total_num_nodes'] = meta['network']['num_input'] + meta['network']['num_config'] + meta['network']['num_output']
    else:
        prm['network']['total_num_nodes'] = prm['network']['num_input'] + prm['network']['num_config'] + prm['network']['num_output']

    #

    #

    # # Extract information about the data set being considtered
    #train_data_X, train_data_Y = Load_Data('all', prm)
    lobj = FetchDataObj(prm)
    class_values = np.unique(lobj.raw_data_Y)
    counts = len(class_values)
    if prm['DE']['data_weighting'] == 1:
        prm['DE']['ClassWeights'] = list(lobj.class_weights)
    elif prm['DE']['data_weighting'] == 0:
        prm['DE']['ClassWeights'] = list(np.ones(len(class_values)))
    # print(class_values, counts)
    prm['DE']['ClassWeights_text'] = str(lobj.class_weights)

    if prm['genome']['BandNumber']['max'] == 'defualt':
        max_number_bands = counts  # assign the number of classes in the data
    else:
        max_number_bands = prm['genome']['BandNumber']['max']

    if max_number_bands < counts:
        raise ValueError(" The number of bands should not be set to less than the number of classes!")
    prm['genome']['BandNumber']['min'] = counts
    prm['DE']['num_classes'] = counts
    prm['genome']['BandNumber']['max'] = max_number_bands
    prm['genome']['BandClass']['min'] = int(class_values.min())
    prm['genome']['BandClass']['max'] = int(class_values.max())

    # If the Interpretation Shcme is Ridged Fit, re-set the FitSchem to Mean Absolut Error
    if prm['DE']['IntpScheme'] == 'Ridged_fit':
        prm['DE']['FitScheme'] = 'MAE'  # Ridged_fit can only do MEA

    if prm['DE']['IntpScheme'] == 'HOW':
        if prm['DE']['num_readout_nodes'] == 'na':
            if prm['network']['num_output'] != len(class_values):
                raise ValueError("For HOW you must use the same number of output nodes as output labels.\n Num output nodes = %d, num classes = %d" % (prm['network']['num_output'], len(class_values)))
        else:
            if prm['DE']['num_readout_nodes'] != len(class_values):
                raise ValueError("For HOW you must use the same number of logican readout output nodes as output labels.\n Num readout nodes = %d, num classes = %d" % (prm['DE']['num_readout_nodes'], len(class_values)))

    if prm['ptype'] == 'EiM' and prm['DE']['num_readout_nodes'] != 'na' and prm['genome']['OutWeight']['active'] != 1:
        if prm['DE']['num_readout_nodes'] != 1:
            # raise ValueError("Cannot use a readout layer unless output weights are turned on!")
            raise ValueError("Cannot use a readout layer (with more then 1 node) unless output weights are turned on!")

    if prm['DE']['IntpScheme'] != 'HOW':
        prm['genome']['HOWperm']['active'] = 0

    if prm['genome']['Shuffle']['scheme'] != 'none' and prm['genome']['Shuffle']['active'] == 0:
        raise ValueError("Error (Settings): Perm_crossp model set, but shuffle gene is off!")

    # save num "final layer" output nodes
    if prm['DE']['num_readout_nodes'] == 'na':
        prm['DE']['num_output_readings'] = prm['network']['num_output']
    else:
        prm['DE']['num_output_readings'] = prm['DE']['num_readout_nodes']

    # # # # # # # # # # # # # # # # # #
    # # # Initialise the bounds matrix

    if NoGenes != 1:
        bounds = []
        j = 0  # Set gene counter and gene loc trackers

        # # Set bounds for voltages
        if prm['network']['num_config'] > 0 and prm['genome']['Config']['active'] == 1:
            temp = []
            for i in range(prm['network']['num_config']):
                temp.append([prm['spice']['Vmin'], prm['spice']['Vmax']])
            bounds.append(temp)
            prm['genome']['Config']['loc'] = j
            j = j + 1

        #

        # # Set bounds for shuffle gene/decision variable
        # perm read by an algorithm (no need to generate whole list)
        num_in_V = prm['network']['num_config'] + prm['network']['num_input']
        print("Number of Input Nodes", num_in_V)
        num_perms = NPerms(num_in_V)
        print("Number of perms:", num_perms)
        # assign the shuffle gene bounds accrotding to num perms
        if prm['genome']['Shuffle']['active'] == 1:
            temp = [0, num_perms-1]
            bounds.append([temp])
            prm['genome']['Shuffle']['loc'] = j  # index for the shuffle gene (Shuffle Gene Index)
            j = j + 1

        #

        # # Produce input weight bounds
        if prm['genome']['InWeight']['active'] == 1:
            temp = []
            for i in range(prm['network']['num_input']):
                temp.append([prm['genome']['InWeight']['min'], prm['genome']['InWeight']['max']])
            bounds.append(temp)
            prm['genome']['InWeight']['loc'] = j
            j = j + 1

        #

        # # Produce input weight bounds
        if prm['genome']['InBias']['active'] == 1:
            temp = []
            for i in range(prm['network']['num_input']):
                temp.append([prm['genome']['InBias']['min'], prm['genome']['InBias']['max']])
            bounds.append(temp)
            prm['genome']['InBias']['loc'] = j
            j = j + 1

        #

        # # Produce output weight bounds
        if prm['genome']['OutWeight']['active'] == 1 and prm['op_layer'] == 'evolved':
            temp = []
            if prm['DE']['num_readout_nodes'] != 'na':
                num_output_weights = prm['network']['num_output']*prm['DE']['num_readout_nodes']
            else:
                num_output_weights = prm['network']['num_output']
            print("Num Output Weights", num_output_weights)
            #print(prm['DE']['num_readout_nodes'] )
            #print(prm['network']['num_output'])
            for i in range(num_output_weights):
                temp.append([prm['genome']['OutWeight']['min'], prm['genome']['OutWeight']['max']])
            bounds.append(temp)
            prm['genome']['OutWeight']['loc'] = j
            j = j + 1
        else:
            prm['genome']['OutWeight']['active'] = 0

        #

        # # Produce input weight bounds
        if prm['genome']['OutBias']['active'] == 1 and prm['op_layer'] == 'evolved':
            temp = []

            # A bias value for each readout node (or per output if no readout used)
            if prm['DE']['num_readout_nodes'] == 'na':
                num_op_bias = prm['network']['num_output']
            else:
                num_op_bias = prm['DE']['num_readout_nodes']

            for i in range(num_op_bias):
                temp.append([prm['genome']['OutBias']['min'], prm['genome']['OutBias']['max']])
            bounds.append(temp)
            prm['genome']['OutBias']['loc'] = j
            j = j + 1
        else:
            prm['genome']['OutBias']['active'] = 0

        #

        #

        # # Produce PulseWidth_gene bounds
        if prm['genome']['PulseWidth']['active'] == 1:
            if prm['spice']['sim_type']  != 'sim_trans_pulse':
                raise ValueError('To vary Pulse Width by a gene, the sim typr must be sim_trans_pulse')
            temp = [prm['genome']['PulseWidth']['min'], prm['genome']['PulseWidth']['max']]
            bounds.append([temp])  # ms  - bounds to select pulse width
            prm['genome']['PulseWidth']['loc'] = j
            j = j + 1

        #

        # # Produce BandNumber_gene bounds
        if prm['genome']['BandNumber']['active'] == 1:
            if prm['genome']['min_number_bands'] == prm['genome']['max_number_bands'] and band_scheme != 'defualt':
                raise ValueError('BandNumber_gene set so number of bands can vary, but max band number not larger then if gene is off.')
            temp = [prm['genome']['min_number_bands'], prm['genome']['max_number_bands']+0.99]
            bounds.append([temp])
            prm['genome']['BandNumber_gene_loc'] = j
            print("BandNumber_gene_loc", prm['genome']['BandNumber_gene_loc'])
            j = j + 1

        #

        # # Produce BandWidth_gene bounds
        if prm['genome']['BandEdge']['active'] == 1:
            if prm['DE']['IntpScheme'] != 'band':
                raise ValueError('To very band edges, use a banding scheme.')
            temp = []
            temp.append([prm['spice']['Vmin'], 0])  # left edge
            temp.append([0, prm['spice']['Vmax']]) # right edge
            bounds.append(temp)
            prm['genome']['BandEdge_gene_loc'] = j
            print("BandEdge_gene_loc", prm['genome']['BandEdge_gene_loc'])
            j = j + 1

        #

        # # Produce BandWidth_gene bounds
        if prm['genome']['BandWidth']['active'] == 1:
            if prm['DE']['IntpScheme'] != 'band':
                raise ValueError('To very band width, use a banding scheme.')
            temp = []
            for i in range(max_number_bands-1):
                temp.append([-1, 1])
            bounds.append(temp)
            prm['genome']['BandWidth_gene_loc'] = j
            print("BandWidth_gene_loc", prm['genome']['BandWidth_gene_loc'])
            j = j + 1

        #

        # # Produce BandClass_gene bounds
        if prm['genome']['BandClass']['active'] == 1:
            if prm['DE']['IntpScheme'] != 'band':
                raise ValueError('To very band labels, use a banding scheme.')
            temp = []
            for i in range(max_number_bands):
                temp.append([prm['genome']['BandClass']['min'], prm['genome']['BandClass']['max']+0.99])
            bounds.append(temp)
            prm['genome']['BandClass']['loc'] = j
            j = j + 1

        #

        # # Set bounds for Highest output wins output order gene
        # perms read by an algorithm (no need to generate whole list)
        num_perms = NPerms(prm['network']['num_output'])
        print("Number HOW output order perms:", num_perms)
        # assign the shuffle gene bounds accrotding to num perms
        if prm['genome']['HOWperm']['active'] == 1:
            temp = [0, num_perms-1]
            bounds.append([temp])
            prm['genome']['HOWperm']['loc'] = j  # index for the HOW_perm_gene_loc
            j = j + 1

        #

        # for a Neuromorphic network (NN) rather then a Random Netowrk (RN)
        # generate a weight value, this allows connection weights (resistance) to
        # be calaculated
        if prm['network']['model'][-2:] == 'NN':
            num_bais_nodes = prm['network']['num_input'] + prm['network']['num_output'] + prm['network']['num_config']
            prm['network']['num_nodes'] = num_bais_nodes + prm['spice']['NodesPerLayer']*(prm['spice']['num_layers']-1) + prm['spice']['ConfigPerLayer']*(prm['spice']['num_layers'])
            temp = []
            for i in range(prm['network']['num_nodes']):
                temp.append([0, 1])
            bounds.append(temp)
            prm['genome']['NN_weight_loc'] = j
            print("NN_weight_loc", prm['genome']['NN_weight_loc'])
            j = j + 1

            if prm['spice']['ConfigPerLayer'] >= 1:
                temp = []
                for i in range(prm['spice']['ConfigPerLayer']*(prm['spice']['num_layers'])):
                    temp.append([-5, 5])
                bounds.append(temp)
                prm['genome']['NN_layerConfig_loc'] = j
                print("NN_layerConfig_loc", prm['genome']['NN_layerConfig_loc'])
                j = j + 1
            else:
                prm['genome']['NN_layerConfig_loc'] = 'na'

        #

        # # Record the number of genomes (i.e dimension of decision variables)
        dim = 0
        grouping = []
        for gen_group in bounds:
            dim = dim + len(gen_group)
            grouping.append(len(gen_group))

        prm['DE']['dimensions'] = dim
        print("Decision vector Dimension:", dim)
        prm['DE']['bounds'] = bounds
        prm['genome']['grouping'] = grouping

        # # Infomation print out
        print("The bounds:\n", bounds)

        # remove additional keys
        if 'encoder' in prm.keys():
            del prm['encoder']

        if 'decoder' in prm.keys():
            del prm['decoder']





    #

    #

    # # Collect time and dat stamp of results
    now = datetime.now()
    d_string = now.strftime("%Y_%m_%d")
    t_string = now.strftime("%H_%M_%S")
    print("\n>> Time Stamp:", t_string, " <<")

    #

    # # Set Results Folder name
    if experiment == 0:
        new_dir = "Results/%s/__%s__%s__%s__%s" % (d_string, t_string, prm['DE']['training_data'], prm['network']['model'], prm['ptype'])
        prm['experiment'] = {'active': 0}

    # # Set Results Folder name, including some experiment information
    elif experiment == 1:
        new_dir = "Results/%s/__%s__%s__EXP_%s__%s" % (d_string, t_string, prm['DE']['training_data'], exp_name, prm['ptype'])

        # # Add Experiment Information
        prm['experiment'] = {}
        prm['experiment']['active'] = 1
        prm['experiment']['name'] = exp_name
        prm['experiment']['file'] = 'Results_Experiments/%s' % (experiment_file)

        # if the new folder does not yet exist, create it
        if not os.path.exists(prm['experiment']['file']):
            os.makedirs(prm['experiment']['file'])

    #

    # # Assign and create the saved data directory
    if make_save == 1:
        os.makedirs(new_dir)
        prm['SaveDir'] = new_dir

    #

    #

    return prm

    #

    #

    # fin
