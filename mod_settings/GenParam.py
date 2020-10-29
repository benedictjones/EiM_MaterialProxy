import numpy as np
import multiprocessing
import yaml

from mod_load.LoadData import Load_Data
from mod_methods.FetchPerm import NPerms
from mod_settings.Set_Change import ChangeSettings


def GenSimParam(param_file='', algorithm='na', save_param_file=0, pt='na', NoGenes=0, **kwargs):

    """
    Function which loads a param_file (using the above argument to select
    the end of the filename param%s). Depending on the parmaters, other
    properties are added such as number of processors, or the bounds for the
    different angorithm decision variables (genes).

    This is then saved to a temporary param file of the same name as the Loaded
    param_file.
    Note that if save_param_file is not zero, then the destination temporary
    paramater file name can be selected independently of the original loaded
    file.

    """

    if save_param_file == 0:
        print("\nGen param file:", param_file)
    else:
        print("\nGen param file:", save_param_file)

    """# # Load Params from file (if not passed in) # #"""
    if isinstance(param_file, dict):
        CompiledDict = param_file
    else:
        with open(r'mod_settings/param%s.yaml' % (str(param_file))) as file:
            CompiledDict = yaml.full_load(file)

    """# # Apply Custom kargs # #"""
    CompiledDict = ChangeSettings(CompiledDict, pt=pt, **kwargs)


    """# # Edit and add to Params # #"""

    # # Add param file name (used for material save name)
    if save_param_file == 0:
        CompiledDict['param_file'] = param_file
    else:
        CompiledDict['param_file'] = save_param_file

    # #  assign the algorithm
    if algorithm != 'na':
        CompiledDict['algorithm'] = algorithm
        CompiledDict['ptype'] = 'na'
    else:
        raise ValueError("Select an Algorithm!")

    # # extract the processor type from algorithm selected
    if 'EiM' in CompiledDict['algorithm']:
        CompiledDict['ptype'] = 'EiM'  # processor type
    else:
        raise ValueError("Error (Settings): The processor type (ptype) cannot be determined from the algorithm: %s" % (CompiledDict['algorithm']))




    # # Multiprocessor Settings (for DE simulation process only, MG currently seperate)
    if CompiledDict['num_processors'] == 'max':
        # fetch the local PC's number of cores
        print("number of cores:", multiprocessing.cpu_count())
        CompiledDict['num_processors'] = int(multiprocessing.cpu_count())
    elif CompiledDict['num_processors'] > multiprocessing.cpu_count():
        print("Maximum number of cores is:", multiprocessing.cpu_count())
        print("Number of corse being set the the maximum possible.")
        CompiledDict['num_processors'] = int(multiprocessing.cpu_count())
    else:
        CompiledDict['num_processors'] = CompiledDict['num_processors']

    # # add model to spice section
    CompiledDict['spice']['model'] = CompiledDict['network']['model']

    # # If only one layer, remove per layer values
    if CompiledDict['spice']['num_layers'] == 1:
        CompiledDict['spice']['ConfigPerLayer'] = 'na'
        CompiledDict['spice']['NodesPerLayer'] = 'na'

    """ # Johns limits
    CompiledDict['spice']['material_a_min'] = 33  # nano (or 3.3e-8)
    CompiledDict['spice']['material_a_max'] = 170  # nano (or 1.7e-7)
    CompiledDict['spice']['material_b_min'] = 280  # nano (or 2.8e-7)
    CompiledDict['spice']['material_b_max'] = 960  # nano (or 9.6e-7)
    # """

    #print(">>>>", CompiledDict['DE']['training_data'])
    train_data_X, train_data_Y = Load_Data('train', CompiledDict)

    class_values = np.unique(train_data_Y)
    counts = len(class_values)
    print(class_values, counts)

    if CompiledDict['genome']['max_number_bands'] == 'defualt':
        max_number_bands = counts  # assign the number of classes in the data
    else:
        max_number_bands = CompiledDict['genome']['max_number_bands']

    if max_number_bands < counts:
        raise ValueError(" The number of bands should not be set to less than the number of classes!")
    CompiledDict['genome']['min_number_bands'] = counts
    CompiledDict['DE']['num_classes'] = counts
    CompiledDict['genome']['max_number_bands'] = max_number_bands
    CompiledDict['genome']['min_class_value'] = int(class_values.min())
    CompiledDict['genome']['max_class_value'] = int(class_values.max())

    if CompiledDict['DE']['IntpScheme'] == 'HOW':
        if CompiledDict['DE']['num_readout_nodes'] == 'na':
            if CompiledDict['network']['num_output'] != len(class_values):
                raise ValueError("For HOW you must use the same number of output nodes as output labels.\n Num output nodes = %d, num classes = %d" % (CompiledDict['network']['num_output'], len(class_values)))
        else:
            if CompiledDict['DE']['num_readout_nodes'] != len(class_values):
                raise ValueError("For HOW you must use the same number of logican readout output nodes as output labels.\n Num readout nodes = %d, num classes = %d" % (CompiledDict['DE']['num_readout_nodes'], len(class_values)))

    if CompiledDict['ptype'] == 'EiM' and CompiledDict['DE']['num_readout_nodes'] != 'na' and CompiledDict['genome']['OutWeight_gene'] != 1:
        if CompiledDict['DE']['num_readout_nodes'] != 1:
            # raise ValueError("Cannot use a readout layer unless output weights are turned on!")
            raise ValueError("Cannot use a readout layer (with more then 1 node) unless output weights are turned on!")

    if CompiledDict['DE']['IntpScheme'] != 'HOW':
        CompiledDict['genome']['HOWperm_gene'] = 0

    if CompiledDict['genome']['perm_crossp_model'] != 'none' and CompiledDict['genome']['shuffle_gene'] == 0:
        raise ValueError("Error (Settings): Perm_crossp model set, but shuffle gene is off!")




    # # # # # # # # # # # # # # # # # #
    # # # Initialise the bounds matrix

    if NoGenes != 1:
        bounds = []
        j = 0  # Set gene counter and gene loc trackers

        # # Set bounds for voltages
        if CompiledDict['network']['num_config'] > 0:
            temp = []
            for i in range(CompiledDict['network']['num_config']):
                temp.append([CompiledDict['spice']['Vmin'], CompiledDict['spice']['Vmax']])
            bounds.append(temp)
            CompiledDict['genome']['config_gene_loc'] = j
            print("config_gene_loc", CompiledDict['genome']['config_gene_loc'])
            j = j + 1

        #

        # # Set bounds for shuffle gene/decision variable
        # perm read by an algorithm (no need to generate whole list)
        num_in_V = CompiledDict['network']['num_config'] + CompiledDict['network']['num_input']
        print("num_in_V", num_in_V)
        num_perms = NPerms(num_in_V)
        print("Number of perms:", num_perms)
        # assign the shuffle gene bounds accrotding to num perms
        if CompiledDict['genome']['shuffle_gene'] == 1:
            temp = [0, num_perms-1]
            bounds.append([temp])
            CompiledDict['genome']['SGI'] = j  # index for the shuffle gene (Shuffle Gene Index)
            print("SGI", CompiledDict['genome']['SGI'])
            j = j + 1

        #

        # # Produce input weight bounds
        if CompiledDict['genome']['InWeight_gene'] == 1:
            temp = []
            for i in range(CompiledDict['network']['num_input']):
                temp.append([-CompiledDict['genome']['MaxInputWeight'], CompiledDict['genome']['MaxInputWeight']])
            bounds.append(temp)
            CompiledDict['genome']['in_weight_gene_loc'] = j
            print("in_weight_gene_loc", CompiledDict['genome']['in_weight_gene_loc'])
            j = j + 1

        #

        # # Produce output weight bounds
        if CompiledDict['genome']['OutWeight_gene'] == 1:
            temp = []
            if CompiledDict['DE']['num_readout_nodes'] != 'na':
                num_output_weights = CompiledDict['network']['num_output']*CompiledDict['DE']['num_readout_nodes']
            else:
                num_output_weights = CompiledDict['network']['num_output']
            print(">>>>", num_output_weights)
            print(CompiledDict['DE']['num_readout_nodes'] )
            print(CompiledDict['network']['num_output'])
            for i in range(num_output_weights):
                temp.append([-CompiledDict['genome']['MaxOutputWeight'], CompiledDict['genome']['MaxOutputWeight']])
            bounds.append(temp)
            CompiledDict['genome']['out_weight_gene_loc'] = j
            print("out_weight_gene_loc", CompiledDict['genome']['out_weight_gene_loc'])
            j = j + 1
        else:
            CompiledDict['genome']['out_weight_gene_loc'] = 'na'

        #

        # # Produce PulseWidth_gene bounds
        if CompiledDict['genome']['PulseWidth_gene'] == 1:
            if CompiledDict['spice']['sim_type']  != 'sim_trans_pulse':
                raise ValueError('To vary Pulse Width by a gene, the sim typr must be sim_trans_pulse')
            temp = [CompiledDict['genome']['MinPulseWidth'], CompiledDict['genome']['MaxPulseWidth']]
            bounds.append([temp])  # ms  - bounds to select pulse width
            CompiledDict['genome']['PulseWidth_gene_loc'] = j
            print("PulseWidth_gene_loc", CompiledDict['genome']['PulseWidth_gene_loc'])
            j = j + 1

        #

        # # Produce BandNumber_gene bounds
        if CompiledDict['genome']['BandNumber_gene'] == 1:
            if CompiledDict['genome']['min_number_bands'] == CompiledDict['genome']['max_number_bands'] and band_scheme != 'defualt':
                raise ValueError('BandNumber_gene set so number of bands can vary, but max band number not larger then if gene is off.')
            temp = [CompiledDict['genome']['min_number_bands'], CompiledDict['genome']['max_number_bands']+0.99]
            bounds.append([temp])
            CompiledDict['genome']['BandNumber_gene_loc'] = j
            print("BandNumber_gene_loc", CompiledDict['genome']['BandNumber_gene_loc'])
            j = j + 1

        #

        # # Produce BandWidth_gene bounds
        if CompiledDict['genome']['BandEdge_gene'] == 1:
            if CompiledDict['DE']['IntpScheme'] != 'band':
                raise ValueError('To very band edges, use a banding scheme.')
            temp = []
            temp.append([CompiledDict['spice']['Vmin'], 0])  # left edge
            temp.append([0, CompiledDict['spice']['Vmax']]) # right edge
            bounds.append(temp)
            CompiledDict['genome']['BandEdge_gene_loc'] = j
            print("BandEdge_gene_loc", CompiledDict['genome']['BandEdge_gene_loc'])
            j = j + 1

        #

        # # Produce BandWidth_gene bounds
        if CompiledDict['genome']['BandWidth_gene'] == 1:
            if CompiledDict['DE']['IntpScheme'] != 'band':
                raise ValueError('To very band width, use a banding scheme.')
            temp = []
            for i in range(max_number_bands-1):
                temp.append([-1, 1])
            bounds.append(temp)
            CompiledDict['genome']['BandWidth_gene_loc'] = j
            print("BandWidth_gene_loc", CompiledDict['genome']['BandWidth_gene_loc'])
            j = j + 1

        #

        # # Produce BandClass_gene bounds
        if CompiledDict['genome']['BandClass_gene'] == 1:
            if CompiledDict['DE']['IntpScheme'] != 'band':
                raise ValueError('To very band labels, use a banding scheme.')
            temp = []
            for i in range(max_number_bands):
                temp.append([class_values.min(), class_values.max()+0.99])
            bounds.append(temp)
            CompiledDict['genome']['BandClass_gene_loc'] = j
            print("BandClass_gene_loc", CompiledDict['genome']['BandClass_gene_loc'])
            j = j + 1

        #

        # # Set bounds for Highest output wins output order gene
        # perms read by an algorithm (no need to generate whole list)
        num_perms = NPerms(CompiledDict['network']['num_output'])
        print("Number HOW output order perms:", num_perms)
        # assign the shuffle gene bounds accrotding to num perms
        if CompiledDict['genome']['HOWperm_gene'] == 1:
            temp = [0, num_perms-1]
            bounds.append([temp])
            CompiledDict['genome']['HOW_perm_gene_loc'] = j  # index for the HOW_perm_gene_loc
            print("HOW_perm_gene_loc", CompiledDict['genome']['HOW_perm_gene_loc'])
            j = j + 1

        #

        # for a Neuromorphic network (NN) rather then a Random Netowrk (RN)
        # generate a weight value, this allows connection weights (resistance) to
        # be calaculated
        if CompiledDict['network']['model'][-2:] == 'NN':
            num_bais_nodes = CompiledDict['network']['num_input'] + CompiledDict['network']['num_output'] + CompiledDict['network']['num_config']
            CompiledDict['network']['num_nodes'] = num_bais_nodes + CompiledDict['spice']['NodesPerLayer']*(CompiledDict['spice']['num_layers']-1) + CompiledDict['spice']['ConfigPerLayer']*(CompiledDict['spice']['num_layers'])
            temp = []
            for i in range(CompiledDict['network']['num_nodes']):
                temp.append([0, 1])
            bounds.append(temp)
            CompiledDict['genome']['NN_weight_loc'] = j
            print("NN_weight_loc", CompiledDict['genome']['NN_weight_loc'])
            j = j + 1

            if CompiledDict['spice']['ConfigPerLayer'] >= 1:
                temp = []
                for i in range(CompiledDict['spice']['ConfigPerLayer']*(CompiledDict['spice']['num_layers'])):
                    temp.append([-5, 5])
                bounds.append(temp)
                CompiledDict['genome']['NN_layerConfig_loc'] = j
                print("NN_layerConfig_loc", CompiledDict['genome']['NN_layerConfig_loc'])
                j = j + 1
            else:
                CompiledDict['genome']['NN_layerConfig_loc'] = 'na'

        #

        # # Record the number of genomes (i.e dimension of decision variables)
        dim = 0
        grouping = []
        for gen_group in bounds:
            dim = dim + len(gen_group)
            grouping.append(len(gen_group))

        CompiledDict['DE']['dimensions'] = dim
        print("Decision vector Dimension:", dim)
        CompiledDict['DE']['bounds'] = bounds
        CompiledDict['genome']['grouping'] = grouping

        # # Infomation print out
        print("The bounds:\n", bounds)

        # remove additional keys
        if 'encoder' in CompiledDict.keys():
            del CompiledDict['encoder']

        if 'decoder' in CompiledDict.keys():
            del CompiledDict['decoder']

    else:
        CompiledDict['DE']['popsize'] = CompiledDict['encoder']['popsize']*CompiledDict['decoder']['popsize']

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Save the dictionary to a file:
    if save_param_file == 0:
        # Save the temp file under the same name as the loaded file
        with open(r'Temp_Param%s.yaml' % (str(param_file)), 'w') as sfile:
            yaml.dump(CompiledDict, sfile)
    else:
        # Save the temp file under a different name than the loaded file
        with open(r'Temp_Param%s.yaml' % (str(save_param_file)), 'w') as sfile:
            yaml.dump(CompiledDict, sfile)

    return CompiledDict, str(CompiledDict['param_file'])

    #

    #

    # fin
