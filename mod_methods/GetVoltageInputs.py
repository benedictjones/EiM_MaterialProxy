import numpy as np

from mod_methods.FetchPerm import IndexPerm


def reorder(arr, index):
    """ Function which re-orders an array acording to the input perm
    """
    # # Error Check inputs
    if len(arr) != len(index):
        print("Error (GetVoltageInputs): reorder input array length does not equal index length")
        raise ValueError(' reorder input array length does not equal index length')

    new_arr = []
    for loc in index:
        new_arr.append(arr[loc])

    return new_arr

def reorder_cols(matrix, index):
    """ Function which re-orders an array acording to the input perm
    """
    new_matrix = np.zeros(matrix.shape)
    for new_idx, old_idx in enumerate(index):
        print(new_idx, old_idx)
        new_matrix[:, new_idx] = matrix[:, old_idx]
    return new_matrix

#

#

#


def get_voltages(x_in, genome, GenomeDict, seq_len):

    # Initialise lists
    Vin = []
    #print(genome)

    # # apply input weights to attribute voltages
    if GenomeDict['InWeight_gene'] == 0:
        for col in x_in:
            Vin.append(col)
    elif GenomeDict['InWeight_gene'] == 1:
        i = 0
        in_weight_list = genome[GenomeDict['in_weight_gene_loc']]

        for col in x_in:
            Vin.append(col*in_weight_list[i])
            i = i + 1

    # # add config voltages
    for v_config in genome[GenomeDict['config_gene_loc']]:
        Vin.append(v_config)
    #print("Vin before shuffle:", Vin)

    # # shuffle
    if GenomeDict['shuffle_gene'] == 1:
        perm_index = genome[GenomeDict['SGI']][0]
        #print("Perm", perm_index)
        order = IndexPerm(seq_len, perm_index)
        Vin_ordered = reorder(Vin, order)
    else:
        Vin_ordered = Vin

    #print(Vin_ordered)

    Vin_ordered = np.asarray(Vin_ordered)
    new_Vin = np.around(Vin_ordered, decimals=5)
    #print("Vin after shuffle:", new_Vin)

    return new_Vin

#

#

def get_voltages_all(X, genome, GenomeDict, seq_len):

    # Initialise lists
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
        #print("Perm", perm_index)
        order = IndexPerm(seq_len, perm_index)
        Vin_ordered = reorder_cols(Vin, order)
    else:
        Vin_ordered = Vin

    #print(Vin_ordered)

    Vin_ordered = np.asarray(Vin_ordered)
    new_Vin = np.around(Vin_ordered, decimals=5)

    return new_Vin
