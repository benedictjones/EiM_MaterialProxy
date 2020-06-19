import numpy as np
import time

from Module_Functions.FetchPerm import IndexPerm

""" Function to take produce the modified V inuts to the material network
"""


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

#

#

#


def get_voltages(x_in, genome, InWeight_gene, InWeight_sheme, in_weight_gene_loc,
                 shuffle_gene, SGI, seq_len, config_gene_loc):

    # Initialise lists
    Vin = []

    # # appy input weights to attribute voltages
    if InWeight_gene == 0:
        for col in x_in:
            Vin.append(col)
    elif InWeight_gene == 1:
        i = 0
        in_weight = np.arange(in_weight_gene_loc[0], in_weight_gene_loc[1])
        for col in x_in:
            Vin.append(col*genome[in_weight[i]])
            i = i + 1

    # # add config voltages
    for j in range(config_gene_loc[0], config_gene_loc[1]):
        Vin.append(genome[j])

    # # shuffle
    if shuffle_gene == 1:
        perm_index = int(genome[SGI])
        order = IndexPerm(seq_len, perm_index)
        Vin_ordered = reorder(Vin, order)
    else:
        Vin_ordered = Vin

    new_Vin = np.around(Vin_ordered, decimals=4)

    return new_Vin

#

#



#

#

#

# fin
