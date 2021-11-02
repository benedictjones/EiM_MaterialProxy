import numpy as np
import random
from mod_methods.FetchPerm import IndexPerm

from mod_load.LoadData import Load_Data

#


def fetch_data(data, CompiledDict, the_data='train',
               input_data='na', output_data='na',
               noise='na', noise_type='per'):


    # #  Load the training data
    if str(input_data) == 'na':
        data_X, data_Y = Load_Data(the_data, CompiledDict, noise=noise, noise_type=noise_type)
    else:
        data_X = input_data
        data_Y = output_data


    # # load target array (no noise on target)
    if the_data == 'train' or the_data == 'veri':

        if str(output_data) == 'input' or CompiledDict['DE']['IntpScheme'] == 'Ridged_fit':
            data_Y, nn = Load_Data(the_data, CompiledDict)  # no noise
        elif str(output_data) == 'na':
            nn, data_Y = Load_Data(the_data, CompiledDict)
        else:
            data_Y = output_data


    return data_X, data_Y

#
