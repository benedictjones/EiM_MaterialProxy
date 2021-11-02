import numpy as np
import pandas as pd

from mod_load.NewWeightedData_Load import Load_NewWeighted_Data

from sklearn.model_selection import train_test_split


def Load_Data(type, CompiledDict, noise='na', noise_type='per'):

    # normalisation limits (i.e the DE voltage lims)
    min = CompiledDict['spice']['Vmin']
    max = CompiledDict['spice']['Vmax']

    test_size = 0.1
    validation_size = 0.1

    # # Load HDF5 files and data ##
    if CompiledDict['DE']['UseCustom_NewAttributeData'] == 0:

        training_data = CompiledDict['DE']['training_data']

        # # Do we flip the class data?
        if CompiledDict['DE']['training_data'] == 'flipped_2DDS':
            training_data = '2DDS'
            flip = 1
        else:
            flip = 0

        # # Load data
        df = pd.read_hdf('mod_load/data/%s_data.h5' % (training_data), 'all')
        df_all = pd.read_hdf('mod_load/data/%s_data.h5' % (training_data), 'all')

        # # Normalise the data
        df_x = df.iloc[:, :-1]
        raw_df_x = df_x.values
        df_all_x = df_all.iloc[:,:-1]

        #df_all_min = df_all_x.min(axis=0)  # min in each col
        df_all_min = np.min(df_all_x.values)  # min in whole array
        df_x = df_x + (df_all_min*-1)  # shift smallest val to 0
        df_all_x = df_all_x + (df_all_min*-1)  # shift ref matrix too

        #df_all_max = df_all_x.max(axis=0)  # max in each col
        df_all_max = np.max(df_all_x.values)  # max in whole array
        nom_data = df_x / df_all_max  # normalise

        # # scale Normalised the data to a +/-5v input
        diff = np.fabs(min-max)
        data = min + (diff*nom_data)
        data = np.around(data, decimals=3)  # round

        # # Sort and assign data to self
        raw_data_X = data.values
        raw_data_Y = df['class'].values
        if flip == 1:
            raw_data_Y = np.flip(raw_data_Y, axis=0)  # FLIP CLASSES



    elif CompiledDict['DE']['UseCustom_NewAttributeData'] == 1:
        raw_data_X, raw_data_Y = Load_NewWeighted_Data('all', CompiledDict['network']['num_input'], CompiledDict['DE']['training_data'])

    #

    #

    #

    """
    #######################################################################
    Produce the 3 sub datasets
    #######################################################################
    """

    if type != 'all':

        # if we aren't batching we only want 2 datasets
        # Use stratify to try and maitain the same label split as the whole dataset
        x, x_test, y, y_test = train_test_split(raw_data_X, raw_data_Y,
                                                test_size=test_size,
                                                random_state=0,
                                                stratify=raw_data_Y)

        if CompiledDict['DE']['batch_size'] != 0:
            # if we are batching we want to split the training set again to make a validation set
            # Use stratify to try and maitain the same label split as the whole dataset
            vprop = validation_size/(1-test_size)  # calc correct split for desired overal validation %
            x_train, x_valid, y_train, y_valid = train_test_split(x, y,
                                                                 test_size=vprop,
                                                                 random_state=0,
                                                                 stratify=y)



        if CompiledDict['DE']['batch_size'] == 0:
            x_train = x
            y_train = y
            x_valid = 0
            y_valid = 0
        else:
            x_train = x_train
            y_train = y_train
            x_valid = x_valid
            y_valid = y_valid

        if CompiledDict['DE']['batch_size'] > len(x_train):
            e = "Batch Size (%d) is larger then the dataset (%d)!" % (CompiledDict['DE']['batch_size'], len(x_train))
            raise ValueError(e)

    #

    #

    #

    """
    #######################################################################
    Select Which set to return
    #######################################################################
    """

    if type == 'train':
        return_x = x_train
        return_y = y_train

    elif type == 'validation':
        if CompiledDict['DE']['batch_size'] == 0:
            raise ValueError('Not using batching so there is no validation set')
        else:
            return_x = x_valid
            return_y = y_valid

    elif type == 'test':
        return_x = x_test
        return_y = y_test

    elif type == 'all':
        return_x = raw_data_X
        return_y = raw_data_Y

    else:
        raise ValueError('Invalid dataset selected: %s', the_data)


    #

    #

    #

    """
    #######################################################################
    Add Noise
    #######################################################################
    """

    # # Adding in noise to the data, if selected
    if noise != 'na':
        range = max - min

        if noise_type == 'per':
            noise_frac = noise/100
            noise_std = noise_frac*range
        elif noise_type =='std':
            noise_std = noise
        else:
            raise ValueError("Invalid noise type")

        noise = np.random.normal(0, noise_std, np.shape(return_x))
        return_x = return_x + noise




    return return_x, return_y

#

#

# fin
