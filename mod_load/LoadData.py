import numpy as np
import pandas as pd

from mod_load.NewWeightedData_Load import Load_NewWeighted_Data

def Load_Data(type, CompiledDict):

    # normalisation limits (i.e the DE voltage lims)
    min = -5
    max = 5

    #full_path = os.path.realpath(__file__)
    #path, filename = os.path.split(full_path)
    #print(path + ' --> ' + filename + "\n")


    # # Translate selected inputs # #
    if type == 'train':
        if CompiledDict['DE']['TestVerify'] == 0:
            data_type = 'all'
        elif CompiledDict['DE']['TestVerify'] == 1:
            data_type = 'training'
    elif type == 'veri':
        data_type = 'veri'
    elif type == 'all':
        data_type = 'all'
        if CompiledDict['DE']['TestVerify'] == 1:
            print('Warning (LoadData): Loaded all data but split for train & test is active')
    elif type == 'training':
        data_type = 'training'
        print("Warning (LoadData): Directly selected training. Use 'train' for data to be auto adjusted by 'Split_for_TrainVerify'.")
    else:
        raise ValueError('(LoadData) invalid type to load: %s' % (type))

    # # Load HDF5 files and data ##
    if CompiledDict['DE']['UseCustom_NewAttributeData'] == 0:

        # # Do we flip the class data?
        if CompiledDict['DE']['training_data'] == 'flipped_2DDS':
            CompiledDict['DE']['training_data'] = '2DDS'
            flip = 1
        else:
            flip = 0

        # # Load data
        df = pd.read_hdf('mod_load/data/%s_data.h5' % (CompiledDict['DE']['training_data']), data_type)
        df_all = pd.read_hdf('mod_load/data/%s_data.h5' % (CompiledDict['DE']['training_data']), 'all')

        # # Normalise the data
        df_x = df.iloc[:, :-1]
        df_all_x = df_all.iloc[:,:-1]
        df_x = df_x + (df_all_x.min(axis=0)*-1)  # shift smallest val to 0
        df_all_x = df_all_x + (df_all_x.min(axis=0)*-1)  # shift ref matrix too
        nom_data = df_x / df_all_x.max(axis=0)  # normalise

        # # scale Normalised the data to a +/-5v input
        diff = np.fabs(min-max)
        data = min + (diff*nom_data)
        data = np.around(data, decimals=3)  # round

        # # Sort and assign data to self
        X_input_attr = data.values
        Y_class = df['class'].values
        if flip == 1:
            Y_class = np.flip(Y_class, axis=0)  # FLIP CLASSES

        """# # error check
        if CompiledDict['network']['num_input'] != len(X_input_attr[0, :]):
            raise ValueError("Error (LoadTrainingData.py): Num inputs not compatible with training data")"""

    elif CompiledDict['DE']['UseCustom_NewAttributeData'] == 1:
        X_input_attr, Y_class = Load_NewWeighted_Data(data_type, CompiledDict['network']['num_input'], CompiledDict['DE']['training_data'])






    """print("X_input_attr\n", X_input_attr[range(3),:])
    print("Y_class\n", Y_class)
    print(CompiledDict['DE']['training_data'])
    exit()"""

    return X_input_attr, Y_class

#

#

# fin
