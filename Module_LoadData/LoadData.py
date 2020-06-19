import numpy as np
import pandas as pd
import h5py
from Module_LoadData.NewWeightedData_Load import Load_NewWeighted_Data


def Load_Data(type, num_input, num_output_readings, training_data, Split_for_TrainVerify, UseCustom_NewAttributeData):

    # normalisation limits (i.e the DE voltage lims)
    min = -5
    max = 5

    if type == 'train':
        if Split_for_TrainVerify == 0:
            data_type = 'all'
        elif Split_for_TrainVerify == 1:
            data_type = 'training'
    elif type == 'veri':
        data_type = 'veri'
    elif type == 'all':
        data_type = 'all'
        if Split_for_TrainVerify == 1:
            print('Warning (LoadData): Loaded all data but split for train & test is active')
    elif type == 'training':
        data_type = 'training'
        print("Warning (LoadData): Directly selected training. Use 'train' for data to be auto adjusted by 'Split_for_TrainVerify'.")
    else:
        raise ValueError('(LoadData) invalid type to load')

    if UseCustom_NewAttributeData == 0:
        # # import the data for 2D class
        if training_data == '2DDS':

            # # Load Data
            df = pd.read_hdf('Module_LoadData/data/2DDS_data.h5', data_type)
            raw_data = df.values

            # # Sort and assign data to self
            X_input_attr = df.iloc[:, :-1].values
            Y_class = df['class'].values
            # Y_class = np.flip(self.Y_class, axis=0)  # FLIP CLASSES

        # # import Mammographic Mass Data Set (mmds) Data, which has been reduced
        elif training_data == 'MMDS':

            df = pd.read_hdf('Module_LoadData/data/MMDS_data.h5', data_type)

            raw_data = df.values
            # print('raw data\n', raw_data)

            df_x = df.iloc[:, :-1]
            nom_data = df_x / df_x.max(axis=0)

            # # scale Normalised the data to a +/-5v input
            diff = np.fabs(min-max)
            data = min + (diff*nom_data)
            data = np.around(data, decimals=3)  # round

            # # Sort and assign data to self
            X_input_attr = data.values
            Y_class = df['class'].values

        # Import the Overlapping 2DDS: gausian about (0,0)-class1 and (1,1)-class2
        elif training_data == 'O2DDS':

            # # Load Data
            df = pd.read_hdf('Module_LoadData/data/O2DDS_data.h5', data_type)
            raw_data = df.values

            # # Sort and assign data to self
            X_input_attr = df.iloc[:, :-1].values
            Y_class = df['class'].values

        # Import the concentric 2DDS
        elif training_data == 'con2DDS':

            # # Load Data
            df = pd.read_hdf('Module_LoadData/data/con2DDS_data.h5', data_type)
            raw_data = df.values

            # # Sort and assign data to self
            X_input_attr = df.iloc[:, :-1].values
            Y_class = df['class'].values

        # # error check
        if num_input != len(raw_data[0, :]) - num_output_readings:
            print(" ")
            print("Error (LoadTrainingData.py): Num inputs not compatible with training data")
            print("ABORTED")
            exit()

    elif UseCustom_NewAttributeData == 1:
        X_input_attr, Y_class = Load_NewWeighted_Data(data_type, num_input, num_output_readings, training_data)
    #

    #
    #print("X_input_attr\n", X_input_attr[range(3),:])
    #print("Y_class\n", Y_class)
    #print("mean x", np.mean(X_input_attr))

    #print("mean y", np.mean(Y_class))
    #exit()

    return X_input_attr, Y_class

#

#

# fin
