import numpy as np
import pandas as pd


def Load_NewWeighted_Data(type, num_input, training_data):


    # # normalising factor
    min = -5
    max = 5

    # # Load Data

    df = pd.read_hdf('Module_LoadData/TempData/TempData.h5', type)

    df_x = df.iloc[:, :-1]
    nom_data = df_x / df_x.max(axis=0)

    # # scale Normalised the data to a +/-5v input
    # diff = np.fabs(min-max)
    data = max*nom_data  # don't need to use diff as data already about zero
    data = np.around(data, decimals=3)  # round

    # # Sort and assign data to self
    df_X_Nom = data
    data_Y = df['class'].values

    if training_data == '2DDS' or training_data == 'O2DDS' or training_data == 'con2DDS':
        df_X_Nom.iloc[:,0:2] = df.iloc[:,0:2]

        data_X = df_X_Nom.values
    else:
        data_X = df_X_Nom.values

    #

    #

    if num_input != len(data_X[0, :]):
        print(" ")
        print("Error (LoadTrainingData.py): Num inputs not compatible with training data")
        print("ABORTED")
        exit()

    return data_X, data_Y
