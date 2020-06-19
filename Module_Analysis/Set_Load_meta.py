import pickle
import os

''' # Load and return the settings parameter dictionary from file
'''


def LoadMetaData(dir):

    save_file = "%s/Experiment_MetaData.dat" % (dir)

    #print("the file:", save_file)
    #print("os.listdir() \n", os.listdir() )

    with open(save_file, 'rb') as f:
        MetaData = pickle.load(f)

    return MetaData

    #

    # fin
