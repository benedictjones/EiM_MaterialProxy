# # Top Matter
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import scipy

import os
from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap

from Module_Analysis.plt_basic import plotting
from Module_Analysis.plt_ani import Animation
from Module_Analysis.plt_mg import material_graphs


''' Description:
This class file is designed to provide various functions to analise saved data
from the looped DE.
'''
#########################################################################
# Plot Average
#########################################################################


class analysis(Animation, plotting, material_graphs):

    def __init__(self, dir_list, dir_RefNameList=0, save_loc=0, format='pdf'):

        plotting.__init__(self)
        #Animation.__init__(self)  # comment this out to stop animation object inclusion
        material_graphs.__init__(self)

        # # Check for a list (i.e multi input)
        if isinstance(dir_list, list) is True:
            self.MultiDirAly = 1
            if save_loc == 0:
                print("Error (Analysis.py): Must use a custom save file for analysis on multiple files")
                print("ABORTED")
                exit()
        else:
            self.MultiDirAly = 0
            dir_list = [dir_list]  # place the non-list dir into a list, so it is in the correct format

        # # Consider input data_dir to see if it is an experiment or not
        not_local = 0
        k = 0
        for dir in dir_list:
            not_local_path = ''
            print('THE DIR:', dir)
            split_data_dir = dir.split("/")

            if split_data_dir[0] != 'Experiment_List' and split_data_dir[0] != 'Results':
                not_local = 1

            # check for the indicaters in the split file dir
            for bit in split_data_dir:
                if bit == 'Experiment_List':
                    current_dir = 'Experiment_List'
                    break
                if bit == 'Results':
                    current_dir = 'Results'
                    break
                elif not_local == 1:
                    not_local_path = not_local_path + bit + '/'

            if current_dir == 'Experiment_List':
                if k == 0:
                    self.exp = 1
                elif k != 0 and self.exp == 0:
                    print("Error (Analysis.py): Input files must be of the same type")
                    print("ABORTED")
                    exit()

            elif current_dir == 'Results':
                if k == 0:
                    self.exp = 0
                elif k != 0 and self.exp == 1:
                    print("Error (Analysis.py): Input files must be of the same type")
                    print("ABORTED")
                    exit()
            k = k + 1

        print("dir_RefNameList", dir_RefNameList)
        dir_RefNameList_original = dir_RefNameList
        # # create generic name for each file if the Namelist is empty
        if dir_RefNameList == 0 and self.MultiDirAly == 1:
            dir_RefNameList = []
            for i in range(len(dir_list)):
                file_name = "file%d" % (i)
                dir_RefNameList.append(file_name)
        elif dir_RefNameList == 0 and self.MultiDirAly == 0:  # leave empty is only one file
            dir_RefNameList = [""]
        elif isinstance(dir_RefNameList, list) is False:  # if on;y one file (not a list) s passed in
            if len([dir_RefNameList]) == len(dir_list):
                dir_RefNameList = [dir_RefNameList]
            else:
                print("Error (Analysis.py): Inputed labels for the files is invalid ")
                print("ABORTED")
                exit()
        elif isinstance(dir_RefNameList, list) is True and len(dir_RefNameList) != len(dir_list) and self.MultiDirAly == 1:
            print("Warning (Analysis.py): Passed in dir_RefNameList is the wrong length \n Setting to defualt and continue... ")
            dir_RefNameList = []
            for i in range(len(dir_list)):
                file_name = "file%d" % (i)
                dir_RefNameList.append(file_name)
        elif isinstance(dir_RefNameList, list) is True and len(dir_RefNameList) == len(dir_list) and self.MultiDirAly == 1:
            print("Passes in file name list is GOOD")
        else:
            print("bad hit")
            print("Warning (Analysis.py): Passed in dir_RefNameList is longer than 1 but only one file is in use \n Setting to nothing and continue... ")
            dir_RefNameList = [""]

        # # # # # # # # # # # #
        # Create object data lists
        self.data_dir_list = []
        self.format = format
        num_exp_loops = []
        self.Param_array = []
        self.ParamChanging = []
        self.dpi = 400

        # Set save location
        if save_loc == 0 and self.MultiDirAly == 0:
            self.save_dir = dir_list[0]
            #self.save_dir = os.path.join(os.pardir, dir_list[0])
        else:
            self.save_dir = 'CustomSave/%s' % (save_loc)
            #self.save_dir = os.path.join(os.pardir, cust_path)

            # if the new folder does not yet exist, create it
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        print("** self.save_dir: ", self.save_dir)


        # Extract Data depending on single run, or experiment run, or from a list of experiments/result files
        for dir in dir_list:

            if self.exp == 0:

                self.num_experiments = 1
                num_exp_loops.append(self.num_experiments)

                self.data_dir_list.append(dir)
                self.Param_array.append("n/a - Not an Exp")

            elif self.exp == 1:

                current_data_dir_list = []
                # READ a list of strings from the DataDir.csv
                data_dir_list_file = '%s/DataDir.csv' % (dir)
                dir_file = open(data_dir_list_file, "r")
                rows = dir_file.read().split("\n") # "\r\n" if needed
                for line in rows:
                    if line != "": # add other needed checks to skip titles
                        cols_list = line.split(",")
                        col = cols_list[0]

                        # if the data is being accessed from an external dir
                        if not_local == 1:
                            col = not_local_path + col

                        self.data_dir_list.append(col)
                        current_data_dir_list.append(col)

                self.num_experiments = len(current_data_dir_list)
                num_exp_loops.append(self.num_experiments)

                # READ the parameter list from the saved Param_array.txt
                Param_array_dir = '%s/Param_array.txt' % (dir)
                with open(Param_array_dir, 'r') as file:
                    param_array = file.read().split('\n')
                param_array.remove('')
                self.ParamChanging.extend(param_array[-1])
                del param_array[-1]
                self.Param_array.extend(param_array)



        # # Create correctly sized list of file name references
        self.FileRefNames = []
        k = 0
        for dir_Name in dir_RefNameList:
            for i in range(num_exp_loops[k]):
                self.FileRefNames.append(dir_Name)
            k = k + 1

        print("self.FileRefNames are:", self.FileRefNames)
        print("self.Param_array is:", self.Param_array)
        print("self.data_dir_list:", self.data_dir_list)
        #print(dir_list)




#

#

#

#

#

#

#

#

# fin
