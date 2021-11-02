import numpy as np
import pandas as pd
import random
import time
import math

from mod_load.NewWeightedData_Load import Load_NewWeighted_Data

from sklearn.model_selection import train_test_split


#

#


class FetchDataObj(object):

    def __init__(self, prm,
                 input_data='na', output_data='na'):
        """
        Initialise object
        """

        self.prm = prm
        self.input_data = input_data
        self.output_data = output_data
        self.test_size = 0.20  # % of total dataset

        self.b_idx = 0  # current batch selected
        self.epoch = 0  # current epoch
        self.n_comp = 0  # the number of computations (i.e instances executed)
        self.num_batches = 0
        self.batch_window_size = self.prm['DE']['batch_window_size']
        self.set_up_fin = 0

        self.epoch_ended = 1  # used to manage whether make_batches() can be called
        self.validation_size = 0.20  # % of total dataset

        self._load_data_()
        self._split_data_()

        """
        print("Raw dataset shape:   ", "x", np.shape(self.raw_data_X), ", y", np.shape(self.raw_data_Y))
        print("Train dataset shape: ", "x", np.shape(self.x_train), ", y", np.shape(self.y_train))
        print("Valid dataset shape: ", "x", np.shape(self.x_valid), ", y", np.shape(self.y_valid))
        print("Test dataset shape:  ", "x", np.shape(self.x_test), ", y", np.shape(self.y_test))
        # """

        # # Assigne the batchsize, but see if it a:
        #      1) absolute size (batch_size>1)
        #      2) fractional size (of training dataset) (batch_size<=1)
        batch_size = self.prm['DE']['batch_size']
        if batch_size <= 1:
            # compute fraction of dataset size
            real_size = self.num_train_instances*batch_size

            """# always round up
            if real_size-int(real_size) != 0:
                real_size = int(real_size) + 1"""

            self.batch_size = math.ceil(real_size)
            self.batch_frac_size = batch_size
        else:
            # use absoute bastch size
            self.batch_size = int(batch_size)
            self.batch_frac_size = batch_size/self.num_train_instances

        # # if Batching, make batches!
        if self.batch_size != 0:
            # don't interate the epoch on the initial batch making
            self.make_batches(iterate_epoch=0)
            print("Num Batches:", self.num_batches, "  (Num Raw Batches:", self.num_raw_batches, ")\nBatch Size: List ", self.batch_size_list)

            return_x, return_y = self.get_batch('train', iterate=0)
            self.prev_train_batch = [[-2], [-1], [self.b_idx, return_x, return_y]]  # [prev2, prev1, current] batch
        else:
            self.prev_train_batch = [[-2], [-1], [0, self.x_train, self.y_train]]  # [prev2, prev1, current] batch

        self.set_up_fin = 1

        return

    #

    #

    def fetch_data(self, the_data='train', iterate=0, noise='na', noise_type='per', n_comp_scale=1):
        """
        This function can manage batching OR not batching.
        THis allows this one function to be used in the algorithms, and it can
        adjust it's operation according to the batch_size paramater

        Can add noise to the data.

        """
        #tic = time.time()
        a = self.b_idx
        b = self.epoch

        if self.batch_size == 0:
            # If we are not batching, return the whole training dataset
            return_x, return_y = self.get_data(the_data, iterate)
            self.epoch_ended = 1
        else:
            # If we are batching, return the a batch from the training dataset
            return_x, return_y = self.get_batch(the_data, iterate)

        # print(">", a ,"(%d)" % b, " after:", self.b_idx, "(%d)" % self.epoch, " iterate:", iterate, ", Data:", the_data)

        # # Manage the number of instances that have been loaded so far
        # # (i.e number of material computations)
        self.n_comp = self.n_comp + len(return_x)*n_comp_scale

        # # Adding in noise to the data, if selected
        if noise != 'na':
            range = self.prm['spice']['Vmax'] - self.prm['spice']['Vmin']

            if noise_type == 'per':
                noise_frac = noise/100
                noise_std = noise_frac*range
            elif noise_type == 'std':
                noise_std = noise
            else:
                raise ValueError("Invalid noise type")

            np.random.seed(0)
            noise = np.random.normal(0, noise_std, np.shape(return_x))
            return_x = return_x + noise

        #print(the_data, "data load Time:", time.time()-tic)

        return return_x, return_y

    #

    #

    def get_data(self, the_data, iterate):
        """
        This function returns the appropraite dataset, and can
        itertae the epoch.
        """

        if the_data == 'train':
            return_x = self.x_train
            return_y = self.y_train

            # # Update the previously used/loaded dataset (if not batching)
            if self.batch_size == 0:
                self.prev_train_batch.append([0, return_x, return_y])
                self.prev_train_batch.pop(0)

            # # iterate epoch
            if iterate == 1:
                self.epoch = self.epoch + 1

        elif the_data == 'validation':
            if self.batch_size == 0:
                raise ValueError('Not using batching so there is no validation set')
            else:
                return_x = self.x_valid
                return_y = self.y_valid

        elif the_data == 'test':
            return_x = self.x_test
            return_y = self.y_test

        elif the_data == 'all':
            return_x = self.raw_data_X
            return_y = self.raw_data_Y

        else:
            raise ValueError('Invalid dataset selected: %s', the_data)

        return return_x, return_y
    #

    #

    def get_batch(self, the_data, iterate):
        """
        This function also manages batching.
        When appropriate the batch number and epoch number itertaes.
        """

        """# if the epoch ended on the last call, make new batches
        if self.epoch_ended == 1:
            self.make_batches()"""


        if the_data == 'train':
            data = self.batch_list[self.b_idx]
            return_x = data[0]
            return_y = data[1]

            # # if the batch is set to be itterated
            if iterate == 1:

                # if we have goen though all the batches
                if self.b_idx >= self.num_batches-1:
                    self.epoch_ended = 1  # set toggle to trigger make_batches()
                    self.make_batches()

                # if we still have batches left, itertae batch
                else:
                    self.b_idx = self.b_idx + 1

                # # Update the previously used/loaded dataset
                self.prev_train_batch.append([self.b_idx, return_x, return_y])
                self.prev_train_batch.pop(0)

            #if self.set_up_fin != 0:
            #    print("prev2 batch", self.prev_train_batch[0][0], "prev1 batch", self.prev_train_batch[1][0], "current batch", self.prev_train_batch[2][0])

        elif the_data == 'validation':
            return_x = self.x_valid
            return_y = self.y_valid

        elif the_data == 'test':
            return_x = self.x_test
            return_y = self.y_test

        else:
            raise ValueError('Invalid dataset selected: %s', the_data)

        return return_x, return_y
    #

    #

    def make_batches(self, iterate_epoch=1):
        """
        This creates a new set of batches (from the shuffled raw training data)
        The epoch counter is iterated is passed in argument selcted it

        This function can only be called if the epoch ended toggle is set
        """

        if self.epoch_ended != 1:
            print("The current epoch has not yet finished. Won't re-load & scrambe batches.")
            return

        if self.batch_size > self.num_train_instances:
            e = "Batch Size (%d) is larger then the dataset (%d)!" % (self.batch_size, len(self.x_train))
            raise ValueError(e)


        # # Make the batches
        # (if it is the first run, or on every run if the batches are being shuffled)
        if self.epoch == 0 or self.prm['DE']['batch_scheme'] == 'shuffle' or self.prm['DE']['batch_scheme'] == 'window':

            # # Generated balanced batches
            raw_batch_list = self._batch_generator_()
            self.num_raw_batches = len(raw_batch_list)

            # # Apply Scheme
            if self.prm['DE']['batch_scheme'] == 'window':
                batch_list = []
                if self.batch_window_size >= len(raw_batch_list):
                    raise ValueError("Window size is the same or larger then the number of raw batches!")

                for i in range(len(raw_batch_list)-self.batch_window_size+1):
                    bx = []
                    by = []
                    for j in range(self.batch_window_size):
                        bx.append(raw_batch_list[i+j][0])
                        by.append(raw_batch_list[i+j][1])

                    batch_list.append([np.concatenate(bx), np.concatenate(by)])
            else:
                batch_list = raw_batch_list

            # # Extract information about the batching
            self.batch_list = batch_list
            self.num_batches = len(self.batch_list)
            self.epoch_ended = 0
            self.batch_size_list = []
            for batch in batch_list:
                self.batch_size_list.append(len(batch[0]))

        # # iterate epoch
        if iterate_epoch == 1:
            self.epoch = self.epoch + 1

        # # Reset Batch index
        self.b_idx = 0

        return

    #

    #

    def _split_data_(self):
        """
        Take the data and create the different datasets:
        - training
        - test
        - validation (if batching)
        """


        # if we aren't batching we only want 2 datasets
        # Use stratify to try and maitain the same label split as the whole dataset
        x, x_test, y, y_test = train_test_split(self.raw_data_X, self.raw_data_Y,
                                                test_size=self.test_size,
                                                random_state=1,
                                                stratify=self.raw_data_Y)

        if self.prm['DE']['batch_size'] != 0:
            # if we are batching we want to split the training set again to make a validation set
            # Use stratify to try and maitain the same label split as the whole dataset
            vprop = self.validation_size/(1-self.test_size)  # calc correct split for desired overal validation %
            x_train, x_valid, y_train, y_valid = train_test_split(x, y,
                                                                 test_size=vprop,
                                                                 random_state=1,
                                                                 stratify=y)


        # assign data
        self.x_test = x_test
        self.y_test = y_test

        if self.prm['DE']['batch_size'] == 0:
            self.x_train = x
            self.y_train = y
            self.x_valid = 0
            self.y_valid = 0
        else:
            self.x_train = x_train
            self.y_train = y_train
            self.x_valid = x_valid
            self.y_valid = y_valid



        # assign lengths
        self.num_train_instances = len(self.x_train)

        return
    #

    #

    def _load_data_(self):
        """
        Load in the whole dataset from the storage file.
        Normalise to the max & min voltages.
        """
        # normalisation limits (i.e the DE voltage lims)
        min = self.prm['spice']['Vmin']
        max = self.prm['spice']['Vmax']

        # # Load HDF5 files and data ##
        if self.prm['DE']['UseCustom_NewAttributeData'] == 0:

            training_data = self.prm['DE']['training_data']

            # # Do we flip the class data?
            if 'flipped' in training_data and '2DDS' in training_data:
                training_data = training_data.replace('flipped_','')
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
            data = np.around(data, decimals=4)  # round

            # # Sort and assign data to self
            X_input_attr = data.values
            Y_class = df['class'].values
            if flip == 1:
                Y_class = np.flip(Y_class, axis=0)  # FLIP CLASSES


        elif self.prm['DE']['UseCustom_NewAttributeData'] == 1:
            X_input_attr, Y_class = Load_NewWeighted_Data('all', self.prm['network']['num_input'], self.prm['DE']['training_data'])

        """print("X_input_attr\n", X_input_attr[range(3),:])
        print("Y_class\n", Y_class)
        print(CompiledDict['DE']['training_data'])
        exit()"""

        """ # dont use, might want to just load, errer check in eim_processor
        if len(X_input_attr[0,:]) != CompiledDict['network']['num_input']:
            raise ValueError("Using %d inputs --> for %d attributes!" % (CompiledDict['network']['num_input'], len(X_input_attr[0,:])))
        """

        #

        # # Calc Data Weight due to imbalance
        # > use this to weight fitness scores, e.g. Weighted Cross Entropy Loss
        class_values = np.unique(Y_class)
        nClass_isnatnces = []
        for cdx, cls in enumerate(class_values):
            nClass_isnatnces.append(len(df[df['class']==cls]))
        data_weightings = np.array(nClass_isnatnces)
        #print("Num class instances", data_weightings)
        data_weightings = data_weightings/data_weightings.max()

        #print("data_weightings", data_weightings)
        #exit()

        # # assign data to self
        self.raw_data_X = X_input_attr
        self.raw_data_Y = Y_class
        self.dataset_length = len(self.raw_data_X)
        self.class_weights = data_weightings
        return

    #

    #

    def _batch_generator_(self, print_report=0):

        x = self.x_train
        y = self.y_train

        if print_report == 1:
            print("\nNumber of Instances:", len(x), len(y))
            print("self.batch_size", self.batch_size)

        xdf = pd.DataFrame(x)
        ydf = pd.DataFrame({'class':y})
        df = pd.concat([xdf, ydf], axis=1)

        # # Oversample dataset if there is an imbalance
        if self.prm['DE']['data_oversample'] == 1:
            df2 = df.copy(deep=True)

        class_values = np.unique(self.y_train)
        num_cls = len(class_values)
        class_slip_list = []

        bsize = 0
        for cdx, cls in enumerate(class_values):
            cls_batch_size = round(self.batch_frac_size*len(df[df['class']==cls]))
            ideal_batch_size = self.batch_frac_size*len(df[df['class']==cls])
            bsize += cls_batch_size
            # # slit class up to the batch size
            split_list = []
            # nBatchs = len(df[df['class']==cls])/math.floor(self.batch_frac_size*len(df[df['class']==cls]))
            nBatchs = int(self.batch_frac_size**(-1))

            used_up = 0
            for i in range(0, nBatchs):

                # # Apply random seed if shuffling
                if self.prm['DE']['batch_scheme'] == 'shuffle':
                    seed = np.random.randint(0, 10000)
                else:
                    seed = 0

                # # Find remaining amount of instances in this epoch
                num_remaining = len(df[df['class']==cls])

                # # Sample normally and only use the data given
                if self.prm['DE']['data_oversample'] == 0:

                    if num_remaining < cls_batch_size :
                        sampled = df[df['class']==cls].sample(n=num_remaining, replace=False, random_state=seed)
                    else:
                        sampled = df[df['class']==cls].sample(n=cls_batch_size, replace=False, random_state=seed)
                    df = df.drop(labels=sampled.index, axis=0)
                    split_list.append(np.array(sampled))

                # # Allow over sampling of the given data
                elif self.prm['DE']['data_oversample'] == 1:

                    if num_remaining < cls_batch_size :
                        num_left = cls_batch_size - num_remaining
                        last_sampled = df[df['class']==cls].sample(n=num_remaining, replace=False, random_state=seed)
                        df = df.drop(labels=last_sampled.index, axis=0)
                        used_up = 1
                        extra_sampled = df2[df2['class']==cls].sample(n=num_left, replace=False, random_state=seed)
                        sampled = pd.concat([last_sampled, extra_sampled], axis=0)
                    elif used_up == 1:
                        sampled = df2[df2['class']==cls].sample(n=cls_batch_size, replace=False, random_state=seed)
                    else:
                        sampled = df[df['class']==cls].sample(n=cls_batch_size, replace=False, random_state=seed)
                        df = df.drop(labels=sampled.index, axis=0)
                    split_list.append(np.array(sampled))


            # print("cls:", cls,"\n", split_list)
            class_slip_list.append(split_list)

            if print_report == 1:
                print("Number of Class %d Instances: %d, batch size: %d, ideal batch size: %f" % (cls, len(df[df['class']==cls]), cls_batch_size, ideal_batch_size))


        raw_batch_list = []
        run_out = 0
        for bidx in range(0, nBatchs):
            bx = []
            by = []
            for cdx, cls in enumerate(class_values):

                # # Append the data of currently selected class type
                dat = class_slip_list[cdx][bidx]
                xx = dat[:, :-1]
                yy = dat[:, -1]
                bx.append(xx)
                by.append(yy)

                if len(yy) == 0:
                    run_out = 1


            bx_array = np.concatenate(bx)
            by_array = np.concatenate(by)

            if len(by_array) >= bsize/2 and run_out == 0:
                raw_batch_list.append([bx_array, by_array])
            else:
                # # if run out, or not more then half the number of
                # instances needed, just group the remaining.
                raw_batch_list[-1][0] = np.concatenate([raw_batch_list[-1][0],bx_array])
                raw_batch_list[-1][1] = np.concatenate([raw_batch_list[-1][1],by_array])


        # # Print the batch number or each class
        if print_report == 1:
            for bdx, raw_batch in enumerate(raw_batch_list):
                report = "Batch: %d -" % (bdx)
                for k in range(num_cls):
                    n_C = len(np.where(raw_batch[1] == k+1)[0])
                    report += " Num Class %d: %d," % (k+1, n_C)
                print(report)

        return raw_batch_list
#

#

#

# fin
