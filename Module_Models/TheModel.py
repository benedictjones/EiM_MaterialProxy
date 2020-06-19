import numpy as np
import h5py
import multiprocessing
import time

import sys
import os
import gc
import pickle

# My imports

from Module_LoadData.LoadData import Load_Data
from Module_Settings.Set_Load import LoadSettings
from Module_SPICE.resnet import ResNet
from Module_SPICE.resnet_LoadRun import LoadRun_model
from Module_InterpSchemes.interpretation_scheme import GetClass
from Module_InterpSchemes.fitness_scheme import GetFitness

from Module_Functions.GetVoltageInputs import get_voltages


# # # # # # # # # # # #
# Produce a print disable function


def blockPrint():  # used to plock print outs
    sys.stdout = open(os.devnull, 'w')


def enablePrint():  # Restore
    sys.stdout = sys.__stdout__


class model(object):

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Initialisation of class
    '''
    def __init__(self):
        # # # # # # # # # # # #
        # Import object and assign to self
        self.network_obj = ResNet()

        # # # # # # # # # # # #
        # Assign setting to self from setting dict file
        self.ParamDict = LoadSettings()

        self.ReUse_dir = self.ParamDict['ReUse_dir']
        self.UseCustom_NewAttributeData = self.ParamDict['UseCustom_NewAttributeData']

        if self.ParamDict['ReUse_Circuits'] == 1 and self.ParamDict['ReUse_dir'] != 'na':
            self.LoadModelDir = self.ParamDict['ReUse_dir']
            print("Circuit is being re_used!!!!")

            # Load param from resnet:
            loadnet_file = "%s/Experiment_MetaData.dat" % (self.ParamDict['ReUse_dir'])
            with open(loadnet_file, 'rb') as f:
                loadnet_ParamDict = pickle.load(f)
                if self.ParamDict['num_input'] != loadnet_ParamDict['num_input']:
                    print("loaded network's number of inputs:", loadnet_ParamDict['num_input'])
                    print("current setting number of inputs:", self.ParamDict['num_input'])
                    raise ValueError('(TheModel): loaded network num inputs != current runs num inputs')
                if self.ParamDict['num_config'] != loadnet_ParamDict['num_config']:
                    print("loaded network's number of config:", loadnet_ParamDict['num_config'])
                    print("current setting number of config:", self.ParamDict['num_config'])
                    raise ValueError('(TheModel): loaded network num_config != current runs num_config')
                if self.ParamDict['num_output'] != loadnet_ParamDict['num_output']:
                    print("loaded network's number of outputs:", loadnet_ParamDict['num_output'])
                    print("current setting number of outputs:", self.ParamDict['num_output'])
                    raise ValueError('(TheModel): loaded network num_output != current runs num_output')


        else:
            self.LoadModelDir = self.ParamDict['SaveDir']
            self.ReUse_dir = 'na'
            #print("No ReUse")

        # # Create Temp data file location for MP data if it doesn't exist
        if not os.path.exists('Temp_MPdata'):
            os.makedirs('Temp_MPdata')


        # # # # # # # # # # # #
        # Load the training data
        train_data_X, train_data_Y = Load_Data('train', self.ParamDict['num_input'],
                                            self.ParamDict['num_output_readings'],
                                            self.ParamDict['training_data'],
                                            self.ParamDict['TestVerify'],
                                            self.UseCustom_NewAttributeData)


        # # # # # # # # # # # #
        # Peform Error checking
        #print("self.ParamDict['num_input']:", self.ParamDict['num_input'])
        #print("train_data_X[0] is:", train_data_X[0], "with length:", len(train_data_X[0]))
        if len(train_data_X[0]) != self.ParamDict['num_input']:
            print(" ")
            print("Error (TheModel): training data dimension must match number of input voltages:")
            raise ValueError('(TheModel): training data dimension must match number of input voltages')

        """# print("width of y data",len(self.train_data_Y[0,:]))
        if self.ParamDict['num_output_readings'] != len(train_data_Y[0,:]):
            print(" ")
            print("Error (TheModel): num output readings does not equal training data output dimensions")
            print("ABORTED")
            exit()"""

    #

    #

    #

    #

    #

    #

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    The objective function which returns a fitness score using MULTIPROCESSING
    '''
    def fobj_mp(self, genome_chunk, pos, SaveNetText):
        tic = time.time()
        test_list = []

        # # Assign Values from object to a 'local' variable
        num_input = self.ParamDict['num_input']
        num_config = self.ParamDict['num_config']
        IntpScheme = self.ParamDict['IntpScheme']
        num_output = self.ParamDict['num_output']
        num_output_readings = self.ParamDict['num_output_readings']
        OutWeight_gene = self.ParamDict['OutWeight_gene']
        OutWeight_scheme = self.ParamDict['OutWeight_scheme']
        shuffle_gene = self.ParamDict['shuffle_gene']
        LoadModelDir = self.LoadModelDir
        num_nodes = num_input + num_config + num_output
        FitScheme = self.ParamDict['FitScheme']
        max_OutResponce = self.ParamDict['max_OutResponce']

        EN_Dendrite = self.ParamDict['EN_Dendrite']
        NumDendrite = self.ParamDict['NumDendrite']

        InWeight_gene = self.ParamDict['InWeight_gene']
        InWeight_sheme = self.ParamDict['InWeight_sheme']
        in_weight_gene_loc = self.ParamDict['in_weight_gene_loc']
        SGI = self.ParamDict['SGI']
        config_gene_loc = self.ParamDict['config_gene_loc']
        out_weight_gene_loc = self.ParamDict['out_weight_gene_loc']

        # Only 1 process can save the circuit topology
        if pos == 0 and SaveNetText == 1:
            SNT = 1
        else:
            SNT = 0


        # Load the training data (faster to load then pass in!)
        train_data_X, train_data_Y = Load_Data('train', num_input, num_output_readings, self.ParamDict['training_data'], self.ParamDict['TestVerify'], self.UseCustom_NewAttributeData)


        # Calculate fitness and error for every genome in the passed in chunk
        fitness_genome = []
        error_genome = []
        responceY_genome = []
        Vout_genome = []

        for genome in genome_chunk:
            #tic_s = time.time()

            # # Initialise arrays
            class_out_list = []
            responceY_list = []  # initialise BWs list
            Vout_list = []

            # # # # # # # # # # # #
            # Iterate over all training data to find elementwise error
            for loop in range(len(train_data_X[:, 0])):


                x_in = train_data_X[loop, :]
                seq_len = num_config + num_input
                V_in_mod = get_voltages(x_in, genome, InWeight_gene, InWeight_sheme, in_weight_gene_loc,
                                        shuffle_gene, SGI, seq_len, config_gene_loc)

                #

                # # # # # # # # # # # #
                # Calculate the voltage output of input training data
                # Load circuit if possible, only generate+run on the first loop
                #print("Pos", pos, "training data loop:", loop)
                #ticr = time.time()

                if self.ParamDict['repetition_loop'] == 0 and self.ReUse_dir == 'na':
                    if SaveNetText == 1:  # for the initial population evaluation loop use the slower (generates each time) model
                        try:
                            Vout = self.network_obj.run_model(Input_V=V_in_mod, SaveNetText=SNT)
                        except Exception as e:
                            print("\n\n**Could not calculate output voltages (bad hit 1)**")
                            print("Original Error:\n", e, "\n\nTry againg:")
                            time.sleep(0.1 + pos*0.1)
                            try:
                                Vout = self.network_obj.run_model(Input_V=V_in_mod, SaveNetText=SNT)
                            except Exception as e2:
                                print("Error (TheModel): Failed to run the model")
                                print("V_in_mod:", V_in_mod)
                                print("** Error Re Throw:\n", e2)
                    else:  # once networy is generated and saved, just load it and compute directly
                        try:
                            Vout = LoadRun_model(V_in_mod, LoadModelDir, self.ParamDict['circuit_loop'], num_output, num_nodes)
                        except Exception as e:
                            print("\n\n**Could not calculate output voltages (bad hit 2) **")
                            print("Original Error:\n", e, "\n\nTry againg:")
                            time.sleep(0.1 + pos*0.1)
                            try:
                                Vout = LoadRun_model(V_in_mod, LoadModelDir, self.ParamDict['circuit_loop'], num_output, num_nodes)
                            except Exception as e2:
                                print("Error (TheModel): Failed to Load/Run the model during rep 0")
                                print("V_in_mod:", V_in_mod)
                                print("LoadModelDir:", LoadModelDir)
                                print("circuit_loop:", self.ParamDict['circuit_loop'], "num_output:", num_output, "num_nodes:", num_nodes)
                                print("** Error Re Throw:\n", e2)
                else:  # load network and compute directly
                    try:
                        Vout = LoadRun_model(V_in_mod, LoadModelDir, self.ParamDict['circuit_loop'], num_output, num_nodes)
                    except Exception as e:
                        print("\n\n**Could not calculate output voltages (bad hit 3)**")
                        print("Original Error:\n", e, "\n\nTry againg:")
                        time.sleep(0.1 + pos*0.1)
                        try:
                            Vout = LoadRun_model(V_in_mod, LoadModelDir, self.ParamDict['circuit_loop'], num_output, num_nodes)
                        except Exception as e2:
                            print("Error (TheModel): Failed to Load/Run the model (after rep 0)")
                            print("V_in_mod:", V_in_mod)
                            print("LoadModelDir:", LoadModelDir)
                            print("circuit_loop:", self.ParamDict['circuit_loop'], "num_output:", num_output, "num_nodes:", num_nodes)
                            print("** Error Re Throw:\n", e2)


                #print("Vout", Vout)

                Vout_list.append(Vout)

                #tocr = time.time()
                #test_list.append(tocr - ticr)
                #if pos == 1:
                    #print("  resistor solve:", pos, "execution time:", tocr - ticr)

                # # # # # # # # # # # #
                # Calculate the output reading from voltage inputs
                #ticg = time.time()
                class_out, responceY = GetClass(genome, out_weight_gene_loc, Vout,
                                                IntpScheme,
                                                num_output,
                                                num_output_readings,
                                                OutWeight_gene,
                                                OutWeight_scheme,
                                                EN_Dendrite, NumDendrite)

                #print("responceY", responceY)
                #print("class_out\n", class_out)

                class_out_list.append(class_out)
                responceY_list.append(responceY)

                #tocg = time.time()


                # # print the developing class_out
                #if pos == 0:
                    #print("\n",pos,"gives:\n", class_out.T)

                #if pos == 1:
                    #print("  time to get:", pos, "execution time:", tocg - ticg)




            # # # # # # # # # # # #
            # Find error and fitness from element wise class output
            class_out_array = np.asarray(np.concatenate(class_out_list))
            train_data_Y = np.asarray(train_data_Y)

            if len(class_out_array) != len(train_data_Y):
                print("Error (TheModel.py): produced class list not same length as real class checking data")
                raise ValueError('(TheModel.py): produced class list not same length as real class checking data')
            error = class_out_array - train_data_Y  # calc error matrix

            #print("class_out_array\n", class_out_array)
            #print("train_data_Y\n", train_data_Y)


            #print(pos, "error\n", error)
            #print(pos, "class_out_list\n", np.concatenate(class_out_list))
            #print(pos, "train_data_Y\n", train_data_Y)
            #exit()
            #fitness = sum(sum(np.abs(error)))/(len(error)*self.ParamDict['num_output_readings'])
            scheme_fitness, err_fit = GetFitness(train_data_Y, error, responceY_list, FitScheme,
                                                 num_output_readings, max_OutResponce, pos)

            #

            # save values to a list
            fitness_genome.append(scheme_fitness)
            error_genome.append(error)
            responceY_genome.append(responceY_list)
            Vout_genome.append(Vout_list)

            #toc_s = time.time()
            #test_list.append(toc_s - tic_s)

        # # # # # # # # # # # #
        # Save local prcoess data to file

        # temp file name
        location = 'Temp_MPdata/process%d_data.hdf5' % (pos)
        for attempt in [1,2,3]:
            try:
                with h5py.File(location, 'w') as hdf:
                    hdf.create_dataset('fitness_genome', data=fitness_genome)
                    hdf.create_dataset('error_genome', data=error_genome)
                    hdf.create_dataset('responceY_genome', data=responceY_genome)
                    hdf.create_dataset('Vout_genome', data=Vout_genome)
                break
            except Exception as e:
                time.sleep(0.5+pos*0.1)
                print("Save processor data", pos)
                print("Attempt", attempt, "failed... \n Re trying to save process data from model.")
                if attempt == 3:
                    print("Error (TheModel): Failed to save process data")
                    print("** Error Thrown is:\n", e)
                else:
                    pass


        # # Set retun value in file
        for attempt in [1,2,3]:
            try:
                tog_loc = 'Temp_MPdata/p%d_return.npy' % (pos)
                np.save(tog_loc, 1)  # set defualt to 0 i.e process not sucessful
                break
            except Exception as e:
                time.sleep(0.5+pos*0.1)
                print("Save processor", pos, " return value")
                print("Attempt", attempt, "failed... \n Re try...")
                if attempt == 3:
                    print("Error (TheModel): Failed to save process return toggle")
                    print("** Error Thrown is:\n", e)
                else:
                    pass


        # # Print individual processor execution time
        toc = time.time()
        #print("  processor:", pos, "execution time:", toc - tic)

        #print("pos", pos, "res run time:", sum(test_list))
        #plt.plot(test_list)
        #title = "Main - Processor: %d" % pos
        #plt.title(title)
        #plt.show()



    #

    #

    #

    #

    #

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Run the multiprocessing fobj function
        > organises the running of the multiprocessors
    '''
    def run_fobj_mp(self, genome_chunk_list, pop_size, SaveNetText):


        ticpt = time.time()

        # # # # # # # # # # # #
        # # Create multiprocessing que variables
        # variables are external hdf5 files

        # # # # # # # # # # # #
        # Initialise and retrieve useful data
        num_chunks = len(genome_chunk_list)

        # # # # # # # # # # # #
        # Setup the list of processes to run
        processes = [multiprocessing.Process(target=self.fobj_mp,
                                             args=(genome_chunk_list[x], x, SaveNetText)) for x in range(num_chunks)]

        #save seperate files for retun values
        for x in range(num_chunks):
            for attempt in [1,2,3]:
                try:
                    tog_loc = 'Temp_MPdata/p%d_return.npy' % (x)
                    np.save(tog_loc, 0)  # set defualt to 0 i.e process not sucessful
                    break
                except Exception as e:
                    time.sleep(0.5)
                    print("Initiate return toggle for processor", x)
                    print("Attempt", attempt, "failed... \n Re try...")
                    if attempt == 3:
                        print("Error (TheModel): Failed to initialise process return toggle as zero")
                        print("** Error Thrown is:\n", e)
                    else:
                        pass




        # Run processes
        for p in processes:
            #ticp = time.time()
            p.start()
            #tocp = time.time()
            if p == processes[0] and self.ParamDict['repetition_loop'] == 0:
                time.sleep(0.2)  # need small delay to ensure circuit has been saved

            #print("Processor start time:", tocp - ticp)

        # Exit the completed processes
        for p in processes:
            p.join()

        # check returned values
        for attempt in [1,2,3]:
            try:
                error_toggle = 0
                error_flag = 0
                for x in range(num_chunks):
                    tog_loc = 'Temp_MPdata/p%d_return.npy' % (x)
                    ret_val = np.load(tog_loc)  # set defualt to 0 i.e process not sucessful
                    if ret_val == 0 and error_toggle == 0:
                        error_toggle = 1
                        error_flag = 1
                break
            except Exception as e:
                time.sleep(0.5)
                print("Attempt", attempt, "failed... \n Re trying to load process return.")
                if attempt == 3:
                    print("\n\nError (TheModel): Failed to load process return val")
                    print("** Error Thrown is:\n", e)
                else:
                    pass

        # # if an error was flagged
        if error_flag == 1:
            print("Error (TheModel): A process retuned a None value.")
            raise ValueError('(TheModel): A process retuned a None value.')

        tocpt = time.time()
        #print("Processor total time:", tocpt - ticpt)

        # # # # # # # # # # # #
        # Format Output results
        for attempt in [1,2,3]:

            fit_genome_all = []
            error_genome_all = []
            responceY_genome_all = []  # i.e the responceY
            Vout_genome_all = []

            try:
                for process in range(num_chunks):
                    location = 'Temp_MPdata/process%d_data.hdf5' % (process)

                    with h5py.File(location, 'r') as hdf:
                        fit_genome_all.append(np.array(hdf.get('fitness_genome')))
                        error_genome_all.append(np.array(hdf.get('error_genome')))
                        responceY_genome_all.append(np.array(hdf.get('responceY_genome')))
                        Vout_genome_all.append(np.array(hdf.get('Vout_genome')))

                break
            except Exception as e:
                time.sleep(0.5)
                print("Attempt", attempt, "failed... \n Re trying to load process data into model.")
                if attempt == 3:
                    print("Error (TheModel): Failed to load process data")
                    print("** Error Thrown is:\n", e)
                else:
                    pass

        fitness_all = np.concatenate(fit_genome_all)
        error_all = np.concatenate(error_genome_all)
        responceY_all = np.concatenate(responceY_genome_all)  # i.e the boundry weight
        Vout_all = np.concatenate(Vout_genome_all)

        #print("fitness:\n", fitness)
        #print("Vout:\n", Vout)
        #exit()

        # Tidy up, and remove now finisged processes
        gc.collect()

        return fitness_all, error_all, responceY_all

#

# fin
