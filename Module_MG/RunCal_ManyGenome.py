# Import
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # allows the creation of a custom cmap
import numpy as np
import h5py
import pickle

import multiprocessing

import time

# My imports

from Module_SPICE.resnet_LoadRun import LoadRun_model
from Module_InterpSchemes.interpretation_scheme import GetClass

from Module_Functions.GetVoltageInputs import get_voltages

#

#

#

''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
Function to calculate the class of each element of a passed in data set
'''
def calc_class(genome_chunk, pos):
    tic = time.time()

    # # load settings (so don't need to pass object)
    # # Clashes on file access seem to be occouring, added in a fail safe
    try:
        save_file = "Temp_MPdata/MG_param.dat"
        with open(save_file, 'rb') as f:
            ParamDict = pickle.load(f)  # this is the edited ParamDisct i.e the bound_dict
    except Exception as e:
        print("Could not open: ", save_file, "(Processor clash?)")
        print("Original Error:\n", e, "\n\nTry againg:")
        time.sleep(0.1 + pos*0.1)
        try:
            save_file = "Temp_MPdata/MG_param.dat"
            with open(save_file, 'rb') as f:
                ParamDict = pickle.load(f)  # this is the edited ParamDisct i.e the bound_dict
        except Exception as e:
            print("Error:\n", e, "\n\nABORTING.")
            exit()

    # # load in x data (faster this way)
    location = 'Temp_MPdata/MG_x_data_all.hdf5'
    try:
        with h5py.File(location, 'r') as hdf:
            x_in_list = np.array(hdf.get('x_in_list'))
    except Exception as e:
        print("Could not open: ", location, "(Processor clash?)")
        print("Original Error:\n", e, "\n\nTry againg:")
        time.sleep(0.1 + pos*0.1)
        try:
            with h5py.File(location, 'r') as hdf:
                x_in_list = np.array(hdf.get('x_in_list'))
        except Exception as e2:
            print("Error:\n", e2, "\n\nABORTING.")
            exit()


    if ParamDict['ReUse_Circuits'] == 1 and ParamDict['ReUse_dir'] != 'na':
        SaveDir = ParamDict['ReUse_dir']
        #print("Circuit is being re_used!!!!")
    else:
        SaveDir = ParamDict['SaveDir']

    # Calculate fitness and error for every genome in the passed in chunk
    responceY_genome = []
    Vout_genome = []
    ClassOut_genome = []

    for genome in genome_chunk:
        # print("")
        # print("The genome being alalised:", genome)

        # # Initialise arrays
        class_out_allx = []
        responceY__allx = []  # initialise BWs list
        Vout__allx = []

        for x_in in x_in_list:
            #print("x_in")
            #print(x_in)
            #print("")

            # # Initialise arrays
            class_out_list = []
            responceY_list = []  # initialise BWs list
            Vout_list = []


            # # # # # # # # # # # #
            # Iterate over all input data to find elementwise error
            for loop in range(len(x_in[:, 0])):

                x_row = x_in[loop, :]
                seq_len = ParamDict['num_config'] + ParamDict['num_input']
                V_in_mod = get_voltages(x_row, genome, ParamDict['InWeight_gene'], ParamDict['InWeight_sheme'], ParamDict['in_weight_gene_loc'],
                                        ParamDict['shuffle_gene'], ParamDict['SGI'], seq_len, ParamDict['config_gene_loc'])

                #

                # # # # # # # # # # # #
                # Calculate the voltage output of input training data
                # Load circuit if possible, only generate+run on the first loop
                try:
                    Vout = LoadRun_model(V_in_mod, SaveDir, ParamDict['circuit_loop'], ParamDict['num_output'], ParamDict['num_nodes'])
                except:
                    time.sleep(0.1 + pos*0.1)
                    try:
                        Vout = LoadRun_model(V_in_mod, SaveDir, ParamDict['circuit_loop'], ParamDict['num_output'], ParamDict['num_nodes'])
                    except:
                        print("Error (TheModel): Failed to Load/Run the model (after rep 0)")
                        print("V_in_mod:", V_in_mod)
                        print("SaveDir:", SaveDir)
                        print("circuit_loop:", ParamDict['circuit_loop'], "num_output:", ParamDict['num_output'], "num_nodes:", ParamDict['num_nodes'])

                Vout_list.append(Vout)

                # # # # # # # # # # # #
                # Calculate the output class & boundry closeness from voltage inputs
                class_out, responceY = GetClass(genome, ParamDict['out_weight_gene_loc'], Vout,
                                                  ParamDict['IntpScheme'],
                                                  ParamDict['num_output'],
                                                  ParamDict['num_output_readings'],
                                                  ParamDict['OutWeight_gene'],
                                                  ParamDict['OutWeight_scheme'],
                                                  ParamDict['EN_Dendrite'], ParamDict['NumDendrite'])
                # save values to a list
                class_out_list.append(class_out)
                responceY_list.append(responceY)

            class_out_allx.append(class_out_list)
            responceY__allx.append(responceY_list)
            Vout__allx.append(Vout_list)

        # save values to a list
        ClassOut_genome.append(class_out_allx)
        responceY_genome.append(responceY__allx)
        Vout_genome.append(Vout__allx)


    toc = time.time()
    # # Print individual processor execution time
    # print("  processor:", pos, "execution time:", toc - tic)

    location = 'Temp_MPdata/MGprocess%d_data.hdf5' % (pos)

    with h5py.File(location, 'w') as hdf:
        hdf.create_dataset('responceY_genome', data=responceY_genome)
        hdf.create_dataset('Vout_genome', data=Vout_genome)
        hdf.create_dataset('ClassOut_genome', data=ClassOut_genome)

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
                print("Error (RunCal_ManyGenome): Failed to save process return toggle")
                print("** Error Thrown is:\n", e)
            else:
                pass

#

#

#

#

#

''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
Run the multiprocessing fobj function
'''
def run_calc_class(genome_chunk_list, pop_size, x_in_list):

    # # save x_list (don't pass to MP)
    location = 'Temp_MPdata/MG_x_data_all.hdf5'
    with h5py.File(location, 'w') as hdf:
        hdf.create_dataset('x_in_list', data=x_in_list)


    # # # # # # # # # # # #
    # Initialise and retrieve useful data
    num_chunks = len(genome_chunk_list)  # genome_chunk_list = list of all of the chunks
    #print("")
    #print("genome_chunk_list")
    #print(genome_chunk_list)

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
                    print("Error (RunCal_1Genome): Failed to initialise process return toggle as zero")
                    print("** Error Thrown is:\n", e)
                else:
                    pass



    # # # # # # # # # # # #
    # Setup the list of processes to run
    # #tic = time.time()
    processes = [multiprocessing.Process(target=calc_class,
                                         args=(genome_chunk_list[x], x)) for x in range(num_chunks)]


    #save seperate files for retun values
    for x in range(num_chunks):
        tog_loc = 'Temp_MPdata/p%d_return.npy' % (x)
        np.save(tog_loc, 0)  # set defualt to 0 i.e process not sucessful


    # Run processes
    for p in processes:
        #ticp = time.time()
        p.start()
        #tocp = time.time()
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

    # if an error was flagged
    if error_flag == 1:
        print("Error (RunCal_ManyGenome): A process retuned a None value.")
        raise ValueError('(RunCal_ManyGenome): A process retuned a None value.')

    # # # # # # # # # # # #
    # Format Output results
    for attempt in [1,2,3]:

        responceY_genome_all = []  # i.e the responceY
        Vout_genome_all = []
        ClassOut_genome_all = []

        try:
            for process in range(num_chunks):
                location = 'Temp_MPdata/MGprocess%d_data.hdf5' % (process)

                with h5py.File(location, 'r') as hdf:
                    responceY_genome_all.append(np.array(hdf.get('responceY_genome')))
                    Vout_genome_all.append(np.array(hdf.get('Vout_genome')))
                    ClassOut_genome_all.append(np.array(hdf.get('ClassOut_genome')))
            break
        except Exception as e:
            time.sleep(0.5)
            print("Attempt", attempt, "failed... \n Re trying to load MG process data into RunCal_ManyGenome.")
            if attempt == 3:
                print("Error (RunCal_ManyGenome): Failed to load process data")
                print("** Error Thrown is:\n", e)
            else:
                pass

    ClassOut_all = np.concatenate(ClassOut_genome_all)
    responceY_all = np.concatenate(responceY_genome_all)  # i.e the boundry weight
    Vout_all = np.concatenate(Vout_genome_all)

    return ClassOut_all, responceY_all  # this is a list of all vals

#

#

#

#

#

#

# fin
