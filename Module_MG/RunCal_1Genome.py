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

''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
Function to calculate the class of each element of a passed in data set
'''
def calc_class1(genome, pos):
    #test_time = []
    #tic = time.time()

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
        except Exception as e2:
            print("Error:\n", e2, "\n\nABORTING.")
            exit()



    # # load in x data chunk
    location = 'Temp_MPdata/MG_x_data%d.hdf5' % (pos)
    try:
        with h5py.File(location, 'r') as hdf:
            x_in = np.array(hdf.get('x_chunk'))
    except Exception as e:
        print("Could not open: ", location, "(Processor clash?)")
        print("Original Error:\n", e, "\n\nTry againg:")
        time.sleep(0.1 + pos*0.1)
        try:
            with h5py.File(location, 'r') as hdf:
                x_in = np.array(hdf.get('x_chunk'))
        except Exception as e2:
            print("Error:\n", e2, "\n\nABORTING.")
            exit()


    if ParamDict['ReUse_Circuits'] == 1 and ParamDict['ReUse_dir'] != 'na':
        SaveDir = ParamDict['ReUse_dir']
        #print("Circuit is being re_used!!!!")
    else:
        SaveDir = ParamDict['SaveDir']

    #print("x_in_list")
    #print(x_in_list)

    # # Initialise arrays
    Vout_list = []
    class_out_list = []
    responceY_list = []



    # # # # # # # # # # # #
    # Iterate over all input data to find elementwise error
    for loop in range(len(x_in[:, 0])):

        x_row = x_in[loop, :]
        seq_len = ParamDict['num_config'] + ParamDict['num_input']
        V_in_mod = get_voltages(x_row, genome, ParamDict['InWeight_gene'], ParamDict['InWeight_sheme'], ParamDict['in_weight_gene_loc'],
                                ParamDict['shuffle_gene'], ParamDict['SGI'], seq_len, ParamDict['config_gene_loc'])


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
        Vout_list.append(Vout)


    # # Print individual processor execution time
    #print("  processor:", pos, "execution time:", toc - tic)

    location = 'Temp_MPdata/MGprocess%d_data.hdf5' % (pos)

    with h5py.File(location, 'w') as hdf:
        hdf.create_dataset('responceY_genome', data=responceY_list)
        hdf.create_dataset('Vout_genome', data=Vout_list)
        hdf.create_dataset('ClassOut_genome', data=class_out_list)

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
                print("Error (RunCal_1Genome): Failed to save process return toggle")
                print("** Error Thrown is:\n", e)
            else:
                pass

    #print("time",pos,":", sum(test_time))
#

#

#

#

#

''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
Run the multiprocessing fobj function
'''
def run_calc_class1(genome, x_in_list):

    save_file = "Temp_MPdata/MG_param.dat"
    with open(save_file, 'rb') as f:
        bound_dict = pickle.load(f)


    # # concatinate x data into one single list
    all_x_in = np.concatenate(x_in_list)

    # split up the x_data into chunks
    x_list = np.array_split(all_x_in, bound_dict['num_processors'])

    # # Save x data (it is too bulky for quick passing in as a variable!)
    for pos in range(bound_dict['num_processors']):
        location = 'Temp_MPdata/MG_x_data%d.hdf5' % (pos)
        with h5py.File(location, 'w') as hdf:
            hdf.create_dataset('x_chunk', data=x_list[pos])

    #save seperate files for retun values
    for x in range(bound_dict['num_processors']):
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
    processes = [multiprocessing.Process(target=calc_class1,
                                         args=(genome, x)) for x in range(bound_dict['num_processors'])]


    #save seperate files for retun values
    for x in range(bound_dict['num_processors']):
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
            for x in range(bound_dict['num_processors']):
                tog_loc = 'Temp_MPdata/p%d_return.npy' % (x)
                ret_val = np.load(tog_loc)  # set defualt to 0 i.e process not sucessful
                if ret_val == 0 and error_toggle == 0:
                    error_toggle = 1
                    error_flag = 1
            break
        except Exception as e:
            time.sleep(0.5)
            print("Attempt", attempt, "failed... \n Re trying to load MG process return.")
            if attempt == 3:
                print("\n\nError (RunCal_1Genome): Failed to load MG process return val")
                print("** Error Thrown is:\n", e)
            else:
                pass

    # if an error was flagged
    if error_flag == 1:
        print("Error (RunCal_1Genome): A process retuned a None value.")
        raise ValueError('(RunCal_1Genome): A process retuned a None value.')

    # # # # # # # # # # # #
    # Format Output results
    for attempt in [1,2,3]:

        responceY_all = []  # i.e the responceY
        Vout_all = []
        ClassOut_all = []

        # load saved MP data
        try:
            for process in range(bound_dict['num_processors']):
                location = 'Temp_MPdata/MGprocess%d_data.hdf5' % (process)
                with h5py.File(location, 'r') as hdf:
                    responceY_all.append(np.array(hdf.get('responceY_genome')))
                    Vout_all.append(np.array(hdf.get('Vout_genome')))
                    ClassOut_all.append(np.array(hdf.get('ClassOut_genome')))
            break
        except Exception as e:
            time.sleep(0.5)
            print("Attempt", attempt, "failed... \n Re trying to load MG process data into RunCal_1Genome.")
            if attempt == 3:
                print("Error (RunCal_1Genome): Failed to load process data")
                print("** Error Thrown is:\n", e)
            else:
                pass


    ClassOut_all = np.concatenate(ClassOut_all)
    responceY_all = np.concatenate(responceY_all)  # i.e the boundry weight
    Vout_all = np.concatenate(Vout_all)

    # # re format into lists of each row of x1
    formatted_ClassOut = []
    formatted_responceY = []
    formatted_Vout = []
    k = 0
    for i in range(bound_dict['len_x1']):
        temp_CO = []
        temp_rY = []
        temp_Vo = []
        for j in range(bound_dict['len_x2']):
            temp_CO.append(ClassOut_all[k])
            temp_rY.append(responceY_all[k])
            temp_Vo.append(Vout_all[k])
            k = k + 1
        formatted_ClassOut.append(temp_CO)
        formatted_responceY.append(temp_rY)
        formatted_Vout.append(temp_Vo)
        #print("temp_rY[0]:\n", temp_rY[0])
        #print("temp_rY:\n", temp_rY)
        #print("formatted_responceY:\n", formatted_responceY)


        #input("Press Enter to continue...")


    return [formatted_ClassOut], [formatted_responceY]  # this is a list of all vals

#

#

#

#

#

#

# fin
