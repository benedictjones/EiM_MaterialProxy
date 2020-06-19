import numpy as np
import time


# My imports
from Module_Settings.Set_Load import LoadSettings
from Module_SPICE.resnet_LoadRun import LoadRun_model
from Module_InterpSchemes.interpretation_scheme import GetClass
from Module_InterpSchemes.fitness_scheme import GetFitness
from Module_LoadData.LoadData import Load_Data
from Module_LoadData.AddAttributeToData import AddAttribute
from Module_Functions.GetVoltageInputs import get_voltages

def FetchVerificationFitness(best_genome):

    ParamDict = LoadSettings()

    # # Assign Values from object to a 'local' variable
    num_config = ParamDict['num_config']
    num_input = ParamDict['num_input']
    num_output = ParamDict['num_output']
    num_output_readings = ParamDict['num_output_readings']
    shuffle_gene = ParamDict['shuffle_gene']
    OutWeight_gene = ParamDict['OutWeight_gene']
    training_data = ParamDict['training_data']
    OutWeight_scheme = ParamDict['OutWeight_scheme']
    IntpScheme = ParamDict['IntpScheme']
    num_nodes = num_input + num_config + num_output
    FitScheme = ParamDict['FitScheme']
    max_OutResponce = ParamDict['max_OutResponce']

    repetition_loop = ParamDict['repetition_loop']
    circuit_loop = ParamDict['circuit_loop']
    #SaveDir = ParamDict['SaveDir']  # the current dir file
    #ReUse_dir = ParamDict['ReUse_dir']  # the file which has the circuit models saved

    # select the dir with the saved circuits in it
    if ParamDict['ReUse_Circuits'] == 1 and ParamDict['ReUse_dir'] != 'na':
        SaveDir = ParamDict['ReUse_dir']
        print("Circuit is being re_used!!!!")
    else:
        SaveDir = ParamDict['SaveDir']

    InWeight_gene = ParamDict['InWeight_gene']
    InWeight_sheme = ParamDict['InWeight_sheme']
    in_weight_gene_loc = ParamDict['in_weight_gene_loc']
    SGI = ParamDict['SGI']
    config_gene_loc = ParamDict['config_gene_loc']
    out_weight_gene_loc = ParamDict['out_weight_gene_loc']


    # Load the training data (faster to load then pass in!)
    train_data_X, train_data_Y = Load_Data('veri', num_input, num_output_readings,
                                           training_data, ParamDict['TestVerify'],
                                           ParamDict['UseCustom_NewAttributeData'])


    # Calculate fitness for the best genome
    genome = best_genome

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

        # # # # # # # # # # # #
        # Calculate the voltage output of input training data
        # Load circuit if possible, only generate+run on the first loop
        try:
            Vout = LoadRun_model(V_in_mod, SaveDir, circuit_loop, num_output, num_nodes)
        except:
            time.sleep(0.1)
            try:
                Vout = LoadRun_model(V_in_mod, SaveDir, circuit_loop, num_output, num_nodes)
            except:
                print("Error (TheModel): Failed to Load/Run the model (after rep 0)")
                print("V_in_mod:", V_in_mod)
                print("SaveDir:", SaveDir)
                print("circuit_loop:", circuit_loop, "num_output:", num_output, "num_nodes:", num_nodes)

        Vout_list.append(Vout)

        # # # # # # # # # # # #
        # Calculate the output reading from voltage inputs
        class_out, responceY = GetClass(genome, out_weight_gene_loc, Vout,
                                        IntpScheme,
                                        num_output,
                                        num_output_readings,
                                        OutWeight_gene,
                                        OutWeight_scheme,
                                        ParamDict['EN_Dendrite'], ParamDict['NumDendrite'])


        class_out_list.append(class_out)
        responceY_list.append(responceY)

    # # # # # # # # # # # #
    # Find error and fitness from element wise class output
    if len(class_out_list) != len(train_data_Y):
        print("Error (Run_Verification.py): produced class list not same length as real class checking data")
        print("ABORTED")
        exit()
    error = np.concatenate(class_out_list) - train_data_Y  # calc error matrix

    fitness, err_fit = GetFitness(train_data_Y, error, responceY_list, FitScheme,
                                  num_output_readings, max_OutResponce, 0)

    if ParamDict['Add_Attribute_To_Data'] == 1:
        AddAttribute('veri', responceY_list)

    print("Verfication fitness =", fitness, "The Error Fitness=", err_fit)

    """ also output:
     > class_out_list
     > responceY_list
    """
    return fitness, err_fit
