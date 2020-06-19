import numpy as np
import multiprocessing
import pickle

from Module_Functions.FetchPerm import NPerms

''' # The Settings available:

Network types =
    'NL_RN':  # non-linear resistor network
    'R_RN':  # random resistor network
    'custom_RN':  # custom defined resistor network
    'D_RN':  # random resistor network with diodes
    'POD_RN'  # random network of parallel oposing diodes in series with a
                random resistor (direction is random (can be toggled), i.e
                the resistor can be before or after the diodes)

perm_crossp_model = 'none'   - random, with a  fixed threshold
                    'linear' - reduces the perm crossp linearly
                    'box'    - perm crossp is 1 for the first half of the
                               total number of iterations, then 0
                    'quad'   - reduces the perm crossp quadratically
                    'unity'  - always crossp=1


If we turn on output weightings (i.e OutWeight_gene=1), then we can select how
this scheme is implemented (within the Interprate_Scheme.py class):

    OutWeight_scheme = 'random'  - normal mode, weighings are randomly selected
                                   by the DE acording to the defined bounds,
                                   and left to evolve.
                     = 'AddSub'  - the output weighting is randomly chosen to
                                   either "+1" or "-1".

Interpretation schemes can be:
Note: Toggle NR_FEC = 0 or 1 to enable a 'network responce for each class'.

scheme = 'band_binary' (or 0) - if  -1<reading<1 class 2, else, class 1
         'pn_binary'   (or 1) - positive responce = class 1, negative = class 2


Fitness evaluation schemes can be:
FitScheme = 'error'     - Defualt, finds mean error from all instances
            'dist'      - Calaculates a value related to net depth of instances
                          into the class (away from the boundary)
            'Err_dist'  - 50/50 weighted split between the above two schemes

# # # # # # #
# This is a script which produces a Temp_SettingsParam.npz file using
# numpy.savez, this allows the selected parameters to be saved to a file which
# can be loaded where needed in the various bits of this code.
# (This should speed up the process, as setting are no longer stored in an
# object, which takes time to pass.)

'''


''' # # # # # # # # # # # # # # #
Run Function to set and save Settings Parameters for a Dictionary in a file
'''


def SaveSettings(mut=0.8, crossp=0.8, popsize=20, its=100,
                 num_input=2, num_output=2, num_config=3,
                 shuffle_gene=1, perm_crossp_model='none',
                 InWeight_gene=0, InWeight_sheme='random',
                 OutWeight_gene=0, OutWeight_scheme='random', num_output_readings=1,
                 the_model='R_RN', shunt_Res=10, R_array_in='null',
                 diode_rand_dir=1, defualt_diode_dir=1, Use_DefualtDiode=1,
                 num_processors='max',  # 'max' to auto select the max number
                 scheme='pn_binary',
                 training_data='2DDS',
                 EN_Dendrite=0, NumDendrite=3,  # allow Dendrite averaging of multiple network responces
                 NumClasses=2,
                 num_circuits=10,
                 num_repetitions=3,
                 ReUse_Circuits=0, ReUse_dir='na',
                 BreakOnZeroFit=1,  # Stop breaking on zero Fit
                 PlotExtraAttribute=0, Add_Attribute_To_Data=0,  # save data with extra attribute, and to plot
                 UseCustom_NewAttributeData=0, UseOriginalNetowrk=0, NewAttrDataDir='na',  # run the DE using previously saved new extra attr data
                 TestVerify=0,
                 FitScheme='error',
                 MG_dict=0):

    # # # # # # # # # # #
    # Create dictionary to store settings in
    ParamDict = {}

    # # # # # # # # # # # #
    # Set DE param
    print("Settings being saved...")
    ParamDict['mut'] = mut
    ParamDict['crossp'] = crossp
    ParamDict['popsize'] = popsize
    ParamDict['its'] = its
    ParamDict['shuffle_gene'] = shuffle_gene
    ParamDict['InWeight_gene'] = InWeight_gene
    ParamDict['InWeight_sheme'] = InWeight_sheme
    ParamDict['OutWeight_gene'] = OutWeight_gene
    ParamDict['OutWeight_scheme'] = OutWeight_scheme  # seledcts how the InterpSheme class uses the OutWeight
    ParamDict['training_data'] = training_data
    ParamDict['NumClasses'] = NumClasses
    ParamDict['num_circuits'] = num_circuits  # the generated circuits by the renet
    ParamDict['num_repetitions'] = num_repetitions  # whether we re-use previously generated circuits
    ParamDict['ReUse_Circuits'] = ReUse_Circuits  # tells the system if subsequent experiments are re-using the same material models
    ParamDict['ReUse_dir'] = ReUse_dir  # file with models in it
    ParamDict['FitScheme'] = FitScheme  # 'error', 'dist', 'Err_dist'


    ParamDict['BreakOnZeroFit'] = BreakOnZeroFit
    ParamDict['TestVerify'] = TestVerify

    if perm_crossp_model != 'none' and shuffle_gene == 0:
        print("")
        print("Error (Settings): Perm_crossp model set, but shuffle gene is off!")
        print("ABORTED")
        exit()
    else:
        ParamDict['perm_crossp_model'] = perm_crossp_model



    # # # # # # # # # # # #
    # Set ResNet Parameters
    ParamDict['num_input'] = num_input
    ParamDict['num_output'] = num_output
    ParamDict['num_config'] = num_config
    ParamDict['shunt_R'] = shunt_Res
    num_nodes = num_input + num_output + num_config
    ParamDict['num_nodes'] = num_nodes
    ParamDict['model'] = the_model
    ParamDict['R_array_in'] = R_array_in
    ParamDict['rand_dir'] = diode_rand_dir  # diode random direction toggle
    ParamDict['defualt_diode_dir'] = defualt_diode_dir
    ParamDict['DefualtDiode'] = Use_DefualtDiode

    sum = 0
    for i in range(num_nodes-2):  # sum from 1 to (N-3)
        sum = sum + i
    ParamDict['num_connections'] = num_nodes + (num_nodes-3) + sum
    print("Number of Connections:", ParamDict['num_connections'])

    # # # # # # # # # # # #
    # Set ResNet Material Parameters
    ParamDict['max_r'] = 10  # k ohms
    ParamDict['min_r'] = 0.1  # k ohms

    # # # # # # # # # # # #
    # Multiprocessor Settings (for DE simulation process only, MG currently seperate)
    if num_processors == 'max':
        # fetch the local PC's number of cores
        print("number of cores:", multiprocessing.cpu_count())
        ParamDict['num_processors'] = multiprocessing.cpu_count()
    elif num_processors > multiprocessing.cpu_count():
        print("Maximum number of cores is:", multiprocessing.cpu_count())
        print("Number of corse being set the the maximum possible.")
        ParamDict['num_processors'] = multiprocessing.cpu_count()
    else:
        ParamDict['num_processors'] = num_processors

    # # # # # # # # # # # #
    # USed to set the current loop of an single experiment run
    ParamDict['loop'] = 'none'
    ParamDict['circuit_loop'] = 0
    ParamDict['repetition_loop'] = 0
    ParamDict['SaveDir'] = 'bin/'

    # # # # # # # # # # # #
    # Set whether to save data with extra attribute, and to plot
    ParamDict['Add_Attribute_To_Data'] = Add_Attribute_To_Data
    ParamDict['PlotExtraAttribute'] = PlotExtraAttribute

    # # # # # # # # # # # #
    # Set whether run the DE using previously saved new data
    ParamDict['UseCustom_NewAttributeData'] = UseCustom_NewAttributeData  # run DE using weighted data saved from a previous run
    ParamDict['UseOriginalNetowrk'] = UseOriginalNetowrk  # use a new material network
    ParamDict['NewAttrDataDir'] = NewAttrDataDir  # the dir from which the data is being loaded


    # # # # # # # # # # # #
    # # enable Demdrite averaging of multiple network responces
    ParamDict['EN_Dendrite'] = EN_Dendrite  # Demdrite enabled
    ParamDict['NumDendrite'] = NumDendrite  # Dendrite number (i.e network responce number)

    # # Add MG properties
    if MG_dict != 0:
        ParamDict.update(MG_dict)

    if EN_Dendrite == 1 and OutWeight_gene == 0:
        print("To enable a Network Responce For Each Class (NR_FEC), output weights must be enabled.")
        print("ABORTED")
        exit()
    elif EN_Dendrite == 1 and scheme == 'HOwins':
        print("Network Responce For Each Class (NR_FEC)not compatible with HOwins.")
        print("ABORTED")
        exit()

    ''' ******************************************************************
    Assign boundaries
        ******************************************************************
    '''

    # # Set some limits
    ParamDict['num_output_readings'] = num_output_readings  # e.g for a full added, we have 2 output readings
    ParamDict['Vmin'] = -5
    ParamDict['Vmax'] = 5
    ParamDict['MaxOutputWeight'] = 1
    ParamDict['MaxInputWeight'] = 1

    # # calc max output responce
    if OutWeight_gene == 0:
        ParamDict['max_OutResponce'] = ParamDict['Vmax'] * ParamDict['num_output']
    elif OutWeight_gene == 1:
        if EN_Dendrite == 0:
            ParamDict['max_OutResponce'] = ParamDict['Vmax'] * ParamDict['num_output'] * ParamDict['MaxOutputWeight']
        elif EN_Dendrite == 1:
            ParamDict['max_OutResponce'] = ParamDict['Vmax'] * ParamDict['num_output'] * ParamDict['MaxOutputWeight'] * NumDendrite

    # # Determin the number of boundaries possibly required
    if EN_Dendrite == 1:  # set of weights for each classes individual output responce?
        num_output_weights = num_output*num_output_readings*NumDendrite
    else:
        num_output_weights = num_output*num_output_readings


    # Initialise the bounds matrix
    #num_row = num_config + 1 + num_output_weights
    #bounds = np.zeros((num_row, 2))
    bounds = []

    # # Set gene counter and gene loc trackers
    j = 0
    config_gene_loc = []
    # SGI - index for the shuffle gene (Shuffle Gene Index)
    in_weight_gene_loc = []
    out_weight_gene_loc = []


    # # Set bounds for voltages
    config_gene_loc.append(j)
    num_in_V = num_config + num_input
    print("num_in_V", num_in_V)
    for i in range(num_config):
        #bounds[i, 0] = ParamDict['Vmin']
        #bounds[i, 1] = ParamDict['Vmax']
        bounds.append([ParamDict['Vmin'], ParamDict['Vmax']])

        j = j + 1
    config_gene_loc.append(j)
    ParamDict['config_gene_loc'] = config_gene_loc

    # # Set bounds for shuffle gene/decision variable
    # perms read by an algorithm (no need to generate whole list)
    num_perms = NPerms(num_in_V)
    print("Number of perms:", num_perms)


    # assign the shuffle gene bounds accrotding to num perms
    if shuffle_gene == 1:
        #bounds[j, 0] = 0
        #bounds[j, 1] = num_perms-1
        bounds.append([0, num_perms-1])
        ParamDict['SGI'] = j  # index for the shuffle gene (Shuffle Gene Index)
        j = j + 1
    else:
        ParamDict['SGI'] = 'na'

    # # Produce input weight bounds
    if InWeight_gene == 1:
        in_weight_gene_loc.append(j)
        for i in range(num_input):
            bounds.append([-ParamDict['MaxInputWeight'], ParamDict['MaxInputWeight']])
            j = j + 1
        in_weight_gene_loc.append(j)
        ParamDict['in_weight_gene_loc'] = in_weight_gene_loc
    else:
        ParamDict['in_weight_gene_loc'] = 'na'


    # # Produce output weight bounds
    if OutWeight_gene == 1:
        out_weight_gene_loc.append(j)
        for i in range(num_output_weights):
            bounds.append([-ParamDict['MaxOutputWeight'], ParamDict['MaxOutputWeight']])
            j = j + 1
        out_weight_gene_loc.append(j)
        ParamDict['out_weight_gene_loc'] = out_weight_gene_loc
    else:
        ParamDict['out_weight_gene_loc'] = 'na'

    # # Record the number of genomes (i.e dimension of decision variables)
    bounds = np.asarray(bounds)
    ParamDict['dimensions'] = len(bounds[:, 0])
    ParamDict['bounds'] = bounds

    # # Infomation print out
    print("")
    print("the bounds:")
    print(bounds)

    #exit()


    ''' Example Genome:
    For 2 inputs, 2 config voltages, shuffel gene on, 2 output nodes and one output dimension
        Vconfig 0 = -3.814
        Vconfig 1 = -2.288
        Permutation 16, which is (2, 3, 0, 1). NOTE: 0 to 1 are the Vinputs
        Vout(node out = 0, dimension out = 0)*W where W=-0.693154
        Vout(node out = 1, dimension out = 0)*W where W=-0.420583
    '''


    ''' ******************************************************************
    '''

    # # # # # # # # # # # #
    # Interpretation Scheme Set up
    # self.IS_obj = InterpretScheme(self, scheme)
    ParamDict['IntpScheme'] = scheme

    # produce number version for meta data
    if scheme == 'band_binary':
        ParamDict['scheme_num'] = 0
    elif scheme == 'pn_binary':
        ParamDict['scheme_num'] = 1
    elif scheme == 'HOwins':
        ParamDict['scheme_num'] = 2
        # check parametrs are correct
        if ParamDict['NumClasses'] != ParamDict['num_output']:
            print("Error (Set_Save): For interpretation scheme 'Highest Output wins', we need NumClasses=NumOutputs ")
            print("ABORTED")
            exit()

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Save the dictionary to a file:
    save_file = "TempDict_SettingsParam.dat"
    with open(save_file, 'wb') as outfile:
        pickle.dump(ParamDict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    print("Setting Parameters Saved as dictionary")

    # Fin setting script
    return

    #

    #

    # fin
