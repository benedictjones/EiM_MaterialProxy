import pickle


''' # Load and return the settings parameter dictionary from file
'''


def LoadSettings():

    save_file = "TempDict_SettingsParam.dat"

    with open(save_file, 'rb') as f:
        ParamDict = pickle.load(f)

    return ParamDict

def LoadSettings_ResnetDebug():

    ParamDict = {}

    ParamDict['num_output'] = 2
    ParamDict['shunt_R'] = 10
    
    num_nodes = 5
    ParamDict['num_nodes'] = num_nodes
    ParamDict['model'] = 'custom_RN'

    sum = 0
    for i in range(num_nodes-2):  # sum from 1 to (N-3)
        sum = sum + i
    ParamDict['num_connections'] = num_nodes + (num_nodes-3) + sum

    ParamDict['R_array_in'] = [5, 1, 10, 3, 0.1, 1, 2, 0.7, 0.5, 1000]  # in kOhm
    ParamDict['rand_dir'] = 0
    ParamDict['defualt_diode_dir'] = 1
    ParamDict['DefualtDiode'] = 1

    ParamDict['loop']  = 0 # the current loop from DE.py
    ParamDict['SaveDir'] = 'CustomSave/debug' # extract save file
    ParamDict['circuit_loop']  = 0  # the generated circuits by the renet
    ParamDict['repetition_loop'] = 0   # whether we re-use previously generated circuits

    return ParamDict

    #

    # fin
