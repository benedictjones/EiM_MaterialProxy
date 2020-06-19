import pickle


''' # # # # # # # # # # # # # # #
Chnage settings in the file:
i.e load dict from file, edit, save dict to file

Note: only parameters which have been needed to be changed are included
'''


def ChangeSettings(AlsoLoad=0, loop='na', SaveDir='na', circuit_loop='na', repetition_loop='na'):

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load dictionary from file
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # define the file which is used
    save_file = "TempDict_SettingsParam.dat"

    # load the current dictionary
    with open(save_file, 'rb') as infile:
        ParamDict = pickle.load(infile)

    #

    #

    #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Change some variables if selected
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if loop != 'na':
        ParamDict['loop'] = loop

    if SaveDir != 'na':
        ParamDict['SaveDir'] = SaveDir

    if circuit_loop != 'na':
        ParamDict['circuit_loop'] = circuit_loop

    if repetition_loop != 'na':
        ParamDict['repetition_loop'] = repetition_loop

    #

    #

    #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Save dictionary back to file
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    with open(save_file, 'wb') as outfile:
        pickle.dump(ParamDict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    #

    #

    #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Also return the new Dictionary if toggled on
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if AlsoLoad == 0:
        return
    elif AlsoLoad == 1:
        return ParamDict

    #

    #

    #

    #

    #

    #

    #

    #

    #

    #

    #

    # fin
