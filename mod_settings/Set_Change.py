import time
import yaml
from mod_settings.Set_Load import LoadSettings


''' # # # # # # # # # # # # # # #
Chnage settings in the file:
i.e load dict from file, edit, save dict to file

Note: only parameters which have been needed to be changed are included
'''


def ChangeSettings(param_file='', pt='na', **kwargs):
    """
    Load Relevant param file, edit it, and save.
    Or, pass in a param file, edit is, and return it.

    Use a **kwargs keyword/value pair to assign a value to any key, ignoring
    the (first layer) parent key grouping.
    """

    # # Load dictionary from file
    if isinstance(param_file, dict):
        CompiledDict = param_file
        save_tog = 0
    else:
        CompiledDict = LoadSettings(param_file)
        save_tog = 1

    # # Check to see if there is anything to change
    if len(kwargs.keys()) != 0:

        # # loop though Key Value **kwargs List
        failed_key = []
        for k, v in kwargs.items():

            if k == 'algorithm':
                raise ValueError("Can't assign algorithm in the kwargs, only in the param file or as a GenParam argument.")
                continue

            c = 0
            if k in CompiledDict.keys():
                CompiledDict[k] = v
                c = c + 1
            for key, value in CompiledDict.items():
                if isinstance(value, dict):
                    if k in value.keys():
                        CompiledDict[key][k] = v
                        c = c + 1

            if c == 0:
                failed_key.append(k)  # record failed keys

            """ # stop double assignment - catually ok
            if c > 1:
                raise ValueError("Assigned a value to two different keys!")
            """

        if len(failed_key) != 0:
            print("\nWarning (Set_Change.py): Failed Assignment of keys include:", failed_key, "\n")

    # ################################################################
    # # Toggle vales if pre-training
    # ################################################################

    # switch to pre-train values
    if pt == 1:
        print("pt_hit 1")
        CompiledDict['DE']['IntpScheme'] = CompiledDict['DE']['pt_IntpScheme']
        CompiledDict['DE']['FitScheme'] = CompiledDict['DE']['pt_FitScheme']
    # switch to final values, to assess the pre-trained reservoir
    elif pt == 0:
        print("pt_hit 0")
        CompiledDict['DE']['IntpScheme'] = CompiledDict['DE']['pt_final_IntpScheme']
        CompiledDict['DE']['FitScheme'] = CompiledDict['DE']['pt_final_FitScheme']

    # ################################################################
    # #  Save dictionary back to file
    # ################################################################
    if save_tog == 1:
        for attempt in [1,2,3]:
            try:
                with open(r'Temp_Param%s.yaml' % (str(param_file)), 'w') as sfile:
                    yaml.dump(CompiledDict, sfile)
                break
            except Exception as e:
                time.sleep(0.2+attempt)
                print("Attempt", attempt, "failed... \n Re trying to Change and save Settings.")
                if attempt == 3:
                    print("Error (Set_Change.py): Failed to Load Settings.\n\n")
                    raise ValueError(e)
                else:
                    pass

    #

    # return the new Dictionary
    return CompiledDict

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
