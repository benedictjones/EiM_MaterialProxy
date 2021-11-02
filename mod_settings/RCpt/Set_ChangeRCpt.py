import yaml
import time
from mod_settings.Set_Load import LoadSettings


''' # # # # # # # # # # # # # # #
Chnage settings in the file:
i.e load dict from file, edit, save dict to file

Note: only parameters which have been needed to be changed are included
'''


def switch_IntrpScheme(pt, param_file=''):

    # load the current dictionary
    CompiledDict = LoadSettings(param_file)

    # switch to pre-train values
    if pt == 1:
        CompiledDict['algorithm'] = 'EiM'
        CompiledDict['ptype'] = 'EiM'  # processor type
        CompiledDict['DE']['IntpScheme'] = CompiledDict['DE']['pt_IntpScheme']
        CompiledDict['DE']['FitScheme'] = CompiledDict['DE']['pt_FitScheme']

    # switch to final values, to assess the pre-trained reservoir
    elif pt == 0:
        CompiledDict['algorithm'] = 'RCpt'
        CompiledDict['ptype'] = 'RC'  # processor type
        CompiledDict['DE']['IntpScheme'] = CompiledDict['DE']['pt_final_IntpScheme']
        CompiledDict['DE']['FitScheme'] = CompiledDict['DE']['pt_final_FitScheme']

    else:
        raise ValueError("(Set_ChangeRCpt.py) Must select pt as 1 (switch to pre-train values) or 0 (switch to final training values)")


    # #  Save dictionary back to file
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

# fin
