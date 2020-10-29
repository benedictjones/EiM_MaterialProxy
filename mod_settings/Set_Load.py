import pickle
import time
import yaml

''' # Load and return the settings parameter dictionary from file
'''


def LoadSettings(param_file=''):

    # make up to 3 attempts to load settings
    for attempt in [1,2,3]:
        try:
            with open(r'Temp_Param%s.yaml' % (str(param_file))) as file:
                CompiledDict = yaml.full_load(file)
            break
        except Exception as e:
            time.sleep(0.2)
            print("Attempt", attempt, "failed... \n Re trying to Load Settings.")
            if attempt == 3:
                print("Error (Set_Load.py): Failed to Load Settings.\n\n")
                raise ValueError(e)
            else:
                pass

    return CompiledDict



    #

    # fin
