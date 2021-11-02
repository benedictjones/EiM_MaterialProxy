import yaml

''' # Load and return the settings parameter dictionary from file
'''


def LoadMetaData(dir, param_file=''):

    with open(r'%s/Experiment_MetaData%s.yaml' % (dir, param_file)) as file:
        # MetaData = yaml.full_load(file)  # broken with classweight obj load
        MetaData = yaml.load(file, Loader=yaml.Loader)

    return MetaData

#

# fin
