import yaml

''' # Load and return the settings parameter dictionary from file
'''


def LoadMetaData(dir):

    with open(r'%s/Experiment_MetaData.yaml' % (dir)) as file:
        MetaData = yaml.full_load(file)

    return MetaData

#

# fin
