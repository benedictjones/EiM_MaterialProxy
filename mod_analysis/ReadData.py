import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import h5py
from mod_analysis.Set_Load_meta import LoadMetaData


def Read_DE(dir):

    prm = LoadMetaData(dir)
    ParamDict = prm['DE']
    NetworkDict = prm['network']
    GenomeDict = prm['genome']


    location = "%s/data.hdf5" % (dir)
    hdf = h5py.File(location, 'r')
    fitness_matrix = np.zeros((prm['num_repetitions'], prm['num_systems']))
    Veri_fitness_matrix = np.zeros((prm['num_repetitions'], prm['num_systems']))
    for syst in range(prm['num_systems']):
        for rep in range(prm['num_repetitions']):
            de_saved = hdf.get('/%d_rep%d/DE_data' % (syst, rep))
            evo_fitness = np.array(de_saved.get('fitness'))
            fitness_matrix[rep, syst] = evo_fitness[-1]
            Veri_fitness_matrix[rep, syst] = np.array(de_saved.get('veri_fit'))

    return fitness_matrix, Veri_fitness_matrix

#

#

def Read_RC(dir):

    prm = LoadMetaData(dir)
    ParamDict = prm['DE']
    NetworkDict = prm['network']
    GenomeDict = prm['genome']


    location = "%s/data.hdf5" % (dir)
    hdf = h5py.File(location, 'r')
    fitness_matrix = np.zeros((prm['num_repetitions'], prm['num_systems']))
    Veri_fitness_matrix = np.zeros((prm['num_repetitions'], prm['num_systems']))
    for syst in range(prm['num_systems']):
        for rep in range(prm['num_repetitions']):
            de_saved = hdf.get('/%d_rep%d/DE_data' % (syst, rep))
            fitness_matrix[rep, syst] = np.array(de_saved.get('best_fitness'))
            Veri_fitness_matrix[rep, syst] = np.array(de_saved.get('veri_fit'))

    return fitness_matrix, Veri_fitness_matrix

#

#

def Read_RCpt(dir):

    prm = LoadMetaData(dir)
    ParamDict = prm['DE']
    NetworkDict = prm['network']
    GenomeDict = prm['genome']


    location = "%s/data.hdf5" % (dir)
    hdf = h5py.File(location, 'r')
    fitness_matrix = np.zeros((prm['num_repetitions'], prm['num_systems']))
    Veri_fitness_matrix = np.zeros((prm['num_repetitions'], prm['num_systems']))
    for syst in range(prm['num_systems']):
        for rep in range(prm['num_repetitions']):
            de_saved = hdf.get('/%d_rep%d/final_data' % (syst, rep))
            fitness_matrix[rep, syst] = np.array(de_saved.get('fitness'))
            Veri_fitness_matrix[rep, syst] = np.array(de_saved.get('veri_fit'))

    return fitness_matrix, Veri_fitness_matrix
# fin
