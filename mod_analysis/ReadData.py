import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import h5py
from mod_analysis.Set_Load_meta import LoadMetaData


def Read_DE(dir):

    CompiledDict = LoadMetaData(dir)
    ParamDict = CompiledDict['DE']
    NetworkDict = CompiledDict['network']
    GenomeDict = CompiledDict['genome']


    location = "%s/data.hdf5" % (dir)
    hdf = h5py.File(location, 'r')
    fitness_matrix = np.zeros((CompiledDict['num_repetitions'], CompiledDict['num_circuits']))
    Veri_fitness_matrix = np.zeros((CompiledDict['num_repetitions'], CompiledDict['num_circuits']))
    for cir in range(CompiledDict['num_circuits']):
        for rep in range(CompiledDict['num_repetitions']):
            de_saved = hdf.get('/%d_rep%d/DE_data' % (cir, rep))
            evo_fitness = np.array(de_saved.get('fitness'))
            fitness_matrix[rep, cir] = evo_fitness[-1]
            Veri_fitness_matrix[rep, cir] = np.array(de_saved.get('veri_fit'))

    return fitness_matrix, Veri_fitness_matrix

#

#

def Read_RC(dir):

    CompiledDict = LoadMetaData(dir)
    ParamDict = CompiledDict['DE']
    NetworkDict = CompiledDict['network']
    GenomeDict = CompiledDict['genome']


    location = "%s/data.hdf5" % (dir)
    hdf = h5py.File(location, 'r')
    fitness_matrix = np.zeros((CompiledDict['num_repetitions'], CompiledDict['num_circuits']))
    Veri_fitness_matrix = np.zeros((CompiledDict['num_repetitions'], CompiledDict['num_circuits']))
    for cir in range(CompiledDict['num_circuits']):
        for rep in range(CompiledDict['num_repetitions']):
            de_saved = hdf.get('/%d_rep%d/DE_data' % (cir, rep))
            fitness_matrix[rep, cir] = np.array(de_saved.get('best_fitness'))
            Veri_fitness_matrix[rep, cir] = np.array(de_saved.get('veri_fit'))

    return fitness_matrix, Veri_fitness_matrix

#

#

def Read_RCpt(dir):

    CompiledDict = LoadMetaData(dir)
    ParamDict = CompiledDict['DE']
    NetworkDict = CompiledDict['network']
    GenomeDict = CompiledDict['genome']


    location = "%s/data.hdf5" % (dir)
    hdf = h5py.File(location, 'r')
    fitness_matrix = np.zeros((CompiledDict['num_repetitions'], CompiledDict['num_circuits']))
    Veri_fitness_matrix = np.zeros((CompiledDict['num_repetitions'], CompiledDict['num_circuits']))
    for cir in range(CompiledDict['num_circuits']):
        for rep in range(CompiledDict['num_repetitions']):
            de_saved = hdf.get('/%d_rep%d/final_data' % (cir, rep))
            fitness_matrix[rep, cir] = np.array(de_saved.get('fitness'))
            Veri_fitness_matrix[rep, cir] = np.array(de_saved.get('veri_fit'))

    return fitness_matrix, Veri_fitness_matrix
# fin
