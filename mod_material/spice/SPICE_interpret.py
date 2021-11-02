import numpy as np
from PySpice.Unit import *

###############################################################################
# Formats the output node values, extracting the voltage (& time) data from_
# the analysis data object
###############################################################################


#

#


def format_Vout(analysis, pulse_Tp, CompiledDict, num_samples):
    '''
    Format the voltages from the spice sim output analysis data object.
    Depending on the sim type, execute different functions to read the Vout.
    '''
    SpiceDict = CompiledDict['spice']
    NetworkDict = CompiledDict['network']
    num_output = NetworkDict['num_output']
    num_nodes = NetworkDict['num_input'] + NetworkDict['num_config'] + NetworkDict['num_output']

    if SpiceDict['sim_type'] == 'sim_dc':
        Vout = interpret_output_dc(analysis, SpiceDict['sim_type'], num_output, num_samples)
        sample_time = 0
    else:
        if SpiceDict['sim_type'] == 'sim_trans_pulse':
            Vout, sample_time = interpret_output_pulse(analysis, num_nodes, num_output, pulse_Tp, SpiceDict,  num_samples)
        elif SpiceDict['sim_type'] == 'sim_trans_wave':
            raise ValueError("trans wave interpretation not yet added")

    return Vout, sample_time

#

#


def interpret_output_dc(analysis, sim_type, num_output, num_samples):
    '''
    Interprets the ouput DC sweep data
    '''

    #print(sim_res_dict)
    if sim_type != 'sim_dc':
        raise ValueError("To interpret ouput DC data, must use a DC simulation!")

    # format output note voltages into array
    # extract output voltages
    Vout = np.zeros((num_samples, num_output))
    for i in range(num_output):
        key = "op%d_conn" % (i+1)
        for inst, val in enumerate(np.array(analysis[key])):
            Vout[inst,i] = val

    return Vout

#

#


def interpret_output_pulse(analysis, num_nodes, num_output,
                           pulse_Tp, SpiceDict, num_samples):
    '''
    Interprets the ouput pulse data.
    The time array is scanned for each subsequent output sample time, and
    output voltages are read for these sample times.
    Note that a windowing method has been used to speed up the searches, this
    is necessary as the simulation time length increases.
    '''

    if SpiceDict['sim_type'] != 'sim_trans_pulse':
        raise ValueError("To interpret ouput pulse data, must use a transient pulse input!")


    # generate output voltage array
    Vout = np.zeros((num_samples, num_output))
    sample_time = np.zeros((num_samples))
    sim_time = np.array(analysis.time)

    chunk_size = int(len(sim_time)/num_samples)

    #print("sim_time:\n", sim_time)
    # loop though sample times
    for s in range(num_samples):
        sample = s + 1
        #print("sample:", sample)
        t_sample = (sample*SpiceDict['pulse_TT']) + ((sample-1)*pulse_Tp) + (SpiceDict['rloc']*pulse_Tp)
        #print(t_sample)

        # estimate the area that the time will occur to speed up argmin finding
        # was idx = find_nearest(sim_time, t_sample, time_unit) which looked at whole array

        #print(s*chunk_size, (s+1)*chunk_size)
        #print("chunk:\n", time[range(s*chunk_size, (s+1)*chunk_size)])
        time_chunk = sim_time[range(s*chunk_size, (s+1)*chunk_size)]
        chunk_idx = find_nearest(time_chunk, t_sample, SpiceDict['trans_t_unit'])
        idx = s*chunk_size + chunk_idx

        sample_time[s] = sim_time[idx]  # this is the real sample time (closesed to the ideal sample time)

        # Loop though output nodes
        for i in range(num_output):
            key = "op%d_conn" % (i+1)
            #print(s, i)
            v_array_temp = np.array(analysis[key])
            Vout[s, i] = v_array_temp[idx]

    return Vout, sample_time

#

#


def find_nearest(array, value, time_u):
    """
    Finds the index of the passed in array which is closest to the passed if __name__ == '__main__':
    value.
    """

    if time_u == 'n':
        scale = 10**(9)
    elif time_u == 'u':
        scale = 10**(6)
    elif time_u == 'm':
        scale = 10**(3)

    array = np.asarray(array)*scale  # scale outputs to real value
    idx = (np.abs(array - value)).argmin()
    return idx
















# fin
