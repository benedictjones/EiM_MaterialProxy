import os, sys
import numpy as np
import math

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__




import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *


###############################################################################
###############################################################################


def apply_time_unit(time_unit, t_end, sim_step_time):

    if time_unit == 'm':
        t_end_out = t_end@u_ms
        sim_step_time_out = sim_step_time@u_ms
    elif time_unit == 'u':
        t_end_out = t_end@u_us
        sim_step_time_out = sim_step_time@u_us
    elif time_unit == 'n':
        t_end_out = t_end@u_ns
        sim_step_time_out = sim_step_time@u_ns
    else:
        raise ValueError('Illegal time unit %s.' % (time_unit))

    return t_end_out, sim_step_time_out

#

#


def define_inputs_trans_Pulse(circuit, input_data, period, SpiceDict):
    """
    **Use period & trans_time to formulate a PULSE TRAIN from data.**
    Produces B voltage source's which allow a DC sweep of voltages.
    DC sweep on Voltage component, this causes B source to select the deired
    input voltage to the input node.

    """

    trans_time = SpiceDict['pulse_TT']
    time_unit = SpiceDict['trans_t_unit']

    # generate slice used in simulator.dc()
    num_in = len(input_data[0,:])
    if num_in == 1:
        num_instances = len(input_data)
    else:
        num_instances = len(input_data[:, 0])

    t_end=num_instances*(period+trans_time) + trans_time

    for inp in range(num_in):
        in_node = inp + 1

        # fetch data
        if num_in == 1:
            data = input_data
        else:
            data = input_data[:, inp]

        if len(set(data)) == 1:
            # static voltage if all vals are the same
            Vin_name = 'c%d' % (in_node)
            circuit.V(Vin_name, 'in%d_conn' % (in_node), circuit.gnd, data[0]@u_V)
        else:
            # format data into PWL
            circuit.raw_spice += "Vpwl%d in%d_conn 0 PWL(0s 0V " % (in_node, in_node)
            current_time = 0
            for val in data:
                current_time = current_time + trans_time
                circuit.raw_spice += '%f%ss %fV ' % (current_time, time_unit, val)  # rise/fall time
                current_time = current_time + period
                circuit.raw_spice += '%f%ss %fV ' % (current_time, time_unit, val)  # pulse fin
            current_time = current_time + trans_time
            circuit.raw_spice += '%f%ss %fV ' % (current_time, time_unit, 0)  # pulse fin
            circuit.raw_spice += 'r=0s td=0.0s)' + os.linesep

    """
    TT_step_lim = trans_time/10
    Tp_step_lim = period/100
    sim_step_time = min([TT_step_lim, Tp_step_lim])
    #"""
    sim_step_time = period/200


    #print("sim_step_time", sim_step_time)
    #print(time_unit, t_end, sim_step_time)

    t_end, sim_step_time = apply_time_unit(time_unit, t_end, sim_step_time)
    #print(time_unit, t_end, sim_step_time)

    return circuit, t_end, sim_step_time

#

#


def define_inputs_trans_Wave(circuit, input_time_data, SpiceDict):
    """
    **Uses raw data to plot voltages at specific time, wave!**
    Produces B voltage source's which allow a DC sweep of voltages.
    DC sweep on Voltage component, this causes B source to select the deired
    input voltage to the input node

    Note: input_time_data must be a list of 2d arrays where
            > col1 = time
            > col2 = voltage data
            > must be equal length
    """

    # Extract meta-paramaters
    data = np.array(input_time_data[0])
    if len(data[0,:]) != 2:
        raise ValueError('Input data is not a list of 2d arrays')

    time_data = data[:,0]
    t_starts = time_data[0]
    t_end = time_data[-1]
    num_instances = len(time_data)
    step_size = (t_end-t_starts)/num_instances  # assuming linear sampling
    sim_step_time = step_size/10

    num_in = len(input_time_data)
    time_unit = SpiceDict['trans_t_unit']
    for inp in range(num_in):
        in_node = inp + 1

        # fetch this inputs 2d data
        data = np.array(input_time_data[inp])

        time_data = data[:,0]
        voltage_data = data[:,1]

        if len(set(voltage_data)) == 1:
            # static voltage if all vals are the same
            Vin_name = 'c%d' % (in_node)
            circuit.V(Vin_name, 'in%d_conn' % (in_node), circuit.gnd, data[0]@u_V)
        else:
            # format data into PWL
            circuit.raw_spice += "Vpwl%d in%d_conn 0 PWL(0ms 0V " % (in_node, in_node)
            loop = 0
            for val in data:
                circuit.raw_spice += '%f%ss %.4fV ' % (time_data[loop], time_unit, voltage_data[loop])  # rise/fall time
                loop += 1
            circuit.raw_spice += 'r=0s td=0.0s)' + os.linesep

    t_end, sim_step_time = apply_time_unit(time_unit, t_end, sim_step_time)

    return circuit, t_end, sim_step_time

#

#


def define_inputs_trans_Sine(circuit, num_in, f_data, period=100, freq_unit='k'):
    """
    **Uses raw data to plot voltages at specific time, wave!**
    Produces B voltage source's which allow a DC sweep of voltages.
    DC sweep on Voltage component, this causes B source to select the deired
    input voltage to the input node

    Note: f_data must be a list of arrays where each array contains an inputs
          changing frequency

    """

    #if freq_unit != 'G' and freq_unit != 'M' and freq_unit != 'k':
    #    raise ValueError('Illegal time unit %s.' % (freq_unit))

    if freq_unit == 'G':
        time_u = 'n'
        scale = 10**9
    elif freq_unit == 'M':
        time_u = 'u'
        scale = 10**6
    elif freq_unit == 'k':
        time_u = 'm'
        scale = 10**3
    else:
        raise ValueError('Illegal time unit %s.' % (freq_unit))


    # enconde frequency change
    steps = 10

    amp = 1
    encoded_data = []
    for inp in range(num_in):
        in_node = inp + 1

        f = max(f_data[inp])*scale
        Fs = f*2     # No. of samples per second, Fs = 50 kHz
        N = int(Fs/f)*200     # No. of samples for 2 ms, N = 100

        temp_encoded_data = []
        current_time = 0
        for val in f_data[inp]:
            samples = np.linspace(current_time, period, N, endpoint=False)
            signal = amp * np.sin(2 * np.pi * val * scale * samples)
            res = (np.concatenate(([samples], [signal]), axis=0)).T
            temp_encoded_data.append(res)
        encoded_data.append(np.concatenate(temp_encoded_data))

    circuit, t_end, sim_step_time = define_inputs_trans_Wave(circuit, num_in, encoded_data, time_unit=time_u)

    return circuit, t_end, sim_step_time

#

#


def define_inputs_dc(circuit, input_data):
    """
    Produces B voltage source's which allow a DC sweep of voltages.
    DC sweep on Voltage component, this causes B source to select the deired
    input voltage to the input node
    """
    num_in = len(input_data[0,:])
    for inp in range(num_in):
        in_node = inp + 1

        # fetch data
        if num_in == 1:
            data = input_data
        else:
            data = input_data[:, inp]

        # produce Vconfig (if all identical) or a Vsource if input
        if len(set(data)) == 1:
            # static voltage
            Vin_name = 'c%d' % (in_node)
            circuit.V(Vin_name, 'in%d_conn' % (in_node), circuit.gnd, data[0]@u_V)

        else:
            # format data into PWL
            v_seq = "pwl(v(im),"
            loop = 1
            for val in data:

                if loop % 10 == 0:
                    v_seq += os.linesep + '+'

                if loop == len(data):
                    v_seq += ' %d,%.5f' % (loop, val)
                else:
                    v_seq += ' %d,%.5f,' % (loop, val)
                loop += 1
            v_seq += ')'

            # produce B voltage soure with custom pwl profile (look up table style)
            b_name = "s%d" % (in_node)
            circuit.B(b_name, 'in%d_conn' % (in_node), circuit.gnd, v=v_seq)

    # produce voltage source
    #new_line = "Vsweep im 0 DC=0"
    new_line = "Vimg im 0"
    circuit.raw_spice += new_line + os.linesep

    # generate slice used in simulator.dc()
    if num_in == 1:
        slice_end = len(input_data)
    else:
        slice_end = len(input_data[:, 0])
    sim_slice = slice(1, slice_end, 1)

    return circuit, sim_slice


















# fin
