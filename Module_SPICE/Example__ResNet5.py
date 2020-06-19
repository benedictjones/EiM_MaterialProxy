####################################################################################################
import warnings
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time

#warnings.filterwarnings("error")
#PYTHONWARNINGS=ignore::yaml.YAMLLoadWarning
#PYTHONWARNINGS=ignore::DeprecationWarning
#PYTHONWARNINGS=ignore

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from SPICE_setup import *


####################################################################################################

logger = Logging.setup_logging()
#f# circuit_macros('resistor-bridge.m4')


v_data = np.array([[0,1],[1,3],[2,-2],[-1,-2],[5,-3],[-5,-3],[-2.5,-0.1]])


node1 = []
node2 = []
node3 = []
node4 = []
node5 = []
ticp = time.time()
for i in range(len(v_data[:,0])):

    circuit = Circuit('Five node resistor network')

    circuit.V('input1', 1, circuit.gnd, v_data[i,0]@u_V)
    circuit.V('input2', 2, circuit.gnd, v_data[i,1]@u_V)
    circuit.V('input3', 3, circuit.gnd, 3@u_V)

    circuit.R(1, 1, 2, 1@u_kOhm)
    circuit.R(2, 1, 3, 1@u_kOhm)
    circuit.R(3, 1, 4, 10@u_kOhm)
    circuit.R(4, 1, 5, 1@u_kOhm)
    circuit.R(5, 2, 3, 1@u_kOhm)
    circuit.R(6, 2, 4, 1@u_kOhm)
    circuit.R(7, 2, 5, 2@u_kOhm)
    circuit.R(8, 3, 4, 1@u_kOhm)
    circuit.R(9, 3, 5, 1@u_kOhm)
    circuit.R(10, 4, 5, 0.8@u_kOhm)


    # ADC Shunt Resistors
    circuit.R(11, 4, circuit.gnd, 10@u_kOhm)
    circuit.R(12, 5, circuit.gnd, 10@u_kOhm)

    enablePrint()
    #print(circuit)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.operating_point()

    sim_res_dict = {}
    i = 0
    for node in analysis.nodes.values():
        #print("node:", str(node))  # the name
        #print("array(node):", np.array(node))  # the data
        data_label = "node_%s" % str(node)
        #print(data_label)
        sim_res_dict[data_label] = np.array(node)

    node1.append(sim_res_dict['node_1'])
    node2.append(sim_res_dict['node_2'])
    node3.append(sim_res_dict['node_3'])
    node4.append(sim_res_dict['node_4'])
    node5.append(sim_res_dict['node_5'])

print("node_1", np.concatenate(node1))
print("node_2", np.concatenate(node2))
print("node_3", np.concatenate(node3))
print("node_4", np.concatenate(node4))
print("node_5", np.concatenate(node5))

tocp = time.time()
o_time = tocp - ticp
print("op sims - repetition. Execute time:", o_time)





# ##############################################
# using a DC sweep
# ##############################################

print("\n\n NEW op sims - DC sweep")
ticp = time.time()
circuit = Circuit('Five node resistor network')

circuit, sim_slice = define_inputs_dc(circuit, 2, v_data)
circuit.V('input3', 3, circuit.gnd, 3@u_V)

circuit.R(1, 1, 2, 1@u_kOhm)
circuit.R(2, 1, 3, 1@u_kOhm)
circuit.R(3, 1, 4, 10@u_kOhm)
circuit.R(4, 1, 5, 1@u_kOhm)
circuit.R(5, 2, 3, 1@u_kOhm)
circuit.R(6, 2, 4, 1@u_kOhm)
circuit.R(7, 2, 5, 2@u_kOhm)
circuit.R(8, 3, 4, 1@u_kOhm)
circuit.R(9, 3, 5, 1@u_kOhm)
circuit.R(10, 4, 5, 0.8@u_kOhm)


# ADC Shunt Resistors
circuit.R(11, 4, circuit.gnd, 10@u_kOhm)
circuit.R(12, 5, circuit.gnd, 10@u_kOhm)

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.dc(Vinput=sim_slice)  # slice = start:step:stop (inclusive)

sim_res_dict = format_output(analysis, 'dc')

tocp = time.time()
o_time = tocp - ticp
print("op sims - DC sweep total time:", o_time)


print("sim_res_dict['node_1']", sim_res_dict['node_1'])
print("sim_res_dict['node_2']", sim_res_dict['node_2'])
print("sim_res_dict['node_3']", sim_res_dict['node_3'])
print("sim_res_dict['node_4']", sim_res_dict['node_4'])
print("sim_res_dict['node_5']", sim_res_dict['node_5'])


# ##############################################
# using a transient pulse sweep
# ##############################################
del circuit

print("\n\n NEW using a transient pulse sweep")
ticp = time.time()
circuit = Circuit('Five node resistor network')

#v_data = [1,0.5,2,0.5]

circuit, t_end, sim_step_time = define_inputs_trans_Pulse(circuit, 2, v_data, 100, 50)  # time in ms
#circuit.V('input2', 2, circuit.gnd, 2@u_V)
circuit.V('input3', 3, circuit.gnd, 3@u_V)

circuit.R(1, 1, 2, 1@u_kOhm)
circuit.R(2, 1, 3, 1@u_kOhm)
circuit.R(3, 1, 4, 10@u_kOhm)
circuit.R(4, 1, 5, 1@u_kOhm)
circuit.R(5, 2, 3, 1@u_kOhm)
circuit.R(6, 2, 4, 1@u_kOhm)
circuit.R(7, 2, 5, 2@u_kOhm)
circuit.R(8, 3, 4, 1@u_kOhm)
circuit.R(9, 3, 5, 1@u_kOhm)
circuit.R(10, 4, 5, 0.8@u_kOhm)


# ADC Shunt Resistors
circuit.R(11, 4, circuit.gnd, 10@u_kOhm)
circuit.R(12, 5, circuit.gnd, 10@u_kOhm)

print(circuit)

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
step_time = sim_step_time@u_ms  # step_time = (t_end/200)@u_us
analysis = simulator.transient(step_time=step_time, end_time=t_end@u_ms)

sim_res_dict = format_output(analysis, 'trans')

tocp = time.time()
o_time = tocp - ticp
print("using a transient pulse sweep total time:", o_time)

f = plt.figure()
plt.plot(sim_res_dict['time'], sim_res_dict['node_1'], label='in1')
plt.plot(sim_res_dict['time'], sim_res_dict['node_2'], label='in2')
plt.plot(sim_res_dict['time'], sim_res_dict['node_3'], label='fixed')
plt.plot(sim_res_dict['time'], sim_res_dict['node_4'], label='node4')
plt.plot(sim_res_dict['time'], sim_res_dict['node_5'], label='node5')
plt.legend()


"""
# ##############################################
# using a transient wave input
# ##############################################
del circuit

print("\n\n NEW using a transient pulse sweep")
ticp = time.time()
circuit = Circuit('Five node resistor network')

t_v_data = [[[0,0],[0.1,0.5],[0.2,2],[0.3,1.5],[0.4,0.25]],
            [[0,0],[0.1,-2],[0.2,0.2],[0.3,1.3],[0.4,0.6]]]

circuit, t_end, sim_step_time = define_inputs_trans_Wave(circuit, 2, t_v_data, time_unit='m')
#circuit.V('input2', 2, circuit.gnd, 2@u_V)
circuit.V('input3', 3, circuit.gnd, 3@u_V)

circuit.R(1, 1, 2, 1@u_kOhm)
circuit.R(2, 1, 3, 1@u_kOhm)
circuit.R(3, 1, 4, 10@u_kOhm)
circuit.R(4, 1, 5, 1@u_kOhm)
circuit.R(5, 2, 3, 1@u_kOhm)
circuit.R(6, 2, 4, 1@u_kOhm)
circuit.R(7, 2, 5, 2@u_kOhm)
circuit.R(8, 3, 4, 1@u_kOhm)
circuit.R(9, 3, 5, 1@u_kOhm)
circuit.R(10, 4, 5, 0.8@u_kOhm)


# ADC Shunt Resistors
circuit.R(11, 4, circuit.gnd, 10@u_kOhm)
circuit.R(12, 5, circuit.gnd, 10@u_kOhm)

print(circuit)

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
step_time = sim_step_time@u_ms  # step_time = (t_end/200)@u_us
analysis = simulator.transient(step_time=step_time, end_time=t_end@u_ms)

sim_res_dict = format_output(analysis, 'trans')

tocp = time.time()
o_time = tocp - ticp
print("using a transient pulse sweep total time:", o_time)

f = plt.figure()
plt.plot(sim_res_dict['time'], sim_res_dict['node_1'], label='in1')
plt.plot(sim_res_dict['time'], sim_res_dict['node_2'], label='in2')
plt.plot(sim_res_dict['time'], sim_res_dict['node_3'], label='fixed')
plt.plot(sim_res_dict['time'], sim_res_dict['node_4'], label='node4')
plt.plot(sim_res_dict['time'], sim_res_dict['node_5'], label='node5')
plt.legend()
"""

# ##############################################
# using a transient sine wave input
# ##############################################
del circuit

print("\n\n NEW using a transient pulse sweep")
ticp = time.time()
circuit = Circuit('Five node resistor network')

f_data = [[1,3,0.2],
            [0.1,0.05,0.6]]

circuit, t_end, sim_step_time = define_inputs_trans_Sine(circuit, 2, f_data, freq_unit='k')
circuit.V('input3', 3, circuit.gnd, 3@u_V)

circuit.R(1, 1, 2, 1@u_kOhm)
circuit.R(2, 1, 3, 1@u_kOhm)
circuit.R(3, 1, 4, 10@u_kOhm)
circuit.R(4, 1, 5, 1@u_kOhm)
circuit.R(5, 2, 3, 1@u_kOhm)
circuit.R(6, 2, 4, 1@u_kOhm)
circuit.R(7, 2, 5, 2@u_kOhm)
circuit.R(8, 3, 4, 1@u_kOhm)
circuit.R(9, 3, 5, 1@u_kOhm)
circuit.R(10, 4, 5, 0.8@u_kOhm)


# ADC Shunt Resistors
circuit.R(11, 4, circuit.gnd, 10@u_kOhm)
circuit.R(12, 5, circuit.gnd, 10@u_kOhm)

print(circuit)

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
step_time = sim_step_time@u_ms  # step_time = (t_end/200)@u_us
analysis = simulator.transient(step_time=step_time, end_time=t_end@u_ms)

sim_res_dict = format_output(analysis, 'trans')

tocp = time.time()
o_time = tocp - ticp
print("using a transient pulse sweep total time:", o_time)

f = plt.figure()
plt.plot(sim_res_dict['time'], sim_res_dict['node_1'], label='in1')
plt.plot(sim_res_dict['time'], sim_res_dict['node_2'], label='in2')
plt.plot(sim_res_dict['time'], sim_res_dict['node_3'], label='fixed')
plt.plot(sim_res_dict['time'], sim_res_dict['node_4'], label='node4')
plt.plot(sim_res_dict['time'], sim_res_dict['node_5'], label='node5')
plt.legend()







#

#
plt.show()

# fin
