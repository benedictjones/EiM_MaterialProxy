# #############################################################################
# Top Matter
# #############################################################################

import numpy as np
import sys
import os
import PySpice
import random

from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Unit import *
import PySpice.Logging.Logging as Logging

from mod_material.spice.SubCir import SeriesRD, SeriesNL, SeriesPOD


# My imports
from mod_settings.Set_Load import LoadSettings
from mod_material.spice.SPICE_setup import *



# change sim program location depending on system
if sys.platform == "linux" or sys.platform == "linux2":
    PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'ngspice-subprocess'  # needed for linux
elif sys.platform == "win32":
    pass

# # # # # # # # # # # #
# Produce a print disable function


def blockPrint():  # used to plock print outs
    sys.stdout = open(os.devnull, 'w')


def enablePrint():  # Restore
    sys.stdout = sys.__stdout__


# Initialise SPICE logger (note it thows a print which is suppressed to tidy output)
import PySpice.Logging.Logging as Logging
blockPrint()
logger = Logging.setup_logging()
enablePrint()

# current bug fix
import locale
locale.setlocale(locale.LC_NUMERIC, "C")


''' # Please Read

# WINDOWS #
This Class file requires PySpice (via pip install) and NGSpice installed in
the directory (for windows):
C:\Program Files\Spice64_dll ,
the correct version must be used and include the ngspice.dll file in:
C:\Program Files\Spice64_dll\dll-vs

# LINUX #
Install PySpice via pip and Ngspice Program
need to change source program to ngspice-subprocess
for PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR

# Convergence
Ngspice uses the Newton-Raphson algorithm to solve nonlinear equations arising
from circuit description. The NR algorithm is interactive and terminates when
both of the following conditions hold:
    1. The nonlinear branch currents converge to within a tolerance of 0.1%
or 1 picoamp (1.0e-12 Amp), whichever is larger.
    2. The node voltages converge to within a tolerance of 0.1% or 1 microvolt
(1.0e-6 Volt), whichever is larger.

# Parameters
shunt_R = resistance value (Kohm) or 'none' for no shunt

'''
# #############################################################################
# End of top matter
# #############################################################################


def calc_num_conn(num_node):

    if num_node == 1:
        return 0

    sum = 0
    if num_node >= 3:
        for i in range(num_node-2):  # sum from 1 to (N-3)
            sum = sum + i
    num_conn = num_node + (num_node-3) + sum

    return num_conn


def generate_neuromorphic_netork(CompiledDict, genome, Input_V, pulse_Tp, cir=0):

    # # # # # # # # # # # #
    # Assign setting to self from setting dict file

    SpiceDict = CompiledDict['spice']
    NetworkDict = CompiledDict['network']
    GenomeDict = CompiledDict['genome']
    #ParamDict = LoadSettings_ResnetDebug()

    NN_weights = genome[GenomeDict['NN_weight_loc']]

    #print(layer_config)


    node_cont = 1

    node_layer_list = []
    total_num_conn = 0
    all_nodes = []
    # Generate Fulley connected material nodes


    if SpiceDict['num_layers'] == 1:
        nodes = []
        for i in range(NetworkDict['num_input']+NetworkDict['num_config']):
            nodes.append("in%d" % (node_cont))
            node_cont = node_cont + 1
        for i in range(NetworkDict['num_output']):
            nodes.append("op%d" % (i+1))

        total_num_conn = total_num_conn + calc_num_conn(len(nodes))
        node_layer_list.append(nodes)
        all_nodes = nodes

    elif SpiceDict['num_layers'] >= 2:

        cont = 0
        mn_cont = 1
        mn_config_count = 1
        for l in range(SpiceDict['num_layers']):
            nodes = []

            if (l % 1) == 0 and l !=0:
                mn_cont = cont*SpiceDict['NodesPerLayer'] + 1
                cont = cont + 1

            if l == 0:
                for i in range(NetworkDict['num_input']+NetworkDict['num_config']):
                    nodes.append("in%d" % (node_cont))
                    node_cont = node_cont + 1
                for i in range(SpiceDict['NodesPerLayer']):
                    nodes.append("mn%d" % (mn_cont))
                    mn_cont = mn_cont + 1
                for i in range(SpiceDict['ConfigPerLayer']):
                    nodes.append("mnC%d" % (mn_config_count))
                    mn_config_count = mn_config_count + 1


            elif l == (SpiceDict['num_layers']-1):
                for i in range(SpiceDict['NodesPerLayer']):
                    nodes.append("mn%d" % (mn_cont))
                    mn_cont = mn_cont + 1
                for i in range(SpiceDict['ConfigPerLayer']):
                    nodes.append("mnC%d" % (mn_config_count))
                    mn_config_count = mn_config_count + 1
                for i in range(NetworkDict['num_output']):
                    nodes.append("op%d" % (i+1))

            else:
                for i in range(SpiceDict['NodesPerLayer']*2):
                    nodes.append("mn%d" % (mn_cont))
                    mn_cont = mn_cont + 1
                for i in range(SpiceDict['ConfigPerLayer']):
                    nodes.append("mnC%d" % (mn_config_count))
                    mn_config_count = mn_config_count + 1


            total_num_conn = total_num_conn + calc_num_conn(len(nodes))
            node_layer_list.append(nodes)
            all_nodes.extend(nodes)

    else:
        raise ValueError('Invalid Number of layers %d' % (SpiceDict['num_layers']))

    #print(node_layer_list)
    #print("SpiceDict['ConfigPerLayer']", SpiceDict['ConfigPerLayer'])
    all_nodes = list(dict.fromkeys(all_nodes))
    #print("calc num_nodes", NetworkDict['num_nodes'], " Total num nodes", len(all_nodes))
    node_weight_dict = {}
    for idx, node in enumerate(all_nodes):
        node_weight_dict[node] = NN_weights[idx]
    #exit()
    # #################################################
    # Check condictions
    # #################################################
    Input_V = np.asarray(Input_V)
    """if len(Input_V[0,:]) != (NetworkDict['num_nodes']-NetworkDict['num_output']):
        print(" ")
        print("Error (resnet): Input Vector doesn't equal input node number")
        raise ValueError("(resnet): Input Vector doesn't equal input node number")"""

    # print(Input_V)

    # #################################################
    # Circuit Creation
    # #################################################

    name = 'NeuromorphicNetwork'

    circuit = Circuit(name)

    # #################################################
    # Generate contact resistances
    # #################################################

    for node in all_nodes:
        if node[0] == 'i':
            circuit.R('%s_contact' % (node), '%s_conn' % (node), node, SpiceDict['contact_Res']@u_Ohm)
        elif node[0] == 'o':
            circuit.R('%s_contact' % (node), node, '%s_conn' % (node), SpiceDict['contact_Res']@u_Ohm)
        elif node[:3] == 'mnC':
            circuit.R('%s_contact' % (node), '%s_conn' % (node), node, SpiceDict['contact_Res']@u_Ohm)

    # #################################################
    # Input Creation
    # #################################################

    num_in = NetworkDict['num_input']+NetworkDict['num_config']  # data inputs + config
    if SpiceDict['sim_type'] == 'sim_dc':
        sim_mode = 'dc'
        circuit, sim_slice = define_inputs_dc(circuit, num_in, Input_V)
    elif SpiceDict['sim_type'] == 'sim_trans_pulse':
        sim_mode = 'trans'
        circuit, t_end, sim_step_time = define_inputs_trans_Pulse(circuit, num_in, Input_V, pulse_Tp, SpiceDict['pulse_TT'], time_unit=SpiceDict['trans_t_unit'])
    elif SpiceDict['sim_type'] == 'sim_trans_wave':
        sim_mode = 'trans'
        circuit, t_end, sim_step_time = define_inputs_trans_Wave(circuit, num_in, input_time_data, time_unit=SpiceDict['trans_t_unit'])

    circuit.model('DefualtDiode', 'D', IS=20@u_uA, RS=0.8@u_mOhm,
                  BV=30@u_V, IBV=200@u_uV, N=1)

    if SpiceDict['ConfigPerLayer'] >= 1:
        layer_config = genome[GenomeDict['NN_layerConfig_loc']]
        for i, val in enumerate(layer_config):
            Vin_name = 'layerConfig%d' % (i)
            circuit.V(Vin_name, 'mnC%d_conn' % (i), circuit.gnd, val@u_V)

    # #################################################
    # Produce network acording to selected network
    # #################################################

    # # Initialise settings
    element_num = 1
    part = 1
    my_diode_num = 0
    node_pairs = []  # save data for result printing
    dir_array = []
    res_array = []
    a_list = []

    for nodes in node_layer_list:

        # # Place a component between every node pair
        for idx1, n1 in enumerate(nodes):

            n1_weight = node_weight_dict[n1]

            for idx2, n2 in enumerate(nodes):
                if [n1,n2] in node_pairs or [n2,n1] in node_pairs or n1 == n2:
                    continue  # skip if this pair has already been done
                node_pairs.append([n1, n2])  # save node pairs in arrays

                n2_weight = node_weight_dict[n2]

                node_pair_strength = n1_weight*n2_weight

                # Neuromorphic resistor network
                if NetworkDict['model'] == 'R_NN':

                    temp_R = 0.001 + node_pair_strength*SpiceDict['NN_max_r']
                    temp_R = np.around(temp_R, decimals=3)
                    res_array.append(temp_R)

                    circuit.R(element_num, n1, n2, temp_R@u_kOhm)  # create resistor element
                    element_num = element_num + 1

                # Neuromorphic resistor-diode network
                elif NetworkDict['model'] == 'D_NN':

                    temp_R = 0.001 + node_pair_strength*SpiceDict['NN_max_r']
                    temp_R = np.around(temp_R, decimals=3)
                    res_array.append(temp_R)

                    sub_name = '%s%d' % ('SubCir', part)
                    dir = 1
                    dir_array.append(dir)
                    SubCirc_obj = SeriesRD(sub_name, Res=temp_R,
                                           direction=dir, DefualtDiode=SpiceDict['DefualtDiode'],
                                           custom_diode_num=element_num,
                                           IS_val=0.1)
                    circuit.subcircuit(SubCirc_obj)
                    circuit.X(element_num, sub_name, n1, n2)  # SubCir diode + res element is therefore

                    element_num = element_num + 1
                    part = part + 1

                # Theoretical non-linear network
                elif NetworkDict['model'] == 'NL_NN':
                    a = SpiceDict['material_a_min'] + node_pair_strength*(SpiceDict['material_a_max']-SpiceDict['material_a_min'])
                    a_list.append(a)
                    a_express = "%.2f%s" % (a, "n")

                    """vdiff = '(v(%s)-v(%s))' % (str(n1), str(n2))
                    i_express = "(%s>0) ? (%s*%s**2) : (-%s*%s**2)" % (vdiff,
                                                                       a_express, vdiff,
                                                                       a_express, vdiff)
                    # print(I_express)
                    circuit.B(element_num, n1, n2, i=i_express)  # create B source element"""

                    sub_name = '%s%d' % ('SubCir', element_num)
                    SubCirc_obj = SeriesNL(sub_name, a_express, '0n')

                    circuit.subcircuit(SubCirc_obj)
                    circuit.X(element_num, sub_name, n1, n2)  # SubCir diode + res element is therefore

                    element_num = element_num + 1


                else:
                    raise ValueError("Invalid Neuromorphic Netowrk (NN) selected! (%s)" % (NetworkDict['model']))

    # #################################################
    # ADC Shunt Resistors & extras
    # #################################################
    if SpiceDict['shunt_R'] != 'none':
        for i in range(NetworkDict['num_output']):
            Out_Node = "op%d_conn" % (i+1)
            # print("Node out", Out_Node)
            circuit.R("shunt%d" % (i+1), Out_Node, circuit.gnd, SpiceDict['shunt_R']@u_kOhm)


    for i in range(SpiceDict['NodesPerLayer']*SpiceDict['num_layers']):
        Out_Node = "mn%d" % (i+1)
        # print("Node out", Out_Node)
        circuit.R("mnShunt%d" % (i+1), Out_Node, circuit.gnd, 10@u_GOhm)

    for i in range(SpiceDict['ConfigPerLayer']*SpiceDict['num_layers']):
        Out_Node = "mnC%d" % (i+1)
        # print("Node out", Out_Node)
        circuit.R("mnConfigShunt%d" % (i+1), Out_Node, circuit.gnd, 10@u_GOhm)

    if SpiceDict['shuntC'] == 1:
        min, max = SpiceDict['ShuntCapLim']

        nom_shuntcap_array = np.random.rand(len(nodes))
        diff = np.fabs(max-min)  # absolute val of each el
        long_shuntcap_array = min + nom_shuntcap_array * diff  # pop with their real values
        shuntcap_array = np.around(long_shuntcap_array, decimals=3)
        i = 1
        for node in nodes:
            circuit.C("shunt%d" % (i), node, circuit.gnd, shuntcap_array[i-1]@u_uF)
            i = i + 1


    if SpiceDict['parallelC'] == 1:
        min, max = SpiceDict['ParallelCapLim']
        el_num = 1
        cap_node_pairs = []
        pcn = 1

        nom_parlcap_array = np.random.rand(total_num_conn)
        diff = np.fabs(max-min)  # absolute val of each el
        long_parlcap_array = min + nom_parlcap_array * diff  # pop with their real values
        parlcap_array = np.around(long_parlcap_array, decimals=3)

        for n1 in nodes:
            for n2 in nodes:
                if [n1,n2] in cap_node_pairs or [n2,n1] in cap_node_pairs or n1 == n2:
                    continue  # skip if this pair has already been done
                cap_node_pairs.append([n1, n2])  # save node pairs in arrays
                circuit.C("parl%d" % (el_num), n1, n2, parlcap_array[pcn-1]@u_uF)
                pcn = pcn + 1
                el_num = el_num + 1

    # #################################################
    # #  Simulate
    # #################################################
    # circuit.raw_spice = '.OPTIONS itl1=1000'
    # circuit.raw_spice = '.OPTIONS itl6=100'

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    #print("Load_run:\n", simulator)
    #exit()

    if sim_mode == 'dc':
        #analysis = simulator.operating_point()
        # sweep an imaginary node, which causes a B source to select the input series
        analysis = simulator.dc(vimg=sim_slice)  # slice = start:step:stop (inclusive)
    elif sim_mode == 'trans':
        analysis = simulator.transient(step_time=sim_step_time, end_time=t_end)

    # #################################################
    # # Save Output
    # #################################################
    if num_in == 1:
        num_samples = len(Input_V)
    else:
        num_samples = len(Input_V[:, 0])

    Vout, sample_time = format_Vout(analysis, pulse_Tp, CompiledDict, num_samples)

    # #################################################
    # # Save the netowrk Topology
    # #################################################
    # print("the circuit: \n", circuit)

    # # make file
    if cir >= 0:
        dir_path = "%s/MD_CircuitTop" % (CompiledDict['SaveDir'])
        if os.path.exists(dir_path) is not True:
            os.makedirs(dir_path)

        # # Print circuit topography to file
        circuit_top = circuit
        circuit_top = '%s' % circuit_top

        Net_top_path = "%s/Network_Topology_%s_Cir_%s.txt" % (dir_path, str(CompiledDict['param_file']), cir)
        print("First save:", Net_top_path)
        file1 = open(Net_top_path, "w")
        file1.write(circuit_top)
        file1.close()

        # save more easily read file too
        if NetworkDict['model'] == 'R_NN':
            text_list = []
            for i in range(len(res_array)):

                text_list.append('R%s%s = %06.4f Kohm,     Node %s--/\/\/--Node %s ' % (
                                 node_pairs[i][0], node_pairs[i][1], res_array[i], node_pairs[i][0], node_pairs[i][1]))


        elif NetworkDict['model'] == 'D_NN':
            text_list = []
            for i in range(len(res_array)):
                if dir_array[i] == 1:
                    text_list.append('R%s%s = %06.3f Kohm,     Node %s--/\/\/----|>--Node %s (i.e direction %d)' % (
                                     node_pairs[i][0], node_pairs[i][1], res_array[i], node_pairs[i][0], node_pairs[i][1], dir_array[i]))
                elif dir_array[i] == 0:
                    text_list.append('R%s%s = %06.3f Kohm,     Node %s--<|----/\/\/--Node %s (i.e direction %d)' % (
                                     node_pairs[i][0], node_pairs[i][1], res_array[i], node_pairs[i][0], node_pairs[i][1], dir_array[i]))

        elif NetworkDict['model'] == 'NL_NN':
            text_list = []
            for i in range(len(a_list)):

                a = a_list[i]
                n1, n2 = node_pairs[i]
                a_express = "%.2f%s" % (a, "n")
                vdiff = '(v(%s)-v(%s))' % (str(n1), str(n2))
                i_express = "(%s>0) ? (%s*%s**2) : (-%s*%s**2)" % (vdiff,
                                                                   a_express, vdiff,
                                                                   a_express, vdiff)
                text_list.append(i_express)

        elif NetworkDict['model'] == 'custom_RN':
            text_list = []
            for i in range(len(res_array)):
                text_list.append('R %s %s = %06.3f Kohm' % (
                                 node_pairs[i][0], node_pairs[i][1], res_array[i]))

        else:
            text_list = []
            for i in range(len(res_array)):
                text_list.append('R %s %s = %06.3f Kohm' % (
                                 node_pairs[i][0], node_pairs[i][1], res_array[i, 0]))


        if SpiceDict['shuntC'] == 1:
            text_list.append('\nShunt Cap:')
            i = 0
            for node in nodes:
                text_list.append('C %s %d = %06.3f uF' % (
                                 str(node), 0, shuntcap_array[i]))
                i = i + 1

        if SpiceDict['parallelC'] == 1:
            text_list.append('\nParallel Cap:')
            pcn = 1
            for cap_pair in cap_node_pairs:
                    text_list.append('C %s %s = %06.3f uF' % (
                                     cap_pair[0], cap_pair[1], parlcap_array[pcn-1]))
                    pcn = pcn + 1


        # now try and write it
        circuit_top_path = "%s/MaterialDescription_%s_Cir_%s.txt" % (dir_path, str(CompiledDict['param_file']), cir)
        res_diode_list = ''
        for list_bit in text_list:
            res_diode_list = '%s%s \n' % (res_diode_list, list_bit)

        file2 = open(circuit_top_path, "w")
        file2.write(res_diode_list)
        file2.close()






    del circuit, simulator, analysis

    return Vout

#

# fin
