# #############################################################################
# Top Matter
# #############################################################################

import numpy as np
import sys
import os
import PySpice

from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Unit import *

from Module_SPICE.DiodeSubCir import SeriesRD
from Module_SPICE.DoubleDiodeSubCir import SeriesPOD
import PySpice.Logging.Logging as Logging

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


# # # # # # # # # # # #
# Initialise SPICE logger (note it thows a print which is suppressed
# to tidy output)
blockPrint()
logger = Logging.setup_logging()
enablePrint()


''' # Please Read
This runs a simulation of a pre-generated circuit (which is loaded from
a .txt file)

'''

# #############################################################################
# End of top matter
# #############################################################################


def LoadRun_model(Input_V, SaveDir, cir_loop, num_output, num_nodes):  # the PySpice + NGSpice model


    # Create File path to load network topology
    path_fi = "%s/MD_CircuitTop/Netowrk_Topology__Loop_%s.txt" % (SaveDir, cir_loop)

    # Create circuit
    circuit = Circuit('Five node B-source Resistor network')

    # Load from file and perform sim op
    with open(path_fi, 'r') as reader:

        # Extract raw string from loaded file
        # NetTop = reader.read()
        NetTop = reader.readlines()  # reads lines, but includes "\n"


        # Break up and re-write Network in raw spice
        lines = [line.rstrip('\n') for line in NetTop]
        try:
            while 1:
                lines.remove('')
        except:
            pass

        # Write lines to spice, and replace voltage if it is a Vnode
        v_cont = 0
        for line in lines:
            if line[0] == 'V':
                fields = line.split(' ')
                fields[3] = str(Input_V[v_cont])
                new_line = fields[0] + ' ' + fields[1] + ' ' + fields[2] + ' ' + fields[3]
                #print(new_line)
                circuit.raw_spice += new_line + os.linesep
                v_cont = v_cont + 1
            else:
                #print(line)
                circuit.raw_spice += line + os.linesep


        # #################################################
        # Simulate
        # #################################################
        # circuit.raw_spice = '.OPTIONS itl1=1000'
        # circuit.raw_spice = '.OPTIONS itl6=100'

        # sim_type = 'sim_op'
        simulator2 = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator2.operating_point()

        # #################################################
        # Save Output
        # #################################################

        sim_res_dict = {}
        i = 0
        for node in analysis.nodes.values():
            #print("node:", str(node))  # the name
            #print("array(node):", np.array(node))  # the data
            data_label = "node_%s" % str(node)
            sim_res_dict[data_label] = np.array(node)

        Vout = np.array([])
        for i in range(num_output):
            pos = num_nodes - num_output + i + 1
            data_label = "node_%s" % str(pos)
            #print(data_label, sim_res_dict[data_label])
            Vout = np.append(Vout, sim_res_dict[data_label])



    #

    # Finish and return voltages
    return Vout

#

#

#

#

#

#

#

#

#

#

#

#

# fin
