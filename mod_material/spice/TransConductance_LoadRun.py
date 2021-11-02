# #############################################################################
# Top Matter
# #############################################################################
import sys
import os
import PySpice
import numpy as np

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from mod_material.spice.SPICE_setup import *
from mod_material.spice.SPICE_interpret import *

from PySpice.Spice.NgSpice.Shared import NgSpiceShared



# change sim program location depending on system
if sys.platform == "linux" or sys.platform == "linux2":
    #PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'ngspice-subprocess'  # needed for linux
    PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'ngspice-subprocess'  # needed for linux
    pass
elif sys.platform == "win32":
    #NgSpiceShared._ngspice_id  = 1
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
This runs a simulation of a pre-generated circuit (which is loaded from
a .txt file)

'''

# #############################################################################
# End of top matter
# #############################################################################


def LoadRun_DC_TransConductance(prm, syst, other_InNodes='float', StaticVin2=1, ShuntR=70, l='na', m='na'):
    """
    This loads in a previously produced material, then :
        - Applies a static voltage to input 2
        - Does a DC Voltage Sweep on input 1
        - Measures the output current

    The material is loaded acording to:
        1) The name of the param_file associated with the prcoessor object
        2) The system loop
        3) The Layer & material-in-layer index
    """

    NetworkDict = prm['network']
    SpiceDict = prm['spice']

    # # Add layer refference
    if l == 'na':
        lref = ''
    else:
        lref = '_l%d' % (l)

    # # Add the material (in a select layer) refference
    if m == 'na':
        mref = ''
    else:
        mref = '_m%d' % (m)

    if prm['ReUse_dir'] != 'na':
        ModelDir = prm['ReUse_dir']
    else:
        ModelDir = prm['SaveDir']

    # # Create File path to load network topology
    path_fi = "%s/MD_CircuitTop/Network_Topology_%s_Syst%d%s%s.txt" % (ModelDir, str(prm['param_file']), syst, lref, mref)
    #print(path_fi)

    # # Create circuit
    name = 'Material Network ReLoad'
    circuit = Circuit(name)

    # Load from file and perform sim op
    with open(path_fi, 'r') as reader:

        # Extract raw string from loaded file
        # NetTop = reader.read()
        NetTop = reader.readlines()  # reads lines, but includes "\n"

        # Break up and re-write Network in raw spice
        lines = [a_line.rstrip('\n') for a_line in NetTop]
        try:
            while 1:
                lines.remove('')
        except:
            pass

        #

        # # Write lines to spice, and replace voltage if it is a Vnode
        for line in lines:
            if line[0] == 'V' or line[:2] == 'Bs' or '.title' in line or line[0] == '+' or 'contact' in line or 'shunt' in line:
                pass  # we will redefine these things
            else:
                #print(line)
                circuit.raw_spice += line + os.linesep  # these are the material properties

        #

        # # Peripheral Resistances # #
        node_list = []  # collect input node list

        # Generate Input contacts
        for i in range(NetworkDict['num_input']+NetworkDict['num_config']+NetworkDict['num_output']):
            node_list.append("n%d" % (i+1))

        #

        # # Define Inputs # #
        in1_node = 1
        circuit.R('in%s_contact' % (1), 'in%s_conn' % (1), 'n%d' % (in1_node), SpiceDict['in_contact_R']@u_Ohm)
        circuit.V('test_source', 'in1_conn', circuit.gnd, 0@u_V)
        node_list.remove('n%d' % (in1_node))

        in2_node = 2
        circuit.R('in%s_contact' % (2), 'in%s_conn' % (2), 'n%d' % (in2_node), SpiceDict['in_contact_R']@u_Ohm)
        circuit.V('static_source', 'in2_conn', circuit.gnd, StaticVin2@u_V)
        node_list.remove('n%d' % (in2_node))

        # # Define Output node
        op_node = len(node_list)
        circuit.R('op%s_contact' % (1), 'op%s_conn' % (1), 'n%d' % (op_node), SpiceDict['op_contact_R']@u_Ohm)
        circuit.R("op%d_shunt" % (1), 'n%d' % (op_node), circuit.gnd, ShuntR@u_kOhm)
        circuit.V('opA', 'op1_conn', circuit.gnd, 0@u_V)
        node_list.remove('n%d' % (op_node))

        # # Define other Input nodes
        if other_InNodes == 'float':
            pass
        else:
            for idx, n in enumerate(node_list):
                circuit.V('in%d' % (idx+1), n, circuit.gnd, other_InNodes@u_V)




        # #################################################
        # #  Simulate
        # #################################################
        # circuit.raw_spice = '.OPTIONS itl1=1000'
        # circuit.raw_spice = '.OPTIONS itl6=100'
        circuit.raw_spice += '.OPTIONS itl1=10000' + os.linesep
        circuit.raw_spice += '.OPTIONS itl6=500' + os.linesep

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        # print("Node To Ops Load_run:\n", simulator)
        # exit()

        diff = prm['spice']['Vmax'] - prm['spice']['Vmin']
        max = prm['spice']['Vmax'] + (0.1*diff)
        min = prm['spice']['Vmin'] - (0.1*diff)
        interval = 0.05
        sim_slice = slice(min, max, interval)
        analysis = simulator.dc(vtest_source=sim_slice)  # slice = start:step:stop (inclusive)

        #

    # # Get output
    in_sweep = np.array(analysis['in1_conn'])
    out_I =  np.array(analysis['VopA'])


    """ # Flawed check!
    if ShuntR != 0:
        Vo = np.array(analysis['op1_conn'])
        print("Vo", Vo)
        io = Vo[0]/(ShuntR*1000)

        if abs(np.round(io,4)) != abs(np.round(out_I[0],4)):
            print("Warning bas comparison:", 'io=%.4f' % io, 'out_I=%.4f' % out_I[0])
            print("Warning:  Calc io:", io, ",  Sim out_I:", out_I[0])

            raise ValueError("Transconductance current wrongly calcualted")
    # """

    # # clean up
    del circuit, simulator, analysis


    # Finish and return voltages
    return in_sweep, out_I


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
