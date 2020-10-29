# #############################################################################
# Top Matter
# #############################################################################
import sys
import os
import PySpice

#from PySpice.Spice.NgSpice.Shared import NgSpiceShared

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from mod_material.spice.SPICE_setup import *
from mod_material.spice.SPICE_interpret import *





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


def LoadRun_model(Input_V, Dir, CompiledDict, pulse_Tp, cir):


    NetworkDict = CompiledDict['network']
    SpiceDict = CompiledDict['spice']

    # Create File path to load network topology
    path_fi = "%s/MD_CircuitTop/Network_Topology_%s_Cir_%d.txt" % (Dir, str(CompiledDict['param_file']), cir)
    #print(path_fi)

    # Create circuit
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

        # Write lines to spice, and replace voltage if it is a Vnode
        for line in lines:
            if line[0] == 'V' or line[:2] == 'Bs' or '.title' in line or line[0] == '+':
                pass  # we will redefine these things
            else:
                #print(line)
                circuit.raw_spice += line + os.linesep  # these are the material properties

        if SpiceDict['sim_type'] == 'sim_dc':
            sim_mode = 'dc'
            circuit, sim_slice = define_inputs_dc(circuit, Input_V)
        elif SpiceDict['sim_type'] == 'sim_trans_pulse':
            sim_mode = 'trans'
            circuit, t_end, sim_step_time = define_inputs_trans_Pulse(circuit, Input_V, pulse_Tp, SpiceDict)
        elif SpiceDict['sim_type'] == 'sim_trans_wave':
            sim_mode = 'trans'
            circuit, t_end, sim_step_time = define_inputs_trans_Wave(circuit, input_time_data, SpiceDict)



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
        num_samples = len(Input_V[:, 0])

        Vout, sample_time = format_Vout(analysis, pulse_Tp, CompiledDict, num_samples)

    del circuit, simulator, analysis

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
