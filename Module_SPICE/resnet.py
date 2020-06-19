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

from Module_SPICE.DiodeSubCir import SeriesRD
from Module_SPICE.DoubleDiodeSubCir import SeriesPOD
import PySpice.Logging.Logging as Logging

# My imports
from Module_Settings.Set_Load import LoadSettings, LoadSettings_ResnetDebug


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


class ResNet(object):

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Initialisation of class
    '''
    def __init__(self):

        # # # # # # # # # # # #
        # Assign setting to self from setting dict file
        ParamDict = LoadSettings()
        #ParamDict = LoadSettings_ResnetDebug()

        self.num_output = ParamDict['num_output']
        self.shunt_R = ParamDict['shunt_R']
        self.num_nodes = ParamDict['num_nodes']
        self.model = ParamDict['model']
        self.num_connections = ParamDict['num_connections']
        self.R_array_in = ParamDict['R_array_in']
        self.rand_dir = ParamDict['rand_dir']
        self.defualt_diode_dir = ParamDict['defualt_diode_dir']
        self.DefualtDiode = ParamDict['DefualtDiode']

        self.loop = ParamDict['loop']  # the current loop from DE.py
        self.SaveDir = ParamDict['SaveDir']  # extract save file
        self.circuit_loop = ParamDict['circuit_loop']  # the generated circuits by the renet



        # # # # # # # # # # # #
        # Create ressistance array accordign to charactersistic
        if self.model == 'NL_RN':  # non-linear resistor network
            num_coef = 2

            # create array of coeficient pairs
            self.res_array = np.random.rand(self.num_connections, num_coef)

        elif self.model == 'R_RN' or self.model == 'D_RN' or self.model == 'POD_RN':  # non-linear resistor network
            num_coef = 1

            # not SWCNT are ~kOhms, with a range of 10Kohms
            max_R = ParamDict['max_r']
            min_R = ParamDict['min_r']
            max_coef = max_R
            min_coef = min_R

            # create array of random numbers [0,1]
            nom_res_array = np.random.rand(self.num_connections, num_coef)

            # scale the random pop by boundry vals/resistances
            diff = np.fabs(min_coef - max_coef)  # absolute val of each el
            long_res_array = min_coef + nom_res_array * diff  # pop with their real values
            self.res_array = np.around(long_res_array, decimals=4)

            # assign diode direction
            if self.model == 'D_RN' or self.model == 'POD_RN':
                if self.rand_dir == 1:
                    # array (lenth size) of random 0 or 1
                    self.dir_array = np.random.randint(2, size=self.num_connections)
                elif self.rand_dir == 0 and self.defualt_diode_dir == 1:
                    self.dir_array = np.ones(self.num_connections)
                elif self.rand_dir == 0 and self.defualt_diode_dir == 0:
                    self.dir_array = np.zeros(self.num_connections)
            # print(self.dir_array)

        elif self.model == 'custom_RN':  # non-linear resistor network
            if self.R_array_in == 'null':
                print(" ")
                print("Error (resnet): must define input array:")
                raise ValueError('(resnet): must define input array')
            if len(self.R_array_in) != self.num_connections:
                print(" ")
                print("Error (resnet): must define input resistor array (and be correct size):")
                raise ValueError('(resnet): must define input resistor array (and be correct size)')
            self.res_array = self.R_array_in

        else:
            print(" ")
            print("Error (resnet): must select a model!!")
            print("ABORTED")
            raise ValueError('(resnet): must select a model!!')

        #print(self.res_array)

    #

    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Run the initialied SPICE model
    '''
    def run_model(self, Input_V, SaveNetText=0):  # the PySpice + NGSpice model

        # #################################################
        # Check condictions
        # #################################################
        if len(Input_V) != (self.num_nodes-self.num_output):
            print(" ")
            print("Error (resnet): Input Vector doesn't equal input node number")
            raise ValueError("(resnet): Input Vector doesn't equal input node number")

        # #################################################
        # Circuit Creation
        # #################################################
        circuit = Circuit('Five node B-source Resistor network')

        circuit.model('DefualtDiode', 'D', IS=20@u_uA, RS=0.8@u_mOhm,
                      BV=30@u_V, IBV=200@u_uV, N=1)

        # #################################################
        # Input Creation
        # #################################################
        for inp in range(self.num_nodes-self.num_output):
            in_node = inp + 1
            in_text = 'input%d' % (in_node)
            the_v = Input_V[inp]
            circuit.V(in_text, in_node, circuit.gnd, the_v@u_V)

        # #################################################
        # Produce Resistor network acording to selected network
        # #################################################
        # # Initialise settings
        pin1 = 1
        element_num = 0
        self.node_pairs_1 = np.array([])  # save data for result printing
        self.node_pairs_2 = np.array([])  # save data for result printing

        if self.model == 'NL_RN':  # non-linear resistor network
            # might need to call external iterative solution here
            for i in range(self.num_connections):
                for ii in range(self.num_nodes - pin1):
                    pin2 = ii + pin1 + 1
                    self.node_pairs_1 = np.append(self.node_pairs_1, pin1)  # save node pairs in arrays
                    self.node_pairs_2 = np.append(self.node_pairs_2, pin2)

                    # create expression i = a*(v(pin1, pin2))**2 + b*(v(pin1, pin2))
                    sqr_term = '**2'
                    s2 = '*(v(%d, %d))' % (pin1, pin2)
                    I_express = '%f%s%s%s%f%s' % (self.res_array[element_num, 0], s2,
                                                  sqr_term, ' + ', self.res_array[element_num, 1], s2)
                    # print(I_express)

                    # circuit.B is therefore
                    circuit.B(element_num, pin1, pin2, i=I_express)
                    element_num = element_num + 1
                # end loop, increment pin1
                pin1 = pin1 + 1

        elif self.model == 'R_RN':  # random resistor network
            for i in range(self.num_connections):
                for ii in range(self.num_nodes - pin1):
                    pin2 = ii + pin1 + 1
                    self.node_pairs_1 = np.append(self.node_pairs_1, pin1)  # save node pairs in arrays
                    self.node_pairs_2 = np.append(self.node_pairs_2, pin2)

                    # circuit.R element is therefore
                    temp_R = self.res_array[element_num, 0]
                    circuit.R(element_num, pin1, pin2, temp_R@u_kOhm)
                    element_num = element_num + 1
                # end loop, increment pin1
                pin1 = pin1 + 1

        elif self.model == 'custom_RN':  # custom defined resistor network
            for i in range(self.num_connections):
                for ii in range(self.num_nodes - pin1):
                    pin2 = ii + pin1 + 1
                    self.node_pairs_1 = np.append(self.node_pairs_1, pin1)  # save node pairs in arrays
                    self.node_pairs_2 = np.append(self.node_pairs_2, pin2)

                    # circuit.R element is therefore
                    temp_R = self.res_array[element_num]
                    #print(temp_R)
                    circuit.R(element_num, pin1, pin2, temp_R@u_kOhm)
                    element_num = element_num + 1
                # end loop, increment pin1
                pin1 = pin1 + 1
        elif self.model == 'D_RN':  # random resistor network with diodes
            part = 1
            for i in range(self.num_connections):
                for ii in range(self.num_nodes - pin1):
                    pin2 = ii + pin1 + 1

                    # SubCir diode + res element is therefore
                    sub_name = '%s%d' % ('SubCir', part)
                    temp_R = self.res_array[element_num, 0]
                    dir = self.dir_array[element_num]
                    self.node_pairs_1 = np.append(self.node_pairs_1, pin1)  # save node pairs in arrays
                    self.node_pairs_2 = np.append(self.node_pairs_2, pin2)

                    SubCirc_obj = SeriesRD(sub_name, Res=temp_R,
                                           direction=dir, DefualtDiode=self.DefualtDiode,
                                           custom_diode_num=element_num,
                                           IS_val=0.1)
                    circuit.subcircuit(SubCirc_obj)
                    circuit.X(element_num, sub_name, pin1, pin2)

                    element_num = element_num + 1
                    part = part + 1
                # end loop, increment pin1
                pin1 = pin1 + 1
        elif self.model == 'POD_RN':  # random resistors in series with opposing direction parallel diodes
            part = 1
            my_diode_num = 0
            for i in range(self.num_connections):
                for ii in range(self.num_nodes - pin1):
                    pin2 = ii + pin1 + 1

                    # SubCir diode + res element is therefore
                    sub_name = '%s%d' % ('SubCir', part)
                    temp_R = self.res_array[element_num, 0]
                    dir = self.dir_array[element_num]
                    self.node_pairs_1 = np.append(self.node_pairs_1, pin1)  # save node pairs in arrays
                    self.node_pairs_2 = np.append(self.node_pairs_2, pin2)

                    SubCirc_obj = SeriesPOD(sub_name, Res=temp_R,
                                            direction=dir, DefualtDiode=0,
                                            custom_diode_num=my_diode_num,
                                            IS_val='rand',
                                            BV_val='rand',
                                            RS_val='rand')
                    circuit.subcircuit(SubCirc_obj)
                    circuit.X(element_num, sub_name, pin1, pin2)

                    # on the first loop only save data
                    if my_diode_num == 0 and SaveNetText == 1:
                        IS_val_lim = SubCirc_obj.IS_val_lim
                        BV_val_lim = SubCirc_obj.BV_val_lim
                        RS_val_lim = SubCirc_obj.RS_val_lim


                    element_num = element_num + 1
                    my_diode_num = my_diode_num + 2
                    part = part + 1
                # end loop, increment pin1
                pin1 = pin1 + 1

        # #################################################
        # ADC Shunt Resistors
        # #################################################
        if self.shunt_R != 'none':
            for i in range(self.num_output):
                Out_Node = self.num_nodes - self.num_output + i + 1  # select last nodes for output
                element_num = element_num + 1  # increment element
                # print("Node out", Out_Node)
                circuit.R(element_num, Out_Node, circuit.gnd, self.shunt_R@u_kOhm)

        # #################################################
        # Print the circuit
        # #################################################
        # print("")
        # print("the circuit:")
        # print(circuit)
        #self.circuit = circuit

        # #################################################
        # Simulate
        # #################################################
        # circuit.raw_spice = '.OPTIONS itl1=1000'
        # circuit.raw_spice = '.OPTIONS itl6=100'

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        sim_type = 'sim_op'

        if sim_type == 'sim_op':
            # print("...")
            # print("Sim - op")
            analysis = simulator.operating_point()

        elif sim_type == 'sim_trans':
            # print("...")
            # print("Sim - trans")
            analysis = simulator.transient(step_time=1@u_us, end_time=100@u_us)

        # #################################################
        # Save Output
        # #################################################

        Vout = self.format_Vout(analysis)
        #print("Vout:", Vout)


        # #################################################
        # print the ResNet
        # #################################################
        if SaveNetText == 1:


            dir_path = "%s/MD_CircuitTop" % (self.SaveDir)

            if os.path.exists(dir_path) is not True:
                os.makedirs(dir_path)

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Print circuit topography to file
            circuit_top = circuit
            circuit_top = '%s' % circuit_top

            try:
                Net_top_path = "%s/Netowrk_Topology__Loop_%s.txt" % (dir_path, self.circuit_loop)
                file1 = open(Net_top_path, "w")
                file1.write(circuit_top)
                file1.close()
            except:
                pass

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Print resistance, diode direction and coeficient vals to file
            res_array = self.res_array
            Node1 = self.node_pairs_1
            Node2 = self.node_pairs_2

            if self.model == 'D_RN':
                dir_array = self.dir_array
                text_list = []
                for i in range(len(res_array)):
                    if dir_array[i] == 1:
                        text_list.append('R%d%d = %06.3f Kohm,   Node%d--/\/\/----|>--Node%d (i.e direction %d)' % (
                                         Node1[i], Node2[i], res_array[i,0], Node1[i], Node2[i], dir_array[i]))
                    elif dir_array[i] == 0:
                        text_list.append('R%d%d = %06.3f Kohm,   Node%d--<|----/\/\/--Node%d (i.e direction %d)' % (
                                         Node1[i], Node2[i], res_array[i,0], Node1[i], Node2[i], dir_array[i]))
                res_diode_list = ''
                for list_bit in text_list:
                    res_diode_list = '%s%s \n' % (res_diode_list, list_bit)

                # create path
                circuit_top_path = "%s/ResistorArray_AND_DiodeDirection__Loop_%s.txt" % (dir_path, self.circuit_loop)

            elif self.model == 'POD_RN':
                dir_array = self.dir_array
                text_list = []
                for i in range(len(res_array)):
                    if dir_array[i] == 1:
                        text_list.append('R%d%d = %06.3f Kohm,   Node%d--/\/\/----BackToBack_ParallelDiodes--Node%d (i.e direction %d)' % (
                                         Node1[i], Node2[i], res_array[i,0], Node1[i], Node2[i], dir_array[i]))
                    elif dir_array[i] == 0:
                        text_list.append('R%d%d = %06.3f Kohm,   Node%d--BackToBack_ParallelDiodes----/\/\/--Node%d (i.e direction %d)' % (
                                      Node1[i], Node2[i], res_array[i,0], Node1[i], Node2[i], dir_array[i]))


                # Save the limits for the random diode IS values
                if IS_val_lim == 'fixed':
                    text_list.append('\nUsed a fixed IS val for all diodes')
                else:
                    text_list.append('\nUsed Random IS vals, with limits: %s [uA]' % (str(IS_val_lim)))

                # Save the limits for the random diode BV values
                if BV_val_lim == 'fixed':
                    text_list.append('Used a fixed BV val for all diodes')
                else:
                    text_list.append('Used Random BV vals, with limits: %s [uV]' % (str(BV_val_lim)))

                # Save the limits for the random diode RS values
                if RS_val_lim == 'fixed':
                    text_list.append('Used a fixed RS val for all diodes')
                else:
                    text_list.append('Used Random IS vals, with limits: %s [mOhm]' % (str(RS_val_lim)))




                res_diode_list = ''
                for list_bit in text_list:
                    res_diode_list = '%s%s \n' % (res_diode_list, list_bit)

                # create path
                circuit_top_path = "%s/ResistorArray_AND_DiodeDirection__Loop_%s.txt" % (dir_path, self.circuit_loop)

            elif self.model == 'NL_RN':
                print("need work")

            elif self.model == 'custom_RN':
                text_list = []
                for i in range(len(res_array)):
                    text_list.append('R%d%d = %06.3f Kohm' % (
                                     Node1[i], Node2[i], res_array[i]))

                res_diode_list = ''
                for list_bit in text_list:
                    res_diode_list = '%s%s \n' % (res_diode_list, list_bit)

                # create path
                circuit_top_path = "%s/ResistorArray__Loop_%s.txt" % (dir_path, self.circuit_loop)

            else:
                text_list = []
                for i in range(len(res_array)):
                    text_list.append('R%d%d = %06.3f Kohm' % (
                                     Node1[i], Node2[i], res_array[i, 0]))

                res_diode_list = ''
                for list_bit in text_list:
                    res_diode_list = '%s%s \n' % (res_diode_list, list_bit)

                # create path
                circuit_top_path = "%s/ResistorArray__Loop_%s.txt" % (dir_path, self.circuit_loop)

            try:
                file2 = open(circuit_top_path, "w")
                file2.write(res_diode_list)
                file2.close()
            except:
                pass

        return Vout


    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Format the output from the spice sim
    '''
    def format_Vout(self, analysis):  # the PySpice + NGSpice model

        #id = random.random()
        sim_res_dict = {}
        i = 0
        for node in analysis.nodes.values():
            #print("node:", str(node))  # the name
            #print("array(node):", np.array(node))  # the data
            data_label = "node_%s" % str(node)
            sim_res_dict[data_label] = np.array(node)


        Vout = np.array([])
        for i in range(self.num_output):
            pos = self.num_nodes - self.num_output + i + 1
            data_label = "node_%s" % str(pos)
            #print(id, data_label, sim_res_dict[data_label])
            Vout = np.append(Vout, sim_res_dict[data_label])


        return Vout


    ''' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    Format the output from the spice sim
    '''
    def format_Vout_old(self, analysis):  # the PySpice + NGSpice model

        #print(analysis.nodes.values(),"\n")
        error_called = 0
        Vout = np.array([])
        Node_name = []
        Node_V = np.array([])
        exit_flag = 0
        # read results from analysis
        for node in analysis.nodes.values():
            #print('node:', node)
            #print("#######")
            name = str(node)
            #if SaveNetText == 1:
                #print('Node_name', name)
                #print('Node_f', float(node))

            # only extrac data from the main nodes, not internal sub-circuit nodes
            if name[0] != 'x':
                #print(name)
                Node_name.append('{}'.format(str(node)))
                try:
                    node_v = '{:5.4f}'.format(float(node))
                    Node_V = np.append(Node_V, node_v)
                except:
                    try:
                        analysis2 = simulator2.operating_point()
                        for node in analysis2.nodes.values():
                            if name[0] != 'x':
                                node_v = '{:5.4f}'.format(float(node))
                                Node_V = np.append(Node_V, node_v)
                    except Exception as e:
                        print('ERROR IN SIM RESULT!')
                        #print(node)
                        print("alanysis 1:\n", analysis.nodes.values())
                        print("alanysis 2:\n", analysis2.nodes.values())
                        print("Input_V", Input_V)
                        print(circuit)

                        exit_flag = 1
                        error_called = 1
                        print("*****************")
                        print("** Error Thrown is:\n", e)




            #print('Voltage:', node_v)
            #print("********")

        #if SaveNetText == 1:
            #print('Node_name', Node_name)
            #print('Node_V', Node_V)


        # find the correct index for the output nodes, and read V from array

        if exit_flag == 0:
            for i in range(self.num_output):
                pos = self.num_nodes - self.num_output + i + 1
                for j in range(len(Node_name)):
                    if str(pos) == Node_name[j]:
                        #if SaveNetText == 1:
                        #print("Pos:", str(pos), "  Node:", Node_name[j])
                        try:
                            Vout = np.append(Vout, float(Node_V[j]))
                        except Exception as e:
                            print('ERROR IN Appending to Vout!')
                            print("Pos:", str(pos), "  Node:", Node_name[j])
                            print(Node_V[j])
                            print("******")
                            print("** Error Thrown is:\n", e)

        return Vout


#

# fin
