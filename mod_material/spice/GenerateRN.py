import numpy as np
import sys
import os
import PySpice
import random
import seaborn as sns


from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Unit import *
import PySpice.Logging.Logging as Logging

from scipy import stats

# My imports
from mod_material.spice.SubCir import SeriesRD, SeriesNL, SeriesPOD
#from Module_SPICE.SPICE_setup import *



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
logger = Logging.setup_logging()  # causes print out?
enablePrint()

"""# current bug fix
import locale
locale.setlocale(locale.LC_NUMERIC, "C")"""


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

#


def generate_random_netork(CompiledDict, cir):

    # # # # # # # # # # # #
    # Assign setting to self from setting dict file

    MaterialDict = CompiledDict['spice']
    NetworkDict = CompiledDict['network']
    GenomeDict = CompiledDict['genome']
    #ParamDict = LoadSettings_ResnetDebug()

    # # make file
    dir_path = "%s/MD_CircuitTop" % (CompiledDict['SaveDir'])
    if os.path.exists(dir_path) is not True:
        os.makedirs(dir_path)



    node_cont = 1

    # Generate Fulley connected material nodes
    nodes = []
    for i in range(NetworkDict['num_input']+NetworkDict['num_config']):
        nodes.append("in%d" % (node_cont))
        node_cont = node_cont + 1
    for i in range(NetworkDict['num_output']):
        nodes.append("op%d" % (i+1))

    num_conn = calc_num_conn(len(nodes))



    # # # # # # # # # # # #
    # Create ressistance array accordign to charactersistic
    if NetworkDict['model'] == 'NL_RN':  # non-linear resistor network

        cov_matrix = [[1., MaterialDict['corr']],
                      [MaterialDict['corr'], 1.]]

        # generate correlated normal distributions
        mvnorm = stats.multivariate_normal([0, 0], cov_matrix, allow_singular=True)
        x = mvnorm.rvs((num_conn,))

        # convert normal distributions to uniform distributions
        norm = stats.norm([0],[1])
        x_unif = norm.cdf(x)
        nom_a_list, nom_b_list = x_unif[:,0], x_unif[:,1]


        # """
        h1 = sns.jointplot(x=x[:, 0], y=x[:, 1]+0.00001*np.random.rand(len(x[:,0])), kind='kde')  # add noise to avoid error
        h1.set_axis_labels('X1', 'X2', fontsize=16)
        h1.ax_joint.plot(x[:, 0], x[:, 1], ".", color="#800000")
        h1.ax_marg_y.set_ylim(-np.max(np.abs(x[:, 1]))-0.5, np.max(np.abs(x[:, 1]))+0.5)
        h1.ax_marg_x.set_xlim(-np.max(np.abs(x[:, 0]))-0.5, np.max(np.abs(x[:, 0]))+0.5)
        h1.fig.suptitle('Correlated Normal Distribution')
        h1.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.95)
        h1.plot_marginals(sns.rugplot, color="#800000", height=.15, clip_on=True)
        h1.savefig("%s/NormalDistr.png" % (dir_path))

        # # plot correlated uniform ditribution, created from the
        # # transformed normal distributions

        h2 = sns.jointplot(x=x_unif[:, 0], y=x_unif[:, 1], kind='hex', gridsize=4)
        h2.ax_joint.plot(x_unif[:, 0], x_unif[:, 1], ".", color="#800000")
        h2.set_axis_labels('Y1', 'Y2', fontsize=16)
        h2.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.95)
        h2.fig.suptitle('Correlated Uniform Distribution')
        h2.savefig("%s/UniformDistr.png" % (dir_path))
        #"""

        # scale to the appropriate a & b ranges
        diff = np.fabs(MaterialDict['material_a_min'] - MaterialDict['material_a_max'])  # absolute val of each el
        long_a_list = MaterialDict['material_a_min'] + nom_a_list * diff  # pop with their real values
        a_list = np.around(long_a_list, decimals=2)

        diff = np.fabs(MaterialDict['material_b_min'] - MaterialDict['material_b_max'])  # absolute val of each el
        long_b_list = MaterialDict['material_b_min'] + nom_b_list * diff  # pop with their real values
        b_list = np.around(long_b_list, decimals=2)

    elif NetworkDict['model'] == 'NL_nRN':

        a_var = MaterialDict['a_std']**2
        b_var = MaterialDict['b_std']**2

        # use the covarience factor [-1,1] to calc the selected cov
        max_covarience = (a_var*b_var)**0.5
        real_covarence = MaterialDict['corr']*max_covarience

        cov_matrix = [[a_var, real_covarence],
                      [real_covarence, b_var]]

        data = np.random.multivariate_normal([MaterialDict['a_mean'], MaterialDict['b_mean']], cov_matrix, size=num_conn)
        a_all = data[:, 0]
        b_all = data[:, 1]
        a_choices, b_choices = a_all, b_all

        a_list = []
        b_list = []
        for i in range(num_conn):
            while 1:
                aa = np.random.choice(a_choices, replace=False)
                bb = np.take(b_choices, np.where(a_choices == aa)[0])
                if aa > 0 and bb > 0:
                    a_list.append(aa)
                    b_list.append(bb)
                    break

        a_list = np.asarray(a_list)
        b_list = np.asarray(b_list)

    elif NetworkDict['model'] == 'NL_uRN':
        nom_a_list = np.random.rand(num_conn)
        nom_b_list = np.random.rand(num_conn)

        diff = np.fabs(MaterialDict['material_a_min'] - MaterialDict['material_a_max'])  # absolute val of each el
        long_a_list = MaterialDict['material_a_min'] + nom_a_list * diff  # pop with their real values
        a_list = np.around(long_a_list, decimals=2)

        diff = np.fabs(MaterialDict['material_b_min'] - MaterialDict['material_b_max'])  # absolute val of each el
        long_b_list = MaterialDict['material_b_min'] + nom_b_list * diff  # pop with their real values
        b_list = np.around(long_b_list, decimals=2)
        #b_list = np.zeros(len(b_list))

    elif NetworkDict['model'] == 'R_RN' or NetworkDict['model'] == 'D_RN' or NetworkDict['model'] == 'POD_RN':  # non-linear resistor network

        # create array of random numbers [0,1]
        nom_res_array = np.random.rand(num_conn, 1)

        # scale the random pop by boundry vals/resistances
        diff = np.fabs(MaterialDict['min_r'] - MaterialDict['max_r'])  # absolute val of each el
        long_res_array = MaterialDict['min_r'] + nom_res_array * diff  # pop with their real values
        res_array = np.around(long_res_array, decimals=4)

        # assign diode direction
        if NetworkDict['model'] == 'D_RN' or NetworkDict['model'] == 'POD_RN':
            if MaterialDict['rand_dir'] == 1:
                # array (lenth size) of random 0 or 1
                dir_array = np.random.randint(2, size=(num_conn))
            elif MaterialDict['rand_dir'] == 0 and MaterialDict['defualt_diode_dir'] == 1:
                dir_array = np.ones(num_conn)
            elif MaterialDict['rand_dir'] == 0 and MaterialDict['defualt_diode_dir'] == 0:
                dir_array = np.zeros(num_conn)
        # print(dir_array)

    elif NetworkDict['model'] == 'custom_RN':  # non-linear resistor network
        if MaterialDict['R_array_in'] == 'null':
            print("\nError (resnet): must define input array:")
            raise ValueError('(resnet): must define input array')
        if len(MaterialDict['R_array_in']) != (num_conn):
            print("\nError (resnet): must define input resistor array (and be correct size):")
            raise ValueError('(resnet): must define input resistor array (and be correct size)')
        res_array = MaterialDict['R_array_in']

    else:
        print("\nError (resnet): must select a model!!")
        print("ABORTED")
        raise ValueError('(resnet): must select a model!!')

    #print(res_array)

#

    # #################################################
    # Circuit Creation
    # #################################################

    name = 'RandomNetwork'

    circuit = Circuit(name)

    # #################################################
    # Input Creation
    # #################################################

    # custom
    """
    circuit.model('DefualtDiode', 'D',
                  IS=20@u_uA, RS=0.8@u_mOhm, BV=30@u_V, IBV=200@u_uV, N=1)
    #"""

    # 1N4148PH (Signal Diode)
    #"""
    circuit.model('DefualtDiode', 'D',
                  IS=4.352@u_nA, RS=0.6458@u_Ohm, BV=110@u_V, IBV=0.0001@u_V, N=1.906)
    #"""

    # 1N4001 (Power Diode)
    """
    circuit.model('DefualtDiode', 'D',
                  IS=29.5@u_nA, RS=73.5@u_mOhm, BV=60@u_V, IBV=10@u_uV, N=1.96)
    #"""

    # #################################################
    # Generate contact resistances
    # #################################################

    for node in nodes:
        if node[0] == 'i':
            circuit.R('%s_contact' % (node), '%s_conn' % (node), node, MaterialDict['contact_Res']@u_Ohm)
        #elif node[0] == 'o':
        #    circuit.R('%s_contact' % (node), node, '%s_conn' % (node), MaterialDict['contact_Res']@u_Ohm)

    # #################################################
    # Produce network acording to selected network
    # #################################################

    # # Initialise settings
    element_num = 1
    part = 1
    my_diode_num = 0
    node_pairs = []  # save data for result printing

    IS_val_lim = 'na'
    BV_val_lim = 'na'
    RS_val_lim = 'na'


    # # Place a component between every node pair
    for idx1, n1 in enumerate(nodes):
        for ind2, n2 in enumerate(nodes):
            if [n1,n2] in node_pairs or [n2,n1] in node_pairs or n1 == n2:
                continue  # skip if this pair has already been done
            node_pairs.append([n1, n2])  # save node pairs in arrays

            # non-linear resistor network
            if 'NL' in NetworkDict['model']:
                a, b = a_list[element_num-1], b_list[element_num-1]
                a_express = "%.2f%s" % (a, "n")
                b_express = "%.2f%s" % (b, "n")

                sub_name = '%s%d' % ('SubCir', element_num)
                SubCirc_obj = SeriesNL(sub_name, a_express, b_express)

                circuit.subcircuit(SubCirc_obj)
                circuit.X(element_num, sub_name, n1, n2)  # SubCir diode + res element is therefore

                element_num = element_num + 1

            # random resistor network
            elif NetworkDict['model'] == 'R_RN':
                temp_R = res_array[element_num-1, 0]
                circuit.R(element_num, n1, n2, temp_R@u_kOhm)  # create resistor element
                element_num = element_num + 1

            # custom defined resistor network
            elif NetworkDict['model'] == 'custom_RN':
                temp_R = res_array[element_num-1]
                circuit.R(element_num, n1, n2, temp_R@u_kOhm) # create resistor element
                element_num = element_num + 1

            # random resistor network with diodes
            elif NetworkDict['model'] == 'D_RN':

                sub_name = '%s%d' % ('SubCir', part)
                temp_R = res_array[element_num-1, 0]
                dir = dir_array[element_num-1]
                SubCirc_obj = SeriesRD(sub_name, Res=temp_R,
                                       direction=dir, DefualtDiode=MaterialDict['DefualtDiode'],
                                       custom_diode_num=element_num,
                                       IS_val=0.1)
                circuit.subcircuit(SubCirc_obj)
                circuit.X(element_num, sub_name, n1, n2)  # SubCir diode + res element is therefore

                element_num = element_num + 1
                part = part + 1

            # random resistors in series with opposing direction parallel diodes
            elif NetworkDict['model'] == 'POD_RN':

                sub_name = '%s%d' % ('SubCir', part)
                temp_R = res_array[element_num-1, 0]
                dir = dir_array[element_num-1]

                SubCirc_obj = SeriesPOD(sub_name, Res=temp_R,
                                        direction=dir, DefualtDiode=0,
                                        custom_diode_num=my_diode_num,
                                        IS_val='rand',
                                        BV_val='rand',
                                        RS_val='rand')
                circuit.subcircuit(SubCirc_obj)
                circuit.X(element_num, sub_name, n1, n2)  # SubCir diode + res element is therefore

                # on the first loop only save data
                if my_diode_num == 0:
                    IS_val_lim = SubCirc_obj.IS_val_lim
                    BV_val_lim = SubCirc_obj.BV_val_lim
                    RS_val_lim = SubCirc_obj.RS_val_lim
                element_num = element_num + 1
                my_diode_num = my_diode_num + 2
                part = part + 1


    # #################################################
    # ADC Shunt Resistors & extras
    # #################################################
    if MaterialDict['shunt_R'] != 'none':
        for i in range(NetworkDict['num_output']):
            Out_Node = "op%d" % (i+1)
            # print("Node out", Out_Node)
            circuit.R("shunt%d" % (i+1), Out_Node, circuit.gnd, MaterialDict['shunt_R']@u_kOhm)

    if MaterialDict['shuntC'] == 1:
        min, max = MaterialDict['ShuntCapLim']

        nom_shuntcap_array = np.random.rand(len(nodes))
        diff = np.fabs(max-min)  # absolute val of each el
        long_shuntcap_array = min + nom_shuntcap_array * diff  # pop with their real values
        shuntcap_array = np.around(long_shuntcap_array, decimals=3)
        i = 1
        for node in nodes:
            circuit.C("shunt%d" % (i), node, circuit.gnd, shuntcap_array[i-1]@u_uF)
            i = i + 1


    if MaterialDict['parallelC'] == 1:
        min, max = MaterialDict['ParallelCapLim']
        el_num = 1
        cap_node_pairs = []
        pcn = 1

        nom_parlcap_array = np.random.rand(num_conn)
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
    # # Save the netowrk Topology
    # #################################################
    # print("the circuit: \n", circuit)

    # # Print circuit topography to file
    circuit_top = circuit
    circuit_top = '%s' % circuit_top


    Net_top_path = "%s/Network_Topology_%s_Cir_%s.txt" % (dir_path, str(CompiledDict['param_file']), cir)
    #print("First save:", Net_top_path)
    file1 = open(Net_top_path, "w")
    file1.write(circuit_top)
    file1.close()

    # #################################################
    # # Save material description to file
    # #################################################

    if NetworkDict['model'] == 'D_RN':
        dir_array = dir_array
        text_list = []
        for i in range(len(res_array)):
            if dir_array[i] == 1:
                text_list.append('R%s%s = %06.3f Kohm,   Node%s--/\/\/----|>--Node%s (i.e direction %d)' % (
                                 node_pairs[i][0], node_pairs[i][1], res_array[i,0], node_pairs[i][0], node_pairs[i][1], dir_array[i]))
            elif dir_array[i] == 0:
                text_list.append('R%s%s = %06.3f Kohm,   Node%s--<|----/\/\/--Node%s (i.e direction %d)' % (
                                 node_pairs[i][0], node_pairs[i][1], res_array[i,0], node_pairs[i][0], node_pairs[i][1], dir_array[i]))

        res_diode_list = ''
        for list_bit in text_list:
            res_diode_list = '%s%s \n' % (res_diode_list, list_bit)


    elif NetworkDict['model'] == 'POD_RN':
        dir_array = dir_array
        text_list = []
        for i in range(len(res_array)):
            if dir_array[i] == 1:
                text_list.append('R%s%s = %06.3f Kohm,   Node%s--/\/\/----BackToBack_ParallelDiodes--Node%s (i.e direction %d)' % (
                                 node_pairs[i][0], node_pairs[i][1], res_array[i,0], node_pairs[i][0], node_pairs[i][1], dir_array[i]))
            elif dir_array[i] == 0:
                text_list.append('R%s%s = %06.3f Kohm,   Node%s--BackToBack_ParallelDiodes----/\/\/--Node%s (i.e direction %d)' % (
                              node_pairs[i][0], node_pairs[i][1], res_array[i,0], node_pairs[i][0], node_pairs[i][1], dir_array[i]))


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

    elif 'NL' in NetworkDict['model']:
        text_list = []
        for i in range(len(a_list)):

            a, b = a_list[i], b_list[i]
            a_express = "%.2f%s" % (a, "n")
            b_express = "%.2f%s" % (b, "n")
            vdiff = '(v(%s)-v(%s))' % (node_pairs[i][0], node_pairs[i][1])
            i_express = "if((%s>0)  THEN  (%s*%s**2 + %s*%s)  ELSE  (-%s*%s**2 + %s*%s )" % (vdiff,
                                                                                             a_express, vdiff, b_express, vdiff,
                                                                                             a_express, vdiff, b_express, vdiff)
            text_list.append(i_express)

            res_diode_list = ''
            for list_bit in text_list:
                res_diode_list = '%s%s \n' % (res_diode_list, list_bit)


    elif NetworkDict['model'] == 'custom_RN':
        text_list = []
        for i in range(len(res_array)):
            text_list.append('R %s %s = %06.3f Kohm' % (
                             node_pairs[i][0], node_pairs[i][1], res_array[i]))

        res_diode_list = ''
        for list_bit in text_list:
            res_diode_list = '%s%s \n' % (res_diode_list, list_bit)

    else:
        text_list = []
        for i in range(len(res_array)):
            text_list.append('R %s %s = %06.3f Kohm' % (
                             node_pairs[i][0], node_pairs[i][1], res_array[i, 0]))


    if MaterialDict['shuntC'] == 1:
        text_list.append('\nShunt Cap:')
        i = 0
        for node in nodes:
            text_list.append('C %s %d = %06.3f uF' % (
                             str(node), 0, shuntcap_array[i]))
            i = i + 1

    if MaterialDict['parallelC'] == 1:
        text_list.append('\nParallel Cap:')
        pcn = 1
        for cap_pair in cap_node_pairs:
                text_list.append('C %s %s = %06.3f uF' % (
                                 cap_pair[0], cap_pair[1], parlcap_array[pcn-1]))
                pcn = pcn + 1

    text_list.append('\nOutput Shunt Resistance = %.3f kOhm' % (MaterialDict['shunt_R']))
    text_list.append('Contact Resistance = %.3f Ohm' % (MaterialDict['contact_Res']))

    # now try and write it
    circuit_top_path = "%s/MaterialDescription_%s_Cir_%s.txt" % (dir_path, str(CompiledDict['param_file']), cir)
    res_diode_list = ''
    for list_bit in text_list:
        res_diode_list = '%s%s \n' % (res_diode_list, list_bit)

    file2 = open(circuit_top_path, "w")
    file2.write(res_diode_list)
    file2.close()

    return

#

# fin
