# ###################################################################################################
# Top Matter

import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os

import PySpice
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Unit import *

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

# change sim program location depending on system
if sys.platform == "linux" or sys.platform == "linux2":
    PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = 'ngspice-subprocess'  # needed for linux
elif sys.platform == "win32":
    pass

''' # Please Read



'''
# ###################################################################################################
# # Define a subcircuit


class SeriesRD(SubCircuit):
    __nodes__ = ('in', 'out')
    def __init__(self, name, Res=1, direction=1,
                 DefualtDiode=1, custom_diode_num=1,
                 IS_val=20, RS_val=0.8, BV_val=20, IBV_val=100):

        SubCircuit.__init__(self, name, *self.__nodes__)


        if DefualtDiode == 1:
            if direction == 1:
                self.R(1, 'in', 'mid', Res@u_kOhm)
                self.raw_spice = 'D1 mid out DefualtDiode'
                #print("DIR OPTION 1")
            elif direction == 0:
                self.R(1, 'mid', 'out', Res@u_kOhm)
                self.raw_spice = 'D1 mid in DefualtDiode'
                #print("DIR OPTION 0")
        elif DefualtDiode == 0:
            Diode_Name = '%s%d%s' % ('My', custom_diode_num, 'Diode')
            self.model(Diode_Name, 'D', IS=IS_val@u_uA, RS=RS_val@u_mOhm,
                          BV=BV_val@u_V, IBV=IBV_val@u_uV, N=1)
            if direction == 1:
                self.R(1, 'in', 'mid', Res@u_kOhm)
                the_raw_spice = 'D%d mid out %s' % (custom_diode_num, Diode_Name)
                self.raw_spice = the_raw_spice
                #print("DIR OPTION 1")
            elif direction == 0:
                self.R(1, 'mid', 'out', Res@u_kOhm)
                the_raw_spice = 'D%d mid in %s' % (custom_diode_num, Diode_Name)
                self.raw_spice = the_raw_spice
                #print("DIR OPTION 0")

    # def DirRet():
    #    return self.the_dir




# fin
