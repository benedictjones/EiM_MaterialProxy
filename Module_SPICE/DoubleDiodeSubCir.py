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
# # random network of parallel oposing diodes in series with a random resistor


class SeriesPOD(SubCircuit):

    __nodes__ = ('in', 'out')

    def __init__(self, name, Res=1, direction=1,
                 DefualtDiode=1, custom_diode_num=1,
                 IS_val='rand', RS_val=0.8, BV_val=20, IBV_val=100):

        SubCircuit.__init__(self, name, *self.__nodes__)

        # PArameters to extract if wanted
        self.IS_val_lim = 'fixed'
        self.BV_val_lim = 'fixed'
        self.RS_val_lim = 'fixed'

        # USe the passed in Defualt diode properties defined in resnet.py
        if DefualtDiode == 1:
            # Resistor, then back to back diode
            if direction == 1:
                self.R(1, 'in', 'mid', Res@u_kOhm)
                ## self.raw_spice = 'D1 mid out DefualtDiode\nD2 out mid DefualtDiode'  # DOES NOT WORK

                self.raw_spice = 'D1 mid out DefualtDiode' + os.linesep
                self.raw_spice += 'D2 out mid DefualtDiode'

                # self.Diode(1, 'mid', 'out', raw_spice=DefualtDiode)
                # self.Diode(2, 'out', 'mid', raw_spice=DefualtDiode)

            # Resistor, then back to back diode
            elif direction == 0:
                self.R(1, 'mid', 'out', Res@u_kOhm)

                self.raw_spice = 'D1 mid in DefualtDiode' + os.linesep
                self.raw_spice += 'D2 in mid DefualtDiode'

                # self.Diode(1, 'mid', 'in', raw_spice=DefualtDiode)
                # self.Diode(2, 'in', 'mid', raw_spice=DefualtDiode)


        # USe custom diode properties
        elif DefualtDiode == 0:

            # # # # # # # # # # # # # # #
            # get random IS bounds
            if IS_val == 'rand':
                max_IS = 100  # uAmps
                min_IS = 0.001  # uAmps
                self.IS_val_lim = [min_IS, max_IS]

                # create array of random numbers [0,1]
                nom_IS_array = np.random.rand(2)

                # scale the random pop by boundry vals/resistances
                diff = np.fabs(min_IS - max_IS)  # fine the difference
                IS_array = min_IS + nom_IS_array * diff  # scale random number (0,1)
                IS_array = np.around(IS_array, decimals=3)  # round
            else:
                IS_array = [IS_val, IS_val]

            # # # # # # # # # # # # # # #
            # get random BV_val bounds
            if BV_val == 'rand':
                max_BV = 30  # V
                min_BV = 1  # V
                self.BV_val_lim = [min_BV, max_BV]

                # create array of random numbers [0,1]
                nom_BV_array = np.random.rand(2)

                # scale the random pop by boundry vals/resistances
                diff = np.fabs(min_BV - max_BV)  # fine the difference
                BV_array = min_BV + nom_BV_array * diff  # scale random number (0,1)
                BV_array = np.around(BV_array, decimals=0)  # round
            else:
                BV_array = [BV_val, BV_val]

            # # # # # # # # # # # # # # #
            # get random RS_val bounds
            if RS_val == 'rand':
                max_RS = 10000000  # mOhm
                min_RS = 0.1  # mOhm
                self.RS_val_lim = [min_RS, max_RS]

                # create array of random numbers [0,1]
                nom_RS_array = np.random.rand(2)

                # scale the random pop by boundry vals/resistances
                diff = np.fabs(min_RS - max_RS)  # fine the difference
                RS_array = min_RS + nom_RS_array * diff  # scale random number (0,1)
                RS_array = np.around(RS_array, decimals=1)  # round
            else:
                RS_array = [RS_val, RS_val]


            # # # # # # # # # # # # # # #
            # Brate Diodes
            if direction == 1:
                self.R(1, 'in', 'mid', Res@u_kOhm)

                # Create new custome diode for each
                Diode_Name1 = '%s%d%s' % ('My', custom_diode_num, 'Diode')
                self.model(Diode_Name1, 'D', IS=IS_array[0]@u_uA, RS=RS_array[0]@u_mOhm, BV=BV_array[0]@u_V, IBV=IBV_val@u_uV, N=1)
                self.Diode(1, 'mid', 'out', raw_spice=Diode_Name1)

                Diode_Name2 = '%s%d%s' % ('My', custom_diode_num+1, 'Diode')
                self.model(Diode_Name2, 'D', IS=IS_array[1]@u_uA, RS=RS_array[1]@u_mOhm, BV=BV_array[1]@u_V, IBV=IBV_val@u_uV, N=1)
                self.Diode(2, 'out', 'mid', raw_spice=Diode_Name2)

            elif direction == 0:
                self.R(1, 'mid', 'out', Res@u_kOhm)

                # Create new custome diode for each
                Diode_Name1 = '%s%d%s' % ('My', custom_diode_num, 'Diode')
                self.model(Diode_Name1, 'D', IS=IS_array[0]@u_uA, RS=RS_array[0]@u_mOhm, BV=BV_array[0]@u_V, IBV=IBV_val@u_uV, N=1)
                self.Diode(1, 'mid', 'in', raw_spice=Diode_Name1)

                Diode_Name2 = '%s%d%s' % ('My', custom_diode_num+1, 'Diode')
                self.model(Diode_Name2, 'D', IS=IS_array[1]@u_uA, RS=RS_array[1]@u_mOhm, BV=BV_array[1]@u_V, IBV=IBV_val@u_uV, N=1)
                self.Diode(2, 'in', 'mid', raw_spice=Diode_Name2)



    # def DirRet():
    #    return self.the_dir




# fin
