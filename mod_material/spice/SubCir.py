import numpy as np
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


# ############################################################################
# # List of defined subcircuit
# ############################################################################


class SeriesNL(SubCircuit):
    """
    Series current source element with shunt resistors on either side to provide
    paths to ground.
    Uses a passed in value of a & b to produce a non-linear characterisitc:
        if V > 0
            I = aV^2 + bV
        elif V < 0
            I = -aV^2 + bV
    """

    __nodes__ = ('in', 'out')
    def __init__(self, name, a, b):

        SubCircuit.__init__(self, name, *self.__nodes__)

        i_express = "((v(in)-v(out))>0) ? (%s*(v(in)-v(out))**2 + %s*(v(in)-v(out))) : (-%s*(v(in)-v(out))**2 + %s*(v(in)-v(out)) )" % (a,b,a,b)

        self.B(name, 'in', 'out', i=i_express)  # create B source element

        # add routes to gnd
        self.R('shuntG1', 'in', self.gnd, 100@u_GOhm)
        self.R('shuntG2', 'out', self.gnd, 100@u_GOhm)

        return

#

#


class SeriesNSNL(SubCircuit):
    """
    Series current source element with shunt resistors on either side to provide
    paths to ground.
    Uses a passed in value of a & b to produce a non-Symetrical non-linear
    characterisitc:
        if V > 0
            I = a_p*V^2 + b_p*V
        elif V < 0
            I = -a_n*V^2 + b_n*V
    """

    __nodes__ = ('in', 'out')
    def __init__(self, name, a_p, b_p, a_n, b_n):

        SubCircuit.__init__(self, name, *self.__nodes__)

        i_express = "((v(in)-v(out))>0) ? (%s*(v(in)-v(out))**2 + %s*(v(in)-v(out))) : (-%s*(v(in)-v(out))**2 + %s*(v(in)-v(out)) )" % (a_p, b_p, a_n, b_n)

        self.B(name, 'in', 'out', i=i_express)  # create B source element

        # add routes to gnd
        self.R('shuntG1', 'in', self.gnd, 100@u_GOhm)
        self.R('shuntG2', 'out', self.gnd, 100@u_GOhm)

        return

#

#




class SeriesTD(SubCircuit):
    """
    Series current source element with shunt resistors on either side to provide
    paths to ground.
    Uses a passed in value of a & b to produce a non-linear characterisitc:
        if V > 0
            I = aV^2 + bV
        elif V < 0
            I = -aV^2 + bV
    """

    __nodes__ = ('in', 'out')
    def __init__(self, name, direction=1, Res=0.2):

        SubCircuit.__init__(self, name, *self.__nodes__)

        # i_express = "0.01776*v(in,out) -0.10379*v(in,out)**2 +0.22962*v(in,out)**3 -0.22631*v(in,out)**4 + 0.0831*v(in,out)**5"
        # self.B(name, 'in', 'out', i=i_express)  # create B source element
        """#
        i_express = "0.01776*v(1,2) -0.10379*v(1,2)**2 +0.22962*v(1,2)**3 -0.22631*v(1,2)**4 + 0.0831*v(1,2)**5"
        if direction == 1:
            n1, n2 = 'in', 'out'
            #i_express = "0.01776*v(%s,%s) -0.10379*v(%s,%s)**2 +0.22962*v(%s,%s)**3 -0.22631*v(%s,%s)**4 + 0.0831*v(%s,%s)**5" % (n1, n2, n1, n2, n1, n2, n1, n2, n1, n2)
            i_express = "v(%s,%s) > 1 ? -10u*v(%s,%s)+20u : ( v(%s,%s) > -1 ? (10u*v(%s,%s)) : ( v(%s,%s) < -2 ? (-2u*v(%s,%s)+6u) : (-20u*v(%s,%s))-30u) ) )" % (n1, n2, n1, n2, n1, n2, n1, n2, n1, n2, n1, n2, n1, n2)
            self.B(name, n1, n2, i=i_express)
        elif direction == 0:
            n1, n2 = 'out', 'in'
            #i_express = "0.01776*v(%s,%s) -0.10379*v(%s,%s)**2 +0.22962*v(%s,%s)**3 -0.22631*v(%s,%s)**4 + 0.0831*v(%s,%s)**5" % (n1, n2, n1, n2, n1, n2, n1, n2, n1, n2)
            i_express = "v(%s,%s) > 1 ? -10u*v(%s,%s)+20u : ( v(%s,%s) > -1 ? (10u*v(%s,%s)) : ( v(%s,%s) < -2 ? (-2u*v(%s,%s)+6u) : (-20u*v(%s,%s))-30u) ) )" % (n1, n2, n1, n2, n1, n2, n1, n2, n1, n2, n1, n2, n1, n2)
            self.B(name, n1, n2, i=i_express)
        self.C(1, 'in', 'out', 0.3@u_pF)
        #"""


        # # tan def
        C1 = 9e-7
        C2 = 62
        C3 = 2e-5
        C4 = 5e-4
        i = 2
        j = 8
        k = 5
        Vt = 0.16
        Vn = 0.185
        R1 = 50
        R2 = 5500
        D = 0.03e-6
        W = 3e-6
        L = 15e-6
        A = (W-2*D)*(L-2*D)
        P = 2*(W+L)


        #"""
        if direction == 1:
            self.R(1, 'in', 'mid', Res@u_kOhm)
            n1, n2 = 'mid', 'out'

            # # % order poly
            #i_express = "0.01776*v(%s,%s) -0.10379*v(%s,%s)**2 +0.22962*v(%s,%s)**3 -0.22631*v(%s,%s)**4 + 0.0831*v(%s,%s)**5" % (n1, n2, n1, n2, n1, n2, n1, n2, n1, n2)

            # # 3 Order Poly
            #i_express = "0.007*v(%s,%s) -0.0196*v(%s,%s)**2 +0.013*v(%s,%s)**3" % (n1, n2, n1, n2, n1, n2)

            # Complex
            #"""
            V = 'v(%s,%s)' % (n1, n2)
            i1 = "(%s*%s**%s)*(atan(%s*(%s-%s))-atan(%s*(%s-%s)))" % (str(C1), V, str(i), str(C2), V, str(Vt), str(C2), V, str(Vn))
            ie = "%s*%s*%s**%s" % (str(A), str(C3), V, str(j))
            iside = "%s*%s*%s**%s" % (str(P), str(C4), V, str(k))
            i_express = "%s + %s + %s" % (i1, ie, iside)
            #"""


            self.B(name, n1, n2, i=i_express)
            self.C(1, n1, n2, 0.3@u_pF)
            #self.R('shuntG1', 'in', self.gnd, 10@u_GOhm)
            #self.R('shuntG2', 'mid', self.gnd, 10@u_GOhm)
            #self.R('shuntG3', 'out', self.gnd, 10@u_GOhm)


        elif direction == 0:
            n1, n2 = 'mid', 'in'

            # # % order poly
            #i_express = "0.01776*v(%s,%s) -0.10379*v(%s,%s)**2 +0.22962*v(%s,%s)**3 -0.22631*v(%s,%s)**4 + 0.0831*v(%s,%s)**5" % (n1, n2, n1, n2, n1, n2, n1, n2, n1, n2)

            # # 3 Order Poly
            #i_express = "0.007*v(%s,%s) -0.0196*v(%s,%s)**2 +0.013*v(%s,%s)**3" % (n1, n2, n1, n2, n1, n2)

            # Complex
            #"""
            V = 'v(%s,%s)' % (n1, n2)
            i1 = "(%s*%s**%s)*(atan(%s*(%s-%s))-atan(%s*(%s-%s)))" % (str(C1), V, str(i), str(C2), V, str(Vt), str(C2), V, str(Vn))
            ie = "%s*%s*%s**%s" % (str(A), str(C3), V, str(j))
            iside = "%s*%s*%s**%s" % (str(P), str(C4), V, str(k))
            i_express = "%s + %s + %s" % (i1, ie, iside)
            #"""

            self.B(name, n1, n2, i=i_express)
            self.R(1, 'mid', 'out', Res@u_kOhm)
            self.C(1, n1, n2, 0.3@u_pF)
            #self.R('shuntG1', 'in', self.gnd, 10@u_GOhm)
            #self.R('shuntG2', 'mid', self.gnd, 10@u_GOhm)
            #self.R('shuntG3', 'out', self.gnd, 10@u_GOhm)
        #"""

        """
        if direction == 1:
            self.R(1, 'in', 2, Res@u_Ohm)
            n1, n2 = 2, 3
            i_express = "0.01776*v(%s,%s) -0.10379*v(%s,%s)**2 +0.22962*v(%s,%s)**3 -0.22631*v(%s,%s)**4 + 0.0831*v(%s,%s)**5" % (n1, n2, n1, n2, n1, n2, n1, n2, n1, n2)
            self.B(name, n1, n2, i=i_express)
            self.R(2, 3, 'out', 5@u_Ohm)
            self.C(1, n1, n2, 0.3@u_pF)
            self.R('shuntG1', 'in', self.gnd, 10@u_GOhm)
            #self.R('shuntG2', 2, self.gnd, 10@u_GOhm)
            #self.R('shuntG3', 3, self.gnd, 10@u_GOhm)
            self.R('shuntG4', 'out', self.gnd, 10@u_GOhm)

        elif direction == 0:
            self.R(1, 'out', 3, Res@u_Ohm)
            n1, n2 = 3, 2
            i_express = "0.01776*v(%s,%s) -0.10379*v(%s,%s)**2 +0.22962*v(%s,%s)**3 -0.22631*v(%s,%s)**4 + 0.0831*v(%s,%s)**5" % (n1, n2, n1, n2, n1, n2, n1, n2, n1, n2)
            self.B(name, n1, n2, i=i_express)
            self.R(2, 2, 'in', 5@u_Ohm)
            self.C(1, n1, n2, 0.3@u_pF)
            self.R('shuntG1', 'in', self.gnd, 10@u_GOhm)
            #self.R('shuntG2', 2, self.gnd, 10@u_GOhm)
            #self.R('shuntG3', 3, self.gnd, 10@u_GOhm)
            self.R('shuntG4', 'out', self.gnd, 10@u_GOhm)
        #"""




        # add routes to gnd
        #self.R('shuntG1', 'in', self.gnd, 100@u_GOhm)
        #self.R('shuntG2', 'out', self.gnd, 100@u_GOhm)

        return

#

#

#


class SeriesPWL3(SubCircuit):
    """
    Piece Wise Linear Function defined my 3 points

    """

    __nodes__ = (1, 3)
    def __init__(self, name, Rs=0.02):

        SubCircuit.__init__(self, name, *self.__nodes__)

        """ # Not Very stable with series resitance > 200Ohm
        '''[] N. M. Kriplani, S. Bowyer, J. Huckaby, and M. B. Steer,
        ‘Modelling of an Esaki Tunnel Diode in a Circuit Simulator’, Active and Passive Electronic Components,
        Apr. 17, 2011. https://www.hindawi.com/journals/apec/2011/830182/ (accessed Feb. 15, 2021).'''
        Ip, Vp = 0.001, 0.1  # A, V
        Iv, Vv = 0.0001, 0.5  # A, V
        Vt = 0.65  # V
        #"""

        #""" # Not Very stable with series resitance > 2500Ohm
        # # # If the step is too fine, it gets worse
        '''
        [] Y. Yan,“Silicon-based tunnel diode technology,”Ph.D. dissertation (Universityof Notre Dame, 2008).
        >>> page 41
        '''

        '''Ip, Vp = 0.00015, 0.12  # A, V
        Iv, Vv = 0.00007, 0.35  # A, V
        Vt = 0.6  # V'''

        Ip, Vp = 0.00015, 0.12  # A, V
        Iv, Vv = 0.00007, 0.35  # A, V
        Vt = 0.6  # V
        #"""

        Cpp =  ( (Ip/Vp) )
        Cnn = ( (Iv - Ip)/(Vv - Vp) )
        Ctt = ( (Ip - Iv)/(Vt - Vv) )

        Ann = (Ip - Vp*Cnn)
        Att = (Iv - Vv*Ctt)

        An = "%.9f" % (Ip - Vp*Cnn)
        At = "%.9f" % (Iv - Vv*Ctt)

        Cp = "%.9f" % ( (Ip/Vp) )
        Cn = "%.9f" % ( (Iv - Ip)/(Vv - Vp) )
        Ct = "%.9f" % ( (Ip - Iv)/(Vt - Vv) )


        i_express = "v(2,3) > %s ? (%s*v(2,3)+%s) : ( v(2,3) > %s ? (%s*v(2,3)+%s) :  (%s*v(2,3)) ) )" % (str(Vv), Ct, At, str(Vp), Cn, An, Cp)
        self.R(1, 1, 2, Rs@u_Ω)
        self.B(1, 2, 3, i=i_express)  # create B source element

        return
#

#

#


class SeriesPWL4(SubCircuit):
    """
    Piece Wise Linear Function defined my 3 points

    """

    __nodes__ = (1, 3)
    def __init__(self, name, Rs=0.02):

        SubCircuit.__init__(self, name, *self.__nodes__)

        Id, Vd = 10, 0.1
        Ip, Vp = 50, 0.4
        Iv, Vv = 20, 0.8
        Vt = 1.4

        Cd = ( Id/Vd )
        Cp =  ( (Ip - Id)/(Vp - Vd) )
        Cn = ( (Iv - Ip)/(Vv - Vp) )
        Ct = ( (Ip - Iv)/(Vt - Vv) )

        Ap = "%.9f" % (Id - Vd*Cp)
        An = "%.9f" % (Ip - Vp*Cn)
        At = "%.9f" % (Iv - Vv*Ct)

        Cd = "%.9f" % ( Id/Vd )
        Cp = "%.9f" % ( (Ip - Id)/(Vp - Vd) )
        Cn = "%.9f" % ( (Iv - Ip)/(Vv - Vp) )
        Ct = "%.9f" % ( (Ip - Iv)/(Vt - Vv) )


        i_express = "v(2,3) > %s ? (%su*v(2,3)%su) : ( v(2,3) > %s ? (%su*v(2,3)+%su) : ( v(2,3) > %s ? (%su*v(2,3)%su) : (%su*v(2,3)) ) )" % (str(Vv), Ct, At, str(Vp), Cn, An, str(Vd), Cp, Ap, Cd)
        #i_express = "v(2,3) > %s ? (%sm*v(2,3)%sm) : ( v(2,3) > %s ? (%sm*v(2,3)+%sm) : ( v(2,3) > %s ? (%sm*v(2,3)%sm) : (%sm*v(2,3)) ) )" % (str(Vv), Ct, At, str(Vp), Cn, An, str(Vd), Cp, Ap, Cd)


        self.R(1, 1, 2, Rs@u_kΩ)
        self.B(1, 2, 3, i=i_express)  # create B source element

        return

#

#


class SeriesRD(SubCircuit):
    """
    Series Resistor and Diode
    """

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

        return

#

#

#


class SeriesPOD(SubCircuit):
    """
    Two parallel opposing diodes, each with a series resistance
    """

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
        return

# fin
