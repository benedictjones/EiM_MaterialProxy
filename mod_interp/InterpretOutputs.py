# # Top Matter
import numpy as np
import time
import pickle


from mod_interp.interp_EiM import compute_EiM_ouputs
from mod_interp.fitness_scheme import GetFitness


class InterpretVoltages(object):

    def __init__(self, CompiledDict):
        self.CompiledDict = CompiledDict
        self.ptype = self.CompiledDict['ptype']
        return

    #

    def run(self, Vout, genome, data_Y, the_data, cir, rep):
        """
        Runs the appropriate Interpretation scheme on the collected output
        data.

        > Results format <

        For the Standard Loaded Data:
            EiM: [scheme_fitness, responceY, err_fit, Vout]

        For the Custom input data (can't determine a fitness):
            EiM: [class_out, responceY, Vout]

        """

        # print("processor type:", self.ptype, ", interp scheme:", self.CompiledDict['DE']['IntpScheme'], ", fitness scheme:", self.CompiledDict['DE']['FitScheme'])

        if str(data_Y) == 'na':
            # Interpret results from the Passed In data
            if self.ptype == 'EiM':
                results_wHandle = compute_EiM_ouputs(genome, Vout, data_Y, self.CompiledDict, the_data=the_data)
                results = results_wHandle[:-1]
            else:
                raise ValueError("Only supports EiM. Selected ptype: %s" % (self.ptype))

        else:
            # Interpret results from the loaded data, and determine fitness
            if self.ptype == 'EiM':
                results = self._interpter_EiM_(Vout, genome, data_Y, the_data)
            else:
                raise ValueError("Only supports EiM. Selected ptype: %s" % (self.ptype))

        return results

    #

    def _interpter_EiM_(self, Vout, genome, data_Y, the_data):
        """
        Intepret the results from the EiM processor output voltages.
        """

        # Calculate the output reading from voltage inputs
        class_out, responceY, Vout, handle = compute_EiM_ouputs(genome, Vout, data_Y, self.CompiledDict, the_data=the_data)

        # get a fitness score
        scheme_fitness, err_fit, error = GetFitness(data_Y, class_out, responceY, Vout, self.CompiledDict, handle, the_data=the_data)

        return scheme_fitness, responceY, err_fit, Vout

    #






# fin
