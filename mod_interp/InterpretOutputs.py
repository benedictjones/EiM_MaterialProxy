# # Top Matter
import numpy as np
import time

from mod_interp.interp_EiM import compute_EiM_ouputs
from mod_interp.fitness_scheme import GetFitness


class InterpretVoltages(object):

    def __init__(self, prm):
        self.prm = prm
        self.ptype = self.prm['ptype']
        self.op_layer = self.prm['op_layer']
        return

    #

    def run(self, Vout, genome, data_Y, syst, rep, ridge_layer, the_data,
            force_RL_ret, force_predY_ret):
        """
        Runs the appropriate Interpretation scheme on the collected output
        data.

        >> Results format <<

        --> For Passed in Data and a Target output:
                When ridge_layer = 0 (OR when force_RL_ret = 1):
                    EiM: [scheme_fitness, responceY, err_fit, Vout, -1]
                Otherwise:
                    EiM: [scheme_fitness, responceY, err_fit, Vout]

        --> For input data with no Target data (i.e. data_Y='na' and IntpScheme != raw).
            We can't determine a supervised fitness, but can compute a reponce.
            (Note: if IntpScheme == raw, then it is assumed a unsupervised
            fit scheme is being used.)
                EiM: [class_out, responceY, Vout]


        Note:
        ridge_layer!=0 means that a output layer [weight, bias] is passed in


        """

        # print("processor type:", self.ptype, ", interp scheme:", self.prm['DE']['IntpScheme'], ", fitness scheme:", self.prm['DE']['FitScheme'])

        if (str(data_Y) == 'na' and self.prm['DE']['IntpScheme'] != 'raw') or force_predY_ret == 1:
            # # Interpret results from the Passed In data,
            # but not a Fitness since no target output is provided
            # enless it is intentionally computing raw aoutputs in
            # combination with an unsupervised fitness scheme

            if force_RL_ret == 1:
                raise ValueError("Won't Return Ridge layer, layer must be passed in!")

            if self.op_layer == 'evolved':
                results_wHandle = compute_EiM_ouputs(genome, Vout, data_Y, self.prm)
                # class_out, responceY, Vout, handle = results_wHandle
                results = results_wHandle[:-1]

            elif self.op_layer == 'wa':
                raise ValueError("No Weight Agnostic Option yet ")

        else:
            # # Interpret results from the loaded data, and determine fitness
            # using the target output

            if self.op_layer == 'evolved':

                results = self._interpter_EiM_(Vout, genome, data_Y, ridge_layer, force_RL_ret)

            elif self.op_layer == 'wa':
                raise ValueError("No Weight Agnostic Option yet ")

        return results

    #

    def _interpter_EiM_(self, Vout, genome, data_Y, ridge_layer, force_RL_ret):
        """
        Intepret the results from the EiM processor output voltages.
        """
        tic = time.time()
        # # Calculate the output reading from voltage inputs
        class_out, responceY, Vout, handle = compute_EiM_ouputs(genome, Vout, data_Y, self.prm)

        # # Determin the scheme threshold
        if self.prm['DE']['IntpScheme'] == 'thresh_binary':
            threshold = self.prm['DE']['threshold']
        elif self.prm['DE']['IntpScheme'] == 'pn_binary':
            threshold = 0
        else:
            threshold = 0

        # # get a fitness score
        scheme_fitness, err_fit, error = GetFitness(data_Y, class_out, responceY, Vout, self.prm, threshold, handle)
        # print(" ~ Time to compute _interpter_EiM_:", time.time()-tic)

        # # The returned 0 is used to denote no ridges output layer is used
        if (isinstance(ridge_layer, list) is False and ridge_layer == 0) or force_RL_ret == 1:
            return scheme_fitness, responceY, err_fit, Vout, -1
        elif isinstance(ridge_layer, list) is False and ridge_layer == -1:
            return scheme_fitness, responceY, err_fit, Vout
        else:
            raise ValueError("Cannot use a passed in Ridge Layer for basic EiM")

#

# fin
