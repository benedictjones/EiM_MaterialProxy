# Import
import numpy as np
import time
from tqdm import tqdm
import h5py
import random
import matplotlib as mpl

from mod_material.eim_processor import material_processor
from mod_settings.Set_Load import LoadSettings
from mod_load.AddAttributeToData import AddAttribute
from mod_MG.MaterialGraphs import materialgraphs

from mod_load.FetchDataObj import FetchDataObj

from mod_algorithm.PopManager import population


class de(object):

    def __init__(self, prm):
        ''' # # # # # # # # # # # # # # #
        Initialisation of class
        '''

        print("Differential Evolution object initialised...")

        # assign values from the loaded dictionary to self
        self.prm = prm
        self.ParamDict = self.prm['DE']

        # assign load object
        self.lobj = FetchDataObj(self.prm)

        mpl.rcParams.update(mpl.rcParamsDefault)

        return

    #

    #

    #

    #

    #

    #

    '''
    ###############################################################
    DE main
    ###############################################################
    '''
    def perform_de(self, syst, rep):

        # Note the current system
        self.syst = syst
        self.rep = rep

        # refresh load in object
        self.lobj.__init__(self.prm)

        # # produce data saved lists
        self.train_bGenomes = []
        self.train_bResults = []
        self.train_meanFits = []
        self.train_stdFits = []

        self.batch_Ncomps = []
        self.epoch_Ncomps = []

        self.vali_bGenomes = []
        self.vali_bResults = []

        self.Gbest_bGenomes = []
        self.Gbest_bResults = []


        # # Call population object and create a random population
        pobj = population(self.prm)

        # initialise the material processor (a configurable analogue processor)
        self.cap = material_processor(self.prm)

        # # Generate a material
        self.cap.gen_material(syst, rep)

        # # evaluating the fitness of the initial population
        print("Evaluate initial population...")
        Sxdata, Sydata = self.lobj.fetch_data('train', iterate=0, n_comp_scale=self.prm['DE']['popsize'])
        results, GenList_results = self.cap.run_processors(pobj.pop, syst, rep, Sxdata, Sydata,
                                                           ridge_layer_list=0, ret_str='both', the_data='train')
        pop_T_fits = results[0]
        pop_rls = results[4]  # ridge layer list for the pop
        # print("Fitness of initial population:", pop_T_fits)

        # # Determine best geneome and fitness from initial pop
        bT_idx = np.argmin(pop_T_fits)  # find best Training idex
        bT_member = pobj.pop[bT_idx]  # remeber this is list of arrays
        bT_result = GenList_results[bT_idx]
        bT_fit = pop_T_fits[bT_idx]
        print("Best Fitness of initial population:", bT_fit, "(Idx=%d)" % (bT_idx))

        # # Save Gen 0 values
        self.train_bGenomes.append(bT_member)
        self.train_meanFits.append(np.mean(pop_T_fits))
        self.train_stdFits.append(np.std(pop_T_fits))
        self.train_bResults.append(bT_result)
        self.batch_Ncomps.append(self.lobj.n_comp)

        # # Produce best genome MG and save array to file
        if self.prm['mg']['MG_animation'] == 1:
            self.temp_MGani_obj = materialgraphs(syst, rep, self.lobj, self.prm, GenMG_animation=1)
            best_res = [bT_member, pop_T_fits[bT_idx], 10, 0]
            self.temp_MGani_obj.MG(best_res, obj_RidgeLayer=pop_rls[bT_idx], it=0)
            prev_best = best_res
            GV_prev_best = [[np.array([0]),np.array([0])]]
        #

        #

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Start Evolutionary Loop
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        print("\nStart Evolutionary Loop... ")
        pbar = tqdm(total=self.ParamDict['epochs'], leave=True, unit='Epoch')
        current_epoch = self.lobj.epoch  # assign the current epoch
        pbar.update(current_epoch)
        bVali_fit = np.nan  # set incase vailidation node being used, or initial batches not having it
        i = 0
        epoch_start = 1
        while 1:

            tic = time.time()

            # # If no Epochs (i.e no evolution) break
            if self.ParamDict['epochs'] == 0:
                pbar.close()  # close the tqdm bare before exiting
                self.vali_bResults.append([[-1],[-1],[-1],[-1],[-1]])  # assign validation fitness as a place holder
                print("Evaluated Initial Population - Finished", pop_T_fits[bT_idx])
                break

            """ # Generate Trial/Child Population
            Loop over the whole population and apply mutation and
            recombination """

            pobj.gen_trial_pop(bT_idx)

            """ # Evaluate Trial Pop
            Measure how good it is and if a mutant is better than the
            parent vector then we replace it with the new one."""

            # # If we are batching, update the parent/current pop fitnesses with the new batch
            # Don't evaluate parents if the whole training dataset (batch_size = 1 or num_train_instances)
            if self.ParamDict['batch_size'] != 0 and self.ParamDict['batch_size'] != self.lobj.num_train_instances and self.ParamDict['batch_size'] != 1:

                if 'window' not in self.ParamDict['batch_scheme'] or epoch_start == 0:
                    Re_xdata, Re_ydata = self.lobj.fetch_data('train', iterate=0, n_comp_scale=self.prm['DE']['popsize'])
                    parent_results, parent_GenList_results = self.cap.run_processors(pobj.pop, syst, rep, Re_xdata, Re_ydata,
                                                                                      ridge_layer_list=pop_rls, ret_str='both',
                                                                                      the_data='train', force_RL_return=1)
                    #
                    pop_T_fits = parent_results[0]  # update parent pop fit
                    bT_idx = np.argmin(pop_T_fits)  # update best member index!!
                    bT_member = pobj.pop[bT_idx]  # remember this is list of arrays
                    bT_result = parent_GenList_results[bT_idx]
                    bT_fit = pop_T_fits[bT_idx]
                    #print("Fin parent new batch calcs")
            #

            # # find the Trial pop fitness using the SPICE simulation
            xdata, ydata = self.lobj.fetch_data('train', iterate=1, n_comp_scale=self.prm['DE']['popsize'])
            results, GenList_results = self.cap.run_processors(pobj.trial_pop, syst, rep, xdata, ydata,
                                                               ridge_layer_list=0, ret_str='both', the_data='train')
            trial_pop_fits = results[0]  # fitness of pop
            trial_rls = results[4]  # ridge layer list for the pop
            # print("Current batch", self.lobj.b_idx, " , stored:", self.lobj.prev_train_batch[2][0], self.lobj.prev_train_batch[1][0], self.lobj.prev_train_batch[0][0])

            # # Compare the children/trial genomes to the parent/target
            pobj.update_pop(pop_T_fits, trial_pop_fits)

            # # Compare the children/trial genomes to the parent/target
            for j in range(self.ParamDict['popsize']):
                # print("compare trial_pop_fits", j, "trial_pop_fits=", trial_pop_fits[j], "to previous fitness=", fitness[j])
                # print("     For Genome:", str(np.around(trial_denorm_array[j], decimals=3)))

                if trial_pop_fits[j] <= pop_T_fits[j]:  # whether to keep parent or child/trial
                    pop_T_fits[j] = trial_pop_fits[j]   # update fitness array
                    pop_rls[j] = trial_rls[j]  # update ridge layer list for the pop

                if trial_pop_fits[j] <= pop_T_fits[bT_idx]:  # assign best genome/pop
                    bT_idx = j
                    bT_member = pobj.pop[bT_idx]
                    bT_result = GenList_results[bT_idx]
                    bT_fit = pop_T_fits[j]

            #


            # Save this generations best values
            # only assign from self (not returned arrays!!)
            self.train_bGenomes.append(bT_member)
            self.train_meanFits.append(np.mean(pop_T_fits))
            self.train_stdFits.append(np.std(pop_T_fits))
            self.train_bResults.append(bT_result)
            self.batch_Ncomps.append(self.lobj.n_comp)

            # #

            # # Save Best Global member (if not batching)
            if self.ParamDict['batch_size'] == 0:
                self.Gbest_bGenomes.append(bT_member)
                self.Gbest_bResults.append(bT_result)  # this includes the RidgeLayer if using it
                self.vali_bResults.append([[-1],[-1],[-1],[-1],[-1]])
            #

            # # Manage/Update progress bar
            if self.lobj.epoch == current_epoch + 1:

                # Compute Validation Fitness
                if self.ParamDict['batch_size'] != 0:

                    """
                    # # Calc vali fit of best pop member
                    xdata, ydata = self.lobj.fetch_data('validation', iterate=0)
                    #results = self.cap.run_processors(self.calc_denorm(pop), syst, rep, xdata, ydata, ret_str='unzip', the_data='validation', re_run=1)
                    results = self.cap.run_processor(bT_member, syst, rep, xdata, ydata,
                                                     ridge_layer=pop_rls[bT_idx], the_data='validation')
                    bVali_fit = results[0]"""

                    # # Calc vali fit of whole pop, and save best & global best
                    Vxdata, Vydata = self.lobj.fetch_data('validation', iterate=0, n_comp_scale=self.prm['DE']['popsize'])
                    results, vali_GenList_results = self.cap.run_processors(pobj.pop, syst, rep, Vxdata, Vydata,
                                                                              ridge_layer_list=pop_rls, ret_str='both',
                                                                              the_data='validation',
                                                                              force_RL_return=1)  # include the Ridge layer
                    # # update parent pop Global Validation fit
                    if self.ParamDict['Gbest_fit'] == 'raw':
                        pop_V_fits = results[0]
                    elif self.ParamDict['Gbest_fit'] == 'error':
                        pop_V_fits = results[2]

                    bV_idx = np.argmin(pop_V_fits)  # update best validation member index
                    bVali_fit = pop_V_fits[bV_idx]
                    #print("bV idx:", bV_idx, " , bT_idx", bT_idx)  # Not The same!

                    # # Save Validation information
                    self.vali_bGenomes.append(pobj.pop[bV_idx])
                    self.vali_bResults.append(vali_GenList_results[bV_idx])

                    # # Update Global Best
                    if self.lobj.epoch == 1:
                        bG_idx = bV_idx
                        bGV_fit = pop_V_fits[bG_idx]
                        bGV_member = pobj.pop[bG_idx]
                        # # Save Best Global member result infomation
                        self.Gbest_bGenomes.append(pobj.pop[bG_idx])
                        self.Gbest_bResults.append(vali_GenList_results[bG_idx])
                    elif bVali_fit <= bGV_fit:
                        bG_idx = bV_idx
                        bGV_fit = pop_V_fits[bG_idx]
                        bGV_member = pobj.pop[bG_idx]
                        # # Save Best Global member result infomation
                        self.Gbest_bGenomes.append(pobj.pop[bG_idx])
                        self.Gbest_bResults.append(vali_GenList_results[bG_idx])  # this includes the RidgeLayer if using it
                    else:
                        # # Re-save previous best, if no improvment
                        self.Gbest_bGenomes.append(self.Gbest_bGenomes[-1])
                        self.Gbest_bResults.append(self.Gbest_bResults[-1])

                    pbar.set_description("Best GVali fit %.3f" % bGV_fit)

                else:
                    pbar.set_description("Best pTrain fit %.3f" % bT_fit)


                current_epoch += 1
                self.epoch_Ncomps.append(self.lobj.n_comp)
                epoch_start = 0

                # Update progress bar by 1 epoch (but not on the last epoch)
                if self.lobj.epoch <= self.ParamDict['epochs']:
                    pbar.update(1)

            elif self.lobj.epoch > current_epoch + 1:
                raise ValueError("Object Epoch is too large? (out of synce with while loop)")

            else:
                epoch_start += 1

            #

            # # Produce best genome MG and save array to file
            if self.prm['mg']['MG_animation'] == 1:

                best_res = [bT_member, bT_fit, 'na', 'na']

                if sum(np.concatenate(best_res[0])) == sum(np.concatenate(prev_best[0])) and i>=1:
                    ani_file_dir = "%s/data_ani.hdf5" % (self.prm['SaveDir'])
                    with h5py.File(ani_file_dir, 'r') as hdf:
                        prev_dataset = "%d_rep%d/Train/MG_dat_it_%d" % (syst, rep, i)
                        dataset = hdf.get(prev_dataset)
                        prev_data = np.asarray(dataset)
                        prev_best_genome = dataset.attrs['best_genome']
                    with h5py.File(ani_file_dir, 'a') as hdf:
                        new_dataset = 'MG_dat_it_%d' % (i+1)
                        new_dataset = "%d_rep%d/Train/MG_dat_it_%d" % (syst, rep, i+1)
                        Y_dat = hdf.create_dataset(new_dataset, data=prev_data)
                        Y_dat.attrs['best_fit'] = bT_fit  # prev_best_fit
                        Y_dat.attrs['best_genome'] = str(prev_best_genome)
                else:
                    # generate MG
                    self.temp_MGani_obj.MG(best_res, obj_RidgeLayer=pop_rls[bT_idx], it=(i+1))
                    prev_best = best_res

                # # Repeate Ani With Validation Global best member
                ep = self.lobj.epoch
                if self.ParamDict['batch_size'] != 0 and epoch_start == 0 and ep >= 1:
                    GV_best_res = [bGV_member, bGV_fit, 'na', 'na']
                    if sum(np.concatenate(GV_best_res[0])) == sum(np.concatenate(GV_prev_best[0])) and ep > 1:
                        ani_file_dir = "%s/data_ani.hdf5" % (self.prm['SaveDir'])
                        with h5py.File(ani_file_dir, 'r') as hdf:
                            prev_dataset = "%d_rep%d/Vali/MG_dat_it_%d" % (syst, rep, ep-1)
                            dataset = hdf.get(prev_dataset)
                            prev_data = np.asarray(dataset)
                            prev_best_genome = dataset.attrs['best_genome']
                        with h5py.File(ani_file_dir, 'a') as hdf:
                            new_dataset = "%d_rep%d/Vali/MG_dat_it_%d" % (syst, rep, ep)
                            Y_dat = hdf.create_dataset(new_dataset, data=prev_data)
                            Y_dat.attrs['best_fit'] = bGV_fit  # prev_best_fit
                            Y_dat.attrs['best_genome'] = str(prev_best_genome)
                    else:
                        # generate MG
                        Gbest_rl = self.Gbest_bResults[-1][4]
                        self.temp_MGani_obj.MG(GV_best_res, obj_RidgeLayer=Gbest_rl, it=ep, ani_label='Vali')
                        GV_prev_best = GV_best_res

            # # Break loop if zero Fitness Genome is found
            if self.ParamDict['BreakOnZeroFit'] == 1:
                if self.ParamDict['batch_size'] == 0 and pop_T_fits[bT_idx] == 0:
                    pbar.close()  # close the tqdm bare before exiting
                    print(" > Best training fit is:", pop_T_fits[bT_idx])
                    break
                elif self.ParamDict['batch_size'] != 0 and bGV_fit == 0:
                    pbar.close()  # close the tqdm bare before exiting
                    print(" > Best GVali fit is:", bGV_fit)
                    break

            # # Break loop if max number of computations is found
            if self.ParamDict['BreakOnNcomp'] != 'na' and self.ParamDict['BreakOnNcomp'] != 0:
                if self.lobj.n_comp >= self.ParamDict['BreakOnNcomp']:
                    pbar.close()  # close the tqdm bare before exiting
                    print(" --> Hit Max num Comps")
                    break


            # # Break if max number of epoch's is reached
            if self.lobj.epoch >= self.ParamDict['epochs']:
                pbar.close()  # close the tqdm bare before exiting
                break

            # print(" ~ Time to run a gen:", time.time()-tic, '\n')
            i += 1  # iterate

        #

        #

        # # Once the DE is finished
        if self.ParamDict['SaveExtraAttribute'] == 1 or self.ParamDict['PlotExtraAttribute'] == 1:
            Axdata, Aydata = self.lobj.get_data(the_data='all', iterate=0)
            res = self.cap.run_processor(bT_member, syst, rep, Axdata, Aydata, ridge_layer=pop_rls[bT_idx], the_data='all')
            whole_train_set_rY = res[1]
            whole_train_set_fit = res[0]
            AddAttribute(self.prm, 'all', whole_train_set_rY, whole_train_set_fit, syst, rep, Axdata, Aydata)

        #

        # # Clean Up
        if self.prm['mg']['MG_animation'] == 1:
            del self.temp_MGani_obj

        #

        # Restructure (i.e unzip) best generation results list
        # i.e. make it a list of properties each in an array according to itteration/generation
        # not a list of all the best properties from each itteration/generation

        # Format Training
        self.train_bResult_unzipped = []
        res_list = list(zip(*self.train_bResults))
        for property in res_list:
            #train_bResult_unzipped.append(np.asarray(property))
            self.train_bResult_unzipped.append(property)

        # # Format Validation
        self.vali_bResult_unzipped = []
        res_list = list(zip(*self.vali_bResults))
        for property in res_list:
            #train_bResult_unzipped.append(np.asarray(property))
            self.vali_bResult_unzipped.append(property)

        # # Format Global Best
        self.Gbest_bResult_unzipped = []
        res_list = list(zip(*self.Gbest_bResults))
        for property in res_list:
            #train_bResult_unzipped.append(np.asarray(property))
            self.Gbest_bResult_unzipped.append(property)

        #

        if self.lobj.prev_train_batch[1][1].all() != xdata.all():
            raise ValueError("No match")

        """res = self.cap.run_processor(bT_member, syst, rep, xdata, ydata, ridge_layer=pop_rls[bT_idx], the_data='train')
        print(">>>> bTrain Fit", bT_idx, res[0])
        print("bmember", bT_member)
        print(">", res[3][:2])"""

        #

        return

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
