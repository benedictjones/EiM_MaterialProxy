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


class population(object):
    """
    Generates and manges a population
    """

    def __init__(self, prm):
        """
        Initialisation of class
        """

        # # assign dictionary to self
        self.prm = prm
        print("Population Generated %s" % (self.prm['param_file']))

        self.ParamDict = self.prm['DE']
        self.GenomeDict = self.prm['genome']

        # # Generate initial pop
        self.gen_pop()

        return

    #

    def gen_pop(self):
        """
        Generate population in a "grouped by gene type" format
        """
        tic = time.time()

        pop = []
        for i in range(self.ParamDict['popsize']):
            member = []
            for gen_group in self.ParamDict['bounds']:
                member.append(np.around(np.random.rand(len(gen_group)), decimals=5))
            pop.append(np.asarray(member, dtype=object))

        pop_out = np.asarray(pop, dtype=object)

        self.pop_raw = pop_out
        self.pop = self._calc_denorm_(pop_out)

        # print("Time to gen pop:", time.time()-tic)
        return

    #

    def gen_trial_pop(self, best_idx='na'):
        """
        Generate a new trial population (both raw and denormalised).
        """
        tic = time.time()
        # # Create all trial_mutants in a list
        trial_denorm_list = []
        trial_list = []
        for j in range(self.ParamDict['popsize']):

            # # Select Indexs to generate mutants from
            """
            Creates an range array(0, popsize) but excludes the current
            value of j, used to randomly select pop involved in mutation.
            i.e idxs is all pop index's except the current one
            """
            idxs = [idx for idx in range(self.ParamDict['popsize']) if idx != j]

            # # Mutation
            mutant = self._mutate_(idxs, best_idx)

            # # Recombination & Replacement
            trial_denorm, trial = self._bin_cross_(j, self.pop_raw, mutant)
            # trial_denorm, trial = self._bin_cross_WCST_(j, pop, mutant, i)

            trial_denorm_list.append(trial_denorm)
            trial_list.append(trial)

        self.trial_pop_raw = np.asarray(trial_list, dtype=object)
        self.trial_pop = np.asarray(trial_denorm_list, dtype=object)
        # print("Time to gen trial pop:", time.time()-tic)
        # exit()
        return

    #

    def update_pop(self, parent_fit, trial_fit):
        """
        Using the passed in fitnesses, the parent pop is updated, replacing
        members with those from the trial pop is a higher fitness was achived.
        """

        # # Compare the children/trial genomes to the parent/target
        for j in range(self.ParamDict['popsize']):
            # print("compare fi", j, "fi=", fi[j], "to previous fitness=", fitness[j])
            # print("     For Genome:", str(np.around(trial_denorm_array[j], decimals=3)))

            if trial_fit[j] <= parent_fit[j]:  # whether to keep parent or child/trial
                self.pop_raw[j] = self.trial_pop_raw[j]  # update corresponding pop

        # # Update denorm pop using the newly updated pop
        self.pop = self._calc_denorm_(self.pop_raw)

        return

    #

    def update_pop_member(self, member, parent_fit, trial_fit):
        """
        Using the passed in fitnesses, the parent pop is updated, replacing
        members with those from the trial pop is a higher fitness was achived.
        """
        #print("self.pop_raw:\n", self.pop_raw[member])
        #print("\npop before:\n", self.pop)
        if trial_fit <= parent_fit:  # whether to keep parent or child/trial
            self.pop_raw[member] = self.trial_pop_raw[member]  # update corresponding pop

        # # Update denorm pop using the newly updated pop
        pop_member = self._calc_denorm_([self.pop_raw[member]])
        #print("\npop_member:\n", pop_member[0])
        self.pop[member] = pop_member[0]

        #print("\npop:\n", self.pop)
        #exit()
        return
    #

    #

    #

    '''
    ###############################################################
    Mutation Functions
    ###############################################################
    '''

    def _mutate_(self, idxs, best_idx):
        """
        Selects which mutation scheme to use, and returns the mutant.
        """

        if self.ParamDict['mut_scheme'] == 'rand1':
            mutant = self._rand1_(idxs, self.pop_raw)
        elif self.ParamDict['mut_scheme'] == 'best1':
            mutant = self._best1_(idxs, self.pop_raw, best_idx)
        elif self.ParamDict['mut_scheme'] == 'rand2':
            mutant = self._rand2_(idxs, self.pop_raw)
        elif self.ParamDict['mut_scheme'] == 'best2':
            mutant = self._best2_(idxs, self.pop_raw, best_idx)
        elif self.ParamDict['mut_scheme'] == 'rtb1':
            mutant = self._rtb1_(idxs, self.pop_raw, best_idx)
        else:
            raise ValueError("Invalit Mutation Scheme: %s" % (self.ParamDict['mut_scheme']))

        return mutant

    def _rand1_(self, idxs, pop):
        """
        Random1 mutation method.
        Randomly choose 3 indexes without replacement.
        """
        selected = np.random.choice(idxs, 3, replace=False)
        np_pop = np.asarray(pop, dtype=object)
        a, b, c = np_pop[selected]  # assign to a variable
        # note this is not the real pop values

        # mutant
        mutant = np.concatenate(a) + self.ParamDict['mut'] * (np.concatenate(b) - np.concatenate(c))

        # now the mutant may no longer be within the normalized boundary
        mutant = np.clip(mutant, 0, 1)

        return mutant  # this is unformatted

    #

    def _rand2_(self, idxs, pop):
        """
        Random2 mutation method.
        Randomly choose 5 indexes without replacement
        """

        selected = np.random.choice(idxs, 5, replace=False)
        np_pop = np.asarray(pop, dtype=object)
        a, b, c, d, e = np_pop[selected]  # assign to a variable
        # note; a, b etc are genomes

        # mutant
        mutant = np.concatenate(a) + self.ParamDict['mut'] * (np.concatenate(b) - np.concatenate(c) + np.concatenate(d) - np.concatenate(e))

        # now the mutant may no longer be within the normalized boundary
        mutant = np.clip(mutant, 0, 1)  # must be a single array to clip

        return mutant  # this is unformatted

    #

    def _best1_(self, idxs, pop, best_idx):
        """
        Best1 mutation method
        Randomly choose 2 indexes without replacement, combined with the best.
        """

        selected = np.random.choice(idxs, 2, replace=False)
        np_pop = np.asarray(pop, dtype=object)
        b, c = np_pop[selected]
        a = pop[best_idx]

        # mutant
        mutant = np.concatenate(a) + self.ParamDict['mut'] * (np.concatenate(b) - np.concatenate(c))

        # now the mutant may no longer be within the normalized boundary
        mutant = np.clip(mutant, 0, 1)

        return mutant  # this is unformatted

    #

    def _best2_(self, idxs, pop, best_idx):
        """
        Best2 mutation method
        Randomly choose 4 indexes without replacement, combined with the best.
        """

        selected = np.random.choice(idxs, 4, replace=False)
        np_pop = np.asarray(pop, dtype=object)
        b, c, d, e = np_pop[selected]
        a = pop[best_idx]

        # mutant
        mutant = np.concatenate(a) + self.ParamDict['mut'] * (np.concatenate(b) - np.concatenate(c) + np.concatenate(d) - np.concatenate(e))

        # now the mutant may no longer be within the normalized boundary
        mutant = np.clip(mutant, 0, 1)

        return mutant  # this is unformatted

    #

    def _rtb1_(self, idxs, pop, best_idx):
        """
        Rand-to-best/1 mutation method
        Randomly choose 4 indexes without replacement, combined with the best.
        """

        selected = np.random.choice(idxs, 4, replace=False)
        np_pop = np.asarray(pop, dtype=object)
        a, c, d, e = np_pop[selected]
        b = pop[best_idx]

        F1 = self.ParamDict['mut']
        F2 = self.ParamDict['mut']

        # mutant
        mutant = np.concatenate(a) + F1 * (np.concatenate(b) - np.concatenate(c)) + F2 * (np.concatenate(d) - np.concatenate(e))

        # now the mutant may no longer be within the normalized boundary
        mutant = np.clip(mutant, 0, 1)

        return mutant  # this is unformatted

    #

    #

    #

    #

    '''
    ###############################################################
    Recombination & Replacement
    ###############################################################
    '''

    def _bin_cross_(self, j, pop, mutant):
        """
        Basic binary crossover
        """

        # returns true or false for each of the random elements
        cross_points = np.random.rand(self.prm['DE']['dimensions']) < self.ParamDict['crossp']

        # Randomly set a paramater to True to ensure a mutation occurs
        a = np.arange(self.prm['DE']['dimensions'])
        x = int(random.choice(a))
        cross_points[x] = True

        # Where True, yield x, otherwise yield y np.where(condition,x,y)
        trial = np.where(cross_points, mutant, np.concatenate(pop[j]))

        # put into the grouped by gene type format used
        trial = self._format_mutant_(trial)

        trial_denorm = self._calc_denorm_([trial])
        #trial_denorm = self.min_b + trial * self.diff  # terms of the real val
        # will this be a problem with binary number/pin posistions

        return trial_denorm[0], trial

    #

    def _bin_cross_WCST_(self, j, pop, mutant, DE_iter):
        """
        Binary crossover but with controlled shuffle crossp threshold
        """

        # returns true or false for each of the random elements
        cross_points = np.random.rand(self.prm['DE']['dimensions']) < self.ParamDict['crossp']  # true means choose mutant

        # Randomly set a paramater to True to ensure a mutation occurs
        a = np.arange(self.prm['DE']['dimensions'])
        if self.GenomeDict['perm_crossp_model'] != 'none' and self.GenomeDict['shuffle_gene'] == 1:
            while 1:
                # not allowed to set cross point for the shuffle Mutation
                # this stops it over riding the perm_crossp_model selected
                x = int(random.choice(a))
                if x != self.GenomeDict['SGI']:
                    break
        else:
            x = int(random.choice(a))
        cross_points[x] = True

        # Vary crossp for perm gene (only) depending if a model & shuffle is
        # selected, and determin variable cross_point for permutation gene
        # only and replace it into the array
        perm_crossp = self.ParamDict['crossp']
        if self.GenomeDict['perm_crossp_model'] == 'linear' and self.GenomeDict['shuffle_gene'] == 1:
            perm_crossp = 1 - DE_iter*(1/self.ParamDict['epochs'])
            cross_points[self.GenomeDict['SGI']] = np.random.rand() < perm_crossp
        elif self.GenomeDict['perm_crossp_model'] == 'box' and self.GenomeDict['shuffle_gene'] == 1:
            lin_val = 1 - DE_iter*(1/self.ParamDict['epochs'])
            if lin_val > 0.5:
                perm_crossp = 1
            else:
                perm_crossp = 0
            cross_points[self.GenomeDict['SGI']] = np.random.rand() < perm_crossp
        elif self.GenomeDict['perm_crossp_model'] == 'quad' and self.GenomeDict['shuffle_gene'] == 1:
            perm_crossp = 1 - (DE_iter*(1/self.ParamDict['epochs']))**2
            cross_points[self.GenomeDict['SGI']] = np.random.rand() < perm_crossp
        elif self.GenomeDict['perm_crossp_model'] == 'unity' and self.GenomeDict['shuffle_gene'] == 1:
            perm_crossp = 1
            cross_points[self.GenomeDict['SGI']] = np.random.rand() < perm_crossp

        # Where True, yield x, otherwise yield y np.where(condition,x,y)
        trial = np.where(cross_points, mutant, np.concatenate(pop[j]))

        # put into the grouped by gene type format used
        trial = self._format_mutant_(trial)

        trial_denorm = self._calc_denorm_([trial])
        #trial_denorm = self.min_b + trial * self.diff  # terms of the real val
        # will this be a problem with binary number/pin posistions

        # print("trial_denorm:", trial_denorm, "Perm index is:", self.ParamDict['SGI'], "perm CR is:", cross_points[self.ParamDict['SGI']])

        return trial_denorm[0], trial

    #

    #

    #

    #

    #

    #

    #

    '''
    ###############################################################
    Population and mutant generation function
    ###############################################################
    '''

    def _format_mutant_(self, mutant):
        """
        Re-format mutant into "grouped by gene type" format
        """
        member = []
        i = 0
        for bound_group in self.ParamDict['bounds']:
            gen_group = []
            for bound in bound_group:
                gen_group.append(mutant[i])
                i = i + 1
            member.append(np.asarray(gen_group, dtype=object))

        return np.asarray(member, dtype=object)

    #

    def _calc_denorm_(self, pop_in):
        """
        Produce the de-normalised population using the bounds.
        Also groups the population into its gene groups, which are indexed
        using the 'loc' variable.

        """

        # pop_denorm = np.around(self.min_b + pop_in * self.diff, decimals=3)  # pop with their real values

        # # Group and scale the Population
        pop_denorm = []
        for i in range(len(pop_in)):
            member = pop_in[i]
            member_denorm = []

            for j in range(len(member)):
                genome_group = member[j]
                group_bounds = self.ParamDict['bounds'][j]

                gen_group_denorm = []
                for k in range(len(genome_group)):
                    lower_b, upper_b = group_bounds[k]
                    val = lower_b + genome_group[k]*(abs(upper_b-lower_b))
                    gen_group_denorm.append(np.around(val, decimals=3))
                j = j + 1
                member_denorm.append(np.asarray(gen_group_denorm))
            pop_denorm.append(np.asarray(member_denorm, dtype=object))

        pop_denorm = np.asarray(pop_denorm, dtype=object)

        # # truncate shuffle gene vale to an integer
        if self.GenomeDict['Shuffle']['active'] == 1:
            for i in range(len(pop_denorm)):
                pop_denorm[i][self.GenomeDict['Shuffle']['loc']][0] = int(pop_denorm[i][self.GenomeDict['Shuffle']['loc']][0])

        # # Apply Input Weight scheme
        pop_denorm = self._apply_InWeight_scheme_(pop_denorm)

        # # Apply output weight scheme
        pop_denorm = self._apply_OutWeight_scheme_(pop_denorm)

        # # Apply banding scheme
        pop_denorm = self._apply_Banding_scheme_(pop_denorm)

        return np.asarray(pop_denorm, dtype=object)

    #

    #

    # # Apply output weight scheme
    def _apply_InWeight_scheme_(self, pop):

        if self.GenomeDict['InWeight']['active'] == 0:
            return pop
        elif self.GenomeDict['InWeight']['active'] == 1:
            if self.GenomeDict['InWeight']['scheme'] == 'random':
                return pop
            elif self.GenomeDict['InWeight']['scheme'] == 'AddSub':
                #if trial == 0:
                # # truncate shuffle gene vale to an integer
                for i in range(len(pop)):
                    for j in range(len(pop[i][self.GenomeDict['InWeight']['loc']])):
                        if pop[i][self.GenomeDict['InWeight']['loc']][j] >= 0:
                            pop[i][self.GenomeDict['InWeight']['loc']][j] = 1
                        else:
                            pop[i][self.GenomeDict['InWeight']['loc']][j] = -1
            else:
                print("Error: Invalid Output Weight Scheme")
                raise ValueError('(FunClass): Invalid Output Weight Scheme.')
        return pop

    #

    # # Apply output weight scheme
    def _apply_OutWeight_scheme_(self, pop):

        if self.GenomeDict['OutWeight']['active'] == 0:
            return pop
        elif self.GenomeDict['OutWeight']['active'] == 1:
            if self.GenomeDict['OutWeight']['scheme'] == 'random':
                return pop
            elif self.GenomeDict['OutWeight']['scheme'] == 'AddSub':
                # # truncate shuffle gene vale to an integer
                for i in range(len(pop)):
                    for j in range(len(pop[i][self.GenomeDict['OutWeight']['loc']])):
                        if pop[i][self.GenomeDict['OutWeight']['loc']][j] >= 0:
                            pop[i][self.GenomeDict['OutWeight']['loc']][j] = 1
                        else:
                            pop[i][self.GenomeDict['OutWeight']['loc']][j] = -1
            else:
                print("Error: Invalid Output Weight Scheme")
                raise ValueError('(FunClass): Invalid Output Weight Scheme.')
        return pop

    #

    # # Apply banding scheme
    def _apply_Banding_scheme_(self, pop):

        if self.GenomeDict['BandClass']['active'] == 0:
            pass
        elif self.GenomeDict['BandClass']['active'] == 1:
            for i in range(len(pop)):
                for j in range(len(pop[i][self.GenomeDict['BandClass']['loc']])):
                    pop[i][self.GenomeDict['BandClass']['loc']][j] = int(pop[i][self.GenomeDict['BandClass']['loc']][j])
        else:
            raise ValueError('(FunClass): Invalid BandClass_gene Scheme.')


        if self.GenomeDict['BandNumber']['active'] == 0:
            pass
        elif self.GenomeDict['BandNumber']['active'] == 1:
            for i in range(len(pop)):
                for j in range(len(pop[i][self.GenomeDict['BandNumber']['loc']])):
                    pop[i][self.GenomeDict['BandNumber']['loc']][j] = int(pop[i][self.GenomeDict['BandNumber']['loc']][j])
        else:
            raise ValueError('(FunClass): Invalid BandNumber_gene Scheme.')

        return pop

#

#

# fin
