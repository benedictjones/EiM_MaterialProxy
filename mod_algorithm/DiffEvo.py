# Import
import numpy as np
import time
from tqdm import tqdm
import h5py
import random

from mod_material.eim_processor import material_processor
from mod_settings.Set_Load import LoadSettings
from mod_load.AddAttributeToData import AddAttribute
from mod_MG.MaterialGraphs import materialgraphs


class de(object):

    def __init__(self, param_file=''):
        ''' # # # # # # # # # # # # # # #
        Initialisation of class
        '''

        print("Differential Evolution object initialised...")

        # assign values from the loaded dictionary to self
        self.param_file = param_file
        self.prm = LoadSettings(param_file)
        self.ParamDict = self.prm['DE']
        self.GenomeDict = self.prm['genome']

        # params
        self.GenMG_animation = self.prm['mg']['MG_animation']
        self.dimensions = self.prm['DE']['dimensions']

        # extracting boundaries and initialisation
        #self.min_b, self.max_b = np.asarray(self.ParamDict['bounds']).T  # arrays of the mim/max of each boundary
        #self.diff = np.fabs(self.min_b - self.max_b)  # absolute val of each el
        self.best_idx = []
        self.best = []
        self.best_responceY = []

        # for MG saving itteratively (for animation)
        self.prev_best = [0]  # set initial: no prev value

        """# create an MG plotter if makign animation
        if self.GenMG_animation == 1:
            self.temp_MGani_obj = materialgraphs(self.GenMG_animation)"""

    #

    #

    #

    #

    #

    ''' # # # # # # # # # # # # # # #
    Mutation Functions
    '''

    # # # # # # # # # # # # # # # #
    # Random1 mutation method
    def rand1(self, idxs, pop):
        # randomly choose 5 indexes without replacement
        selected = np.random.choice(idxs, 3, replace=False)
        np_pop = np.asarray(pop)
        a, b, c = np_pop[selected]  # assign to a variable
        # note this is not the real pop values

        # mutant
        mutant = np.concatenate(a) + self.ParamDict['mut'] * (np.concatenate(b) - np.concatenate(c))

        # now the mutant may no longer be within the normalized boundary
        mutant = np.clip(mutant, 0, 1)

        return mutant  # this is unformatted

    # # # # # # # # # # # # # # # #
    # Random2 mutation method
    def rand2(self, idxs, pop):
        # randomly choose 5 indexes without replacement
        selected = np.random.choice(idxs, 5, replace=False)
        np_pop = np.asarray(pop)
        a, b, c, d, e = np_pop[selected]  # assign to a variable
        # note; a, b etc are genomes

        # mutant
        mutant = np.concatenate(a) + self.ParamDict['mut'] * (np.concatenate(b) - np.concatenate(c) + np.concatenate(d) - np.concatenate(e))

        # now the mutant may no longer be within the normalized boundary
        mutant = np.clip(mutant, 0, 1)

        return mutant  # this is unformatted

    # # # # # # # # # # # # # # # #
    # Best1 mutation method
    def best1(self, idxs, pop):
        # randomly choose 5 indexes without replacement
        selected = np.random.choice(idxs, 2, replace=False)
        np_pop = np.asarray(pop)
        b, c = np_pop[selected]
        a = pop[self.best_idx]

        # mutant
        mutant = np.concatenate(a) + self.ParamDict['mut'] * (np.concatenate(b) - np.concatenate(c))

        # now the mutant may no longer be within the normalized boundary
        mutant = np.clip(mutant, 0, 1)

        return mutant  # this is unformatted

    ''' # # # # # # # # # # # # # # #
    Recombination & Replacement
    '''

    # # # # # # # # # # # # # # # #
    # Basic binary crossover
    def bin_cross(self, j, pop, mutant):

        # returns true or false for each of the random elements
        cross_points = np.random.rand(self.dimensions) < self.ParamDict['crossp']

        # Randomly set a paramater to True to ensure a mutation occurs
        a = np.arange(self.dimensions)
        x = int(random.choice(a))
        cross_points[x] = True

        # Where True, yield x, otherwise yield y np.where(condition,x,y)
        trial = np.where(cross_points, mutant, np.concatenate(pop[j]))

        # put into the grouped by gene type format used
        trial = self.format_mutant(trial)

        trial_denorm = self.calc_denorm([trial])
        #trial_denorm = self.min_b + trial * self.diff  # terms of the real val
        # will this be a problem with binary number/pin posistions

        return trial_denorm[0], trial

    # # # # # # # # # # # # # # # #
    # Binary crossover but with controlled shuffle crossp threshold
    def bin_cross_WCST(self, j, pop, mutant, DE_iter):

        # returns true or false for each of the random elements
        cross_points = np.random.rand(self.dimensions) < self.ParamDict['crossp']  # true means choose mutant

        # Randomly set a paramater to True to ensure a mutation occurs
        a = np.arange(self.dimensions)
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
            perm_crossp = 1 - DE_iter*(1/self.ParamDict['its'])
            cross_points[self.GenomeDict['SGI']] = np.random.rand() < perm_crossp
        elif self.GenomeDict['perm_crossp_model'] == 'box' and self.GenomeDict['shuffle_gene'] == 1:
            lin_val = 1 - DE_iter*(1/self.ParamDict['its'])
            if lin_val > 0.5:
                perm_crossp = 1
            else:
                perm_crossp = 0
            cross_points[self.GenomeDict['SGI']] = np.random.rand() < perm_crossp
        elif self.GenomeDict['perm_crossp_model'] == 'quad' and self.GenomeDict['shuffle_gene'] == 1:
            perm_crossp = 1 - (DE_iter*(1/self.ParamDict['its']))**2
            cross_points[self.GenomeDict['SGI']] = np.random.rand() < perm_crossp
        elif self.GenomeDict['perm_crossp_model'] == 'unity' and self.GenomeDict['shuffle_gene'] == 1:
            perm_crossp = 1
            cross_points[self.GenomeDict['SGI']] = np.random.rand() < perm_crossp

        # Where True, yield x, otherwise yield y np.where(condition,x,y)
        trial = np.where(cross_points, mutant, np.concatenate(pop[j]))

        # put into the grouped by gene type format used
        trial = self.format_mutant(trial)

        trial_denorm = self.calc_denorm([trial])
        #trial_denorm = self.min_b + trial * self.diff  # terms of the real val
        # will this be a problem with binary number/pin posistions

        # print("trial_denorm:", trial_denorm, "Perm index is:", self.ParamDict['SGI'], "perm CR is:", cross_points[self.ParamDict['SGI']])

        return trial_denorm[0], trial

    #

    #

    '''
    ###############################################################
    DE main
    ###############################################################
    '''
    def perform_de(self, cir, rep):

        # Load in the data again, this is mainly to update the new SaveDir
        self.prm = LoadSettings(self.param_file)
        self.ParamDict = self.prm['DE']
        self.GenomeDict = self.prm['genome']


        # # produce return lists
        best_gen_list = []
        best_results_list = []
        mean_fit_list = []
        std_fit_list = []

        # # create a random population
        pop = self.gen_pop()

        # # Scale the random pop by boundry vals
        pop_denorm = self.calc_denorm(pop)

        # initialise the material processor (a configurable analogue processor)
        self.cap = material_processor(self.prm)

        # # Generate a material
        self.cap.gen_material(cir, rep)

        # # evaluating the fitness of the initial population
        print("Evaluate initial population...")
        results, GenList_results = self.cap.run_processor(pop_denorm, cir, rep, ret_str='both')
        pop_fitnesses = results[0]
        responceY_list = results[1]
        print("Fitness of initial population:", pop_fitnesses)

        # # Determine best geneome and fitness from initial pop
        self.best_idx = np.argmin(pop_fitnesses)  # gives index of the smallest val
        self.best = pop_denorm[self.best_idx]  # remeber this is list of arrays
        self.best_result = GenList_results[self.best_idx]
        self.best_fit = pop_fitnesses[self.best_idx]
        self.best_responceY = responceY_list[self.best_idx]

        # # Save Gen 0 values
        best_gen_list.append(self.best)
        mean_fit_list.append(np.mean(pop_fitnesses))
        std_fit_list.append(np.std(pop_fitnesses))
        best_results_list.append(self.best_result)


        # # Produce best genome MG and save array to file
        if self.GenMG_animation == 1:
            self.temp_MGani_obj = materialgraphs(cir, rep, self.GenMG_animation)
            best_res = [self.best, pop_fitnesses[self.best_idx]]
            self.temp_MGani_obj.MG(0, best_res, it=0)
            self.prev_best = best_res

        # # # # # # # # # # # # # # # #
        # Start Evolutionary Loop
        print("\nStart Evolutionary Loop... ")
        pbar = tqdm(range(self.ParamDict['its']), leave=True, unit='Gen')
        for i in pbar:  # loop for the number of itterations

            ''' Loop over the whole population - apply mutation and
            recombination before assessing whether the parent/target should
            be replaced with the child/trial '''

            # # Create all trial_mutants in a list
            trial_denorm_list = []
            trial_list = []
            for j in range(self.ParamDict['popsize']):

                # creates an range array(0, popsize) but excludes the current
                # value of j, used to randomly select pop involved in mutation
                idxs = [idx for idx in range(self.ParamDict['popsize']) if idx != j]
                # i.e indx is all pop index's except the current one

                # # Mutation
                mutant = self.rand2(idxs, pop)

                # # Recombination & Replacement
                trial_denorm, trial = self.bin_cross(j, pop, mutant)
                #trial_denorm, trial = self.bin_cross_WCST(j, pop, mutant, i)

                trial_denorm_list.append(trial_denorm)
                trial_list.append(trial)
            trial_array = np.asarray(trial_list)
            trial_denorm_array = np.asarray(trial_denorm_list)
            #print(trial_list)
            #print(trial_denorm_list)
            #exit()

            ''' After generating our new child/trial vectors, we evaluate it to
            measure how good it is. If this mutant is better than the
            current vector (pop[j]) then we replace it with the new one.'''

            # # find the population fitness using the SPICE simulation
            results, GenList_results = self.cap.run_processor(trial_denorm_array, cir, rep, ret_str='both')
            fi = results[0]
            responceY_list = results[1]

            # # Compare the children/trial genomes to the parent/target
            for j in range(self.ParamDict['popsize']):
                # print("compare fi", j, "fi=", fi[j], "to previous fitness=", fitness[j])
                # print("     For Genome:", str(np.around(trial_denorm_array[j], decimals=3)))

                if fi[j] <= pop_fitnesses[j]:  # whether to keep parent or child/trial
                    pop_fitnesses[j] = fi[j]   # update fitness array
                    pop[j] = trial_array[j]  # update corresponding pop

                if fi[j] <= pop_fitnesses[self.best_idx]:  # assign best genome/pop
                    self.best_idx = j
                    self.best = trial_denorm_list[j]
                    self.best_fit = pop_fitnesses[j]
                    self.best_result = GenList_results[self.best_idx]
                    self.best_responceY = responceY_list[self.best_idx]

            # # Print/Update loop infomation
            pbar.set_description("Best fitness %.5f" % self.best_fit)

            # print("\nbest:", self.best)
            # print("best_idx:", self.best_idx)
            # print("\nFitness:\n", fitness)
            # print("pop:\n", self.calc_denorm(pop))

            # Save this generations best values
            # only assign from self (not returned arrays!!)
            best_gen_list.append(self.best)
            mean_fit_list.append(np.mean(pop_fitnesses))
            std_fit_list.append(np.std(pop_fitnesses))
            best_results_list.append(self.best_result)


            # # Produce best genome MG and save array to file
            if self.GenMG_animation == 1:

                best_res = [self.best, self.best_fit]
                #print(">>>best_res", best_res)
                """print("self.prev_best[0]", self.prev_best[0])
                print("sum(best_res[0])", sum(best_res[0]))
                print("sum(self.prev_best[0])", sum(self.prev_best[0]))
                """
                if sum(np.concatenate(best_res[0])) == sum(np.concatenate(self.prev_best[0])):
                    # load prev and Copy
                    ani_file_dir = "%s/MG_AniData/%d_Rep%d.hdf5" % (self.prm['SaveDir'], cir, rep)

                    with h5py.File(ani_file_dir, 'r') as hdf:
                        prev_dataset = 'MG_dat_it_%d' % (i)
                        dataset = hdf.get(prev_dataset)
                        prev_data = np.asarray(dataset)

                        prev_best_fit = dataset.attrs['best_fit']
                        prev_best_genome = dataset.attrs['best_genome']

                    with h5py.File(ani_file_dir, 'a') as hdf:
                        new_dataset = 'MG_dat_it_%d' % (i+1)
                        Y_dat = hdf.create_dataset(new_dataset, data=prev_data)
                        Y_dat.attrs['best_fit'] = prev_best_fit
                        Y_dat.attrs['best_genome'] = str(prev_best_genome)

                else:
                    # generate MG
                    self.temp_MGani_obj.MG(0, best_res, it=i+1)

                    self.prev_best = best_res

            # # Break loop if ideal Genome is found
            if self.ParamDict['BreakOnZeroFit'] == 1:
                if pop_fitnesses[self.best_idx] == 0:
                    pbar.close()  # close the tqdm bare before exiting
                    print("Best fit is:", pop_fitnesses[self.best_idx])
                    break

        # # Once the DE is finished
        if self.ParamDict['Add_Attribute_To_Data'] == 1:
            AddAttribute('train', self.best_responceY, pop_fitnesses[self.best_idx])

        if self.GenMG_animation == 1:
            del self.temp_MGani_obj

        # Restructure (i.e unzip) best generation results list
        # i.e. make it a list of properties each in an array according to itteration/generation
        # not a list of all the best properties from each itteration/generation
        unzipped_results = []
        res_list = list(zip(*best_results_list))
        for property in res_list:
            unzipped_results.append(np.asarray(property))

        return best_gen_list, mean_fit_list, std_fit_list, unzipped_results

    #

    #

    #

    #

    # # Re-format mutant into "grouped by gene type" format
    def format_mutant(self, mutant):
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

    # # Generate population in a "grouped by gene type" format
    def gen_pop(self):

        pop = []
        for i in range(self.ParamDict['popsize']):
            member = []
            for gen_group in self.ParamDict['bounds']:
                member.append(np.around(np.random.rand(len(gen_group)), decimals=5))
            #member.append(np.asarray([]))
            pop.append(np.asarray(member, dtype=object))
            #pop.append(member)


        pop_out = np.asarray(pop, dtype=object)
        #print(pop_out.shape)
        #exit()
        return pop_out

    # # produce the denormalised population using the bounds
    def calc_denorm(self, pop_in):

        # # Denormalaise the population according the the bounds
        #pop_denorm = np.around(self.min_b + pop_in * self.diff, decimals=3)  # pop with their real values
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
        if self.GenomeDict['shuffle_gene'] == 1:
            for i in range(len(pop_denorm)):
                pop_denorm[i][self.GenomeDict['SGI']][0] = int(pop_denorm[i][self.GenomeDict['SGI']][0])

        # # Apply Input Weight scheme
        pop_denorm = self.apply_InWeight_scheme(pop_denorm)

        # # Apply output weight scheme
        pop_denorm = self.apply_OutWeight_scheme(pop_denorm)

        # # Apply banding scheme
        pop_denorm = self.apply_Banding_scheme(pop_denorm)

        return np.asarray(pop_denorm, dtype=object)

    # # Apply output weight scheme
    def apply_InWeight_scheme(self, pop):

        if self.GenomeDict['InWeight_gene'] == 0:
            return pop
        elif self.GenomeDict['InWeight_gene'] == 1:
            if self.GenomeDict['InWeight_sheme'] == 'random':
                return pop
            elif self.GenomeDict['InWeight_sheme'] == 'AddSub':
                #if trial == 0:
                # # truncate shuffle gene vale to an integer
                for i in range(len(pop)):
                    for j in range(len(pop[i][self.GenomeDict['in_weight_gene_loc']])):
                        if pop[i][self.GenomeDict['in_weight_gene_loc']][j] >= 0:
                            pop[i][self.GenomeDict['in_weight_gene_loc']][j] = 1
                        else:
                            pop[i][self.GenomeDict['in_weight_gene_loc']][j] = -1
            else:
                print("Error: Invalid Output Weight Scheme")
                raise ValueError('(FunClass): Invalid Output Weight Scheme.')
        return pop

    # # Apply output weight scheme
    def apply_OutWeight_scheme(self, pop):

        if self.GenomeDict['OutWeight_gene'] == 0:
            return pop
        elif self.GenomeDict['OutWeight_gene'] == 1:
            if self.GenomeDict['OutWeight_scheme'] == 'random':
                return pop
            elif self.GenomeDict['OutWeight_scheme'] == 'AddSub':
                # # truncate shuffle gene vale to an integer
                for i in range(len(pop)):
                    for j in range(len(pop[i][self.GenomeDict['out_weight_gene_loc']])):
                        if pop[i][self.GenomeDict['out_weight_gene_loc']][j] >= 0:
                            pop[i][self.GenomeDict['out_weight_gene_loc']][j] = 1
                        else:
                            pop[i][self.GenomeDict['out_weight_gene_loc']][j] = -1
            else:
                print("Error: Invalid Output Weight Scheme")
                raise ValueError('(FunClass): Invalid Output Weight Scheme.')
        return pop

    # # Apply banding scheme
    def apply_Banding_scheme(self, pop):

        if self.GenomeDict['BandClass_gene'] == 0:
            pass
        elif self.GenomeDict['BandClass_gene'] == 1:
            for i in range(len(pop)):
                for j in range(len(pop[i][self.GenomeDict['BandClass_gene_loc']])):
                    pop[i][self.GenomeDict['BandClass_gene_loc']][j] = int(pop[i][self.GenomeDict['BandClass_gene_loc']][j])
        else:
            raise ValueError('(FunClass): Invalid BandClass_gene Scheme.')


        if self.GenomeDict['BandNumber_gene'] == 0:
            pass
        elif self.GenomeDict['BandNumber_gene'] == 1:
            for i in range(len(pop)):
                for j in range(len(pop[i][self.GenomeDict['BandNumber_gene_loc']])):
                    pop[i][self.GenomeDict['BandNumber_gene_loc']][j] = int(pop[i][self.GenomeDict['BandNumber_gene_loc']][j])
        else:
            raise ValueError('(FunClass): Invalid BandNumber_gene Scheme.')

        #print("BandWidth_gene_loc", self.GenomeDict['BandWidth_gene_loc'])
        #print("BandClass_gene_loc", self.GenomeDict['BandClass_gene_loc'])
        #raise ValueError("fin")

        return pop

#

#

# fin
