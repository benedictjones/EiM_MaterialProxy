# Import
import numpy as np
import time
from tqdm import tqdm
import csv
import h5py
import random

from Module_Models.TheModel import model
from Module_Settings.Set_Load import LoadSettings
from Module_LoadData.AddAttributeToData import AddAttribute
from Module_MG.MaterialGraphs import materialgraphs

class f(object):

    ''' # # # # # # # # # # # # # # #
    Initialisation of class
    '''
    def __init__(self, func=None):
        print("f object initialised...")

        # assign values from the loaded dictionary to self
        # Not running enough times to make significant impact
        ParamDict = LoadSettings()

        # DE param
        self.mut = ParamDict['mut']
        self.crossp = ParamDict['crossp']
        self.popsize = ParamDict['popsize']
        self.its = ParamDict['its']
        self.bounds = ParamDict['bounds']

        self.perm_crossp_model = ParamDict['perm_crossp_model']
        self.save_dir = ParamDict['SaveDir']
        self.repetition_loop = ParamDict['repetition_loop']
        self.circuit_loop = ParamDict['circuit_loop']

        self.shuffle_gene = ParamDict['shuffle_gene']
        self.SGI = ParamDict['SGI']  # Shuffle gene index
        self.InWeight_gene = ParamDict['InWeight_gene']
        self.InWeight_sheme = ParamDict['InWeight_sheme']
        self.OutWeight_gene = ParamDict['OutWeight_gene']
        self.OutWeight_scheme = ParamDict['OutWeight_scheme']
        self.in_weight_gene_loc = ParamDict['in_weight_gene_loc']
        self.config_gene_loc = ParamDict['config_gene_loc']
        self.out_weight_gene_loc = ParamDict['out_weight_gene_loc']


        self.BreakOnZeroFit = ParamDict['BreakOnZeroFit']

        self.GenMG_animation = ParamDict['MG_animation']

        # num cores being used
        self.num_processors = ParamDict['num_processors']

        # extracting boundaries and initialisation
        self.min_b, self.max_b = np.asarray(self.bounds).T  # arrays of the mim/max of each boundary
        self.diff = np.fabs(self.min_b - self.max_b)  # absolute val of each el
        self.best_idx = []
        self.best = []
        self.best_responceY = []

        # for MG saving itteratively (for animation)
        self.prev_best = [0]  # set initial: no prev value

        # Model params
        self.num_config = ParamDict['num_config']
        self.num_input = ParamDict['num_input']
        self.dimensions = ParamDict['dimensions']

        # save data with extra attribute, or plot?
        self.Add_Attribute_To_Data = ParamDict['Add_Attribute_To_Data']

        # initialise the model
        self.model_obj = model()

        # create an MG plotter if makign animation
        if self.GenMG_animation == 1:
            self.temp_MG_obj = materialgraphs(self.model_obj.network_obj, self.GenMG_animation)

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
        # randomly choose 3 indexes without replacement
        selected = np.random.choice(idxs, 3, replace=False)
        a, b, c = pop[selected]  # assign to a variable
        # note this is not the real pop values

        # mutant
        mutant = a + self.mut * (b - c)

        # now the mutant may no longer be within the normalized boundary
        mutant = np.clip(mutant, 0, 1)

        return mutant

    # # # # # # # # # # # # # # # #
    # Random2 mutation method
    def rand2(self, idxs, pop):
        # randomly choose 5 indexes without replacement
        selected = np.random.choice(idxs, 5, replace=False)
        a, b, c, d, e = pop[selected]  # assign to a variable
        # note; a, b etc are genomes

        # mutant
        mutant = a + self.mut * (b - c + d - e)

        # now the mutant may no longer be within the normalized boundary
        mutant = np.clip(mutant, 0, 1)

        return mutant

    # # # # # # # # # # # # # # # #
    # Best1 mutation method
    def best1(self, idxs, pop):

        a = pop[self.best_idx]
        # randomly choose 2 indexes without replacement
        selected = np.random.choice(idxs, 2, replace=False)
        b, c = pop[selected]  # assign to a variable
        # note this is not the real pop values

        # mutant
        mutant = a + self.mut * (b - c)

        # now the mutant may no longer be within the normalized boundary
        mutant = np.clip(mutant, 0, 1)

        return mutant

    ''' # # # # # # # # # # # # # # #
    Recombination & Replacement
    '''

    # # # # # # # # # # # # # # # #
    # Basic binary crossover
    def bin_cross(self, j, pop, mutant):

        # returns true or false for each of the random elements
        cross_points = np.random.rand(self.dimensions) < self.crossp

        # Randomly set a paramater to True to ensure a mutation occurs
        a = np.arange(self.dimensions)
        x = int(random.choice(a))
        cross_points[x] = True

        # Where True, yield x, otherwise yield y np.where(condition,x,y)
        trial = np.where(cross_points, mutant, pop[j])

        trial_denorm = self.calc_denorm(trial, trial=1)
        #trial_denorm = self.min_b + trial * self.diff  # terms of the real val
        # will this be a problem with binary number/pin posistions

        return trial_denorm, trial

    # # # # # # # # # # # # # # # #
    # Binary crossover but with controlled shuffle crossp threshold
    def bin_cross_WCST(self, j, pop, mutant, DE_iter):

        # returns true or false for each of the random elements
        cross_points = np.random.rand(self.dimensions) < self.crossp  # true means choose mutant

        # Randomly set a paramater to True to ensure a mutation occurs
        a = np.arange(self.dimensions)
        if self.perm_crossp_model != 'none' and self.shuffle_gene == 1:
            while 1:
                # not allowed to set cross point for the shuffle Mutation
                # this stops it over riding the perm_crossp_model selected
                x = int(random.choice(a))
                if x != self.SGI:
                    break
        else:
            x = int(random.choice(a))
        cross_points[x] = True

        # Vary crossp for perm gene (only) depending if a model & shuffle is
        # selected, and determin variable cross_point for permutation gene
        # only and replace it into the array
        perm_crossp = self.crossp
        if self.perm_crossp_model == 'linear' and self.shuffle_gene == 1:
            perm_crossp = 1 - DE_iter*(1/self.its)
            cross_points[self.SGI] = np.random.rand() < perm_crossp
        elif self.perm_crossp_model == 'box' and self.shuffle_gene == 1:
            lin_val = 1 - DE_iter*(1/self.its)
            if lin_val > 0.5:
                perm_crossp = 1
            else:
                perm_crossp = 0
            cross_points[self.SGI] = np.random.rand() < perm_crossp
        elif self.perm_crossp_model == 'quad' and self.shuffle_gene == 1:
            perm_crossp = 1 - (DE_iter*(1/self.its))**2
            cross_points[self.SGI] = np.random.rand() < perm_crossp
        elif self.perm_crossp_model == 'unity' and self.shuffle_gene == 1:
            perm_crossp = 1
            cross_points[self.SGI] = np.random.rand() < perm_crossp

        # Where True, yield x, otherwise yield y np.where(condition,x,y)
        trial = np.where(cross_points, mutant, pop[j])

        trial_denorm = self.calc_denorm(trial, trial=1)
        #trial_denorm = self.min_b + trial * self.diff  # terms of the real val
        # will this be a problem with binary number/pin posistions

        # print("trial_denorm:", trial_denorm, "Perm index is:", self.SGI, "perm CR is:", cross_points[self.SGI])

        return trial_denorm, trial

    #

    #

    #

    #

    #

    ''' # # # # # # # # # # # # # # #
    Evaluate the fitness and find the error matrix of a population (of genomes)
    Use multiprocessing to speed up evaluation of the population
    '''
    def Find_Fitness(self, pop_denorm_in, SaveNetText):

        # # Divide the population in n chunks, where n = num_processors
        pop_size = len(pop_denorm_in[:, 0])

        # detemin the number of number of chunks
        if pop_size < self.num_processors:
            num_chunks = pop_size
        else:
            num_chunks = self.num_processors

        # split up the population into chunks
        genome_chunk_list = np.array_split(pop_denorm_in, num_chunks)

        # Evaluate the fitness of each genome
        # Use a try loop to re-atempt if error occurs.
        # >> fitness, error_list, responceY = self.model_obj.run_fobj_mp(genome_chunk_list, pop_size, SaveNetText)
        for attempt in [1,2,3]:
            try:
                fitness, error_list, responceY = self.model_obj.run_fobj_mp(genome_chunk_list, pop_size, SaveNetText)
                break
            except Exception as e:
                time.sleep(0.5)
                print("Failed to run self.model_obj.run_fobj_mp in Find_Fitness (FunClass)")
                print("Attempt", attempt, "failed... \n Re try...")
                if attempt == 3:
                    print("Error (FunClass): Failed to get fitnesses using run_fobj_mp.")
                    print("** Error Thrown is:\n", e)
                else:
                    pass

        return fitness, error_list, responceY
    #

    #

    #

    #

    #

    '''
    ######################################
    DE main
    ######################################
    '''
    def de(self):
        print(" ")

        # create a random population
        pop = np.random.rand(self.popsize, self.dimensions)

        # Scale the random pop by boundry vals
        pop_denorm = self.calc_denorm(pop)

        '''
        # # Print Population
        print("Initial pop:")
        print(pop)
        print("diff", self.diff)
        print("Initial pop denorm:")
        print(pop_denorm)
        # # '''

        # evaluating the fitness of the initial population
        print(" ")
        print("Evaluate initial population...")

        # find the population fitness using the SPICE simulation
        fitness, error_list, responceY = self.Find_Fitness(pop_denorm, 1)

        # # # # # # # # # # # # # # # #
        # Determine best geneome and fitness from initial pop, and return them
        self.best_idx = np.argmin(fitness)  # gives index of the smallest val
        self.best = pop_denorm[self.best_idx]  # remeber this is list of arrays
        self.best_error = error_list[self.best_idx]
        self.best_responceY = responceY[self.best_idx]

        # Send back values
        yield self.best, fitness[self.best_idx], np.mean(fitness), np.std(fitness)
        print("Fitness of initial population:", fitness)

        # # Produce best genome MG and save array to file
        if self.GenMG_animation == 1:
            best_res = [self.best, fitness[self.best_idx]]
            self.temp_MG_obj.MG(0, best_res, it=0, GenMG_animation=self.GenMG_animation)
            self.prev_best = best_res

        # # # # # # # # # # # # # # # #
        # Start Evolutionary Loop
        print(" ")
        print("Start Evolutionary Loop... ")

        for i in range(self.its):  # loop for the number of itterations
            tic = time.time()

            ''' Loop over the whole population - apply mutation and
            recombination before assessing whether the parent/target should
            be replaced with the child/trial '''

            # Create all trial_mutants in a list
            trial_denorm_list = []
            trial_list = []
            for j in range(self.popsize):

                # creates an range array(0, popsize) but excludes the current
                # value of j, used to randomly select pop involved in mutation
                idxs = [idx for idx in range(self.popsize) if idx != j]
                # i.e indx is all pop index's except the current one

                # # Mutation
                mutant = self.rand2(idxs, pop)

                # # Recombination & Replacement
                trial_denorm, trial = self.bin_cross_WCST(j, pop, mutant, i)

                trial_denorm_list.append(trial_denorm)
                trial_list.append(trial)

            ''' After generating our new child/trial vectors, we evaluate it to
            measure how good it is. If this mutant is better than the
            current vector (pop[j]) then we replace it with the new one.'''


            # find the population fitness using the SPICE simulation
            fi = []
            error_list = []
            trial_denorm_array = np.array(trial_denorm_list)
            fi, error_list, responceY = self.Find_Fitness(trial_denorm_array, 0)  # needs a np.array input

            # # Compare the children/trial genomes to the parent/target
            for j in range(self.popsize):
                #print("compare fi", j, "fi=", fi[j], "to previous fitness=", fitness[j])
                #print("     For Genome:", str(np.around(trial_denorm_array[j], decimals=3)))
                if fi[j] <= fitness[j]:  # whether to keep parent or child/trial
                    fitness[j] = fi[j]
                    pop[j] = trial_list[j]
                if fi[j] <= fitness[self.best_idx]:  # assign best genome/pop
                    self.best_idx = j
                    self.best = trial_denorm_list[j]
                    self.best_error = error_list[j]
                    self.best_responceY = responceY[j]


            # # # # # # # # # # # # # # # #
            # Print loop infomation
            toc = time.time()
            print("Itteration:", i, "best fitness:", fitness[self.best_idx], "  Execution time:", toc - tic)
            print("  Evolved pop fitness", np.around(fitness, decimals=4))
            # print("best:", self.best)
            # print("best_idx:", self.best_idx)
            # print("best fitness:", fitness[self.best_idx])
            # print("fitness:", fitness)
            # print("Best error:")
            # print(self.best_error)
            # print(" ")

            # # Send back values to DE.py
            yield self.best, fitness[self.best_idx], np.mean(fitness), np.std(fitness)

            # # Send back all vals so an animation can be produced
            # yield self.min_b + pop * self.diff, fitness, self.best_idx

            # # Produce best genome MG and save array to file
            if self.GenMG_animation == 1:

                best_res = [self.best, fitness[self.best_idx]]
                #print(">>>best_res", best_res)
                """print("self.prev_best[0]", self.prev_best[0])
                print("sum(best_res[0])", sum(best_res[0]))
                print("sum(self.prev_best[0])", sum(self.prev_best[0]))
                """
                if sum(best_res[0]) == sum(self.prev_best[0]):
                    # load prev and Copy
                    ani_file_dir = "%s/MG_AniData/%d_Rep%d.hdf5" % (self.save_dir, self.circuit_loop, self.repetition_loop)

                    with h5py.File(ani_file_dir, 'r') as hdf:
                        prev_dataset = 'MG_dat_it_%d' % (i)
                        dataset = hdf.get(prev_dataset)
                        prev_data = np.array(dataset)

                        prev_best_fit = dataset.attrs['best_fit']
                        prev_best_genome = dataset.attrs['best_genome']


                    with h5py.File(ani_file_dir, 'a') as hdf:
                        new_dataset = 'MG_dat_it_%d' % (i+1)
                        Y_dat = hdf.create_dataset(new_dataset, data=prev_data)
                        Y_dat.attrs['best_fit'] = prev_best_fit
                        Y_dat.attrs['best_genome'] = prev_best_genome

                    """
                    prev_file = "%s/MG_AniData/%d_Rep%d/it_%d.npy" % (self.save_dir, self.circuit_loop, self.repetition_loop, i)
                    old_data = np.load(prev_file)
                    new_file = "%s/MG_AniData/%d_Rep%d/it_%d.npy" % (self.save_dir, self.circuit_loop, self.repetition_loop, i+1)
                    np.save(new_file, old_data)"""
                else:
                    # generate MG
                    self.temp_MG_obj.MG(0, best_res, it=i+1, GenMG_animation=self.GenMG_animation)

                    self.prev_best = best_res

            # # Break loop if ideal Genome is found
            if self.BreakOnZeroFit == 1:
                if fitness[self.best_idx] == 0:
                    break

        # # Once the DE is finished
        if self.Add_Attribute_To_Data == 1:
            AddAttribute('train', self.best_responceY)

        if self.GenMG_animation == 1:
            del self.temp_MG_obj


    #

    #

    #

    #

    #

    # # produce the denormalised population using the bounds
    def calc_denorm(self, pop_in, trial=0):

        # # Denormalaise the population according the the bounds
        pop_denorm = np.around(self.min_b + pop_in * self.diff, decimals=3)  # pop with their real values

        # # truncate shuffle gene vale to an integer
        if self.shuffle_gene == 1:
            if trial == 0:
                pop_denorm[:, self.SGI] = pop_denorm[:, self.SGI].astype(int)  # for a matrix of gemomes (i.e a population)
            else:
                pop_denorm[self.SGI] = pop_denorm[self.SGI].astype(int)  # for only a sinly genome

        # # Apply Input Weight scheme
        pop_with_in_weight = self.apply_InWeight_scheme(pop_denorm, trial=trial)

        # # Apply output weight scheme
        output_pop = self.apply_OutWeight_scheme(pop_with_in_weight, trial)

        return output_pop

    # # Apply output weight scheme
    def apply_InWeight_scheme(self, pop_in, trial=0):

        if self.InWeight_gene == 0:
            return pop_in
        elif self.InWeight_gene == 1:

            if self.InWeight_sheme == 'random':
                return pop_in
            elif self.InWeight_sheme == 'AddSub':
                pop_out = pop_in
                if trial == 0:
                    for i in range(self.in_weight_gene_loc[0], self.in_weight_gene_loc[1]):
                        for row in range(len(pop_in[:, 0])):
                            if pop_in[row, i] >= 0:
                                pop_out[row, i] = 1
                            elif pop_in[row, i] < 0:
                                pop_out[row, i] = -1

                else:
                    for i in range(self.in_weight_gene_loc[0], self.in_weight_gene_loc[1]):
                        if pop_in[i] >= 0:
                            pop_out[i] = 1
                        elif pop_in[i] < 0:
                            pop_out[i] = -1
            else:
                print("Error: Invalid Output Weight Scheme")
                raise ValueError('(FunClass): Invalid Output Weight Scheme.')
        return pop_out

    # # Apply output weight scheme
    def apply_OutWeight_scheme(self, pop_in, trial=0):

        if self.OutWeight_gene == 0:
            return pop_in
        elif self.OutWeight_gene == 1:

            if self.OutWeight_scheme == 'random':
                return pop_in
            elif self.OutWeight_scheme == 'AddSub':
                pop_out = pop_in
                if trial == 0:
                    for i in range(self.out_weight_gene_loc[0], self.out_weight_gene_loc[1]):
                        for row in range(len(pop_in[:, 0])):
                            if pop_in[row, i] >= 0:
                                pop_out[row, i] = 1
                            elif pop_in[row, i] < 0:
                                pop_out[row, i] = -1

                else:
                    for i in range(self.out_weight_gene_loc[0], self.out_weight_gene_loc[1]):
                        if pop_in[i] >= 0:
                            pop_out[i] = 1
                        elif pop_in[i] < 0:
                            pop_out[i] = -1
            else:
                print("Error: Invalid Output Weight Scheme")
                raise ValueError('(FunClass): Invalid Output Weight Scheme.')
        return pop_out
#

#

# fin
