# Model Modules

These modules contain the core element which allow the evolutionary process to be performed.

- FunClass runs the evolutionary loop. Generates an initial population, tests there fitnesses, the considers a set of children generated though mutation & recombination.

- Run_Verification takes the best population member after training, and uses the test dataset to find a verification (test) fitness.

- THeModel generates the initial network, and manages the execution of the SPICE network and retrieval of the population fitness.
