These modules deal with generation of a temporary settings dictionary which is saved to a file. 
This helps reduce Multiprocessing set up time as paramaters can be directly loaded into spawned processors.

At the end of a DE run, the most important paramaters are saved to text in the results (or/and Experiment) folder,
along with the dictionary itself.
