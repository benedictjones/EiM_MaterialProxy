# Setting Modules

These modules deal with generation of a temporary settings dictionary which is saved to a file.
This helps reduce Multiprocessing set up time as parameters can be directly loaded into spawned processors.

At the end of a DE run, the most important parameters are saved to text in the results (or/and Experiment) folder,
along with the dictionary itself.

Note: a temporary file called TempDict_SettingsParam.dat will be created in the root directory.
This is maintains the active paramater settings while the program is running.