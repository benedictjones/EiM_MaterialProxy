# SPICE network modules

These modules deal with the set up and running of the SPICE networks.

- resnet.py actually generates the network, which is saved to the results file.
This is normally done only for the first repetition that the SPICE circuit is run for.

- resnet_LoadRun.py can then load and run previously saved SPICE networks, which removes
overhead of having to re-generate them.
