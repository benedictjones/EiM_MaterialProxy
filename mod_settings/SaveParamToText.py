import yaml
from datetime import datetime


def save_param(CompiledDict, experiment, now):

    ParamDict = CompiledDict['DE']
    NetworkDict = CompiledDict['network']
    GenomeDict = CompiledDict['genome']
    SpiceDict = CompiledDict['spice']

    text_list = []

    text_list.append('Part of a series of experiments: %d' % (experiment))
    if experiment == 1:
        text_list.append('  > dir from which saved material models were used:   %s' % (CompiledDict['ReUse_dir']))


    text_list.append('\nNumber of loops = %d' % (CompiledDict['num_circuits']*CompiledDict['num_repetitions']))
    text_list.append('  > Number of generated circuits = %d' % (CompiledDict['num_circuits']))
    text_list.append('  > Number of times the generated circuits where tested (num_repetitions) = %d' % (CompiledDict['num_repetitions']))

    text_list.append('Iteratiosn per loop = %d' % (ParamDict['its']))
    text_list.append('Population Size = %s' % (str(ParamDict['popsize'])))
    text_list.append('Simulation Type --> %s' % (SpiceDict['sim_type']))
    text_list.append('mut = %f' % (ParamDict['mut']))
    if GenomeDict['perm_crossp_model'] == 'none':
        text_list.append('crossp = %f   (Note: perm_crossp_model=none)' % (ParamDict['crossp']))
    else:
        text_list.append('crossp = %f' % (ParamDict['crossp']))
        text_list.append('perm_crossp_model = %s' % (ParamDict['perm_crossp_model']))
    text_list.append('num input nodes = %s' % (str(NetworkDict['num_input'])))
    text_list.append('num configuration nodes = %s' % (str(NetworkDict['num_config'])))
    text_list.append('num output nodes = %s' % (str(NetworkDict['num_output'])))

    if ParamDict['num_readout_nodes'] != 'na' and CompiledDict['ptype'] == 'EiM':
        text_list.append('Using a readout layer')
        text_list.append('   > num readout layer output nodes = %d' % (ParamDict['num_readout_nodes']))

    text_list.append('Training Data used: %s' % (ParamDict['training_data']))
    text_list.append('  > Number of classes to optimise for: %s' % (ParamDict['num_classes']))
    if ParamDict['TestVerify'] == 0:
        text_list.append('  > All of data is being used for training')
    elif ParamDict['TestVerify'] == 1:
        text_list.append('  > Data is split for training and verification')
        text_list.append('  > Every instance of circuit + repetition is verified (read out in order)')
    text_list.append('Using a shuffle gene = %d' % (GenomeDict['shuffle_gene']))

    text_list.append('Using input weightings = %d' % (GenomeDict['InWeight_gene']))
    if GenomeDict['InWeight_gene'] == 1:
        text_list.append('  > Input weightings scheme is: %s' % (GenomeDict['InWeight_sheme']))
        if GenomeDict['InWeight_sheme'] == 'random':
            text_list.append('  > Input weightings randomly chosen from +/- %f' % (GenomeDict['MaxInputWeight']))
        elif GenomeDict['InWeight_sheme'] == 'AddSub':
            text_list.append('  > Input weightings randomly chosen either +1 or -1')

    text_list.append('Using output weightings = %d' % (GenomeDict['OutWeight_gene']))
    if GenomeDict['OutWeight_gene'] == 1:
        text_list.append('  > Output weightings scheme is: %s' % (GenomeDict['OutWeight_scheme']))
        if GenomeDict['OutWeight_scheme'] == 'random':
            text_list.append('  > Output weightings randomly chosen from +/- %f' % (GenomeDict['MaxOutputWeight']))
        elif GenomeDict['OutWeight_scheme'] == 'AddSub':
            text_list.append('  > Output weightings randomly chosen either +1 or -1')

    if SpiceDict['sim_type'] == 'sim_trans_pulse':
        text_list.append('Using variable pulse width = %d' % (GenomeDict['PulseWidth_gene']))
        if GenomeDict['PulseWidth_gene'] == 1:
            text_list.append('  > Bounds are [%f, %f] ms' % (GenomeDict['MinPulseWidth'], GenomeDict['MaxPulseWidth']))
        else:
            text_list.append('  > Pulse Width is %f ms' % (GenomeDict['pulse_Tp']))

    text_list.append('Interpretation Scheme is:  %s' % (ParamDict['IntpScheme']))
    if ParamDict['IntpScheme'][:6] == 'Ridged':
        text_list.append('  > Ridged Alpha = %f' % (ParamDict['RidgedAlpha']))
        text_list.append('  > Ridged Intercept (Bias) = %s' % (str(ParamDict['RidgedIntercept'])))
    text_list.append('Fitness Scheme is:  %s' % (ParamDict['FitScheme']))

    text_list.append('Value of output node shunt resistance = %f kOhm\n' % (SpiceDict['shunt_R']))
    text_list.append('Value of Contact Resistance = %f Ohm\n' % (SpiceDict['contact_Res']))

    if ParamDict['UseCustom_NewAttributeData'] == 1:
        text_list.append('Data being loaded from previously saved data that included an added attribute (the new attribute is the Network Responce i.e boundry weight)')
        text_list.append('  > New Data loaded from:  %s' % (ParamDict['NewAttrDataDir']))

    text_list.append('The Material netowk model used = %s' % (SpiceDict['model']))
    if SpiceDict['model'][0] == 'D':
        text_list.append('Where the Default diodes used? = %d' % (SpiceDict['DefualtDiode']))
        text_list.append('      Default diodes properties: IS=20@u_uA, RS=0.8@u_mOhm, BV=30@u_V, IBV=200@u_uV, N=1')

    text_list.append('Material Properties are:')
    if SpiceDict['model'][:2] == 'NL':
        text_list.append(' > i = aV^2 + bV where:')
        text_list.append(' > Using distributions A~N(%f, %f) & B~N(%f, %f), a & b are selected np.random.multivariate_normal with covariance = %f' % (SpiceDict['a_mean'], SpiceDict['a_var'], SpiceDict['b_mean'], SpiceDict['b_var'], SpiceDict['cov']))
        text_list.append(' > These values then normalized between 0 and 1, before being scaled to between the selected range:')
        text_list.append('  >> a selected between [%f, %f]' % (SpiceDict['material_a_min'], SpiceDict['material_a_max']))
        text_list.append('  >> b selected between [%f, %f]' % (SpiceDict['material_b_min'], SpiceDict['material_b_max']))
    elif SpiceDict['model'][-2:] == 'NN':
        text_list.append('A evolvable Neuromorphic Network was used:')
        text_list.append(' > Number of layers = %d' % SpiceDict['num_layers'])
        text_list.append(' > Number of nodes connecting layers = %d' % SpiceDict['NodesPerLayer'])
        text_list.append(' > Number of config nodes per layer = %d' % SpiceDict['ConfigPerLayer'])
        if SpiceDict['model'][0] == 'R':
            text_list.append('   >> Maximum resistance between nodes = %d kOhm' % SpiceDict['NN_max_r'])

    else:
        text_list.append('  > Resistors randomly chosen between [%f, %f]' % (SpiceDict['min_r'], SpiceDict['max_r']))

    text_list.append('\nNote: the saved data includes the initial population best fitness and genome')
    text_list.append('      So read outputs in chunks of %d\n' % (1 + ParamDict['its']))

    text_list.append('Data Directory is:')
    text_list.append('%s\n' % (CompiledDict['SaveDir']))

    # append finishing time

    d_string_fin = now.strftime("%d_%m_%Y")
    t_string_fin = now.strftime("%H_%M_%S")
    text_list.append('Finished on %s, at %s' % (d_string_fin, t_string_fin))

    Deets = ''
    for Deets_Line in text_list:
        Deets = '%s%s \n' % (Deets, Deets_Line)

    path_deets = "%s/Details.txt" % (CompiledDict['SaveDir'])
    file1 = open(path_deets, "w")
    file1.write(Deets)
    file1.close()

    # Save data to be extracted in analysis
    with open(r'%s/Experiment_MetaData.yaml' % (CompiledDict['SaveDir']), 'w') as sfile:
        yaml.dump(CompiledDict, sfile)

    return Deets
