import pickle
from datetime import datetime


def save_param(ParamDict, experiment, now,
               MG_obj, plotMG, MG_vary_Vconfig, MG_vary_PermConfig,
               exp_num_circuits, exp_num_repetitions,
               MG_vary_Vconfig_NumGraph, MG_vary_PermConfig_NumGraph):

    text_list = []

    text_list.append('Part of a series of experiments: %d' % (experiment))
    if experiment == 1:
        text_list.append('  > Re Use of circuits in subsequent experiments:  %d' % (ParamDict['ReUse_Circuits']))
        text_list.append('  > dir from which saved material models were used:   %s' % (ParamDict['ReUse_dir']))


    text_list.append('\nNumber of loops = %d' % (ParamDict['num_circuits']*ParamDict['num_repetitions']))
    text_list.append('  > Number of generated circuits = %d' % (ParamDict['num_circuits']))
    text_list.append('  > Number of times the generated circuits where tested (num_repetitions) = %d' % (ParamDict['num_repetitions']))

    text_list.append('Iteratiosn per loop = %d' % (ParamDict['its']))
    text_list.append('Population Size = %d' % (ParamDict['popsize']))
    text_list.append('mut = %f' % (ParamDict['mut']))
    if ParamDict['perm_crossp_model'] == 'none':
        text_list.append('crossp = %f   (Note: perm_crossp_model=none)' % (ParamDict['crossp']))
    else:
        text_list.append('crossp = %f' % (ParamDict['crossp']))
        text_list.append('perm_crossp_model = %s' % (ParamDict['perm_crossp_model']))
    text_list.append('num input nodes = %d' % (ParamDict['num_input']))
    text_list.append('num configuration nodes = %d' % (ParamDict['num_config']))
    text_list.append('num output nodes = %d' % (ParamDict['num_output']))
    text_list.append('num output readings (i.e dimensions) = %d' % (ParamDict['num_output_readings']))
    # num output readins, one for each class??

    text_list.append('Training Data used: %s' % (ParamDict['training_data']))
    text_list.append('  > Number of classes to optimise for: %s' % (ParamDict['NumClasses']))
    if ParamDict['TestVerify'] == 0:
        text_list.append('  > All of data is being used for training')
    elif ParamDict['TestVerify'] == 1:
        text_list.append('  > Data is split for training and verification')
        text_list.append('  > Every instance of circuit + repetition is verified (read out in order)')
    text_list.append('Using a shuffle gene = %d' % (ParamDict['shuffle_gene']))

    text_list.append('Using input weightings = %d' % (ParamDict['InWeight_gene']))
    if ParamDict['InWeight_gene'] == 1:
        text_list.append('  > Input weightings scheme is: %s' % (ParamDict['InWeight_sheme']))
        if ParamDict['InWeight_sheme'] == 'random':
            text_list.append('  > Input weightings randomly chosen from +/- %f' % (ParamDict['MaxInputWeight']))
        elif ParamDict['InWeight_sheme'] == 'AddSub':
            text_list.append('  > Input weightings randomly chosen either +1 or -1')

    text_list.append('Using output weightings = %d' % (ParamDict['OutWeight_gene']))
    if ParamDict['OutWeight_gene'] == 1:
        text_list.append('  > Output weightings scheme is: %s' % (ParamDict['OutWeight_scheme']))
        if ParamDict['OutWeight_scheme'] == 'random':
            text_list.append('  > Output weightings randomly chosen from +/- %f' % (ParamDict['MaxOutputWeight']))
        elif ParamDict['OutWeight_scheme'] == 'AddSub':
            text_list.append('  > Output weightings randomly chosen either +1 or -1')

    if ParamDict['EN_Dendrite'] == 1:
        text_list.append('Using %d Dendrites where Y = sum( (Y_i)^i )    from i = 1,2,...,%d' % (ParamDict['NumDendrite'], ParamDict['NumDendrite']))

    text_list.append('Interpretation Scheme is:  %s' % (ParamDict['IntpScheme']))
    text_list.append('Fitness Scheme is:  %s' % (ParamDict['FitScheme']))

    text_list.append('Value of output node shunt resistance = %f kOhm\n' % (ParamDict['shunt_R']))

    if ParamDict['UseCustom_NewAttributeData'] == 1:
        text_list.append('Data being loaded from previously saved data that included an added attribute (the new attribute is the Network Responce i.e boundry weight)')
        text_list.append('  > New Data loaded from:  %s' % (ParamDict['NewAttrDataDir']))

    text_list.append('The Material netowk model used = %s' % (ParamDict['model']))
    if ParamDict['model'] == 'D_RN':
        text_list.append('Where the Default diodes used? = %d' % (ParamDict['DefualtDiode']))
        text_list.append('      Default diodes properties: IS=20@u_uA, RS=0.8@u_mOhm, BV=30@u_V, IBV=200@u_uV, N=1')
    text_list.append('Material Properties are:')
    text_list.append('  > Resistors randomly chosen between [%f, %f]' % (ParamDict['min_r'], ParamDict['max_r']))

    text_list.append('\nNote: the saved data includes the initial population best fitness and genome')
    text_list.append('      So read outputs in chunks of %d\n' % (1 + ParamDict['its']))

    text_list.append('Data Directory is:')
    text_list.append('%s\n' % (ParamDict['SaveDir']))

    # Add details about producing a grid of varied Vconfig graphs
    if MG_vary_Vconfig == 1 and plotMG == 1:
        text_list.append('A variable Vconfig material graph computation was carried out:')
        try:
            text_list.append('  Length x1 data = %d' % (MG_obj.len_x1_VC))
            text_list.append('  Length x2 data = %d' % (MG_obj.len_x2_VC))
            text_list.append('  Vconfig_1 considered = %s' % (str(MG_obj.Vconfig_1)))
            text_list.append('  Vconfig_2 considered = %s' % (str(MG_obj.Vconfig_2)))
            text_list.append('  Total number of chunks to split up = %d\n' % (len(MG_obj.Vconfig_1)*len(MG_obj.Vconfig_2)))
        except:
            text_list.append('  But it failed')

    # Add details about producing a grid of varied input permutations graphs
    if MG_vary_PermConfig == 1 and plotMG == 1:
        text_list.append('A variable Input permutation/organisation material graph computation was carried out:')
        text_list.append('  Length x1 data = %d' % (MG_obj.len_x1_PC))
        text_list.append('  Length x2 data = %d' % (MG_obj.len_x2_PC))
        text_list.append('  Total number of chunks to split up = %d\n' % (MG_obj.PC_num_chunks))

    # append finishing time

    d_string_fin = now.strftime("%d_%m_%Y")
    t_string_fin = now.strftime("%H_%M_%S")
    text_list.append('Finished on %s, at %s' % (d_string_fin, t_string_fin))

    Deets = ''
    for Deets_Line in text_list:
        Deets = '%s%s \n' % (Deets, Deets_Line)

    path_deets = "%s/Details.txt" % (ParamDict['SaveDir'])
    file1 = open(path_deets, "w")
    file1.write(Deets)
    file1.close()

    # Save data to be extracted in analysis
    # this used to be saved as a .csv
    MetaData = ParamDict
    MetaData['num_loops'] = exp_num_circuits*exp_num_repetitions
    MetaData['MG_vary_Vconfig_NumGraph'] = MG_vary_Vconfig_NumGraph
    MetaData['MG_vary_PermConfig_NumGraph'] = MG_vary_PermConfig_NumGraph
    MetaData['exp_num_circuits'] = exp_num_circuits
    MetaData['exp_num_repetitions'] = exp_num_repetitions


    path_MetaData = "%s/Experiment_MetaData.dat" % (ParamDict['SaveDir'])
    with open(path_MetaData, 'wb') as outfile:
        pickle.dump(MetaData, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    return Deets
