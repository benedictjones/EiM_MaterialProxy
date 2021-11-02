# Import
import numpy as np
import pickle
import h5py

# import sklearn.cluster as cl
# import sklearn.metrics as met

import sklearn.cluster as cl
from mod_methods.FetchPerm import IndexPerm, reorder


''' # NOTES
Different schemes can be used to interpret the readings (which is
the (weighted) sum of the output voltages).
These include:
    'band_binary' ( or 0 )
    'pn_binary' ( or 1 )

'''


def compute_EiM_ouputs(genome, Vout, data_Y, prm):


    ParamDict = prm['DE']
    NetworkDict = prm['network']
    GenomeDict = prm['genome']

    #print(">", genome)

    # # # # # # # # # # # #
    # Apply Output Weights

    if ParamDict['num_readout_nodes'] == 'na':

        op = Vout

        if GenomeDict['OutWeight']['active'] == 1:
            OW = genome[GenomeDict['OutWeight']['loc']]
            for i in range(len(op[0,:])):
                op[:,i] = op[:,i]*OW[i]

        if GenomeDict['OutBias']['active'] == 1:
            OB = genome[GenomeDict['OutBias']['loc']]
            for i in range(len(op[0,:])):
                op[:,i] = op[:,i] + OB[i]

    else:
        if GenomeDict['OutWeight']['active'] == 1:
            op = np.zeros((len(Vout[:,0]), ParamDict['num_readout_nodes']))
            OW = genome[GenomeDict['OutWeight']['loc']]
            #print(">", OW)
            for out_dim in range(ParamDict['num_readout_nodes']):
                temp = np.zeros(Vout.shape)
                for i in range(len(Vout[0,:])):
                    idx = i + out_dim
                    temp[:,i] = Vout[:,i]*OW[idx]
                temp_sum = np.sum(temp, axis=1)
                temp_final = apply_readout_scheme(ParamDict['readout_scheme'], temp_sum)
                op[:,out_dim] = temp_final

        elif GenomeDict['OutWeight']['active'] == 0:
            op = np.zeros((len(Vout[:,0]), ParamDict['num_readout_nodes']))
            for out_dim in range(ParamDict['num_readout_nodes']):
                temp_sum = np.sum(Vout, axis=1)
                temp_final = apply_readout_scheme(ParamDict['readout_scheme'], temp_sum)
                op[:,out_dim] = temp_final

        # # introduce output bias to each readout node
        if GenomeDict['OutBias']['active'] == 1:
            OB = genome[GenomeDict['OutBias']['loc']]
            for i in range(len(op[0,:])):
                op[:,i] = op[:,i] + OB[i]

    #

    # # # # Schemes # # # #
    # Interpret for band binary
    if prm['DE']['IntpScheme'] == 'band_binary':
        class_out, responceY, handle = binary(genome, op, prm)
    #

    # Interpret for +/- binary
    elif prm['DE']['IntpScheme'] == 'pn_binary' or prm['DE']['IntpScheme'] == 'thresh_binary':
        class_out, responceY, handle = binary(genome, op, prm)
    #

    # Interpret for using varying bands
    elif prm['DE']['IntpScheme'] == 'band':
        class_out, responceY = band(genome, op, prm)

    # Interpret for using highest output wins
    elif prm['DE']['IntpScheme'] == 'HOW':
        class_out, responceY, handle = highest_output_wins(genome, op, prm)

    # if we just want the output retuned with no class assignment
    elif prm['DE']['IntpScheme'] == 'raw':
        class_out = np.ones(len(Vout[:,0]))*-1
        responceY = op
        handle = 0
        #print(">> EiM Raw hit \n", responceY)

    # Interpret for using clustering
    elif prm['DE']['IntpScheme'] == 'Kmean':
        class_out, responceY, handle = cluster(genome, op, prm)

    else:  # i.e input is wrong / not vailable
        raise ValueError('(interpretation_scheme): Invalid scheme %s' % (prm['DE']['IntpScheme']))


    class_out = np.asarray(class_out)
    responceY = np.asarray(responceY)

    """ # To see the op graphs
    if ParamDict['num_readout_nodes'] == 2 and GenomeDict['OutWeight']['active'] == 1:
        Vout = op
    #"""


    return class_out, responceY, Vout, handle

#

#

#

def apply_readout_scheme(scheme, array):

    if scheme == 'none':
        op = array

    elif scheme == 'sigmoid':
        op = 1/(1+np.exp(-array)) - 0.5

    elif scheme == 'sigmoid_scale':
        op = 1/(1+np.exp(-array)) - 0.5
        op = op*10

    return op

#

#

#

def binary(genome, op, prm):

    scheme = prm['DE']['IntpScheme']

    class_out = []
    responceY = []

    for reading in op:
        if np.shape(reading)[0] == 1:
            reading = reading[0]
        elif np.shape(reading)[0] > 1:
            raise ValueError("Use a single readout or output node")
        #print(reading)


        # # Calculate the class output from the reading
        # Note: using bounded decision (nessercary for XOR gates)
        # ALSO! Determine class boundary closeness, 0=close to boundary

        if scheme == 'band_binary':
            if reading > -1 and reading < 1:
                class_out.append(2)
                responceY.append(1 - abs(reading))  # find weight as a decimal
            else:
                class_out.append(1)
                # responceY[loop, out_dim] = (abs(reading[out_dim]) -1)/((MaxOutputWeight*Vmax)-1)  # find weight as a decimal percentage of full range
                responceY.append(-(abs(reading) -1))  # find weight, ensuring opposite class is negative
        elif scheme == 'pn_binary':
            if reading >= 0:
                class_out.append(2)
                responceY.append(reading)
            else:
                class_out.append(1)
                responceY.append(reading)
        elif scheme == 'thresh_binary':
            if reading >= prm['DE']['threshold']:
                class_out.append(2)
                responceY.append(reading)
            else:
                class_out.append(1)
                responceY.append(reading)

    return class_out, responceY, 0

#

#

#

#

#

def band(genome, op,  prm):

    ParamDict = prm['DE']
    NetworkDict = prm['network']
    GenomeDict = prm['genome']

    if GenomeDict['BandNumber']['active'] == 1:
        num_bands = int(genome[GenomeDict['BandNumber']['loc']])
    else:
        num_bands = int(GenomeDict['BandNumber']['max'])

    #print(">> num_bands", num_bands)

    if GenomeDict['BandClass']['active'] == 1:
        BandClass_gene_loc = GenomeDict['BandClass']['loc']
        band_classes = genome[BandClass_gene_loc[0]:BandClass_gene_loc[1]]
        band_classes = band_classes.astype(int)
    else:
        basic_classes = np.arange(GenomeDict['BandClass']['min'], GenomeDict['BandClass']['max']+1)
        band_classes = basic_classes
        while 1:
            if len(band_classes) < num_bands:
                band_classes = np.concatenate((band_classes, basic_classes))
            else:
                break

    #print("min, max is:", ParamDict['min_class_value'], ParamDict['max_class_value'])
    #print("basic_classes =", np.arange(ParamDict['min_class_value'], ParamDict['max_class_value']+1))
    #print("band_classes:", band_classes)
    #raise ValueError("Fin")

    # extract bad edges a & b
    if GenomeDict['BandEdge']['active'] == 1:
        BandEdge_genes = genome[GenomeDict['BandEdge']['loc']]
        a = BandEdge_genes[0]
        b = BandEdge_genes[1]
    else:
        a, b = GenomeDict['BandEdge']['lims']


    # for band positions
    unit_b = (b-a)/num_bands
    unit_offset = unit_b/2

    # see if the boundatry posisitions eveolves
    if GenomeDict['BandWidth']['active'] == 1:
        b_offset = genome[GenomeDict['BandWidth']['loc']]  # determin the bounary offset
    else:
        b_offset = np.zeros(GenomeDict['BandNumber']['max']-1)



    # initialise variables
    class_out = []
    responceY = []

    for reading in op:
        if np.shape(reading)[0] == 1:
            reading = reading[0]
        elif np.shape(reading)[0] > 1:
            raise ValueError("Use a single readout or output node")

        # cycle though bands and assign class
        lowest_b = (a+unit_b+b_offset[0]*unit_offset)
        highest_b = (a + (num_bands-1)*unit_b + b_offset[num_bands-2]*unit_offset)

        if reading < lowest_b:
            #print(reading[out_dim], "<", lowest_b, "So class=", band_classes[0])
            class_out.append(band_classes[0])
            responceY.append(reading)
            #responceY.append(band_classes[0])

        elif reading >= highest_b:
            #print(reading[out_dim], ">=", highest_b,"So class=", band_classes[num_bands-1])
            class_out.append(band_classes[num_bands-1])
            responceY.append(reading)
            #responceY.append(band_classes[num_bands-1])

        elif num_bands > 2:
            for band in range(num_bands):  # sweep accros the bands to check if it is right
                lower_b = (a + (band+1)*unit_b + b_offset[band]*unit_offset)
                upper_b = (a + (band+2)*unit_b + b_offset[band+1]*unit_offset)

                if reading >= lower_b and reading < upper_b:
                    class_out.append(band_classes[band+1])
                    #responceY.append(band_classes[band+1])
                    responceY.append(reading)
                    break

    #print(class_out)

    return class_out, class_out, 0

#

#

#

#


def highest_output_wins(genome, op, prm):
    """
    Simply measure the outputs, using a particular perm (defualt is just the output
    node sequence) assign a class to the highest output.
    """

    prm = prm
    ParamDict = prm['DE']
    NetworkDict = prm['network']
    GenomeDict = prm['genome']

    if ParamDict['num_readout_nodes'] != 'na':
        num_logical_op = ParamDict['num_readout_nodes']
    else:
        num_logical_op = NetworkDict['num_output']

    # create class options
    out_class = np.arange(GenomeDict['BandClass']['min'], GenomeDict['BandClass']['max']+1)
    if len(out_class) != num_logical_op:
        print(out_class)
        raise ValueError("Generated output class array selection is incorrect!")

    # check if output class order is being changed
    if GenomeDict['HOWperm']['active'] == 1:
        HOW_perm_gene_loc = GenomeDict['HOWperm']['loc']
        HOW_perm_gene = genome[HOW_perm_gene_loc]
        order = IndexPerm(num_logical_op, HOW_perm_gene)
        out_class_ordered = reorder(out_class, order)
    else:
        HOW_perm_gene = 0
        out_class_ordered = out_class

    class_out = []
    responceY = []

    for row in op:
        # Find output reading for each output dimension
        #for out_dim in range(num_logical_op):
        #print(np.argmax(row), row)
        indx_largest = np.argmax(row)
        selected_class = out_class_ordered[indx_largest]
        class_out.append(selected_class)
        responceY.append(selected_class)

    return class_out, responceY, 0

#

#

#

#

#

def cluster(genome, op, prm):
    """
    Apply a clustring algorithm to produce a responce which is the assigned
    classes.

    Note: If clustering is to be used as a fitness metric only. Generate raw
    responce values and use clustring in the FitScheme!
    """

    ParamDict = prm['DE']

    class_out = []

    model = cl.KMeans(ParamDict['num_classes'], n_jobs=1)
    model.fit(op)
    yhat = model.predict(op) # assign a cluster to each example

    yhat = yhat + 1

    # retrieve unique clusters
    clusters = np.unique(yhat)
    #print(clusters)


    """# create scatter plot for samples from each cluster
    matplotlib.use('TkAgg')
    num_instances, num_features = np.shape(op)
    print(">>", num_instances, num_features)
    fig = plt.figure()

    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        if cluster == -1:
            plt.scatter(op[row_ix, 0], op[row_ix, 1], color='k')  # error/unkown
        else:
            plt.scatter(op[row_ix, 0], op[row_ix, 1])
    plt.xlabel('Attr 1')
    plt.ylabel('Attr 2')
    plt.show()
    exit()
    #homo = met.homogeneity_score(y, yhat)
    #com = met.completeness_score(y, yhat)
    #plt.title("Agglomerative Clustering\nHomo Schore = %f\nCom Score = %f" %(homo, com))"""

    class_out = yhat
    responceY = class_out
    handle = model

    return class_out, responceY, handle


#

#

#

#

#

# fin
