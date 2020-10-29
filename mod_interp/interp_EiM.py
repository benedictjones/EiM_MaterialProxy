# Import
import numpy as np
import pickle
import h5py

# import sklearn.cluster as cl
# import sklearn.metrics as met

from sklearn.linear_model import Ridge

from mod_methods.FetchPerm import IndexPerm, reorder


''' # NOTES
Different schemes can be used to interpret the readings (which is
the (weighted) sum of the output voltages).
These include:
    'band_binary' ( or 0 )
    'pn_binary' ( or 1 )

'''


def compute_EiM_ouputs(genome, Vout, data_Y, CompiledDict, the_data=0):


    ParamDict = CompiledDict['DE']
    NetworkDict = CompiledDict['network']
    GenomeDict = CompiledDict['genome']

    # # # # # # # # # # # #
    # Apply Output Weights

    if ParamDict['num_readout_nodes'] == 'na':
        if GenomeDict['OutWeight_gene'] == 1:
            op = np.zeros(Vout.shape)
            OW = genome[GenomeDict['out_weight_gene_loc']]
            #print("<", OW)
            for i in range(len(Vout[0,:])):
                op[:,i] = Vout[:,i]*OW[i]

        elif GenomeDict['OutWeight_gene'] == 0:
            op = Vout

    else:
        if GenomeDict['OutWeight_gene'] == 1:
            op = np.zeros((len(Vout[:,0]), ParamDict['num_readout_nodes']))
            OW = genome[GenomeDict['out_weight_gene_loc']]
            #print(">", OW)
            for out_dim in range(ParamDict['num_readout_nodes']):
                temp = np.zeros(Vout.shape)
                for i in range(len(Vout[0,:])):
                    idx = i + out_dim
                    temp[:,i] = Vout[:,i]*OW[idx]
                temp_sum = np.sum(temp, axis=1)
                temp_final = apply_readout_scheme(ParamDict['readout_scheme'], temp_sum)
                op[:,out_dim] = temp_final

        elif GenomeDict['OutWeight_gene'] == 0:
            op = np.zeros((len(Vout[:,0]), ParamDict['num_readout_nodes']))
            for out_dim in range(ParamDict['num_readout_nodes']):
                temp_sum = np.sum(Vout, axis=1)
                temp_final = apply_readout_scheme(ParamDict['readout_scheme'], temp_sum)
                op[:,out_dim] = temp_final

    #

    # # # # Schemes # # # #
    # Interpret for band binary
    if CompiledDict['DE']['IntpScheme'] == 'band_binary':
        class_out, responceY = binary(genome, op, CompiledDict)
    #

    # Interpret for +/- binary
    elif CompiledDict['DE']['IntpScheme'] == 'pn_binary':
        class_out, responceY = binary(genome, op, CompiledDict)
    #

    # Interpret for using varying bands
    elif CompiledDict['DE']['IntpScheme'] == 'band':
        class_out, responceY = band(genome, op, CompiledDict)

    # Interpret for using highest output wins
    elif CompiledDict['DE']['IntpScheme'] == 'HOW':
        class_out, responceY = highest_output_wins(genome, op, CompiledDict)

    # if we just want the output retuned with no class assignment
    elif CompiledDict['DE']['IntpScheme'] == 'raw':
        class_out = np.zeros(len(Vout[:,0]))*-1
        responceY = op
        #print(">> EiM Raw hit ")

    # Interpret for using clustering
    elif CompiledDict['DE']['IntpScheme'] == 'clustering':
        class_out, responceY = cluster(genome, op, CompiledDict)

    else:  # i.e input is wrong / not vailable
        raise ValueError('(interpretation_scheme): Invalid scheme %s' % (CompiledDict['DE']['IntpScheme']))


    class_out = np.asarray(class_out)
    responceY = np.asarray(responceY)

    """ # To see the op graphs
    if ParamDict['num_readout_nodes'] == 2 and GenomeDict['OutWeight_gene'] == 1:
        Vout = op
    #"""


    return class_out, responceY, Vout

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

def binary(genome, op, Dict):

    CompiledDict = Dict

    ParamDict = CompiledDict['DE']
    NetworkDict = CompiledDict['network']
    GenomeDict = CompiledDict['genome']
    scheme = CompiledDict['DE']['IntpScheme']

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


    return class_out, responceY

#

#

#

#

#

def band(genome, op,  Dict):

    CompiledDict = Dict

    ParamDict = CompiledDict['DE']
    NetworkDict = CompiledDict['network']
    GenomeDict = CompiledDict['genome']

    if GenomeDict['BandNumber_gene'] == 1:
        num_bands = int(genome[GenomeDict['BandNumber_gene_loc']])
    else:
        num_bands = int(GenomeDict['max_number_bands'])

    #print(">> num_bands", num_bands)

    if GenomeDict['BandClass_gene'] == 1:
        BandClass_gene_loc = GenomeDict['BandClass_gene_loc']
        band_classes = genome[BandClass_gene_loc[0]:BandClass_gene_loc[1]]
        band_classes = band_classes.astype(int)
    else:
        basic_classes = np.arange(GenomeDict['min_class_value'], GenomeDict['max_class_value']+1)
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
    if GenomeDict['BandEdge_gene'] == 1:
        BandEdge_genes = genome[GenomeDict['BandEdge_gene_loc']]
        a = BandEdge_genes[0]
        b = BandEdge_genes[1]
    else:
        a, b = GenomeDict['BandEdge_lims']


    # for band positions
    unit_b = (b-a)/num_bands
    unit_offset = unit_b/2

    # see if the boundatry posisitions eveolves
    if GenomeDict['BandWidth_gene'] == 1:
        b_offset = genome[GenomeDict['BandWidth_gene_loc']]  # determin the bounary offset
    else:
        b_offset = np.zeros(GenomeDict['max_number_bands']-1)



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

    return class_out, class_out

#

#

#

#

"""
Simply measure the outputs, using a particular perm (defualt is just the output
node sequence) assign a class to the highest output.
"""
def highest_output_wins(genome, op, Dict):

    CompiledDict = Dict
    ParamDict = CompiledDict['DE']
    NetworkDict = CompiledDict['network']
    GenomeDict = CompiledDict['genome']

    if ParamDict['num_readout_nodes'] != 'na':
        num_logical_op = ParamDict['num_readout_nodes']
    else:
        num_logical_op = NetworkDict['num_output']

    # create class options
    out_class = np.arange(GenomeDict['min_class_value'], GenomeDict['max_class_value']+1)
    if len(out_class) != num_logical_op:
        print(out_class)
        raise ValueError("Generated output class array selection is incorrect!")

    # check if output class order is being changed
    if GenomeDict['HOWperm_gene'] == 1:
        HOW_perm_gene_loc = GenomeDict['HOW_perm_gene_loc']
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

    return class_out, responceY

#

#

#

#

#
"""
Simply measure the outputs, using a particular perm (defualt is just the output
node sequence) assign a class to the highest output.
"""
def cluster(genome, op, Dict):


    CompiledDict = Dict
    ParamDict = CompiledDict['DE']
    NetworkDict = CompiledDict['network']
    GenomeDict = CompiledDict['genome']



    class_out = []

    # cluster vout
    # define the model
    model = cl.AgglomerativeClustering(n_clusters=ParamDict['num_classes'])

    # fit model and predict clusters
    yhat = model.fit_predict(op)

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

    return class_out, responceY


#

#

#

#

#

# fin
