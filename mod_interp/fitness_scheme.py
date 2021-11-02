# Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn import preprocessing

import sklearn.cluster as cl
import sklearn.metrics as met
import itertools
import scipy.spatial.distance as ds

from math import log2

# import warnings filter
from warnings import simplefilter


''' # NOTES
Fitness evaluation schemes can be:
FitScheme = 'error'     - Defualt, finds mean error from all instances
            'dist'      - Calaculates a value related to net depth of instances
                          into the class (away from the boundary)
            'Err_dist'  - 50/50 weighted split between the above two schemes


            'expansionism' - Function which seeks to push responce values away
                             from each other Calaculates the sum of all pair
                             differences (divided by number of data points).
                             This is UNSUPERVISED!

'''


def centr_dist(centers):
    """
    Find the distance between every combination of centres and return the mean
    """
    iterable = np.arange(len(centers))
    center_pairs = list(itertools.combinations(iterable, 2))

    dists = []
    for pair in center_pairs:
        d = ds.euclidean(centers[pair[0]], centers[pair[1]])
        dists.append(d)

    score = np.mean(dists)

    return score


def dist_per_axis(centers):
    """
    Find the distance between the different dimension between every combination
    of centres.
    This is normalised by dividing by the euclidean/hypotinues, then inversed
    to punish small values of distance on any dimension/axis.
    These are combined as a sqrt or the sum of squares
    """
    iterable = np.arange(len(centers))
    center_pairs = list(itertools.combinations(iterable, 2))

    dists = []
    for pair in center_pairs:

        # exract info from pairing
        d = ds.euclidean(centers[pair[0]], centers[pair[1]])
        axis_diff = np.abs(centers[pair[0]]-centers[pair[1]])

        # using pythag: 1 = (a1/c)^2 + (a2/c)^2 + ... + (aN/c)^2
        nom_axis_diff = (axis_diff/d)**2  # normalise each side by the hypotenuse
        axis_diff_invs = nom_axis_diff**-1  # find the inverse (punish small distances)

        #dists.append(np.mean(axis_diff_invs))
        dists.append(np.sum(axis_diff_invs**2)**0.5)

    dists = np.asarray(dists)
    #print(dists)
    #score = np.sum(dists**2)**0.5  # combine scores
    score = np.mean(dists)  # combine scores
    return score


def equal_dist_per_axis(centers):
    """
    Find the distance between the different dimension between every combination
    of centres.
    This is normalised by dividing by the euclidean/hypotinues.
    A fitness score is then generated which equates to how evenly distributes
    the distances are "spread out" between the different dimensions/axis.
    These are combined as a sqrt or the sum of squares
    """
    iterable = np.arange(len(centers))
    center_pairs = list(itertools.combinations(iterable, 2))

    scores = []
    for pair in center_pairs:

        # exract info from pairing
        d = ds.euclidean(centers[pair[0]], centers[pair[1]])
        axis_diff = np.abs(centers[pair[0]]-centers[pair[1]])

        # using pythag: 1 = (a1/c)^2 + (a2/c)^2 + ... + (aN/c)^2
        nom_axis_diff = (axis_diff/d)**2  # normalise each side by the hypotenuse
        per = (1/len(nom_axis_diff))  # we want each side to be of an equal size, find the nomalisation factor
        closeness_ration = nom_axis_diff/per  # normalise by the desired factor (for equal weightings)
        cr_centred = abs(closeness_ration-1)  # centre around zero
        fit = np.sum(cr_centred)  # sum the values
        fit_nom = fit*per  # use the scale to nomalis eback to a maximum of 1

        scores.append(fit_nom)

    scores = np.asarray(scores)
    print(scores)
    score = np.sum(scores**2)**0.5  # combine scores

    return score








"""
Fitness schemes:
"""

def GetFitness(real_class, predicted_class, responceY, Vout, prm, threshold='na', handle=0):

    ParamDict = prm['DE']
    NetworkDict = prm['network']
    FitScheme = ParamDict['FitScheme']
    # max_OutResponce =  # calc
    class_values = np.unique(real_class)
    #
    class_weights = prm['DE']['ClassWeights']

    weight_array = np.ones(len(real_class))
    for cdx, cls in enumerate(class_values):
        idxs = np.where(real_class == cls)
        weight_array[idxs] = weight_array[idxs]*class_weights[cdx]


    if len(predicted_class) != len(real_class) and str(real_class) != 'na':
        raise ValueError('(TheModel.py): produced class list not same length as real class checking data')

    elif str(real_class) != 'na':

        error = np.asarray(np.abs(predicted_class - real_class))  # calc error matrix, mke it posotive

        # # set errored values to 1, if they are not correct (i.e zero)
        for loc, el in enumerate(error):
            if int(el) != 0:
                error[loc] = 1

        err_fit = np.mean(error)

    else:
        error = np.ones(len(Vout[:,0]))*-1
        err_fit = -1

    # # ignore all future warnings
    # these are generated by sklearn KMeans n_jobs argument
    # need to set n_jobs=1 to avoid strange error where KMeans hangs
    # this seems to happen if a second RunAE is run (which is using multiprocessing)
    # I think this is a clash between python mp (forking) and skleans inbuild mp
    # https://github.com/scikit-learn/scikit-learn/issues/636
    # simplefilter(action='ignore', category=FutureWarning)
    # os.environ["OMP_NUM_THREADS"] = "1"

    if FitScheme == 'error':
        """ Fitness is the raw error """
        fitness = err_fit


    elif FitScheme == 'PCAH':
        """ Maximise PCA computed Entropy """
        scaled_data = preprocessing.scale(Vout)
        pca = PCA() # create a PCA object
        pca.fit(scaled_data) # do the math

        eigenvalues_raw = pca.explained_variance_
        eigenvalues = eigenvalues_raw/np.max(eigenvalues_raw)
        max_entropy = np.log(len(eigenvalues_raw))
        entropy = 0
        for landa in eigenvalues:
            entropy = entropy - (landa*np.log(landa))

        per_H = entropy/max_entropy

        fitness = 1-per_H

    elif FitScheme == 'KmeanDist':
        """
        Apply KMeans Clustering to a raw responce value, then calculate the distance
        between clusters.
        Use this as a fitness metric.
        """

        # Re shape the 1d array into a 2d numpy array
        if len(responceY.shape) == 1:
            responceY = np.reshape(responceY, (responceY.shape[0], 1))
        #model = cl.KMeans(ParamDict['num_classes'], n_jobs=1)
        model = cl.KMeans(ParamDict['num_classes'])

        model.fit(responceY)

        yhat = model.predict(responceY) # assign a cluster to each example

        mean_dist = centr_dist(model.cluster_centers_)
        fitness = 1/mean_dist  # we want a big distance!
        #fitness = 0.5

    elif FitScheme == 'KmeanEqualSpace':
        """
        Apply KMeans Clustering to a raw responce value, then calculate the distance
        between clusters.
        Use this as a fitness metric.
        """

        # Re shape the 1d array into a 2d numpy array
        if len(responceY.shape) == 1:
            responceY = np.reshape(responceY, (responceY.shape[0], 1))

        # model = cl.KMeans(ParamDict['num_classes'], n_jobs=1)
        model = cl.KMeans(ParamDict['num_classes'])
        model.fit(responceY)
        yhat = model.predict(responceY) # assign a cluster to each example

        fitness = equal_dist_per_axis(model.cluster_centers_)

    elif FitScheme == 'KmeanSpace':
        """
        Apply KMeans Clustering to a raw responce value, then calculate the distance
        between clusters.
        Use this as a fitness metric.
        """

        # Re shape the 1d array into a 2d numpy array
        if len(responceY.shape) == 1:
            responceY = np.reshape(responceY, (responceY.shape[0], 1))

        # model = cl.KMeans(ParamDict['num_classes'], n_jobs=1)
        model = cl.KMeans(ParamDict['num_classes'])
        model.fit(responceY)
        yhat = model.predict(responceY) # assign a cluster to each example

        fitness = dist_per_axis(model.cluster_centers_)

    elif FitScheme == 'Vsep':
        """
        Seperateong the "distance" between the average class voltages
        """
        #real_class, predicted_class, responceY, Vout

    elif FitScheme == 'Ysep':
        """
        Seperateong the "distance" between the average class responces
        """
        #real_class, predicted_class, responceY

        if len(class_values) > 2:
            raise ValueError("Ysep can only be used for 2 classes (binary problems)")
        class_values.sort()
        classes_data = []
        classes_mean = []
        classes_std = []

        for cla in class_values:
            class_rY = []
            for instance, yclass in enumerate(real_class):
                if yclass == cla:
                    class_rY.append(responceY[instance])
            class_group = np.asarray(class_rY)

            mean_class_rY = np.mean(class_group)
            std_class_rY = np.std(class_group)

            classes_data.append(class_group)
            classes_mean.append(mean_class_rY)
            classes_std.append(std_class_rY)

        diff = abs(classes_mean[0]-classes_mean[1])
        std = (classes_std[0]**2+classes_std[1]**2)**0.5

        """
        lower_edge = classes_mean[np.argmin(classes_mean)] + classes_std[np.argmin(classes_mean)]
        upper_edge = classes_mean[np.argmax(classes_mean)] + classes_std[np.argmax(classes_mean)]
        print(lower_edge, upper_edge)
        diff = abs(lower_edge-upper_edge)
        fitness = 1/(diff+0.000001)
        """

        print(1/(diff+0.000001), 1/(std+0.000001))

        #fitness = 1/(diff+0.000001) + 1/(std+0.000001)
        fitness = 1/(diff+0.000001) + 1/(classes_std[0]+0.000001)+ 1/(classes_std[1]+0.000001)





    # requires a clustering model handle to be passed in
    elif FitScheme == 'ComScore':
        """
        Take the outpus from clustring applied in the InterpScheme (i.e the
        assigned classes) and compute some fitness.
        Take the completness score as the fitness.

        i.e Scores the InterpScheme clustering effort.
        """

        if len(responceY.shape) == 1:
            responceY = np.reshape(responceY, (responceY.shape[0], 1))
        model = handle
        model.fit(responceY)
        yhat = model.predict(responceY) # assign a cluster to each example

        hom, com, v_mes = met.homogeneity_completeness_v_measure(real_class, yhat)

        per = (hom+com)/2
        fit = 1 - per

        fitness = fit

        #print("homo:", hom, " complt:", com, " perc:", per, " fit:", fitness)

        """error = np.asarray(np.abs(yhat - real_class))  # calc error matrix, mke it posotive
        for loc, el in enumerate(error):
            if int(el) != 0:
                error[loc] = 1
        fitness = np.mean(error)"""

        #fitness = og_com/(com+0.0001)  # ratio to clustering not using the material as a kernal



    elif FitScheme == 'BinCrossEntropy':
        """ Using the sigmoid function emphasises the effect of data points
        close to the boundary, and reduces the effect of points far away
        from it.
        """

        if len(np.shape(responceY)) != 1:
            raise ValueError("Can only have one output node for BinCrossEntropy")


        # # Centre the responce on the threshold!
        Centred_rY = responceY - threshold
        num_instances = len(Centred_rY)

        # # Sigmoid factor to scale curve
        #f = 1  # 80% @ ~1.4, 98% @ ~3.9
        #f = 10  # 80% @ ~0.2, 98% @ ~0.4
        #f = 20  # 80% @ ~0.07, 98% @ ~0.195
        f = prm['DE']['SigmoidCorner']

        # # Group Correct Responces
        correct = (error-1)*Centred_rY  # if error, then set to zero
        correct = correct[np.where(correct != 0)]  # filter out zeros
        G_CrossEntropy = 0
        if len(correct) != 0:
            good_sig_array = 1/(1+np.exp(-abs(correct)*-1*f)) # Find array of sig of each value
            #print(">> OG good sig: ", np.mean(good_sig_array))
            good_sig_array = 1-good_sig_array
            G_CrossEntropy = -np.log(good_sig_array+1e-100)
            G_CrossEntropy = G_CrossEntropy*weight_array[np.where(correct != 0)]
        G_CrossEntropy = np.sum(G_CrossEntropy)


        # # Group InCorrect Responces
        incorrect = (error)*Centred_rY  # if NO error, then set to zero
        incorrect = incorrect[np.where(incorrect != 0)] # filter out zeros
        B_CrossEntropy = 0
        if len(incorrect) != 0:
            bad_sig_array = 1/(1+np.exp(-abs(incorrect)*f)) # Find array of sig of each value
            #print(">> OG bad sig: ", np.mean(bad_sig_array))
            bad_sig_array = 1 - bad_sig_array
            B_CrossEntropy = -np.log(bad_sig_array+1e-100)
            B_CrossEntropy = B_CrossEntropy*weight_array[np.where(incorrect != 0)]
        B_CrossEntropy = np.sum(B_CrossEntropy)

        #print(">>  ", np.mean(good_sig_array), np.mean(bad_sig_array))

        fitness = (G_CrossEntropy + B_CrossEntropy)/num_instances
        #print(">", G_CrossEntropy+B_CrossEntropy, ", mean:", fitness, num_instances)
        #exit()

    elif FitScheme == 'BCE':
        """ Using the sigmoid function emphasises the effect of data points
        close to the boundary, and reduces the effect of points far away
        from it.

        The output weights adjust the Responce (Y) value to make sure the
        sigmoid is being exploited to minimise the entropy.
        """

        if len(np.shape(responceY)) != 1:
            raise ValueError("Can only have one output node for BinCrossEntropy")


        # # Centre the responce on the threshold!
        Centred_rY = responceY - threshold
        num_instances = len(Centred_rY)

        sig_array = 1/(1+np.exp(-Centred_rY)) # Find array of sig of each value

        H_cl1 = -np.log(1 - sig_array[np.where(real_class == 1)])
        H_cl2 = -np.log(sig_array[np.where(real_class == 2)])

        # Add Weighting
        H_cl1 = H_cl1*weight_array[np.where(real_class == 1)]
        H_cl2 = H_cl2*weight_array[np.where(real_class == 2)]

        H_total = np.sum(H_cl1) + np.sum(H_cl2)

        fitness = H_total/num_instances
        #print(">New BCE>", H_total, ", mean:", fitness, num_instances)
        #exit()

    elif FitScheme == 'dist':
        """ Using the sigmoid function emphasises the effect of data points
        close to the boundary, and reduces the effect of points far away
        from it.
        """

        # # Centre the responce on the threshold!
        Centred_rY = responceY - threshold
        num_instances = len(Centred_rY)

        # # Sigmoid factor to scale curve
        #f = 1  # 80% @ ~1.4, 98% @ ~3.9
        #f = 10  # 80% @ ~0.2, 98% @ ~0.4
        #f = 20  # 80% @ ~0.07, 98% @ ~0.195
        f = prm['DE']['SigmoidCorner']

        # # Group Correct Responces
        correct = (error-1)*Centred_rY  # if error, then set to zero
        correct = correct[np.where(correct != 0)]  # filter out zeros
        if len(correct) == 0:
            correct = 0
            good_sig_array = 0
        else:
            good_sig_array = 1/(1+np.exp(-abs(correct)*-1*f)) # Find array of sig of each value

        # # Group InCorrect Responces
        incorrect = (error)*Centred_rY  # if NO error, then set to zero
        incorrect = incorrect[np.where(incorrect != 0)] # filter out zeros
        if len(incorrect) == 0:
            incorrect = 0
            bad_sig_array = 0
        else:
            bad_sig_array = 1/(1+np.exp(-abs(incorrect)*f)) # Find array of sig of each value

        #print(Centred_rY)
        #print(correct)
        #print(incorrect)

        # # Find mean of each abs value
        good_reading = np.mean(abs(correct))
        bad_reading = np.mean(abs(incorrect))
        #print("good_reading", good_reading, " ,bad_reading", bad_reading)



        # # Find sigmoid of the mean raw value
        good_sig = 1/(1+np.exp(-good_reading*f))
        bad_sig = 1/(1+np.exp(-bad_reading*f))
        #print("good_sig", good_sig, ", bad_sig", bad_sig)



        #print("good_sig_array", np.mean(good_sig_array), ", bad_sig_array", np.mean(bad_sig_array))
        #print("good_sig_array2", np.sum(good_sig_array)/num_instances, ", bad_sig_array2", np.sum(bad_sig_array)/num_instances)

        #fit1 = bad_reading + 1/good_reading  # we want smaller fitnesses to be better
        #fit2 = bad_sig + 1/good_sig
        #fit3 = np.mean(good_sig_array)+np.mean(bad_sig_array)
        fit4 = (np.sum(good_sig_array) + np.sum(bad_sig_array))/num_instances
        # print("> Raw Fit:", fit1, ", Sig Fit:", fit2, ", Sig2 Fit:", fit3, ", Sig3 Fit:", fit4)

        #fitness = fitness/max_OutResponce
        fitness = fit4

        #exit()

    elif FitScheme == 'mse':
        """ This emphasises the effect of data points further alway
        from the boundary.
        """

        # # Centre the responce on the threshold!
        Centred_rY = responceY - threshold
        num_instances = len(Centred_rY)

        # # Sigmoid factor to scale curve
        #f = 1  # 80% @ ~1.4, 98% @ ~3.9
        #f = 10  # 80% @ ~0.2, 98% @ ~0.4
        #f = 20  # 80% @ ~0.07, 98% @ ~0.195
        f = prm['DE']['SigmoidCorner']

        # # Group Correct Responces
        correct = (error-1)*Centred_rY  # if error, then set to zero
        correct = correct[np.where(correct != 0)]  # filter out zeros
        if len(correct) == 0:
            correct = 0

        # # Group InCorrect Responces
        incorrect = (error)*Centred_rY  # if NO error, then set to zero
        incorrect = incorrect[np.where(incorrect != 0)] # filter out zeros
        if len(incorrect) == 0:
            incorrect = 0

        # Square
        correct_sq = np.asarray(correct)**2
        incorrect_sq = np.asarray(incorrect)**2

        # Find Mean
        good_reading = np.mean(correct_sq)
        bad_reading = np.mean(incorrect_sq)

        fitness = bad_reading + 1/good_reading

    elif FitScheme == 'none':
        fitness = 0


    #print("\nerror\n", error)
    #print("FitScheme", FitScheme)
    #print("Fitness:", fitness)
    #print("responceY\n", responceY)
    #exit()

    return fitness, err_fit, error

#

#

# fin
