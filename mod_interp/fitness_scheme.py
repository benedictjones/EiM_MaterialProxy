# Import
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn import preprocessing

from mod_load.LoadData import Load_Data

import sklearn.cluster as cl
import sklearn.metrics as met
import itertools
import scipy.spatial.distance as ds

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
    iterable = np.arange(len(centers))
    center_pairs = list(itertools.combinations(iterable, 2))

    dists = []
    for pair in center_pairs:
        d = ds.euclidean(centers[pair[0]], centers[pair[1]])
        dists.append(d)

    score = np.mean(dists)

    return score















"""
Fitness schemes:
"""

def GetFitness(real_class, predicted_class, responceY, Vout, CompiledDict, handle=0, the_data=0):

    ParamDict = CompiledDict['DE']
    NetworkDict = CompiledDict['network']
    FitScheme = ParamDict['FitScheme']
    # max_OutResponce =  # calc

    if ParamDict['FitScheme'] == 'orth':
        er = []
        for row_idx in range(len(real_class[:, 0])):
            for col_idx in range(len(real_class[0, :])):
                er.append(abs(real_class[row_idx, col_idx] - Vout[row_idx, col_idx])/10)
        fitness = np.mean(er)
        return  fitness, fitness, er

    if len(predicted_class) != len(real_class):
       raise ValueError('(TheModel.py): produced class list not same length as real class checking data')
    error = np.asarray(np.abs(predicted_class - real_class))  # calc error matrix, mke it posotive

    # # set errored values to 1, if they are not correct (i.e zero)
    for loc, el in enumerate(error):
        if int(el) != 0:
            error[loc] = 1
    err_fit = np.mean(error)


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
        Apply Clustering to a raw responce value, then calculate the distance
        between clusters.
        Use this as a fitness metric.
        """

        # Re shape the 1d array into a 2d numpy array
        if len(responceY.shape) == 1:
            responceY = np.reshape(responceY, (responceY.shape[0], 1))

        model = cl.KMeans(ParamDict['num_classes'])
        model.fit(responceY)
        yhat = model.predict(responceY) # assign a cluster to each example

        mean_dist = centr_dist(model.cluster_centers_)
        fitness = 1/mean_dist  # we want a big distance!



    # requires a clustering model handle to be passed in
    elif FitScheme == 'ComScore':
        """
        Take the outpus from clustring applied in the InterpScheme (i.e the
        assigned classes) and compute some fitness.
        Take the completness score as the fitness.

        i.e Scores the InterpScheme clustering effort.
        """

        data_X, nn = Load_Data(the_data, CompiledDict)

        if len(responceY.shape) == 1:
            responceY = np.reshape(responceY, (responceY.shape[0], 1))
        model = handle
        model.fit(responceY)
        yhat = model.predict(responceY) # assign a cluster to each example

        hom, com, v_mes = met.homogeneity_completeness_v_measure(real_class, yhat)

        model = cl.KMeans(ParamDict['num_classes'])
        model.fit(data_X)
        yhat_og = model.predict(data_X) # assign a cluster to each example
        og_hom, og_com, og_v_mes = met.homogeneity_completeness_v_measure(real_class, yhat_og)


        per = (hom+com)/2

        og_per = (og_hom+og_com)/2
        og_fit = 1-og_per
        fit = 1 - per

        fitness = fit/og_fit

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

        #print(responceY)

        if np.size(responceY) != np.size(error):
            raise ValueError(" The Error and Responce Y arrays do not match!")

        # filter out errors
        correct = np.asarray((error-1)*responceY)

        # error test
        j = 0
        for val in error:
            tt = (val-1)*responceY[j]
            if correct[j] != tt:
                raise ValueError("Wrong!")
            j += 1

        #correct = correct[~np.all(correct == 0)]*-1
        correct = correct[np.argwhere(correct)]  # select non-zero elements
        if len(correct) == 0:
            correct = np.asarray([0])

        # filter out non-error values
        incorrect = (error)*responceY
        #incorrect = incorrect[~np.all(incorrect == 0)]
        incorrect = incorrect[np.argwhere(incorrect)]  # select non-zero elements
        if len(incorrect) == 0:
            incorrect = np.asarray([0])

        good_reading = abs(correct)
        bad_reading = abs(incorrect)*-1

        good_sig_array = 1/(1+np.exp(-good_reading))
        bad_sig_array = 1/(1+np.exp(-bad_reading))

        CrossEntropy = 0
        for el in good_sig_array.flatten():
            CrossEntropy = CrossEntropy - np.log(el)
            #CrossEntropy = CrossEntropy + el

        for el in bad_sig_array.flatten():
            CrossEntropy = CrossEntropy - np.log(el)
            #CrossEntropy = CrossEntropy + el

        fitness = CrossEntropy



        """print(pos, "len correct", len(correct))
        print(pos, "np.mean(abs(correct))", np.mean(abs(correct)))
        exit()#"""

    elif FitScheme == 'dist':
        """ Using the sigmoid function emphasises the effect of data points
        close to the boundary, and reduces the effect of points far away
        from it.
        """

        #responceY = responceY  # scale it?

        correct = (error-1)*responceY
        correct = correct[~np.all(correct == 0, axis=1)]*-1
        if len(correct) == 0:
            correct = 0

        incorrect = (error)*responceY
        incorrect = incorrect[~np.all(incorrect == 0, axis=1)]
        if len(incorrect) == 0:
            incorrect = 0

        good_reading = np.mean(abs(correct))
        bad_reading = np.mean(abs(incorrect))

        good_sig = 1/(1+np.exp(-good_reading))
        bad_sig = 1/(1+np.exp(-bad_reading))


        #fitness = bad_reading + 1/good_reading  # we want smaller fitnesses to be better
        #fitness = fitness/max_OutResponce
        fitness = bad_sig # + 1/good_sig

        """print(pos, "len correct", len(correct))
        print(pos, "np.mean(abs(correct))", np.mean(abs(correct)))
        exit()#"""


    elif FitScheme == 'mse':
        """ This emphasises the effect of data points further alway
        from the boundary.
        """

        #responceY = responceY  # scale it?

        correct = (error-1)*responceY
        correct = correct[~np.all(correct == 0, axis=1)]*-1
        if len(correct) == 0:
            correct = 0

        incorrect = (error)*responceY
        incorrect = incorrect[~np.all(incorrect == 0, axis=1)]
        if len(incorrect) == 0:
            incorrect = 0

        correct_sq = np.asarray(correct)**2
        incorrect_sq = np.asarray(incorrect)**2

        good_reading = np.mean(correct_sq)
        bad_reading = np.mean(incorrect_sq)

        #good_sig = 1/(1+np.exp(-good_reading))
        #bad_sig = 1/(1+np.exp(-bad_reading))

        # # Don't include correct points?
        # # Just "depth" of bad points?
        fitness = bad_reading # + 1/good_reading


    elif FitScheme == 'mre':
        """ This supresses the effect of data points further alway
        from the boundary.
        """

        #responceY = responceY  # scale it?

        correct = (error-1)*responceY
        correct = correct[~np.all(correct == 0, axis=1)]*-1
        if len(correct) == 0:
            correct = 0
        else:
            correct = np.asarray(abs(correct))**0.5

        incorrect = (error)*responceY
        incorrect = incorrect[~np.all(incorrect == 0, axis=1)]
        if len(incorrect) == 0:
            incorrect = 0
        else:
            incorrect = np.asarray(abs(incorrect))**0.5



        good_reading = np.mean(correct)
        bad_reading = np.mean(incorrect)

        #good_sig = 1/(1+np.exp(-good_reading))
        #bad_sig = 1/(1+np.exp(-bad_reading))


        fitness = bad_reading # + 1/good_reading
        #fitness = fitness/max_OutResponce
        #fitness = bad_sig + 1/good_sig


    elif FitScheme == 'BinCrossEntropy2':


        responceY = (responceY/max_OutResponce)*5  # scale
        real_class = np.asarray(real_class)
        #print("##", pos, " len(real_class)", len(real_class))
        #print("##", pos, " len(responceY)", len(responceY))

        # find cross entropy for class 2's
        Y_class2 = (real_class-1)*responceY
        Y_class2 = Y_class2[~np.all(Y_class2 == 0, axis=1)]
        #print("##", pos, " after len(Y_class2)", len(Y_class2))

        CrossEntropy = []
        #Y_class2 = Y_class2*-1  # needed to make sigmoid correct
        loc = 0
        for el in Y_class2:
            x = el[0]  # /max_OutResponce
            sig = 1/(1+np.exp(-x))  # not doing logistic regression, as we are already about zero, but trying to shift zero
            CrossEntropy.append(-np.log(sig))
            """print(pos, "val:", x)
            print(pos, "sig:", sig)
            print(pos, "cross entropy=", - np.log(sig))
            exit()#"""
            loc = loc + 1


        # find cross entropy for class 1's
        Y_class1 = (real_class-2)*responceY
        Y_class1 = Y_class1[~np.all(Y_class1 == 0, axis=1)]*-1  # multiply by -1 to counter the shift filtering above
        #print("##", pos, " after len(Y_class1)", len(Y_class1))

        Y_class1 = Y_class1*-1
        for el in Y_class1:
            x = el[0]  # /max_OutResponce
            sig = 1/(1+np.exp(-x))  # not doing logistic regression, as we are already about zero, but trying to shift zero
            CrossEntropy.append(-np.log(1-sig))
            """print(pos, "val:", x)
            print(pos, "sig:", sig)
            print(pos, "cross entropy=", - np.log(1-sig))
            exit()#"""

        # fitness is mean cross entropy
        fitness = np.mean(CrossEntropy)
        """print(pos, "len CEnt", len(CrossEntropy))
        print(pos, "fit", fitness)
        print(pos, "CrossEntropy array:\n", CrossEntropy)
        exit()#"""

        exit()

    elif FitScheme == 'none':
        fitness = 0


    #print("\nerror\n", error)
    #print("Fitness:", fitness)
    #print("responceY\n", responceY)
    #exit()

    return fitness, err_fit, error

#

#

# fin
