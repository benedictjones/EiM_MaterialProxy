# Import
import numpy as np
import matplotlib.pyplot as plt

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



def GetFitness(the_class, error, responceY, scheme,
               num_output_readings, max_OutResponce, pos):

    # # Format error array
    error = np.abs(error)  # make it posotive
    error = np.asarray(error)

    # # check for errors
    for el in error:
        if el != 0 and el != 1:
            #print("Error (TheModel/fitness_scheme): Illegal value for error:", el)
            #print("element type:", type(el))
            #print("error fit:\n", np.mean(error))
            #exit()
            raise ValueError('(TheModel/fitness_scheme): Illegal value for error')


    # find erro fitness (i.e mean or error) of 1d or 2d array
    err_fit = np.mean(error)

    if scheme == 'error':
        fitness = err_fit
        #print(np.asarray(responceY))

    elif scheme == 'dist':
        """ Using the sigmoid function emphasises the effect of data points
        close to the boundary, and reduces the effect of points far away
        from it.
        """
        responceY = np.asarray(responceY)
        responceY = responceY  # scale it?

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

    elif scheme == 'mse':
        """ This emphasises the effect of data points further alway
        from the boundary.
        """
        responceY = np.asarray(responceY)
        responceY = responceY  # scale it?

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


    elif scheme == 'mre':
        """ This supresses the effect of data points further alway
        from the boundary.
        """
        responceY = np.asarray(responceY)
        responceY = responceY  # scale it?

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

    elif scheme == 'punisher':
        """ This emphasises the effect of data points further alway
        from the boundary if they are bad/wrong (using a mse), but suppressess the
        effect of point swhich are far into the correct class (using a sigmoid).
        """
        responceY = np.asarray(responceY)
        responceY = responceY  # scale it?

        correct = (error-1)*responceY
        correct = correct[~np.all(correct == 0, axis=1)]*-1
        if len(correct) == 0:
            correct = 0
        else:
            correct = np.asarray(correct)

        incorrect = (error)*responceY
        incorrect = incorrect[~np.all(incorrect == 0, axis=1)]
        if len(incorrect) == 0:
            incorrect = 0
        else:
            incorrect = np.asarray(incorrect)**2

        good_reading = np.mean(abs(correct))
        good_sig = 1/(1+np.exp(-good_reading))
        bad_reading = np.mean(abs(incorrect))

        fitness = bad_reading + 1/good_sig


    elif scheme == 'punisher2':
        """ This emphasises the effect of data points further alway
        from the boundary if they are bad/wrong (using a mse), but suppressess the
        effect of point swhich are far into the correct class (using a sigmoid).
        """
        responceY = np.asarray(responceY)
        responceY = responceY  # scale it?

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
            incorrect = np.asarray(incorrect)**2

        good_reading = np.mean(abs(correct))
        bad_reading = np.mean(abs(incorrect))

        fitness = bad_reading + 1/good_reading


    elif scheme == 'BinCrossEntropy':
        responceY = np.asarray(responceY)
        responceY = (responceY/max_OutResponce)*5  # scale
        the_class = np.asarray(the_class)
        #print("##", pos, " len(the_class)", len(the_class))
        #print("##", pos, " len(responceY)", len(responceY))

        # find cross entropy for class 2's
        Y_class2 = (the_class-1)*responceY
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
        Y_class1 = (the_class-2)*responceY
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

        if pos == 1:

            print("Error is:\n", error)

            fig = plt.figure()
            plt.plot(Y_class2, CrossEntropy[:loc], 'r*')
            plt.plot(Y_class1, CrossEntropy[loc:], 'bo')
            plt.title("%d" % (pos))
            plt.xlabel("Weight (i.e Y)")
            plt.ylabel("Entropy ")

            fig = plt.figure()
            plt.plot(error, responceY, 'x')
            plt.title("class vs weight for : %d" % (pos))
            plt.ylabel("Weight (i.e Y)")
            plt.xlabel("Class")

            plt.show()
        exit()

    elif scheme == 'Err_dist':

        # find error fitness
        err_fit = sum(sum(np.abs(error)))/(len(error)*num_output_readings)

        # find distance fitess
        responceY = np.asarray(responceY)
        responceY = responceY  # scale it?

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


        #dist_fit = bad_reading + 1/good_reading  # we want smaller fitnesses to be better
        #dist_fit = dist_fit/max_OutResponce
        dist_fit = bad_sig + 1/good_sig


        fitness = err_fit + 0.1*dist_fit

    elif scheme == 'expansionism':
        """ Function which seeks to push responce values away from each other
        Calaculates the sum of all pair differences (divided by number
        of data points).
        > Pushes boundary far away from data i.e not great <
        """
        responceY = np.asarray(responceY)
        Y = responceY  # scale it?

        # # Find the sum of all pairs
        diffs = []
        cont = 0
        for i, x in enumerate(Y):
            for j, y in enumerate(Y):
                if i != j:
                    diffs.append(abs(x-y))
                    cont += 1

        sum_diffs =  float(sum(diffs)/2)
        sum_diffs = sum_diffs/cont  # reduce down depending on the data set size

        # larger dispersion is better
        fitness = 1/sum_diffs  # + err_fit



    #print("error\n", error)
    #print("responceY\n", responceY)
    #exit()

    return fitness, err_fit

#

#

# fin
